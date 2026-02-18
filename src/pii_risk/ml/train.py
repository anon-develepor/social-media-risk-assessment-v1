from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from pii_risk.data.loader import iter_parquet_records
from pii_risk.labels.weak import weak_label_from_rules
from pii_risk.ml.features import (
    NUMERIC_FEATURE_NAMES,
    build_numeric_features,
    fit_vectorizer,
)


def _split_by_time(records: list[dict]) -> tuple[list[dict], list[dict]]:
    sorted_records = sorted(records, key=lambda item: item.get("created_at", ""))
    split_index = int(len(sorted_records) * 0.8)
    return sorted_records[:split_index], sorted_records[split_index:]


def _prepare_features(texts: list[str], vectorizer) -> csr_matrix:
    numeric = build_numeric_features(texts)
    tfidf = vectorizer.transform(texts)
    return hstack([csr_matrix(numeric), tfidf])


def _print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"accuracy: {accuracy:.3f}")
    print(f"precision: {precision:.3f}")
    print(f"recall: {recall:.3f}")
    print(f"f1: {f1:.3f}")
    print(f"confusion_matrix: tn={tn} fp={fp} fn={fn} tp={tp}")


def train_model(
    input_dir: str, max_rows: int | None = None, models_dir: Path | None = None
) -> dict:
    records = list(iter_parquet_records(input_dir, max_rows=max_rows))
    if not records:
        raise ValueError("No valid records found to train on.")

    train_records, test_records = _split_by_time(records)
    if not train_records or not test_records:
        raise ValueError("Insufficient records for train/test split.")

    train_texts = [record["text"] for record in train_records]
    test_texts = [record["text"] for record in test_records]

    vectorizer = fit_vectorizer(train_texts)
    x_train = _prepare_features(train_texts, vectorizer)
    x_test = _prepare_features(test_texts, vectorizer)

    y_train = np.array([weak_label_from_rules(text)["y_risk"] for text in train_texts])
    y_test = np.array([weak_label_from_rules(text)["y_risk"] for text in test_texts])

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    _print_metrics(y_test, y_pred)

    if models_dir is None:
        models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    with (models_dir / "pii_risk_model.pkl").open("wb") as f:
        pickle.dump(model, f)

    with (models_dir / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)

    metadata = {
        "label_rule": "y_risk=1 if rule_score>=25 or PII types include SSN/CREDIT_CARD; else 0",
        "feature_groups": {
            "numeric": NUMERIC_FEATURE_NAMES,
            "tfidf": {
                "type": "word",
                "min_df": vectorizer.min_df,
                "max_df": vectorizer.max_df,
                "lowercase": vectorizer.lowercase,
                "stop_words": "english",
            },
        },
        "train_test_split": "sorted by created_at, last 20% as test",
    }

    with (models_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return {
        "model_path": str(models_dir / "pii_risk_model.pkl"),
        "vectorizer_path": str(models_dir / "vectorizer.pkl"),
        "metadata_path": str(models_dir / "metadata.json"),
    }
