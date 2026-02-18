from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from pii_risk.data.loader import iter_parquet_records
from pii_risk.labels.weak import weak_label_from_rules
from pii_risk.ml.predict import predict_risk
from pii_risk.pii.detector import detect_pii_spans, redact_text


BUCKET_LABELS = ("TP", "FP", "TN", "FN")


def bucket(pred: int, y: int) -> str:
    if pred == 1 and y == 1:
        return "TP"
    if pred == 1 and y == 0:
        return "FP"
    if pred == 0 and y == 0:
        return "TN"
    return "FN"


def _normalize_models_dir(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_file():
        return path.parent
    return path


def audit_records(
    input_dir: str,
    model_path: str,
    out_path: str,
    max_rows: int | None = None,
    seed: int = 0,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    models_dir = _normalize_models_dir(model_path)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bucket_counts = {label: 0 for label in BUCKET_LABELS}
    total_rows = 0
    p_risk_values: list[float] = []

    fieldnames = [
        "record_id",
        "created_at",
        "bucket",
        "y_risk",
        "pred_risk",
        "p_risk",
        "rule_score",
        "pii_types",
        "text",
        "redacted_text",
        "community",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in iter_parquet_records(input_dir, max_rows=max_rows):
            text = record.get("text", "")
            label = weak_label_from_rules(text)
            ml_result = predict_risk(text, models_dir=models_dir)
            p_risk = float(ml_result["p_risk"])
            pred = 1 if p_risk >= 0.5 else 0
            y = int(label["y_risk"])
            bucket_str = bucket(pred, y)
            spans = detect_pii_spans(text)
            redacted = redact_text(text, spans)

            pii_types = label.get("pii_types", [])
            pii_str = "|".join(sorted({str(pii) for pii in pii_types}))

            community = record.get("community") or ""

            writer.writerow(
                {
                    "record_id": record.get("record_id", ""),
                    "created_at": record.get("created_at", ""),
                    "bucket": bucket_str,
                    "y_risk": y,
                    "pred_risk": pred,
                    "p_risk": p_risk,
                    "rule_score": int(label["rule_score"]),
                    "pii_types": pii_str,
                    "text": text,
                    "redacted_text": redacted,
                    "community": community,
                }
            )

            bucket_counts[bucket_str] += 1
            total_rows += 1
            p_risk_values.append(p_risk)

    mean_p_risk = float(np.mean(p_risk_values)) if p_risk_values else 0.0

    print(f"total_rows: {total_rows}")
    for label in BUCKET_LABELS:
        print(f"{label}: {bucket_counts[label]}")
    print(f"mean_p_risk: {mean_p_risk:.4f}")

    return {
        "total_rows": total_rows,
        "bucket_counts": bucket_counts,
        "mean_p_risk": mean_p_risk,
    }
