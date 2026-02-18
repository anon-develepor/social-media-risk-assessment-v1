from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack

from pii_risk.ml.features import NUMERIC_FEATURE_NAMES, build_numeric_features


def _load_artifacts(models_dir: Path) -> tuple[object, object]:
    with (models_dir / "pii_risk_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (models_dir / "vectorizer.pkl").open("rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def _top_terms(
    tfidf_vector,
    tfidf_coefficients: np.ndarray,
    feature_names: np.ndarray,
    limit: int = 5,
) -> list[str]:
    if tfidf_vector.nnz == 0:
        return []
    contributions = tfidf_vector.multiply(tfidf_coefficients)
    contributions_array = contributions.toarray().ravel()
    positive_indices = np.where(contributions_array > 0)[0]
    if positive_indices.size == 0:
        return []
    ranked = positive_indices[np.argsort(contributions_array[positive_indices])[::-1]]
    top_indices = ranked[:limit]
    return [str(feature_names[idx]) for idx in top_indices]


def predict_risk(text: str, models_dir: Path | None = None) -> dict:
    if models_dir is None:
        models_dir = Path("models")

    model, vectorizer = _load_artifacts(models_dir)
    numeric = build_numeric_features([text])
    tfidf_vector = vectorizer.transform([text])
    features = hstack([csr_matrix(numeric), tfidf_vector])
    proba = model.predict_proba(features)[0][1]

    coefficients = model.coef_[0]
    tfidf_coefficients = coefficients[len(NUMERIC_FEATURE_NAMES) :]
    feature_names = vectorizer.get_feature_names_out()
    top_terms = _top_terms(tfidf_vector, tfidf_coefficients, feature_names)

    return {"p_risk": float(proba), "top_terms": top_terms}
