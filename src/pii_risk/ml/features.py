from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pii_risk.pii.detector import PIIType, detect_pii_spans


NUMERIC_FEATURE_NAMES = [
    "length_chars",
    "length_words",
    "count_digits",
    "num_pii_spans",
    "num_unique_pii_types",
    "count_emails",
    "count_phones",
    "count_urls",
]

_VECTORIZER: TfidfVectorizer | None = None


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _numeric_features_for_text(text: str) -> list[float]:
    spans = detect_pii_spans(text)
    types = [span.type for span in spans]
    count_emails = sum(1 for t in types if t == PIIType.EMAIL)
    count_phones = sum(1 for t in types if t == PIIType.PHONE)
    count_urls = sum(1 for t in types if t == PIIType.URL)

    return [
        float(len(text)),
        float(_count_words(text)),
        float(sum(char.isdigit() for char in text)),
        float(len(spans)),
        float(len(set(types))),
        float(count_emails),
        float(count_phones),
        float(count_urls),
    ]


def build_numeric_features(texts: Iterable[str]) -> np.ndarray:
    return np.array([_numeric_features_for_text(text) for text in texts], dtype=float)


def fit_vectorizer(texts: list[str]) -> TfidfVectorizer:
    global _VECTORIZER

    # Make TF-IDF robust for tiny test corpora:
    # sklearn requires min_df <= number of documents.
    n_docs = max(1, len(texts))
    min_df = 5 if n_docs >= 20 else 1

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=0.8,
        lowercase=True,
        stop_words="english",
    )
    vectorizer.fit(texts)
    _VECTORIZER = vectorizer
    return vectorizer


def transform_texts(texts: list[str]):
    if _VECTORIZER is None:
        raise RuntimeError("Vectorizer has not been fit. Call fit_vectorizer first.")
    return _VECTORIZER.transform(texts)
