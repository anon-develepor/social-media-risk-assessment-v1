from __future__ import annotations

from collections import Counter

from pii_risk.pii.detector import PIISpan, PIIType, detect_pii_spans


WEIGHTS: dict[str, int] = {
    PIIType.SSN: 60,
    PIIType.CREDIT_CARD: 60,
    PIIType.EMAIL: 25,
    PIIType.PHONE: 25,
    PIIType.DOB: 15,
    PIIType.ADDRESS_HINT: 15,
    PIIType.IPV4: 10,
    PIIType.URL: 5,
}


def _build_explanation(counts: Counter[str]) -> str:
    if not counts:
        return "No PII detected."

    ranked = sorted(
        counts.items(),
        key=lambda item: (WEIGHTS.get(item[0], 0), item[1]),
        reverse=True,
    )
    top_types = [item[0] for item in ranked[:2]]
    if len(top_types) == 1:
        return f"Detected {top_types[0]} indicators."
    return f"Detected {top_types[0]} and {top_types[1]} indicators."


def score_record(text: str | None) -> dict:
    spans = detect_pii_spans(text)
    counts = Counter(span.type for span in spans)
    score = 0
    for span_type, count in counts.items():
        score += WEIGHTS.get(span_type, 0) * count

    score = min(score, 100)

    return {
        "score": score,
        "findings": spans,
        "counts_by_type": dict(counts),
        "explanation": _build_explanation(counts),
    }
