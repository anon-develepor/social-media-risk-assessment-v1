from __future__ import annotations

from pii_risk.pii.detector import PIIType, detect_pii_spans
from pii_risk.pii.scoring import score_record


HIGH_SEVERITY_TYPES = {PIIType.SSN, PIIType.CREDIT_CARD}


def weak_label_from_rules(text: str) -> dict:
    """Create weak supervision labels from rule-based PII detection.

    This label represents explicit or likely PII exposure in text, not intent or
    correctness. The definition is centralized here for consistency across
    training and evaluation.
    """
    scoring = score_record(text)
    spans = detect_pii_spans(text)
    pii_types = sorted({span.type for span in spans})
    rule_score = int(scoring["score"])
    y_risk = int(rule_score >= 25 or any(t in HIGH_SEVERITY_TYPES for t in pii_types))

    return {
        "pii_types": pii_types,
        "rule_score": rule_score,
        "y_risk": y_risk,
    }
