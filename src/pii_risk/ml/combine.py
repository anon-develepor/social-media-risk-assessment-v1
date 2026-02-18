from __future__ import annotations


MARGIN = 10


def combined_score(rule_score: int, p_risk: float) -> dict:
    ml_score = round(p_risk * 100)
    final_score = max(rule_score, ml_score)

    if rule_score > ml_score + MARGIN:
        interpretation = "explicit_pii_dominant"
    elif ml_score > rule_score + MARGIN:
        interpretation = "contextually_concerning"
    else:
        interpretation = "ambiguous_context"

    return {
        "rule_score": rule_score,
        "ml_score": ml_score,
        "final_score": final_score,
        "interpretation": interpretation,
    }
