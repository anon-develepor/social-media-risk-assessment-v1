from pii_risk.labels.weak import weak_label_from_rules


def test_weak_labeling_benign_text() -> None:
    label = weak_label_from_rules("Just sharing a project update with no sensitive data.")
    assert label["y_risk"] == 0
    assert label["rule_score"] == 0
    assert label["pii_types"] == []


def test_weak_labeling_explicit_pii() -> None:
    label = weak_label_from_rules("My SSN is 123-45-6789, please keep it private.")
    assert label["y_risk"] == 1
    assert "SSN" in label["pii_types"]
    assert label["rule_score"] >= 25
