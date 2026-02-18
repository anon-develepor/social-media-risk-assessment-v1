from pii_risk.pii.scoring import WEIGHTS, score_record


def test_score_caps_at_100() -> None:
    text = "SSN 123-45-6789 and card 4111 1111 1111 1111"
    result = score_record(text)
    assert result["score"] == 100


def test_weights_applied() -> None:
    text = "Email me at jane@example.com or call 415-555-1234"
    result = score_record(text)
    expected = WEIGHTS["EMAIL"] + WEIGHTS["PHONE"]
    assert result["score"] == expected
    assert result["counts_by_type"]["EMAIL"] == 1
    assert result["counts_by_type"]["PHONE"] == 1


def test_explanation_mentions_top_type() -> None:
    text = "Email me at jane@example.com"
    result = score_record(text)
    assert "EMAIL" in result["explanation"]
