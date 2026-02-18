from pii_risk.pii.detector import PIISpan, PIIType, detect_pii_spans, redact_text


def test_detect_email() -> None:
    text = "Contact me at jane.doe@example.com for info."
    spans = detect_pii_spans(text)
    assert len(spans) == 1
    assert spans[0].type == PIIType.EMAIL
    assert spans[0].match == "jane.doe@example.com"


def test_detect_phone() -> None:
    text = "Call 415-555-1234 tomorrow."
    spans = detect_pii_spans(text)
    assert len(spans) == 1
    assert spans[0].type == PIIType.PHONE
    assert spans[0].match == "415-555-1234"


def test_detect_url() -> None:
    text = "See https://example.com/docs for details."
    spans = detect_pii_spans(text)
    assert len(spans) == 1
    assert spans[0].type == PIIType.URL
    assert spans[0].match == "https://example.com/docs"


def test_overlap_prefers_longer_span() -> None:
    text = "Visit https://example.com/user@example.com for details."
    spans = detect_pii_spans(text)
    assert len(spans) == 1
    assert spans[0].type == PIIType.URL
    assert spans[0].match == "https://example.com/user@example.com"


def test_redact_text() -> None:
    text = "Email jane.doe@example.com or call 415-555-1234."
    spans = [
        PIISpan(type=PIIType.EMAIL, start=6, end=26, match="jane.doe@example.com"),
        PIISpan(type=PIIType.PHONE, start=35, end=47, match="415-555-1234"),
    ]
    redacted = redact_text(text, spans)
    assert (
        redacted
        == "Email [REDACTED:EMAIL] or call [REDACTED:PHONE]."
    )
