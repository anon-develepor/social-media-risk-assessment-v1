from pii_risk.schema import Record


def test_record_text_preserved() -> None:
    record = Record(
        platform="twitter",
        record_type="post",
        record_id="123",
        author_id_hash="abc",
        created_at="2024-01-01T00:00:00Z",
        text="hello world",
    )

    assert record.text == "hello world"
