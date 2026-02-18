from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as ds

from datetime import datetime

from pii_risk.ingest.mastodon import ingest_mastodon


def test_ingest_mastodon_jsonl(tmp_path: Path) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "mastodon_sample.jsonl"
    output_dir = tmp_path / "output"

    ingest_mastodon(str(fixture_path), str(output_dir))

    dataset = ds.dataset(output_dir, format="parquet", partitioning="hive")
    table = dataset.to_table()
    records = table.to_pylist()

    assert len(records) == 3

    record = next(r for r in records if r["record_id"] == "100")
    assert record["text"] == "Hello world"

    reply = next(r for r in records if r["record_id"] == "101")
    assert reply["parent_record_id"] == "100"

    base_partition = output_dir / "platform=mastodon" / "record_type=post"
    assert base_partition.exists()

    for r in records:
        dt = datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
        expected = base_partition / f"year={dt.year:04d}" / f"month={dt.month:02d}"
        assert expected.exists()

