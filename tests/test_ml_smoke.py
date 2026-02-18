from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def iter_parquet_records(root: Path) -> list[dict[str, object]]:
    dataset = ds.dataset(root, format="parquet", partitioning="hive")
    return dataset.to_table().to_pylist()


def test_iter_parquet_records_smoke(tmp_path: Path) -> None:
    records = [
        {
            "platform": "reddit",
            "record_type": "post",
            "record_id": "p1",
            "author_id_hash": "a1",
            "created_at": "2025-01-01T00:00:00Z",
            "text": "hello world",
            "community": "test",
            "parent_record_id": None,
            "thread_id": "t1",
        }
    ]

    table = pa.Table.from_pylist(records)
    output_dir = tmp_path / "output"
    pq.write_to_dataset(table, root_path=output_dir, partition_cols=["platform", "record_type"])

    roundtrip = iter_parquet_records(output_dir)

    assert roundtrip == records
