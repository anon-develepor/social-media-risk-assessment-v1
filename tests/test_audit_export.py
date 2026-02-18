from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pii_risk.eval.audit import audit_records
from pii_risk.ml.train import train_model


def _write_parquet_dataset(output_dir: Path) -> None:
    records = [
        {
            "platform": "reddit",
            "record_type": "post",
            "record_id": "r1",
            "created_at": "2024-01-01T00:00:00Z",
            "text": "Email me at alice@example.com for details.",
            "community": "alpha",
        },
        {
            "platform": "reddit",
            "record_type": "post",
            "record_id": "r2",
            "created_at": "2024-01-02T00:00:00Z",
            "text": "Just a normal update with no sensitive info.",
            "community": "alpha",
        },
        {
            "platform": "reddit",
            "record_type": "comment",
            "record_id": "r3",
            "created_at": "2024-01-03T00:00:00Z",
            "text": "Call me at 415-555-1234 to follow up.",
            "community": "beta",
        },
        {
            "platform": "reddit",
            "record_type": "comment",
            "record_id": "r4",
            "created_at": "2024-01-04T00:00:00Z",
            "text": "Planning a meetup tomorrow afternoon.",
            "community": None,
        },
        {
            "platform": "reddit",
            "record_type": "post",
            "record_id": "r5",
            "created_at": "2024-01-05T00:00:00Z",
            "text": "Here is the project summary everyone asked for.",
            "community": "gamma",
        },
        {
            "platform": "reddit",
            "record_type": "comment",
            "record_id": "r6",
            "created_at": "2024-01-06T00:00:00Z",
            "text": "Reach me at bob@example.org any time.",
            "community": "gamma",
        },
    ]

    table = pa.Table.from_pylist(records)
    pq.write_to_dataset(
        table, root_path=output_dir, partition_cols=["platform", "record_type"]
    )


def test_audit_export(tmp_path: Path) -> None:
    data_dir = tmp_path / "processed"
    _write_parquet_dataset(data_dir)

    models_dir = tmp_path / "models"
    train_model(str(data_dir), models_dir=models_dir)

    audit_path = tmp_path / "audit.csv"
    summary = audit_records(str(data_dir), str(models_dir / "pii_risk_model.pkl"), str(audit_path))

    assert audit_path.exists()

    with audit_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 6

    required_columns = {
        "record_id",
        "created_at",
        "bucket",
        "y_risk",
        "pred_risk",
        "p_risk",
        "rule_score",
        "pii_types",
        "text",
        "redacted_text",
        "community",
    }
    assert required_columns.issubset(set(reader.fieldnames or []))

    bucket_values = {row["bucket"] for row in rows}
    assert bucket_values.issubset({"TP", "FP", "TN", "FN"})

    for row in rows:
        assert isinstance(row["pii_types"], str)

    bucket_counts = summary["bucket_counts"]
    assert sum(bucket_counts.values()) == 6
