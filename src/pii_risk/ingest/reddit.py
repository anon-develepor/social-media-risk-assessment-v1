from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from pii_risk.schema import Record


def ingest_reddit(input_path: str, output_dir: str, max_rows: int | None = None) -> None:
    input_file = Path(input_path)
    suffix = input_file.suffix.lower()
    total_read = 0
    total_written = 0
    total_skipped = 0
    file_counter = 0

    def process_records(records: Iterable[dict[str, Any]]) -> None:
        nonlocal total_read, total_written, total_skipped, file_counter
        rows: list[dict[str, Any]] = []
        for raw in records:
            if max_rows is not None and total_read >= max_rows:
                break
            total_read += 1
            normalized = _normalize_record(raw)
            if normalized is None:
                total_skipped += 1
                continue
            try:
                record = Record(**normalized)
            except Exception:
                total_skipped += 1
                continue
            if hasattr(record, "model_dump"):
                row = record.model_dump()
            else:
                row = record.dict()
            created = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
            row["year"] = f"{created.year:04d}"
            row["month"] = f"{created.month:02d}"
            rows.append(row)

        if not rows:
            return

        table = pa.Table.from_pylist(rows)
        ds.write_dataset(
            table,
            base_dir=output_dir,
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema(
                    [
                        ("platform", pa.string()),
                        ("record_type", pa.string()),
                        ("year", pa.string()),
                        ("month", pa.string()),
                    ]
                ),
                flavor="hive",
            ),
            existing_data_behavior="overwrite_or_ignore",
            basename_template=f"part-{file_counter:03d}-{{i}}.parquet",
        )
        total_written += len(rows)
        file_counter += 1

    if suffix == ".csv":
        for chunk in pd.read_csv(input_file, chunksize=1000, keep_default_na=False):
            records = chunk.to_dict(orient="records")
            process_records(records)
            if max_rows is not None and total_read >= max_rows:
                break
    elif suffix == ".jsonl":
        with input_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if max_rows is not None and total_read >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    total_read += 1
                    total_skipped += 1
                    continue
                process_records([raw])
    else:
        raise ValueError("Unsupported input format. Use .jsonl or .csv")

    print(
        f"total_read={total_read} total_written={total_written} total_skipped={total_skipped}"
    )


def _normalize_record(raw: dict[str, Any]) -> dict[str, Any] | None:
    record_id = raw.get("id")
    author = raw.get("author")
    created_utc = raw.get("created_utc")
    subreddit = raw.get("subreddit")
    title = raw.get("title")
    selftext = raw.get("selftext")
    body = raw.get("body")
    parent_id = raw.get("parent_id")
    link_id = raw.get("link_id")

    author_id_hash = _hash_author(author)
    created_at = _created_at_iso(created_utc)
    has_post_text = bool(title) or bool(selftext)

    if has_post_text:
        text = f"{title or ''}\n\n{selftext or ''}".strip()
        record_type = "post"
    else:
        text = (body or "").strip()
        record_type = "comment"

    return {
        "platform": "reddit",
        "record_type": record_type,
        "record_id": str(record_id) if record_id is not None else None,
        "author_id_hash": author_id_hash,
        "created_at": created_at,
        "community": subreddit if subreddit not in ("", None) else None,
        "parent_record_id": parent_id if record_type == "comment" else None,
        "thread_id": link_id if link_id not in ("", None) else None,
        "text": text,
    }


def _hash_author(author: Any) -> str:
    value = "unknown" if author in (None, "") else str(author)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _created_at_iso(created_utc: Any) -> str | None:
    if created_utc in (None, ""):
        return None
    try:
        timestamp = float(created_utc)
    except (TypeError, ValueError):
        return None
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")
