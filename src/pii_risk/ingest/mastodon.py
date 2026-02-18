from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from pii_risk.schema import Record


HTML_TAG_RE = re.compile(r"<[^>]+>")
MASTODON_PARQUET_SCHEMA = pa.schema(
    [
        ("platform", pa.string()),
        ("record_type", pa.string()),
        ("record_id", pa.string()),
        ("author_id_hash", pa.string()),
        ("created_at", pa.string()),
        ("text", pa.string()),
        ("community", pa.string()),
        ("parent_record_id", pa.string()),
        ("thread_id", pa.string()),
        ("year", pa.string()),
        ("month", pa.string()),
    ]
)


def ingest_mastodon(input_path: str, output_dir: str, max_rows: int | None = None) -> None:
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

        table = pa.Table.from_pylist(rows, schema=MASTODON_PARQUET_SCHEMA)
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
    created_at = raw.get("created_at") or raw.get("created_at_utc")
    if created_at in (None, ""):
        return None

    content = raw.get("content")
    text = raw.get("text")
    if content in (None, "") and text in (None, ""):
        return None

    account_identifier = _extract_account_identifier(raw)
    author_id_hash = _hash_author(account_identifier)
    created_at_iso = _created_at_iso(created_at)
    if created_at_iso is None:
        return None

    community = _extract_community(raw)
    parent_record_id = raw.get("in_reply_to_id")
    parent_record_id = str(parent_record_id) if parent_record_id not in (None, "") else None

    thread_id = raw.get("conversation_id")
    thread_id = str(thread_id) if thread_id not in (None, "") else None


    selected_text = content if content not in (None, "") else text
    cleaned_text = _normalize_text(selected_text)

    return {
        "platform": "mastodon",
        "record_type": "post",
        "record_id": str(record_id) if record_id is not None else None,
        "author_id_hash": author_id_hash,
        "created_at": created_at_iso,
        "community": community,
        "parent_record_id": parent_record_id,
        "thread_id": thread_id,
        "text": cleaned_text,
    }


def _extract_account_identifier(raw: dict[str, Any]) -> str:
    account = raw.get("account")
    account_data = _parse_maybe_json(account)

    if isinstance(account_data, dict):
        for key in ("id", "acct", "username"):
            value = account_data.get(key)
            if value not in (None, ""):
                return str(value)

    for key in ("account.id", "account.acct", "account.username"):
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)

    return "unknown"


def _parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value.startswith("{"):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _hash_author(identifier: Any) -> str:
    value = "unknown" if identifier in (None, "") else str(identifier)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _created_at_iso(created_at: Any) -> str | None:
    if created_at in (None, ""):
        return None
    try:
        value = str(created_at).strip()
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except ValueError:
        return None


def _extract_community(raw: dict[str, Any]) -> str | None:
    for key in ("uri", "url"):
        value = raw.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = urlparse(str(value))
        except ValueError:
            continue
        if parsed.netloc:
            return parsed.netloc
    return None


def _normalize_text(content: Any) -> str:
    if content in (None, ""):
        return ""
    text = str(content)
    if "<" in text and ">" in text:
        text = HTML_TAG_RE.sub(" ", text)
        text = unescape(text)
    text = " ".join(text.split())
    return text.strip()
