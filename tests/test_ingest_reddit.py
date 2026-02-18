from __future__ import annotations

from pathlib import Path

import pandas as pd


def _write_partition(output_dir: Path, record_type: str, year: str, month: str, rows: list[dict]) -> None:
    partition_dir = (
        output_dir
        / "platform=reddit"
        / f"record_type={record_type}"
        / f"year={year}"
        / f"month={month}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(partition_dir / "part-0.parquet", index=False)


def test_ingest_reddit_output_layout(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    post_row = {
        "record_type": "post",
        "record_id": "post-1",
        "text": "Title line\n\nBody line",
        "created_at": "2024-01-01T00:00:00Z",
    }
    comment_row = {
        "record_type": "comment",
        "record_id": "comment-1",
        "text": "Comment body",
        "created_at": "2024-01-01T01:00:00Z",
    }

    _write_partition(output_dir, "post", "2024", "01", [post_row])
    _write_partition(output_dir, "comment", "2024", "01", [comment_row])

    platform_dir = output_dir / "platform=reddit"
    post_partition = platform_dir / "record_type=post"
    comment_partition = platform_dir / "record_type=comment"

    post_files = list(platform_dir.rglob("record_type=post/**/*.parquet"))
    comment_files = list(platform_dir.rglob("record_type=comment/**/*.parquet"))

    assert len(post_files) > 0
    assert len(comment_files) > 0

    assert (post_partition / "year=2024" / "month=01").is_dir()
    assert (comment_partition / "year=2024" / "month=01").is_dir()

    all_records = pd.concat((pd.read_parquet(path) for path in post_files + comment_files))
    assert len(all_records) == 2

    post_texts = all_records[all_records["record_type"] == "post"]["text"].tolist()
    comment_texts = all_records[all_records["record_type"] == "comment"]["text"].tolist()

    assert any("\n\n" in text for text in post_texts)
    assert any(text.strip() for text in comment_texts)
