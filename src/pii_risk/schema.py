from __future__ import annotations

from pydantic import BaseModel


class Record(BaseModel):
    platform: str
    record_type: str
    record_id: str
    author_id_hash: str
    created_at: str
    text: str
    community: str | None = None
    parent_record_id: str | None = None
    thread_id: str | None = None
