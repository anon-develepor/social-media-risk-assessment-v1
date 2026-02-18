import json
from pathlib import Path

INP = Path("data/raw/mastodon_hf.jsonl")
OUT = Path("data/raw/mastodon_for_ingest.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Use a fixed timestamp so partitioning works deterministically.
# (This dataset doesn't include created_at.)
DEFAULT_CREATED_AT = "2025-01-01T00:00:00Z"

n = 0
with INP.open("r", encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)

        text = obj.get("text") or ""
        out = {
            "id": obj.get("id"),
            "created_at": DEFAULT_CREATED_AT,
            "text": text,
            # keep source labels (optional; harmless for ingestion if ignored)
            "binary_label": obj.get("binary label"),
            "multi_class_label": obj.get("multi-class label"),
            "multi_label_label": obj.get("multi-label label"),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        n += 1

print(f"Wrote {n} rows to {OUT}")
