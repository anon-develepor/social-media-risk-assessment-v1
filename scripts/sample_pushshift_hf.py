from datasets import load_dataset
import json
from pathlib import Path
from itertools import islice

N = 10000
OUT = Path("data/raw/reddit_pushshift_sample.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Stream: avoids downloading the full dataset
stream = load_dataset("fddemarco/pushshift-reddit", split="train", streaming=True)

with OUT.open("w", encoding="utf-8") as f:
    for ex in islice(stream, N):
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {N} rows to {OUT}")
