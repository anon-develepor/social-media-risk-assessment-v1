from datasets import load_dataset
import json
from pathlib import Path

OUT = Path("data/raw/mastodon_hf.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Download (small dataset)
ds = load_dataset("ChaseLabs/Harmful-Texts-On-Mastodon", split="train")

with OUT.open("w", encoding="utf-8") as f:
    for ex in ds:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(ds)} rows to {OUT}")
