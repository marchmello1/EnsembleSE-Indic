#!/usr/bin/env python3
"""Generate the mandated Track 2 transcripts.jsonl using SraVaani-1.0."""
import argparse
import json
from pathlib import Path

from huggingface_hub import get_token
from transformers import AutoModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--timestamps", action="store_true")
    p.add_argument("--model-source", default="ARTPARK-IISc/SraVaani-1.0")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args()

    paths = sorted(a.audio_dir.glob("*.wav"))
    if a.num_shards < 1 or not 0 <= a.shard_index < a.num_shards:
        raise SystemExit("require num-shards >= 1 and 0 <= shard-index < num-shards")
    paths = paths[a.shard_index :: a.num_shards]
    if a.limit:
        paths = paths[: a.limit]
    done = set()
    if a.output.exists():
        for line in a.output.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["clip_id"])
            except Exception:
                pass
    paths = [x for x in paths if x.stem not in done]

    model = AutoModel.from_pretrained(
        a.model_source,
        trust_remote_code=True,
        token=get_token(),
    ).cuda().eval()

    with a.output.open("a", encoding="utf-8") as f:
        for i in range(0, len(paths), a.batch_size):
            batch = paths[i : i + a.batch_size]
            hyps = model.transcribe(
                [str(x) for x in batch],
                batch_size=a.batch_size,
                return_hypotheses=True,
                timestamps=a.timestamps,
            )
            for path, hyp in zip(batch, hyps):
                text = str(getattr(hyp, "text", hyp) or "").strip()
                row = {"clip_id": path.stem, "text": text}
                if a.timestamps:
                    row["timestamp"] = getattr(hyp, "timestamp", {}) or {}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if (i + len(batch)) % 100 < a.batch_size:
                print("DONE", len(done) + i + len(batch), flush=True)

    print("WROTE", len(done) + len(paths), a.output, flush=True)


if __name__ == "__main__":
    main()
