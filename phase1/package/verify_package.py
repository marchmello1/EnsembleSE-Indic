#!/usr/bin/env python3
"""Offline structural and checkpoint-integrity check for organizer delivery."""
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "main.py", "README.md", "requirements.txt", "VALIDATION_RESULT.md",
    "checkpoints/t2_binary_detector_full695_0924.pt",
    "checkpoints/t2_mp_full_asr_w1_0924.pt",
    "checkpoints/t2_mp_full_asr_w2_0924.pt",
    "vendor/dataset.py", "vendor/env.py", "vendor/utils.py",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")
    sums = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        sums[name] = expected
    bad = [name for name, expected in sums.items()
           if digest(ROOT / name) != expected]
    if bad:
        raise SystemExit(f"Checksum mismatch: {bad}")
    print(f"PACKAGE_OK files={len(sums)}")


if __name__ == "__main__":
    main()
