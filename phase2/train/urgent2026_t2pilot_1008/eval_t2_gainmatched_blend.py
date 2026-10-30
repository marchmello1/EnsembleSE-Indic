#!/usr/bin/env python3
"""Audio-only output gain alignment; references are used solely for evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def match_gain(primary, secondary, method):
    a = primary.astype(np.float64) - primary.mean()
    b = secondary.astype(np.float64) - secondary.mean()
    energy = float(np.dot(b, b))
    if method == "rms":
        gain = np.sqrt(float(np.dot(a, a)) / max(energy, 1e-12))
    elif method == "ls":
        gain = float(np.dot(a, b)) / max(energy, 1e-12)
    elif method == "none":
        gain = 1.0
    else:
        raise ValueError(method)
    return float(np.clip(gain, 0.1, 10.0))


def blend(primary, secondary, method, weight):
    gain = match_gain(primary, secondary, method)
    y = (1 - weight) * primary + weight * gain * secondary
    y /= max(1.0, float(np.max(np.abs(y))) / 0.99)
    return y.astype(np.float32), gain


def sisdr(y, clean):
    y = y.astype(np.float64) - y.mean()
    clean = clean.astype(np.float64) - clean.mean()
    target = clean * (np.dot(y, clean) / (np.dot(clean, clean) + 1e-12))
    return float(10 * np.log10((np.dot(target, target) + 1e-12) /
                              (np.dot(y - target, y - target) + 1e-12)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("validation", type=Path)
    p.add_argument("primary", type=Path)
    p.add_argument("secondary", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--method", choices=("none", "rms", "ls"), default="ls")
    p.add_argument("--weight", type=float, default=0.3)
    p.add_argument("--natural", action="store_true",
                   help="Render without clean references; algorithm is unchanged.")
    a = p.parse_args()
    if not 0 <= a.weight <= 1:
        p.error("weight must be in [0,1]")
    names = sorted(path.name for path in a.secondary.glob("*.wav"))
    settings = [(method, weight) for method in ("none", "rms", "ls")
                for weight in (0.1, 0.3, 0.5, 0.7, 0.9)]
    scores = {f"{method}_{weight}": [] for method, weight in settings}
    gains = {method: [] for method in ("none", "rms", "ls")}
    if a.output:
        a.output.mkdir(parents=True, exist_ok=True)
    for name in names:
        primary, sr = sf.read(a.primary / name, dtype="float32")
        secondary, sr2 = sf.read(a.secondary / name, dtype="float32")
        clean = None
        if not a.natural:
            clean, sr3 = sf.read(a.validation / "syntheticCleanRefAudio" / name, dtype="float32")
            if sr3 != sr or len(clean) != len(primary):
                raise ValueError(f"incompatible reference: {name}")
        if sr != 16000 or sr2 != sr or len(primary) != len(secondary):
            raise ValueError(f"incompatible sample lengths/rates: {name}")
        for method in gains:
            gains[method].append(match_gain(primary, secondary, method))
        if clean is not None:
            for method, weight in settings:
                y, _ = blend(primary, secondary, method, weight)
                scores[f"{method}_{weight}"].append(sisdr(y, clean))
        if a.output:
            y, _ = blend(primary, secondary, a.method, a.weight)
            sf.write(a.output / name, y, sr, subtype="PCM_16")
    report = {"clips": len(names), "means": {key: float(np.mean(value)) for key, value in scores.items() if value},
              "gain_quantiles": {key: np.quantile(value, [0, .25, .5, .75, 1]).tolist()
                                 for key, value in gains.items()},
              "alignment_uses_references": False}
    a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
