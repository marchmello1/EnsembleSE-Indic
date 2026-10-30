#!/usr/bin/env python3
"""Pilot the official URGENT 2026 BSRNN on paired IndoML validation."""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from baseline_code.d_model import SEModel
from baseline_code.flow_model import FlowSEModel


def sisdr(pred, clean):
    n = min(len(pred), len(clean))
    p = np.asarray(pred[:n], np.float64).copy()
    c = np.asarray(clean[:n], np.float64).copy()
    p -= p.mean()
    c -= c.mean()
    target = c * (np.dot(p, c) / (np.dot(c, c) + 1e-12))
    return float(10 * np.log10((np.dot(target, target) + 1e-12) /
                                (np.dot(p - target, p - target) + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("validation", type=Path)
    ap.add_argument("mp_dir", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--flow", action="store_true")
    ap.add_argument("--flow-steps", type=int, default=15)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cls = FlowSEModel if args.flow else SEModel
    model = cls.load_from_checkpoint(str(args.checkpoint), map_location="cuda").eval().cuda()
    names = sorted(p.name for p in args.mp_dir.glob("*.wav"))[:args.limit]
    scores = {"urgent": [], "mp": []}
    for i, name in enumerate(names, 1):
        noisy_subdir = "naturalNoisyAudio" if args.natural else "syntheticNoiseAudio"
        noisy, sr = sf.read(args.validation / noisy_subdir / name, dtype="float32")
        if not args.natural:
            clean, sr_clean = sf.read(args.validation / "syntheticCleanRefAudio" / name, dtype="float32")
        else:
            clean, sr_clean = None, 16000
        mp, sr_mp = sf.read(args.mp_dir / name, dtype="float32")
        if sr != 16000 or sr_clean != 16000 or sr_mp != 16000:
            raise ValueError(f"unexpected sample rate for {name}")
        wav = torch.from_numpy(np.asarray(noisy, np.float32)).cuda().view(1, -1)
        length = torch.tensor([wav.shape[-1]], device="cuda")
        with torch.inference_mode():
            if args.flow:
                enhanced = model.enhance(wav, sr, length, N=args.flow_steps)
            else:
                enhanced, _ = model.se_model(wav, length, sr)
        out = enhanced.cpu().numpy().reshape(-1)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.9:
            out = out * (0.9 / peak)
        sf.write(args.output / name, out, sr, subtype="PCM_16")
        if clean is not None:
            scores["urgent"].append(sisdr(out, clean))
            scores["mp"].append(sisdr(mp, clean))
        if i % 4 == 0:
            print("DONE", i, flush=True)
    print("FINAL", len(names),
          {k: float(np.mean(v)) for k, v in scores.items() if v}, flush=True)


if __name__ == "__main__":
    main()
