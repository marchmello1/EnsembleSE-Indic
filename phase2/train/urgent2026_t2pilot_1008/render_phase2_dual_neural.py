#!/usr/bin/env python3
"""Render every input with a fixed MP-SENet/URGENT BSRNN neural blend."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def _load_mp_core(package):
    sys.path.insert(0, str(package))
    if (package / "mp_core.py").is_file():
        import mp_core as core
    else:
        import main as core
    return core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("package", type=Path)
    ap.add_argument("mp_checkpoint", type=Path)
    ap.add_argument("urgent_repo", type=Path)
    ap.add_argument("urgent_checkpoint", type=Path)
    ap.add_argument("--weight", type=float, default=0.2)
    ap.add_argument("--gain-match", choices=("none", "ls"), default="none")
    ap.add_argument("--memory-fraction", type=float, default=1.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names-dir", type=Path)
    args = ap.parse_args()
    if not 0 <= args.weight <= 1 or not 0 <= args.shard < args.shards:
        raise ValueError("invalid weight or shard")
    core = _load_mp_core(args.package)
    sys.path.insert(0, str(args.urgent_repo))
    from baseline_code.d_model import SEModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("invalid memory fraction")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    cfg = core.AttrDict(json.loads((core.MP_REPO / "best_ckpt/config.json").read_text()))
    mp = core.load_enhancer(args.mp_checkpoint, cfg, device)
    urgent = SEModel.load_from_checkpoint(str(args.urgent_checkpoint), map_location=device).eval().to(device)
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input.glob("*.wav"))[args.shard::args.shards]
    if args.names_dir:
        paths = [p for p in paths if (args.names_dir / p.name).is_file()]
    if args.limit:
        paths = paths[:args.limit]
    for i, path in enumerate(paths, 1):
        dest = args.output / path.name
        if dest.exists():
            continue
        wave = core.read_audio(path)
        mp_out = core.enhance_full_clip(wave, mp, cfg, device)
        wav = torch.from_numpy(wave).to(device).view(1, -1)
        length = torch.tensor([wav.shape[-1]], device=device)
        with torch.inference_mode():
            urgent_out, _ = urgent.se_model(wav, length, 16000)
        urgent_out = urgent_out.detach().cpu().numpy().reshape(-1).astype(np.float32)
        peak = float(np.max(np.abs(urgent_out))) if urgent_out.size else 0.0
        if peak > 0.9:
            urgent_out *= 0.9 / peak
        if len(urgent_out) < len(wave):
            urgent_out = np.pad(urgent_out, (0, len(wave) - len(urgent_out)))
        primary = mp_out[:len(wave)]
        secondary = urgent_out[:len(wave)]
        if args.gain_match == "ls":
            left = primary.astype(np.float64) - primary.mean()
            right = secondary.astype(np.float64) - secondary.mean()
            gain = np.clip(np.dot(left, right) / max(float(np.dot(right, right)), 1e-12), 0.1, 10.0)
            output = (1 - args.weight) * primary + args.weight * gain * secondary
            output /= max(1.0, float(np.max(np.abs(output))) / 0.99)
        else:
            output = (1 - args.weight) * primary + args.weight * secondary
            output = np.clip(output, -1, 1)
        if len(output) != len(wave) or not np.isfinite(output).all():
            raise ValueError(f"invalid enhanced waveform: {path.name}")
        sf.write(dest, output, 16000, subtype="PCM_16")
        if i % 25 == 0:
            print(f"SHARD {args.shard} DONE {i}/{len(paths)}", flush=True)
    print(f"FINAL SHARD {args.shard} {len(paths)}", flush=True)


if __name__ == "__main__":
    main()
