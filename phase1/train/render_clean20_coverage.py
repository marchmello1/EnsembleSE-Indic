#!/usr/bin/env python3
"""Render MP-SENet with broader Phase-2 event coverage.

This remains noisy-audio-only and reproducible.  ``full`` marks the complete
clip as the conditioning event; threshold modes reuse the learned detector at
the requested posterior cutoff.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio


def decode(wave, detector, mel, device, threshold):
    x = torch.from_numpy(wave).to(device)[None]
    with torch.inference_mode():
        feat = torch.log(mel(x).clamp_min(1e-7))
        feat = (feat-feat.mean((1, 2), keepdim=True))/feat.std((1, 2), keepdim=True).clamp_min(.1)
        prob = detector(feat).sigmoid()
        prob = F.avg_pool1d(prob[:, None], 5, 1, 2)[0, 0].cpu().numpy()
    pred = prob >= threshold
    z = torch.from_numpy(pred.astype(np.float32))[None, None]
    z = F.max_pool1d(z, 9, 1, 4)
    pred = (-F.max_pool1d(-z, 9, 1, 4)[0, 0].numpy()) > .5
    change = np.diff(np.pad(pred.astype(np.int8), (1, 1)))
    return [(l*.01, r*.01) for l, r in zip(np.flatnonzero(change == 1),
            np.flatnonzero(change == -1)) if r-l >= 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("package", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--mode", choices=("full", "threshold"), required=True)
    p.add_argument("--threshold", type=float, default=.25)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)

    module = types.ModuleType("transformers"); module.AutoModel = object
    sys.modules.setdefault("transformers", module)
    hub = types.ModuleType("huggingface_hub"); hub.get_token = lambda: None
    sys.modules.setdefault("huggingface_hub", hub)
    sys.path.insert(0, str(a.package))
    import main as core

    device = torch.device("cuda")
    state = torch.load(a.package/"checkpoints/t2_binary_detector_full695_0924.pt",
                       map_location="cpu", weights_only=False)
    detector = core.Detector(state.get("n_mels", 80), state.get("hidden", 128)).to(device).eval()
    detector.load_state_dict(state["model"])
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, win_length=400,
        hop_length=160, n_mels=80, f_min=40, f_max=7800, power=2, center=True).to(device)
    cfg = core.AttrDict(json.loads((core.MP_REPO/"best_ckpt/config.json").read_text()))
    model = core.load_enhancer(a.checkpoint, cfg, device)

    paths = sorted(a.input.glob("*.wav"))[a.shard::a.shards]
    for index, path in enumerate(paths, 1):
        if (a.output/path.name).is_file(): continue
        wave = core.read_audio(path)
        events = ([(0., len(wave)/16000.)] if a.mode == "full" else
                  decode(wave, detector, mel, device, a.threshold))
        out = core.render(wave, events, model, cfg, device)
        sf.write(a.output/path.name, out, 16000, subtype="PCM_16")
        if index % 50 == 0: print(f"SHARD {a.shard} DONE {index}/{len(paths)}", flush=True)


if __name__ == "__main__": main()
