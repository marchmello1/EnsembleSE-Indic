#!/usr/bin/env python3
"""Render one clean Track-2 MP-SENet checkpoint with the binary detector."""
import argparse
import json
import sys
import types
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # The renderer does not invoke ASR, so avoid importing its optional stack.
    module = types.ModuleType("transformers")
    module.AutoModel = object
    sys.modules.setdefault("transformers", module)
    hub = types.ModuleType("huggingface_hub")
    hub.get_token = lambda: None
    sys.modules.setdefault("huggingface_hub", hub)
    # MP-SENet's training module imports PESQ at module load time, although
    # inference never calls it. Keep rendering independent of that evaluator.
    if "pesq" not in sys.modules:
        pesq_module = types.ModuleType("pesq")
        pesq_module.pesq = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("PESQ is unavailable during inference"))
        sys.modules["pesq"] = pesq_module
    sys.path.insert(0, str(args.package))
    import main as core

    device = torch.device("cuda")
    state = torch.load(
        args.package / "checkpoints/t2_binary_detector_full695_0924.pt",
        map_location="cpu", weights_only=False,
    )
    detector = core.Detector(state.get("n_mels", 80), state.get("hidden", 128)).to(device).eval()
    detector.load_state_dict(state["model"])
    mel = torchaudio.transforms.MelSpectrogram(
        16000, n_fft=400, win_length=400, hop_length=160, n_mels=80,
        f_min=40, f_max=7800, power=2, center=True,
    ).to(device)
    cfg = core.AttrDict(json.loads((core.MP_REPO / "best_ckpt/config.json").read_text()))
    enhancer = core.load_enhancer(args.checkpoint, cfg, device)

    paths = sorted(args.input.glob("*.wav"))[args.shard::args.shards]
    for index, path in enumerate(paths, 1):
        if (args.output / path.name).is_file():
            continue
        wave = core.read_audio(path)
        events = core.decode_events(wave, detector, mel, device)
        output = core.render(wave, events, enhancer, cfg, device)
        sf.write(args.output / path.name, output, 16000, subtype="PCM_16")
        if index % 50 == 0:
            print(f"SHARD {args.shard} DONE {index}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
