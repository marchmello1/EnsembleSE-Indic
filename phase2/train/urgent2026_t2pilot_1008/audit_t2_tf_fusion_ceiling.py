#!/usr/bin/env python3
"""Released-validation-only convex TF oracle. Never an inference/submission method."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from eval_t2_gainmatched_blend import blend, match_gain, sisdr


def main():
    p = argparse.ArgumentParser()
    for name in ("validation", "primary", "secondary", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    names = sorted(path.name for path in a.secondary.glob("*.wav"))
    if a.limit:
        np.random.default_rng(1009).shuffle(names)
        names = names[:a.limit]
    settings = [(freq, time) for freq in (8, 16, 32, 64) for time in (4, 8, 16, 32)]
    scores = {f"f{freq}_t{time}": [] for freq, time in settings}
    baseline = []
    window = torch.hann_window(512, device=a.device)
    with torch.inference_mode():
        for i, name in enumerate(names, 1):
            primary, sr = sf.read(a.primary / name, dtype="float32")
            secondary, sr2 = sf.read(a.secondary / name, dtype="float32")
            ref, sr3 = sf.read(a.validation / "syntheticCleanRefAudio" / name, dtype="float32")
            if sr != 16000 or sr2 != sr or sr3 != sr or len(primary) != len(secondary) or len(ref) != len(primary):
                raise ValueError(name)
            fixed, _ = blend(primary, secondary, "ls", 0.7)
            baseline.append(sisdr(fixed, ref))
            secondary *= match_gain(primary, secondary, "ls")
            # SI-SDR ignores global gain; fit target speech to the primary's gain
            # so the convex oracle measures complementary errors, not gain drift.
            target = ref * (np.dot(primary.astype(np.float64), ref.astype(np.float64)) /
                            max(float(np.dot(ref.astype(np.float64), ref.astype(np.float64))), 1e-12))
            def stft(wave):
                return torch.stft(torch.as_tensor(wave, device=a.device), 512,
                                  hop_length=128, window=window, return_complex=True)
            left, right, clean = stft(primary), stft(secondary), stft(target.astype(np.float32))
            difference = right - left
            numerator = (difference.conj() * (clean - left)).real
            denominator = difference.abs().square()
            for freq, time in settings:
                def pool(value):
                    return F.avg_pool2d(value[None, None], (freq, time), stride=(freq, time),
                                        ceil_mode=True, count_include_pad=False)[0, 0]
                weight = (pool(numerator) / pool(denominator).clamp_min(1e-12)).clamp(0, 1)
                weight = weight.repeat_interleave(freq, 0).repeat_interleave(time, 1)
                weight = weight[:left.shape[0], :left.shape[1]]
                wave = torch.istft(left + weight * difference, 512, hop_length=128,
                                   window=window, length=len(primary)).cpu().numpy()
                scores[f"f{freq}_t{time}"].append(sisdr(wave, ref))
            if i % 20 == 0:
                print("DONE", i, flush=True)
    report = {"clips": len(names), "baseline_ls70_sisdr": float(np.mean(baseline)),
              "oracle_means": {key: float(np.mean(value)) for key, value in scores.items()},
              "uses_clean_validation_references": True,
              "deployable": False,
              "cell_resolution": "FFT 512/hop128 at16kHz; f bins=31.25Hz each, t frames=8ms each"}
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
