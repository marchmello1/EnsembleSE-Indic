#!/usr/bin/env python3
"""Fine-tune public URGENT BSRNN on source-disjoint Vaani paired/remixed audio."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from baseline_code.d_model import SEModel


def crop_pair(row, length, rng, center_event=False):
    noisy = np.asarray(row["waveform"], np.float32).reshape(-1)
    clean = np.asarray(row["clean"], np.float32).reshape(-1)
    n = min(len(noisy), len(clean))
    if n < length:
        noisy = np.pad(noisy[:n], (0, length - n))
        clean = np.pad(clean[:n], (0, length - n))
        return noisy, clean
    if center_event and row.get("events"):
        event = rng.choice(row["events"])
        if isinstance(event, dict):
            start = float(event.get("start", 0))
            end = float(event.get("end", start))
        else:
            start, end = float(event[0]), float(event[1])
        center = int((start + end) * 8000)
        left = max(0, min(n - length, center - length // 2 + rng.randint(-4000, 4000)))
    else:
        left = rng.randint(0, n - length)
    return noisy[left:left + length].copy(), clean[left:left + length].copy()


def sample_example(pairs, speech_rows, noise_rows, length, rng, remix_prob):
    if rng.random() >= remix_prob:
        return crop_pair(rng.choice(pairs), length, rng, center_event=True)
    _, speech = crop_pair(rng.choice(speech_rows), length, rng)
    noisy_noise, clean_noise = crop_pair(rng.choice(noise_rows), length, rng,
                                         center_event=True)
    noise = noisy_noise - clean_noise
    snr = rng.uniform(-6.0, 12.0)
    speech_pow = float(np.mean(speech * speech) + 1e-9)
    noise_pow = float(np.mean(noise * noise) + 1e-9)
    scale = float(np.clip(np.sqrt(speech_pow / (noise_pow * 10 ** (snr / 10))), .05, 12.0))
    mixture = speech + scale * noise
    peak = max(1.0, float(np.max(np.abs(mixture)) / .95))
    return (mixture / peak).astype(np.float32), (speech / peak).astype(np.float32)


def neg_sisdr(pred, clean):
    pred = pred - pred.mean(-1, keepdim=True)
    clean = clean - clean.mean(-1, keepdim=True)
    target = clean * ((pred * clean).sum(-1, keepdim=True) /
                      (clean.square().sum(-1, keepdim=True) + 1e-8))
    residual = pred - target
    return (-10 * torch.log10((target.square().sum(-1) + 1e-8) /
                              (residual.square().sum(-1) + 1e-8))).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", type=Path)
    ap.add_argument("speech", type=Path)
    ap.add_argument("noise", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--length", type=int, default=48000)
    ap.add_argument("--remix-prob", type=float, default=0.7)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--sisdr-weight", type=float, default=0.02)
    ap.add_argument("--save-every", type=int, default=200)
    args = ap.parse_args()
    rng = random.Random(1008)
    torch.manual_seed(1008)
    pairs = torch.load(args.pairs, map_location="cpu", weights_only=False)
    print("LOADED pairs", len(pairs), flush=True)
    speech_rows = torch.load(args.speech, map_location="cpu", weights_only=False)
    print("LOADED speech", len(speech_rows), flush=True)
    noise_rows = torch.load(args.noise, map_location="cpu", weights_only=False)
    print("LOADED noise", len(noise_rows), flush=True)
    model = SEModel.load_from_checkpoint(str(args.checkpoint), map_location="cpu").cuda().train()
    opt = torch.optim.AdamW(model.se_model.parameters(), lr=args.lr, weight_decay=1e-5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        examples = [sample_example(pairs, speech_rows, noise_rows, args.length, rng,
                                   args.remix_prob) for _ in range(args.batch_size)]
        mix = torch.from_numpy(np.stack([x for x, _ in examples])).cuda()
        clean = torch.from_numpy(np.stack([y for _, y in examples])).cuda()
        lengths = torch.full((args.batch_size,), args.length, device="cuda", dtype=torch.long)
        enhanced, _ = model.se_model(mix, lengths, 16000)
        spectral = model.mr_l1_loss(clean, enhanced).mean()
        si = neg_sisdr(enhanced, clean)
        loss = spectral + args.sisdr_weight * si
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.se_model.parameters(), 5.0)
        opt.step()
        if step % 20 == 0:
            print(f"STEP {step} loss={loss.item():.5f} spec={spectral.item():.5f} "
                  f"sisdr={-si.item():.3f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_path = args.output.with_name(args.output.stem + f"_s{step}" + args.output.suffix)
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            checkpoint["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(checkpoint, save_path)
            print("SAVED", save_path, flush=True)


if __name__ == "__main__":
    main()
