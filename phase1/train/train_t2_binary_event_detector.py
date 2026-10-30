#!/usr/bin/env python3
"""Train a noisy-audio-only binary event detector for Track 2 gating."""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class Detector(nn.Module):
    def __init__(self, n_mels=80, hidden=128):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.GroupNorm(8, hidden), nn.SiLU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=.1)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, mel):
        x = self.front(mel).transpose(1, 2)
        x, _ = self.gru(x)
        return self.head(x).squeeze(-1)


def waveform_mask(row):
    n = len(row["waveform"])
    mask = np.zeros(n, np.float32)
    for event in row.get("events", []):
        left = max(0, int(round(float(event["start"]) * 16000)))
        right = min(n, int(round(float(event["end"]) * 16000)))
        if right > left:
            mask[left:right] = 1
    return mask


def crop(row, length, rng):
    x = np.asarray(row["waveform"], np.float32).reshape(-1)
    mask = waveform_mask(row)
    n = len(x)
    if n <= length:
        return np.pad(x, (0, length-n)), np.pad(mask, (0, length-n))
    events = row.get("events", [])
    if events and rng.random() < .7:
        event = rng.choice(events)
        centre = .5 * (float(event["start"]) + float(event["end"])) * 16000
        start = int(centre - length/2 + rng.uniform(-.3, .3) * length)
    else:
        start = rng.randrange(n-length+1)
    start = max(0, min(n-length, start))
    return x[start:start+length], mask[start:start+length]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train_cache", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seconds", type=float, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=2701)
    a = p.parse_args()
    rows = torch.load(a.train_cache, map_location="cpu", weights_only=False)
    rng = random.Random(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    model = Detector().cuda().train()
    mel = torchaudio.transforms.MelSpectrogram(
        16000, n_fft=400, win_length=400, hop_length=160, n_mels=80,
        f_min=40, f_max=7800, power=2, center=True).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    length = int(a.seconds * 16000)
    for step in range(1, a.steps+1):
        batch = [crop(rng.choice(rows), length, rng) for _ in range(a.batch_size)]
        x = torch.from_numpy(np.stack([z[0] for z in batch])).cuda()
        y = torch.from_numpy(np.stack([z[1] for z in batch])).cuda()
        # Gain augmentation prevents absolute-volume shortcuts.
        gain = torch.empty(len(x), 1, device=x.device).uniform_(.35, 1.5)
        x = (x * gain).clamp(-1, 1)
        feat = torch.log(mel(x).clamp_min(1e-7))
        feat = (feat - feat.mean((1,2), keepdim=True)) / feat.std((1,2), keepdim=True).clamp_min(.1)
        target = F.interpolate(y[:,None], size=feat.shape[-1], mode="linear", align_corners=False).squeeze(1)
        logits = model(feat)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = logits.sigmoid()
        dice = 1 - (2*(prob*target).sum()+1) / (prob.sum()+target.sum()+1)
        # Boundary derivative encourages precise transitions without dominating regions.
        edge = F.l1_loss(prob[:,1:]-prob[:,:-1], target[:,1:]-target[:,:-1])
        loss = bce + .6*dice + .15*edge
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3); opt.step()
        if step % 20 == 0:
            with torch.no_grad():
                pred = prob > .5; truth = target > .5
                inter = (pred & truth).sum(); union = pred.sum()+truth.sum()
                d = (2*inter.float()/union.clamp_min(1)).item()
            print(f"STEP {step} loss={loss.item():.4f} bce={bce.item():.4f} dice={d:.4f}", flush=True)
        if step % 500 == 0 or step == a.steps:
            out = a.output.with_name(a.output.stem+f"_s{step}.pt")
            torch.save({"model":model.state_dict(),"step":step,"n_mels":80,"hidden":128}, out)
    torch.save({"model":model.state_dict(),"step":a.steps,"n_mels":80,"hidden":128}, a.output)
    print("WROTE", a.output, flush=True)


if __name__ == "__main__":
    main()
