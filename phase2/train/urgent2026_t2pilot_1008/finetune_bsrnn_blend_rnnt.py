#!/usr/bin/env python3
"""Optimize BSRNN through the deployed uniform blend, on released training data."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from transformers import AutoModel

from finetune_mpsenet_rnnt import acoustic_features, clean_text, rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr


def main():
    p = argparse.ArgumentParser()
    for name in ("pairs", "natural", "metadata", "package", "checkpoint", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--holdout-cache", type=Path, required=True)
    p.add_argument("--asr-dir", type=Path, default=Path("/mnt/nvme_disk1/sravaani_model"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--rnnt-weight", type=float, default=3.0)
    p.add_argument("--natural-prob", type=float, default=0.5)
    p.add_argument("--identity-weight", type=float, default=0.5)
    p.add_argument("--urgent-weight", type=float, default=0.3)
    p.add_argument("--gain-match", choices=("none", "ls"), default="none")
    p.add_argument("--encoder-weight", type=float, default=0.0,
                   help="Match frozen ASR features to paired clean speech.")
    p.add_argument("--max-seconds", type=float, default=8.0)
    p.add_argument("--save-every", type=int, default=100)
    a = p.parse_args()
    if not 0 <= a.natural_prob <= 1 or not 0 < a.urgent_weight <= 1:
        p.error("invalid probability or blend weight")
    rng = random.Random(1009)
    torch.manual_seed(1009)
    sys.path.insert(0, str(a.package))
    import mp_core as core
    from baseline_code.d_model import SEModel

    device = torch.device("cuda")
    cfg = core.AttrDict(json.loads((a.package / "vendor/best_ckpt/config.json").read_text()))
    mp = core.load_enhancer(a.package / "checkpoints/t2_mpdns_publicnoise_hilr_1008.pt", cfg, device)
    mp.eval()
    for param in mp.parameters():
        param.requires_grad_(False)
    urgent = SEModel.load_from_checkpoint(str(a.checkpoint), map_location="cpu").to(device).train()
    urgent.mr_l1_loss.reduction = "mean"
    asr = AutoModel.from_pretrained(str(a.asr_dir), trust_remote_code=True).to(device).eval()
    asr._ensure_loaded()
    for module in (asr.encoder, asr.decoder_joint):
        for param in module.parameters():
            param.requires_grad_(False)
    tokenizer = asr._get_tokenizer()
    metadata = {r["segmentFileName"].removesuffix(".wav"): r
                for r in json.loads(a.metadata.read_text())}
    held = {r["id"] for r in torch.load(a.holdout_cache, map_location="cpu", weights_only=False)}
    held_states = {metadata[clip]["state"] for clip in held}
    pools = []
    for path in (a.pairs, a.natural):
        rows = torch.load(path, map_location="cpu", weights_only=False)
        if held.intersection(r["id"] for r in rows):
            raise ValueError("training and holdout IDs overlap")
        if any(metadata[r["id"]]["state"] in held_states for r in rows if r["id"] in metadata):
            raise ValueError("training and holdout states overlap")
        pool = []
        for row in rows:
            if row["id"] not in metadata or len(row["waveform"]) > a.max_seconds * 16000:
                continue
            tokens = tokenizer.encode(clean_text(metadata[row["id"]]["transcript"]))
            if tokens:
                pool.append((row, tokens))
        if not pool:
            raise ValueError(f"empty training pool: {path}")
        pools.append(pool)
    print("POOLS", [len(pool) for pool in pools], "HELDOUT", len(held), flush=True)
    opt = torch.optim.AdamW(urgent.se_model.parameters(), lr=a.lr, weight_decay=1e-6)
    mp_cache = {}
    teacher_cache = {}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, a.steps + 1):
        natural = rng.random() < a.natural_prob
        row, tokens = rng.choice(pools[int(natural)])
        wave = np.asarray(row["waveform"], np.float32).reshape(-1)
        if row["id"] not in mp_cache:
            mp_cache[row["id"]] = core.enhance_full_clip(wave, mp, cfg, device)
        x = torch.from_numpy(wave).to(device)[None]
        lengths = torch.tensor([x.shape[-1]], device=device)
        out, _ = urgent.se_model(x, lengths, 16000)
        if out.shape[-1] < x.shape[-1]:
            out = torch.nn.functional.pad(out, (0, x.shape[-1] - out.shape[-1]))
        out = out[:, :x.shape[-1]]
        out = out * (0.9 / out.abs().amax(-1, keepdim=True).clamp_min(0.9))
        fixed = torch.as_tensor(mp_cache[row["id"]], device=device)[None, :x.shape[-1]]
        if a.gain_match == "ls":
            left = fixed - fixed.mean(-1, keepdim=True)
            right = out - out.mean(-1, keepdim=True)
            gain = ((left * right).sum(-1, keepdim=True) /
                    right.square().sum(-1, keepdim=True).clamp_min(1e-12)).clamp(0.1, 10)
            out = gain * out
        y = (1 - a.urgent_weight) * fixed + a.urgent_weight * out
        if a.gain_match == "ls":
            y = y / (y.abs().amax(-1, keepdim=True) / 0.99).clamp_min(1)
        else:
            y = y.clamp(-1, 1)
        encoded = None
        phonetic = y.new_zeros(())
        if a.encoder_weight > 0 and not natural:
            features, feature_lengths = acoustic_features(asr, y)
            encoded = asr.encoder(features.to(asr._io_dtype), feature_lengths)
            if row["id"] not in teacher_cache:
                with torch.no_grad():
                    reference = torch.as_tensor(np.asarray(row["clean"], np.float32), device=device)[None]
                    features, feature_lengths = acoustic_features(asr, reference)
                    teacher_cache[row["id"]] = asr.encoder(features.to(asr._io_dtype), feature_lengths)[0].cpu()
            teacher = teacher_cache[row["id"]].to(device)
            if teacher.shape != encoded[0].shape:
                raise ValueError("clean/enhanced ASR feature shapes differ")
            phonetic = torch.nn.functional.smooth_l1_loss(encoded[0].float(), teacher.float(), beta=0.1)
        recognition = rnnt_loss(asr, y, tokens, encoded=encoded)
        if natural:
            si = y.new_zeros(())
            preserve = (y - x).abs().mean() / x.abs().mean().clamp_min(1e-4)
            spectral = y.new_zeros(())
        else:
            clean = torch.as_tensor(np.asarray(row["clean"], np.float32), device=device)[None, :y.shape[-1]]
            si = neg_sisdr(y, clean)
            preserve = y.new_zeros(())
            spectral = urgent.mr_l1_loss(clean, y).mean()
        loss = (si + a.rnnt_weight * recognition + a.identity_weight * preserve +
                0.1 * spectral + a.encoder_weight * phonetic)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite objective at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(urgent.se_model.parameters(), 3.0)
        if not torch.isfinite(grad) or float(grad) == 0:
            raise RuntimeError(f"invalid enhancement gradient at step {step}: {grad}")
        opt.step()
        if step == 1 or step % 10 == 0:
            print("STEP", step, "natural", natural, "loss", float(loss.detach()),
                  "rnnt", float(recognition.detach()), "sisdr", float(-si.detach()),
                  "phonetic", float(phonetic.detach()),
                  "gradient", float(grad), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
            ckpt["state_dict"] = {k: v.detach().cpu() for k, v in urgent.state_dict().items()}
            ckpt["blend_rnnt_training"] = {"step": step, "weight": a.urgent_weight,
                                           "rnnt_weight": a.rnnt_weight,
                                           "gain_match": a.gain_match,
                                           "encoder_weight": a.encoder_weight}
            dest = a.output.with_name(a.output.stem + f"_s{step}" + a.output.suffix)
            torch.save(ckpt, dest)
            print("SAVED", dest, flush=True)


if __name__ == "__main__":
    main()
