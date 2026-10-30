#!/usr/bin/env python3
"""Noisy-WAV-only Track 2 inference: event detector + two MP-SENet experts.

Usage: python main.py INPUT_FOLDER OUTPUT_FOLDER
The output folder receives one 16-kHz mono WAV per input and transcripts.jsonl.
"""
import argparse
import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from huggingface_hub import get_token

# SraVaani and the enhancement stack are PyTorch-only.  Some audit hosts have
# a system TensorFlow installation linked against a different cuDNN; prevent
# Transformers from probing that unrelated backend during import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent
MP_REPO = ROOT / "vendor"
sys.path.insert(0, str(MP_REPO))
from dataset import mag_pha_istft, mag_pha_stft  # noqa: E402
from env import AttrDict  # noqa: E402
from models.model import DenseEncoder, MPNet  # noqa: E402

TAGS = ['<horn>', '[breathing]', '<bird sound>', '<bird squawking>', '<child talking>',
        '[inhaling]', '<child yelling>', '<barking>', '<vehicle noise>', '<animal sound>',
        '<machine noise>', '<dog barking>', '<phone ringing>', '<ringing>', '[lip smacking]',
        '<music>', '<child crying>', '<baby talking>', '<bird chirping>', '<bell sound>', 'other']


class Detector(nn.Module):
    def __init__(self, n_mels=80, hidden=128):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 5, padding=2), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4), nn.GroupNorm(8, hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, 2, batch_first=True, bidirectional=True, dropout=.1)
        self.head = nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, mel):
        x, _ = self.gru(self.front(mel).transpose(1, 2))
        return self.head(x).squeeze(-1)


class TagMP(MPNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dense_encoder = DenseEncoder(cfg, in_channel=2+len(TAGS))

    def forward(self, amp, pha, masks):
        maps = masks[:, :, :, None].expand(-1, -1, -1, amp.shape[1])
        features = torch.stack((amp, pha), -1).permute(0, 3, 2, 1)
        x = self.dense_encoder(torch.cat((features, maps), 1))
        for block in self.TSTransformer:
            x = block(x)
        return self.mask_decoder(x)*amp, self.phase_decoder(x)


def read_audio(path):
    wave, sr = sf.read(path, dtype="float32", always_2d=False)
    if wave.ndim == 2:
        wave = wave.mean(1)
    if sr != 16000:
        wave = torchaudio.functional.resample(torch.from_numpy(wave), sr, 16000).numpy()
    return np.asarray(wave, np.float32)


def decode_events(wave, detector, mel, device):
    x = torch.from_numpy(wave).to(device)[None]
    with torch.inference_mode():
        feat = torch.log(mel(x).clamp_min(1e-7))
        feat = (feat-feat.mean((1, 2), keepdim=True))/feat.std((1, 2), keepdim=True).clamp_min(.1)
        prob = detector(feat).sigmoid()
        prob = F.avg_pool1d(prob[:, None], 5, 1, 2)[0, 0].cpu().numpy()
    pred = prob >= .50
    z = torch.from_numpy(pred.astype(np.float32))[None, None]
    z = F.max_pool1d(z, 5, 1, 2)
    pred = (-F.max_pool1d(-z, 5, 1, 2)[0, 0].numpy()) > .5
    change = np.diff(np.pad(pred.astype(np.int8), (1, 1)))
    events = []
    for left, right in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)):
        if right-left < 3:
            continue
        confidence = float(prob[left:right].mean())
        duration = (right-left)*.01
        if duration < .4 and confidence < .85:
            continue
        events.append((left*.01, right*.01))
    return events


def tag_mask(events, samples):
    mask = np.zeros((len(TAGS), samples), np.float32)
    for start, end in events:
        left = max(0, int(round(start*16000)))
        right = min(samples, int(round(end*16000)))
        mask[-1, left:right] = 1
    return mask


def soft_mask(samples, events, pad=.1, fade=.05):
    mask = np.zeros(samples, np.float32)
    for start, end in events:
        left = max(0, int(round((start-pad)*16000)))
        right = min(samples, int(round((end+pad)*16000)))
        mask[left:right] = 1
    width = max(1, int(round(fade*16000)))
    kernel = np.r_[np.arange(1, width+1), np.arange(width-1, 0, -1)].astype(np.float32)
    return np.clip(np.convolve(mask, kernel/kernel.sum(), mode="same"), 0, 1)


def enhance_segment(wave, events, model, cfg, device):
    x = torch.from_numpy(wave).to(device)[None]
    masks = torch.from_numpy(tag_mask(events, len(wave))).to(device)[None]
    with torch.inference_mode():
        norm = torch.sqrt(x.shape[-1]/x.square().sum(-1).clamp_min(1e-8))
        xn = x*norm[:, None]
        amp, pha, _ = mag_pha_stft(xn, cfg.n_fft, cfg.hop_size, cfg.win_size, cfg.compress_factor)
        masks = F.interpolate(masks, size=amp.shape[-1], mode="linear", align_corners=False)
        amp, pha = model(amp, pha, masks)
        out = mag_pha_istft(amp, pha, cfg.n_fft, cfg.hop_size, cfg.win_size,
                            cfg.compress_factor).squeeze()/norm.squeeze()
    return out.cpu().numpy()


def render(wave, events, model, cfg, device, context=1.5):
    if not events:
        return wave.copy()
    total = np.zeros(len(wave), np.float32)
    count = np.zeros(len(wave), np.float32)
    ctx = int(context*16000)
    for start, end in events:
        left = max(0, int(start*16000)-ctx)
        right = min(len(wave), int(end*16000)+ctx)
        local = [(max(left, int(s*16000))/16000-left/16000,
                  min(right, int(e*16000))/16000-left/16000)
                 for s, e in events if min(right, int(e*16000)) > max(left, int(s*16000))]
        estimate = enhance_segment(wave[left:right], local, model, cfg, device)
        n = min(right-left, len(estimate))
        total[left:left+n] += estimate[:n]
        count[left:left+n] += 1
    estimate = wave.copy()
    active = count > 0
    estimate[active] = total[active]/count[active]
    mask = soft_mask(len(wave), events)
    return np.clip(wave+mask*(estimate-wave), -1, 1)


def word_tokens(text):
    text = re.sub(r"<[^>]*>", " ", unicodedata.normalize("NFC", text or ""))
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text)
    return text.lower().split()


def edit_distance(a, b):
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = cur
    return prev[-1]


def transcribe(model, paths, batch_size):
    output = {}
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i+batch_size]
        hyps = model.transcribe([str(x) for x in batch], batch_size=batch_size,
                                return_hypotheses=True, timestamps=False)
        for path, hyp in zip(batch, hyps):
            output[path.stem] = str(getattr(hyp, "text", hyp) or "").strip()
    return output


def load_enhancer(path, cfg, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = TagMP(cfg).to(device).eval()
    model.load_state_dict(state.get("model", state["generator"]))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", type=Path)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_folder.glob("*.wav"))
    if not paths:
        raise SystemExit("No WAV files found")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dstate = torch.load(ROOT/"checkpoints/t2_binary_detector_full695_0924.pt",
                        map_location="cpu", weights_only=False)
    detector = Detector(dstate.get("n_mels", 80), dstate.get("hidden", 128)).to(device).eval()
    detector.load_state_dict(dstate["model"])
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=400, win_length=400,
        hop_length=160, n_mels=80, f_min=40, f_max=7800, power=2, center=True).to(device)
    cfg = AttrDict(json.loads((MP_REPO/"best_ckpt/config.json").read_text()))
    model_a = load_enhancer(ROOT/"checkpoints/t2_mp_full_asr_w1_0924.pt", cfg, device)
    model_b = load_enhancer(ROOT/"checkpoints/t2_mp_full_asr_w2_0924.pt", cfg, device)
    with tempfile.TemporaryDirectory(prefix="clean20_") as td:
        temp = Path(td); adir = temp/"a"; bdir = temp/"b"; ndir = temp/"noisy"
        adir.mkdir(); bdir.mkdir(); ndir.mkdir()
        for index, path in enumerate(paths, 1):
            wave = read_audio(path); events = decode_events(wave, detector, mel, device)
            sf.write(ndir/path.name, wave, 16000, subtype="PCM_16")
            sf.write(adir/path.name, render(wave, events, model_a, cfg, device), 16000, subtype="PCM_16")
            sf.write(bdir/path.name, render(wave, events, model_b, cfg, device), 16000, subtype="PCM_16")
            if index % 50 == 0:
                print(f"enhanced {index}/{len(paths)}", flush=True)
        asr = AutoModel.from_pretrained("ARTPARK-IISc/SraVaani-1.0",
                                        trust_remote_code=True, token=get_token()).to(device).eval()
        # The organizer interface requires exactly two command-line arguments.
        # Keep ASR batching deterministic and internal to the implementation.
        batch_size = 8
        noisy_text = transcribe(asr, sorted(ndir.glob("*.wav")), batch_size)
        a_text = transcribe(asr, sorted(adir.glob("*.wav")), batch_size)
        b_text = transcribe(asr, sorted(bdir.glob("*.wav")), batch_size)
        rows = []
        for path in paths:
            clip = path.stem; noisy = word_tokens(noisy_text[clip])
            use_b = edit_distance(noisy, word_tokens(b_text[clip])) < edit_distance(noisy, word_tokens(a_text[clip]))
            source = bdir if use_b else adir
            wave, _ = sf.read(source/path.name, dtype="float32")
            sf.write(args.output_folder/path.name, wave, 16000, subtype="PCM_16")
            rows.append({"clip_id": clip, "text": b_text[clip] if use_b else a_text[clip]})
        with (args.output_folder/"transcripts.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n")
    print(f"Wrote {len(paths)} WAVs and transcripts.jsonl to {args.output_folder}")


if __name__ == "__main__":
    main()
