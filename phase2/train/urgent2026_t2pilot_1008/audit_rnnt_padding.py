#!/usr/bin/env python3
"""Measure RNNT padding failures on released training clips only."""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
import numpy as np
import torch
import torchaudio
from transformers import AutoModel
from finetune_mpsenet_rnnt import acoustic_features, clean_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("metadata", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--limit", type=int, default=32)
    a = p.parse_args()
    meta = {r["segmentFileName"].removesuffix(".wav"): r
            for r in json.loads(a.metadata.read_text())}
    rows = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows = [r for r in rows if r["id"] in meta and len(r["waveform"]) <= 8 * 16000]
    random.Random(1009).shuffle(rows)
    asr = AutoModel.from_pretrained("/mnt/nvme_disk1/sravaani_model", trust_remote_code=True).cuda().eval()
    asr._ensure_loaded()
    sp = asr._get_tokenizer()
    results = []
    with torch.inference_mode():
        for row in rows[:a.limit]:
            tokens = sp.encode(clean_text(meta[row["id"]]["transcript"]))
            if not tokens:
                continue
            wav = torch.as_tensor(np.asarray(row["waveform"], np.float32), device="cuda")[None]
            features, lengths = acoustic_features(asr, wav)
            enc, elen = asr.encoder(features.to(asr._io_dtype), lengths)
            tok = torch.tensor(tokens, dtype=torch.int32, device="cuda")
            ids = torch.cat((tok.new_tensor([asr.config.blank_id]), tok))[None]
            h = torch.zeros(asr.config.pred_rnn_layers, 1, asr.config.pred_hidden,
                            device="cuda", dtype=asr._io_dtype)
            logits = asr.decoder_joint(enc, ids, tok.new_tensor([len(tokens) + 1]), h, h.clone())[0]
            logits = logits[..., :asr.config.blank_id + 1].float().contiguous()
            result = {"id": row["id"], "tensor_frames": logits.shape[1],
                      "valid_frames": int(elen.max()), "tokens": len(tokens)}
            def loss(value):
                return torchaudio.functional.rnnt_loss(value, tok[None], elen.int(),
                    tok.new_tensor([len(tokens)]), blank=asr.config.blank_id) / len(tokens)
            try:
                result["old_loss"] = float(loss(logits))
            except RuntimeError as error:
                result["old_error"] = str(error)
            trimmed = logits[:, :int(elen.max()), :len(tokens) + 1].contiguous()
            result["trimmed_loss"] = float(loss(trimmed))
            if not np.isfinite(result["trimmed_loss"]):
                raise RuntimeError("nonfinite trimmed loss")
            if "old_loss" in result and abs(result["old_loss"] - result["trimmed_loss"]) > 1e-5:
                raise RuntimeError("trimming changed an already valid loss")
            results.append(result)
    report = {"clips": len(results), "old_failures": sum("old_error" in r for r in results),
              "trimmed_failures": 0, "results": results}
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
