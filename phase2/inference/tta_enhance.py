"""Uniform audio-only enhancement: four networks, test-time augmentation, fixed ensemble.

Every clip is processed identically. For each input view (polarity +/-1 x leading-zero shift 0/80
samples) the four networks run on the transformed input, the transform is undone on each output, and
the branch outputs are averaged across views. The fixed ensemble arithmetic of the frozen candidate is
then applied unchanged. BSRNN receives its input normalised to RMS 0.10 (it is input-level sensitive).
No reference, metadata, router, dictionary or cached output is used.
"""
import numpy as np
import torch
from tri_neural_enhance import pcm16_roundtrip, fit_length

VIEWS = [(k, s) for k in (0, 40, 80, 120) for s in (1, -1)]
BSRNN_INPUT_RMS = 0.10
LOCO_WEIGHT = 0.7
PRED_WEIGHT = 0.7
LOCO_MIX = 0.5            # share of the second (original-data) Locoformer inside the Locoformer slot


def branches(wave, mp, bsm, pred, loco, enc, dec, cfg, device, core):
    n = len(wave)
    mp_out = core.enhance_full_clip(wave, mp, cfg, device)
    t = torch.from_numpy(np.ascontiguousarray(wave)).to(device).view(1, -1)
    lens = torch.tensor([n], device=device)
    with torch.inference_mode():
        tb = t * (BSRNN_INPUT_RMS / t.pow(2).mean().sqrt().clamp_min(1e-8))
        bs, _ = bsm.se_model(tb, lens, 16000)
        pr = pred(t, sr=16000)
        std = t.std(1, keepdim=True).clamp_min(1e-8)
        z, _ = enc(t / std, lens)
        models = loco if isinstance(loco, (tuple, list)) else (loco,)
        outs = []
        for m in models:
            o, _ = dec(m(z)[:, 0], lens)
            outs.append((o * std)[0].cpu().numpy())
        n1 = np.abs(outs[0]).max() + 1e-8
        lo = sum(o / (np.abs(o).max() + 1e-8) for o in outs) / len(outs) * n1
    return dict(mp=fit_length(np.asarray(mp_out, np.float32).reshape(-1), n),
                bs=fit_length(bs.detach().cpu().numpy().reshape(-1).astype(np.float32), n),
                pr=fit_length(pr.detach().cpu().numpy().reshape(-1).astype(np.float32), n),
                lo=fit_length(np.asarray(lo, np.float32).reshape(-1), n))


def averaged_branches(wave, *models):
    acc = None
    for k, s in VIEWS:
        x = s * np.concatenate([np.zeros(k, wave.dtype), wave])
        out = {name: s * v[k:] for name, v in branches(x, *models).items()}
        acc = out if acc is None else {name: acc[name] + out[name] for name in acc}
    return {name: v / len(VIEWS) for name, v in acc.items()}


def ensemble(wave, br):
    """Fixed ensemble arithmetic of the frozen candidate (MP .147, BSRNN .343, TFGrid .21, Loco .30)."""
    mp_out = pcm16_roundtrip(br['mp'])
    bs = br['bs'].copy()
    bs *= min(1.0, .9 / max(float(np.max(np.abs(bs))), 1e-12))
    bs = pcm16_roundtrip(bs)
    left = mp_out.astype(np.float64) - mp_out.mean()
    right = bs.astype(np.float64) - bs.mean()
    gain = float(np.clip(np.dot(left, right) / max(float(np.dot(right, right)), 1e-12), .1, 10))
    base = .3 * mp_out + .7 * gain * bs
    base /= max(1.0, float(np.max(np.abs(base))) / .99)
    base = pcm16_roundtrip(base.astype(np.float32))
    pr = br['pr'] / (np.max(np.abs(br['pr'])) + 1e-8) * np.max(np.abs(wave))
    pr = pcm16_roundtrip(pr)
    best = (1 - PRED_WEIGHT) * base + PRED_WEIGHT * pr
    best /= max(1.0, float(np.max(np.abs(best))) / .99)
    best = pcm16_roundtrip(best.astype(np.float32))
    lo = br['lo'].copy()
    lo /= max(1, float(np.max(np.abs(lo))) / .99)
    lo = pcm16_roundtrip(lo)
    out = pcm16_roundtrip((1 - LOCO_WEIGHT) * best.astype(np.float64) + LOCO_WEIGHT * lo.astype(np.float64))
    if len(out) != len(wave) or not np.isfinite(out).all():
        raise ValueError('Invalid enhanced waveform')
    return out
