"""Uniform audio-only three-network enhancement; no reference/metadata inputs."""
import io
import numpy as np
import soundfile as sf
import torch

def pcm16_roundtrip(wave):
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return sf.read(buffer, dtype='float32')[0]

def fit_length(wave, length):
    return np.pad(wave, (0, max(0, length - len(wave))))[:length]

def enhance(wave, mp, urgent, predictor, cfg, device, core):
    length = len(wave)
    mp_out = pcm16_roundtrip(core.enhance_full_clip(wave, mp, cfg, device))
    tensor = torch.from_numpy(wave).to(device).view(1, -1)
    with torch.inference_mode():
        bs, _ = urgent.se_model(tensor, torch.tensor([length], device=device), 16000)
        pred = predictor(tensor, sr=16000)
    bs = fit_length(bs.detach().cpu().numpy().reshape(-1).astype(np.float32), length)
    bs *= min(1.0, .9 / max(float(np.max(np.abs(bs))), 1e-12))
    bs = pcm16_roundtrip(bs)
    left = mp_out.astype(np.float64) - mp_out.mean()
    right = bs.astype(np.float64) - bs.mean()
    gain = float(np.clip(np.dot(left, right) / max(float(np.dot(right, right)), 1e-12), .1, 10))
    base = .3 * mp_out + .7 * gain * bs
    base /= max(1.0, float(np.max(np.abs(base))) / .99)
    base = pcm16_roundtrip(base.astype(np.float32))
    pred = fit_length(pred.detach().cpu().numpy().reshape(-1), length)
    pred = pred / (np.max(np.abs(pred)) + 1e-8) * np.max(np.abs(wave))
    pred = pcm16_roundtrip(pred)
    output = .7 * base + .3 * pred
    output /= max(1.0, float(np.max(np.abs(output))) / .99)
    if len(output) != length or not np.isfinite(output).all():
        raise ValueError('Invalid enhanced waveform')
    return output.astype(np.float32)
