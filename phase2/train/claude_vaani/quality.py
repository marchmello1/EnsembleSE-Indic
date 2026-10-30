"""Audio-only 'cleanliness' score: speech-to-floor dynamic range of short-frame energies (p95 - p10, dB),
plus the fraction of near-silent frames. Calibrated on released CLEAN refs vs natural NOISY recordings."""
import os, numpy as np, soundfile as sf, torch


def score(x):
    x = np.asarray(x, np.float64)
    n = 1 + max(0, (len(x)-400)//160); idx = np.arange(400)[None, :] + 160*np.arange(n)[:, None]
    e = 10*np.log10((x[idx]**2).mean(1) + 1e-12)
    return float(np.percentile(e, 95) - np.percentile(e, 10))


if __name__ == '__main__':
    held = torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt', map_location='cpu', weights_only=False)
    clean = [score(np.asarray(r['clean']).reshape(-1)) for r in held]
    noisy = [score(np.asarray(r['waveform']).reshape(-1)) for r in held]
    nd = '/mnt/nvme_disk1/official_validation_20260907/naturalNoisyAudio'
    nat = [score(sf.read(os.path.join(nd, f), dtype='float64')[0]) for f in sorted(os.listdir(nd))[:400]]
    for name, v in (('released CLEAN refs', clean), ('synthetic noisy inputs', noisy), ('natural recordings', nat)):
        v = np.array(v); print(f'{name:<24} p10/p50/p90 = {np.percentile(v,10):.1f}/{np.percentile(v,50):.1f}/{np.percentile(v,90):.1f} dB')
