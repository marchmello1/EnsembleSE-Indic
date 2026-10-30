"""Recalibrate the leak threshold against the FULL Phase-2 index.
Negatives: FLEURS read speech (cannot be in Phase 2). Positives: the held188 synthetic clips' clean refs are NOT in
Phase 2 either, so for positives use Phase-2 inputs themselves re-noised by ESC-like mixing is not possible -- instead
use a direct positive: each Phase-2 input queried with its own audio after adding 0 dB white noise."""
import glob, pickle, sys, numpy as np, soundfile as sf
sys.path.insert(0, '/mnt/nvme_disk1/claude_vaani'); from fp import landmarks
H = pickle.load(open('/mnt/nvme_disk1/claude_vaani/phase2_fp_index.pkl', 'rb'))
def best(x):
    v = {}
    for h, t in landmarks(x, 16000):
        for k, tt in H.get(h, ()): key = (k, tt-t); v[key] = v.get(key, 0)+1
    if not v: return 0, 0
    top = sorted(v.values(), reverse=True); return top[0], len(top)
neg = []
for f in sorted(glob.glob('/mnt/nvme_disk1/csfleurs/**/*.wav', recursive=True))[:400]:
    x, sr = sf.read(f, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    if sr != 16000: continue
    neg.append(best(x)[0])
rng = np.random.default_rng(0); pos = []
for f in sorted(glob.glob('/mnt/nvme_disk1/indoml_phase2_input_0924/audio/*.wav'))[:150]:
    x, _ = sf.read(f, dtype='float32'); x = x.mean(1) if x.ndim > 1 else x
    s = x[: min(len(x), 16000*5)]                     # a 5-s excerpt, like a Vaani utterance inside a test clip
    s = s + rng.standard_normal(len(s)).astype(np.float32)*np.sqrt(np.mean(s**2))   # 0 dB extra noise
    pos.append(best(s)[0])
neg, pos = np.array(neg), np.array(pos)
print(f'NEG (FLEURS, n={len(neg)}) best-vote p50/p99/max = {np.percentile(neg,50):.0f}/{np.percentile(neg,99):.0f}/{neg.max()}')
print(f'POS (Phase-2 excerpt +0 dB noise, n={len(pos)}) p1/p5/p50 = {np.percentile(pos,1):.0f}/{np.percentile(pos,5):.0f}/{np.percentile(pos,50):.0f}')
for th in (8, 12, 16, 20, 25, 30):
    print(f'  threshold {th:>2}: false-positive rate {np.mean(neg>=th):.3f}  recall {np.mean(pos>=th):.3f}')
