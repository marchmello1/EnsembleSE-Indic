"""Validate the fingerprint: clean ref must match its own noisy clip (held188 + proxy with LOUD d32 noise),
and must not match anything when its own clip is removed from the index (negative control)."""
import sys, numpy as np, torch
sys.path.insert(0, '/mnt/nvme_disk1/claude_vaani'); from fp import Index
for name, path in (('held188', '/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt'), ('proxy188 (-9 dB SNR)', '/mnt/nvme_disk1/claude_d32proxy188_1013.pt')):
    rows = torch.load(path, map_location='cpu', weights_only=False)
    ix = Index(); ix_neg = Index()
    for i, r in enumerate(rows):
        ix.add(i, np.asarray(r['waveform'], np.float32).reshape(-1))
        if i % 2 == 0: ix_neg.add(i, np.asarray(r['waveform'], np.float32).reshape(-1))
    pos, neg = [], []
    for i, r in enumerate(rows):
        c = np.asarray(r['clean'], np.float32).reshape(-1)
        k, cnt, n = ix.query(c); pos.append((k == i, cnt, n))
        if i % 2 == 1:
            k2, c2, _ = ix_neg.query(c); neg.append(c2)
    P = np.array([p[1] for p in pos]); hit = np.mean([p[0] for p in pos]); N = np.array(neg)
    print(f'{name}: self-match rate {hit:.3f}; matched-vote count p5/p50 {np.percentile(P,5):.0f}/{np.percentile(P,50):.0f}; '
          f'NEGATIVE best-vote p50/p99/max {np.percentile(N,50):.0f}/{np.percentile(N,99):.0f}/{N.max():.0f}')
