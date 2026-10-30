"""Proxy test set for 'new speech, reused noise' = the Phase-2 situation.
Clean speech: held188 clean refs (six states never used in training). Noise: d32 atoms (the pool the
organizers reused), placed as events at Phase-2-like statistics: coverage ~0.34, event RMS ~9 dB above
speech RMS (spread 3 dB), LS-style random gain. Fixed seed. References kept for scoring only."""
import numpy as np, torch, random
SR = 16000; rng = random.Random(1013); nr = np.random.default_rng(1013)
z = np.load('/mnt/nvme_disk1/t2_d32_bank.npz', allow_pickle=True)
atoms = [np.asarray(z[k], np.float64) for k in z.keys()]; atoms = [a for a in atoms if np.abs(a).max() > 0]
out = []
for r in torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt', map_location='cpu', weights_only=False):
    c = np.asarray(r['clean'], np.float64).reshape(-1); n = len(c)
    srms = np.sqrt(np.mean(c[np.abs(c) > 0.02*np.abs(c).max()]**2)) + 1e-9
    noise = np.zeros(n); placed = 0; guard = 0
    while placed < 0.34*n and guard < 60:
        guard += 1; a = atoms[rng.randrange(len(atoms))]; L = min(len(a), n)
        p = rng.randrange(0, max(1, n-L)); seg = a[:L]*np.hanning(L)**0.1
        want = srms*10**(nr.normal(9.0, 3.0)/20)
        noise[p:p+L] += seg*want/(np.sqrt(np.mean(seg**2))+1e-12); placed += L
    x = c + noise; pk = max(1.0, np.abs(x).max()/0.99); x /= pk; c2 = c/pk
    out.append(dict(id=str(r['id']), waveform=torch.from_numpy(x.astype(np.float32)), clean=torch.from_numpy(c2.astype(np.float32)), state=r['state']))
torch.save(out, '/mnt/nvme_disk1/claude_d32proxy188_1013.pt')
snr = [10*np.log10(np.sum(d['clean'].numpy().astype(float)**2)/np.sum((d['waveform'].numpy()-d['clean'].numpy()).astype(float)**2)) for d in out]
print(f'PROXY n={len(out)} input SNR p10/p50/p90 = {np.percentile(snr,10):.2f}/{np.percentile(snr,50):.2f}/{np.percentile(snr,90):.2f} dB; '
      f'share below 0 dB {np.mean(np.array(snr)<0):.2f}')
