import numpy as np, soundfile as sf, torch
def si(e, r):
    n = min(len(e), len(r)); e, r = e[:n]-e[:n].mean(), r[:n]-r[:n].mean()
    s = r*np.dot(e, r)/(np.dot(r, r)+1e-12); x = e-s
    return 10*np.log10((np.dot(s, s)+1e-12)/(np.dot(x, x)+1e-12))
A = []
for r in torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt', map_location='cpu', weights_only=False):
    c = np.asarray(r['clean'], np.float64).reshape(-1); x = np.asarray(r['waveform'], np.float64).reshape(-1); i = str(r['id'])
    A.append([si(x[:len(c)], c)] + [si(sf.read(f'/mnt/nvme_disk1/claude_cfg_{t}_syn/{i}.wav', dtype='float64')[0], c) for t in ('base', 'lBtg5', 'wer5')])
A = np.array(A)
for lo, hi in ((-99, -5), (-5, 0), (0, 10), (10, 99)):
    m = (A[:, 0] >= lo) & (A[:, 0] < hi)
    print(f'SNR [{lo:>3},{hi:>3}) n={m.sum():>3}  base {A[m,1].mean():.3f}  cand {A[m,2].mean()-A[m,1].mean():+.3f}  wer5 {A[m,3].mean()-A[m,1].mean():+.3f}')
h = A[:, 0] < 0
print(f'below 0 dB: cand {A[h,2].mean()-A[h,1].mean():+.3f}  wer5 {A[h,3].mean()-A[h,1].mean():+.3f}')
