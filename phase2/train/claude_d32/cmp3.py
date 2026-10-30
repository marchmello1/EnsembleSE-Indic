import json, sys, numpy as np, soundfile as sf, torch
def si(e, r):
    n = min(len(e), len(r)); e, r = e[:n]-e[:n].mean(), r[:n]-r[:n].mean()
    s = r*np.dot(e, r)/(np.dot(r, r)+1e-12); x = e-s
    return 10*np.log10((np.dot(s, s)+1e-12)/(np.dot(x, x)+1e-12))
T = ["base", "hlrtg6"] + sys.argv[1:]; A = []
for r in torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt', map_location='cpu', weights_only=False):
    c = np.asarray(r['clean'], np.float64).reshape(-1); x = np.asarray(r['waveform'], np.float64).reshape(-1); i = str(r['id'])
    A.append([si(x[:len(c)], c)] + [si(sf.read(f'/mnt/nvme_disk1/claude_cfg_{t}_syn/{i}.wav', dtype='float64')[0], c) for t in T])
A = np.array(A); h = A[:, 0] < 0
for k, t in enumerate(T):
    s = json.load(open(f'/mnt/nvme_disk1/claude_wer_cfg_{t}_syn.json')); n = json.load(open(f'/mnt/nvme_disk1/claude_wer_cfg_{t}_nat.json'))
    d = A[h, k+1].mean() - A[h, 2].mean(); se = s['errors']['enhanced']; ne = n['errors']['enhanced']
    fc = 1.8*d + 0.8*0.32*100*(627-se)/2190
    print(f'{t:<7} hard SI {A[h,k+1].mean():.3f} ({d:+.3f} vs hlrtg6)  syn err {se}  nat err {ne}  -> forecast vs 18.893: {fc:+.3f}')
