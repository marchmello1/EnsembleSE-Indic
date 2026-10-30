#!/bin/bash
# Candidate: TTA4 package + d32 Loco(B s1600) + d32 TFGrid(s800) + final Loco weight .5. Verify on held188.
set -eu
SRC=/mnt/nvme_disk1/t2_claude_tta4_codepkg_1012/marchmello01
DST=/mnt/nvme_disk1/t2_claude_d32cand_codepkg_1013/marchmello01
ENV=/mnt/nvme_disk1/t2_four_neural_codepkg_1011
rm -rf "$(dirname $DST)"; mkdir -p "$(dirname $DST)"; cp -a "$SRC" "$DST"
/usr/bin/python3 - <<'PY'
import torch
D = '/mnt/nvme_disk1/t2_claude_d32cand_codepkg_1013/marchmello01/checkpoints/'
lo = torch.load('/mnt/nvme_disk1/claude_d32/loco_d32b_1013_s1600.pt', map_location='cpu', weights_only=False)['model']
torch.save({k: v.contiguous() for k, v in lo.items()}, D+'locoformer_d32.pth')            # plain tensor dict, weights_only-loadable
tg = torch.load('/mnt/nvme_disk1/claude_d32/tfgrid_d32_1013_s800.pt', map_location='cpu', weights_only=False)
torch.save(dict(cfg=tg['cfg'], model=tg['model']), D+'predictor_d32_s800.pt')
print('CHECKPOINTS written')
PY
/usr/bin/python3 - <<'PY'
p = '/mnt/nvme_disk1/t2_claude_d32cand_codepkg_1013/marchmello01/'
m = open(p+'main.py').read()
for a, b in (("checkpoints/predictor_fulladapt_s800.pt", "checkpoints/predictor_d32_s800.pt"),
             ("checkpoints/locoformer_dns.pth", "checkpoints/locoformer_d32.pth")):
    assert m.count(a) == 1, a; m = m.replace(a, b)
open(p+'main.py', 'w').write(m)
t = open(p+'tta_enhance.py').read()
a = "    out = pcm16_roundtrip(.7 * best.astype(np.float64) + .3 * lo.astype(np.float64))"
assert t.count(a) == 1
t = t.replace(a, "    out = pcm16_roundtrip((1 - LOCO_WEIGHT) * best.astype(np.float64) + LOCO_WEIGHT * lo.astype(np.float64))")
t = t.replace("BSRNN_INPUT_RMS = 0.10", "BSRNN_INPUT_RMS = 0.10\nLOCO_WEIGHT = 0.5")
open(p+'tta_enhance.py', 'w').write(t); print('PATCHED main.py + tta_enhance.py')
PY
rm -f "$DST/checkpoints/locoformer_dns.pth" "$DST/checkpoints/predictor_fulladapt_s800.pt"
OUT=/mnt/nvme_disk1/claude_h188_d32cand_pkg_1013; rm -rf "$OUT"
cd "$DST"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 TRACK2_ASR_PYTHON=$ENV/asrenv/bin/python PYTHONNOUSERSITE=1 "$ENV/cleanenv/bin/python" main.py /mnt/nvme_disk1/claude_h188_input_1012 "$OUT" 2>&1 | grep -E "COMPLETE|Error|Traceback" || true
/usr/bin/python3 - <<'PY'
import numpy as np, soundfile as sf, torch
def si(e, r):
    n = min(len(e), len(r)); e, r = e[:n]-e[:n].mean(), r[:n]-r[:n].mean()
    s = r*np.dot(e, r)/(np.dot(r, r)+1e-12); x = e-s
    return 10*np.log10((np.dot(s, s)+1e-12)/(np.dot(x, x)+1e-12))
A = []
for r in torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt', map_location='cpu', weights_only=False):
    c = np.asarray(r['clean'], np.float64).reshape(-1); x = np.asarray(r['waveform'], np.float64).reshape(-1)
    A.append((si(x[:len(c)], c), si(sf.read('/mnt/nvme_disk1/claude_h188_d32cand_pkg_1013/'+str(r['id'])+'.wav', dtype='float64')[0], c)))
A = np.array(A); h = A[:, 0] < 0
print(f'PACKAGED d32cand held188 SI {A[:,1].mean():.4f} (hard {A[h,1].mean():.4f}); refs: tta4 pkg 18.8134, lBtg5 no-TTA 18.5805')
PY
echo BUILD_CAND_DONE
