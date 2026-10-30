#!/bin/bash
set -eu
SRC=/mnt/nvme_disk1/t2_claude_d32cand_codepkg_1013/marchmello01
DST=/mnt/nvme_disk1/t2_claude_long6_codepkg_1013/marchmello01
ENV=/mnt/nvme_disk1/t2_four_neural_codepkg_1011
rm -rf "$(dirname $DST)"; mkdir -p "$(dirname $DST)"; cp -a "$SRC" "$DST"
/usr/bin/python3 - <<'PY'
import torch
D = '/mnt/nvme_disk1/t2_claude_long6_codepkg_1013/marchmello01/'
lo = torch.load('/mnt/nvme_disk1/claude_d32/loco_d32long_1013_s3200.pt', map_location='cpu', weights_only=False)['model']
torch.save({k: v.contiguous() for k, v in lo.items()}, D+'checkpoints/locoformer_d32.pth')
tg = torch.load('/mnt/nvme_disk1/claude_d32/tfgrid_d32long_1013_s2000.pt', map_location='cpu', weights_only=False)
torch.save(dict(cfg=tg['cfg'], model=tg['model']), D+'checkpoints/predictor_d32_s800.pt')
t = open(D+'tta_enhance.py').read(); assert t.count('LOCO_WEIGHT = 0.5') == 1
open(D+'tta_enhance.py', 'w').write(t.replace('LOCO_WEIGHT = 0.5', 'LOCO_WEIGHT = 0.6')); print('LONG6 package written')
PY
OUT=/mnt/nvme_disk1/claude_h188_long6_pkg_1013; rm -rf "$OUT"; cd "$DST"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 TRACK2_ASR_PYTHON=$ENV/asrenv/bin/python PYTHONNOUSERSITE=1 "$ENV/cleanenv/bin/python" main.py /mnt/nvme_disk1/claude_h188_input_1012 "$OUT" 2>&1 | grep -E "COMPLETE|Error|Traceback" || true
/usr/bin/python3 - <<'PY'
import numpy as np, soundfile as sf, torch
def si(e, r):
    n = min(len(e), len(r)); e, r = e[:n]-e[:n].mean(), r[:n]-r[:n].mean()
    s = r*np.dot(e, r)/(np.dot(r, r)+1e-12); x = e-s
    return 10*np.log10((np.dot(s, s)+1e-12)/(np.dot(x, x)+1e-12))
A = []
for r in torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt', map_location='cpu', weights_only=False):
    c = np.asarray(r['clean'], np.float64).reshape(-1); x = np.asarray(r['waveform'], np.float64).reshape(-1); i = str(r['id'])
    A.append((si(x[:len(c)], c), si(sf.read('/mnt/nvme_disk1/claude_h188_long6_pkg_1013/'+i+'.wav', dtype='float64')[0], c),
              si(sf.read('/mnt/nvme_disk1/claude_h188_d32cand_pkg_1013/'+i+'.wav', dtype='float64')[0], c)))
A = np.array(A); h = A[:, 0] < 0
print(f'PACKAGED long6 held188 SI {A[:,1].mean():.4f} hard {A[h,1].mean():.4f} | shipped d32cand {A[:,2].mean():.4f} hard {A[h,2].mean():.4f} | hard delta {A[h,1].mean()-A[h,2].mean():+.4f}')
PY
echo BUILD_LONG6_DONE
