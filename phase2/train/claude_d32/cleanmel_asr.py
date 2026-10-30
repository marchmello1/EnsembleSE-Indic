"""Does a GENERATIVE (vocoder) enhancer recognise better on this data? CleanMel L-mask + Vocos, the official
inference path (app.py), on held188 synthetic. Writes WAVs for SraVaani scoring; also reports SI-SDR."""
import os, sys
import numpy as np, soundfile as sf, torch, yaml
sys.path.insert(0, '/mnt/nvme_disk1/t2_cleanmel_1010')
os.chdir('/mnt/nvme_disk1/t2_cleanmel_1010')
from model.arch.cleanmel import CleanMel
from model.vocos.offline.pretrained import Vocos
from model.io.stft import InputSTFT, TargetMel
DEV = torch.device('cuda'); W = '/mnt/nvme_disk1/t2_cleanmel_weights_1010/ckpts/'
torch.cuda.set_per_process_memory_fraction(.35)
cfg = yaml.safe_load(open('configs/model/cleanmel_offline.yaml'))['model']['arch']['init_args']
model = CleanMel(**cfg); model.load_state_dict(torch.load(W+'CleanMel/offline_CleanMel_L_mask.ckpt', map_location='cpu', weights_only=False), strict=True)
for m in model.modules():
    if hasattr(m, 'use_fast_path'): m.use_fast_path = False
model = model.eval().to(DEV)
vocos = Vocos.from_hparams(config_path='./configs/model/vocos_offline.yaml')
vocos = Vocos.from_pretrained(None, W+'Vocos/vocos_offline.pt', model=vocos).eval().to(DEV)
stft = InputSTFT(n_fft=512, n_win=512, n_hop=128, normalize=False, center=True, onesided=True, online=False).eval().to(DEV)
mel = TargetMel(sample_rate=16000, n_fft=512, n_win=512, n_hop=128, n_mels=80, f_min=0, f_max=8000, power=2, center=True,
                normalize=False, onesided=True, mel_norm='slaney', mel_scale='slaney', librosa_mel=True, online=False).eval().to(DEV)
OUT = '/mnt/nvme_disk1/claude_cleanmel_h188'; os.makedirs(OUT, exist_ok=True)


def si(e, r):
    n = min(len(e), len(r)); e, r = e[:n]-e[:n].mean(), r[:n]-r[:n].mean()
    s = r*np.dot(e, r)/(np.dot(r, r)+1e-12); x = e-s
    return 10*np.log10((np.dot(s, s)+1e-12)/(np.dot(x, x)+1e-12))


S = []
for r in torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt', map_location='cpu', weights_only=False):
    x = torch.as_tensor(np.asarray(r['waveform'], np.float32)).reshape(1, -1).to(DEV)
    with torch.inference_mode():
        X, Xn = stft(x); M = torch.sigmoid(model(X, inference=False)).squeeze()
        Y = torch.square(M*(torch.sqrt(mel(x, Xn).squeeze())+1e-10))
        y = vocos(torch.log(torch.clamp(Y, min=1e-5)).unsqueeze(0), Xn).clamp(-1, 1).flatten().cpu().numpy()
    L = x.numel(); y = np.pad(y, (0, max(0, L-len(y))))[:L]
    sf.write(f"{OUT}/{r['id']}.wav", y, 16000, subtype='PCM_16')
    S.append(si(y, np.asarray(r['clean'], np.float64).reshape(-1)))
print(f'CleanMel vocoder held188 SI mean {np.mean(S):.3f}; files {len(os.listdir(OUT))}', flush=True)
