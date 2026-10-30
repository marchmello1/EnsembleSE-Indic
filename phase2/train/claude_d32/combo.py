"""Is the frozen four-network pipeline sensitive to INPUT LEVEL?

Domain fact (t2_audio_domain_diagnostic_1011.json): every set is peak-normalised to ~1.0, but Phase-2
RMS is 3.6 dB below training at the median (0.094 vs 0.142) and 4.5 dB below at p10. Longer Phase-2
clips let one loud transient set the peak and push the speech down. If the networks are level-
sensitive, that alone could explain part of the 3.7 dB local->official SI gap (18.45 vs 14.77).

Test: released held188 synthetic clips (clean refs exist), input attenuated by 0/-3/-6/-9 dB, run the
EXACT main.py per-clip path, score every component and the ensemble with SI-SDR vs clean. SI-SDR is
scale-invariant, so a flat curve means level-invariant and this hypothesis dies.
Uses only package code; references enter only at scoring.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, yaml
ROOT = Path('/mnt/nvme_disk1/t2_four_neural_codepkg_1011/marchmello01')
sys.path.insert(0, str(ROOT))
import mp_core as core
from baseline_code.d_model import SEModel
from predictor_tfgridnet import TFGridNet
from tri_neural_enhance import pcm16_roundtrip, fit_length
from standalone.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder

dev = torch.device('cuda')
cfg = core.AttrDict(json.loads((ROOT/'vendor/best_ckpt/config.json').read_text()))
mp = core.load_enhancer(ROOT/'checkpoints/t2_mpdns_publicnoise_hilr_1008.pt', cfg, dev).eval()
bsm = SEModel.load_from_checkpoint(str(ROOT/'checkpoints/bsrnn_extended_s1600.ckpt'), map_location='cpu').to(dev).eval()
ck = torch.load(ROOT/'checkpoints/predictor_fulladapt_s800.pt', map_location='cpu', weights_only=False)
pred = TFGridNet(**ck['cfg']).to(dev).eval(); pred.load_state_dict(ck['model'], strict=True)
lc = yaml.safe_load((ROOT/'locoformer_config.yaml').read_text())
loco = TFLocoformerSeparator(**lc['separator_conf']).to(dev).eval()
w = torch.load(ROOT/'checkpoints/locoformer_dns.pth', map_location='cpu', weights_only=True)
loco.load_state_dict({k.removeprefix('separator.'): v for k, v in w.items()}, strict=True)
enc = STFTEncoder(**lc['encoder_conf']).to(dev).eval(); dec = STFTDecoder(**lc['decoder_conf']).to(dev).eval()



def branches_raw(wave, bs_rms=0.10):
    """Raw outputs of the four networks for one input, before the ensemble arithmetic."""
    L = len(wave)
    mp_out = core.enhance_full_clip(wave, mp, cfg, dev)
    t = torch.from_numpy(np.ascontiguousarray(wave)).to(dev).view(1, -1)
    with torch.inference_mode():
        tb = t*(bs_rms/float(t.pow(2).mean().sqrt().clamp_min(1e-8))) if bs_rms else t
        bs, _ = bsm.se_model(tb, torch.tensor([L], device=dev), 16000)
        pr = pred(t, sr=16000)
        std = t.std(1, keepdim=True).clamp_min(1e-8)
        z, _ = enc(t/std, torch.tensor([L], device=dev)); z = loco(z)[:, 0]
        a, _ = dec(z, torch.tensor([L], device=dev)); a = (a*std)[0].cpu().numpy()
    return dict(mp=fit_length(np.asarray(mp_out, np.float32).reshape(-1), L),
                bs=fit_length(bs.detach().cpu().numpy().reshape(-1).astype(np.float32), L),
                pr=fit_length(pr.detach().cpu().numpy().reshape(-1).astype(np.float32), L),
                lo=fit_length(np.asarray(a, np.float32).reshape(-1), L))


def tta(wave, views):
    """Average each branch over input views; each view is (transform, inverse) on 1-D arrays."""
    acc = None
    for fwd, inv in views:
        o = {k: inv(v) for k, v in branches_raw(fwd(wave)).items()}
        acc = o if acc is None else {k: acc[k]+o[k] for k in acc}
    return {k: v/len(views) for k, v in acc.items()}


def ensemble(wave, br):
    """main.py / tri_neural_enhance arithmetic, applied to (possibly averaged) branch outputs."""
    mp_out = pcm16_roundtrip(br['mp'])
    bs = br['bs'].copy(); bs *= min(1.0, .9/max(float(np.max(np.abs(bs))), 1e-12)); bs = pcm16_roundtrip(bs)
    l_, r_ = mp_out.astype(np.float64)-mp_out.mean(), bs.astype(np.float64)-bs.mean()
    g = float(np.clip(np.dot(l_, r_)/max(float(np.dot(r_, r_)), 1e-12), .1, 10))
    base = .3*mp_out + .7*g*bs; base /= max(1.0, float(np.max(np.abs(base)))/.99)
    base = pcm16_roundtrip(base.astype(np.float32))
    pr = br['pr']/(np.max(np.abs(br['pr']))+1e-8)*np.max(np.abs(wave)); pr = pcm16_roundtrip(pr)
    best = .7*base + .3*pr; best /= max(1.0, float(np.max(np.abs(best)))/.99)
    best = pcm16_roundtrip(best.astype(np.float32))
    a = br['lo'].copy(); a /= max(1, float(np.max(np.abs(a)))/.99); a = pcm16_roundtrip(a)
    return pcm16_roundtrip(.7*best.astype(np.float64) + .3*a.astype(np.float64))



import os

def si(e, r):
    e = np.asarray(e, np.float64); r = np.asarray(r, np.float64); n = min(len(e), len(r))
    e, r = e[:n]-e[:n].mean(), r[:n]-r[:n].mean()
    s = r*np.dot(e, r)/(np.dot(r, r)+1e-12); x = e-s
    return 10*np.log10((np.dot(s, s)+1e-12)/(np.dot(x, x)+1e-12))


def ens_w(wave, br, wl):
    """Frozen ensemble arithmetic, but final Loco weight wl instead of .3."""
    mp_out = pcm16_roundtrip(br['mp'])
    bs = br['bs'].copy(); bs *= min(1.0, .9/max(float(np.max(np.abs(bs))), 1e-12)); bs = pcm16_roundtrip(bs)
    l_, r_ = mp_out.astype(np.float64)-mp_out.mean(), bs.astype(np.float64)-bs.mean()
    g = float(np.clip(np.dot(l_, r_)/max(float(np.dot(r_, r_)), 1e-12), .1, 10))
    base = .3*mp_out + .7*g*bs; base /= max(1.0, float(np.max(np.abs(base)))/.99)
    base = pcm16_roundtrip(base.astype(np.float32))
    pr = br['pr']/(np.max(np.abs(br['pr']))+1e-8)*np.max(np.abs(wave)); pr = pcm16_roundtrip(pr)
    best = .7*base + .3*pr; best /= max(1.0, float(np.max(np.abs(best)))/.99)
    best = pcm16_roundtrip(best.astype(np.float32))
    lo = br['lo'].copy(); lo /= max(1, float(np.max(np.abs(lo)))/.99); lo = pcm16_roundtrip(lo)
    return pcm16_roundtrip((1-wl)*best.astype(np.float64) + wl*lo.astype(np.float64))


frozen_lo = {k: v.clone() for k, v in loco.state_dict().items()}
frozen_pr = {k: v.clone() for k, v in pred.state_dict().items()}
D = '/mnt/nvme_disk1/claude_d32/'
configs = [
    ('frozen all, wL.3',            None,                         None,                          0.3),
    ('loco s600, wL.5',             D+'loco_d32_1013_s600.pt',    None,                          0.5),
    ('loco B s1600, wL.3',          D+'loco_d32b_1013_s1600.pt',  None,                          0.3),
    ('loco B s1600, wL.5',          D+'loco_d32b_1013_s1600.pt',  None,                          0.5),
    ('tfgrid d32, loco frozen .3',  None,                         D+'tfgrid_d32_1013_s800.pt',   0.3),
    ('tfgrid d32 + loco B, wL.5',   D+'loco_d32b_1013_s1600.pt',  D+'tfgrid_d32_1013_s800.pt',   0.5),
    ('tfgrid d32 + loco s600, wL.5',D+'loco_d32_1013_s600.pt',    D+'tfgrid_d32_1013_s800.pt',   0.5),
]
sets = {'held188': torch.load('/mnt/nvme_disk1/t2_binary_pred_hold_0924_t0.50.pt', map_location='cpu', weights_only=False),
        'proxy188': torch.load('/mnt/nvme_disk1/claude_d32proxy188_1013.pt', weights_only=False)}
for name, lp, pp, wl in configs:
    loco.load_state_dict(torch.load(lp, map_location='cpu', weights_only=False)['model'] if lp else frozen_lo, strict=True); loco.eval()
    pred.load_state_dict(torch.load(pp, map_location='cpu', weights_only=False)['model'] if pp else frozen_pr, strict=True); pred.eval()
    out = []
    for sname, rows in sets.items():
        E, LO, PR, snr = [], [], [], []
        for r in rows:
            x = pcm16_roundtrip(np.asarray(r['waveform'], np.float32).reshape(-1)); c = np.asarray(r['clean'], np.float64).reshape(-1)
            br = branches_raw(x, 0.10)
            E.append(si(ens_w(x, br, wl), c)); LO.append(si(br['lo'], c)); PR.append(si(br['pr'], c)); snr.append(si(x[:len(c)], c))
        E, LO, PR, snr = map(np.array, (E, LO, PR, snr)); h = snr < 0
        out.append(f'{sname}: ENS {E.mean():.3f} hard {E[h].mean():.3f} | loco {LO[h].mean():.3f} tfgrid {PR[h].mean():.3f} (hard)')
    print(f'{name:<30} ' + '   '.join(out), flush=True)
print('COMBO_DONE')
