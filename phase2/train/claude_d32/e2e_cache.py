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



# Stage 1 of end-to-end ASR training: cache the NON-Locoformer part of the shipped ensemble on d32 remixes.
import json, random
from pathlib import Path
pred.load_state_dict(torch.load('/mnt/nvme_disk1/claude_d32/tfgrid_d32_1013_s800.pt', map_location='cpu', weights_only=False)['model'], strict=True); pred.eval()
loco.load_state_dict(torch.load('/mnt/nvme_disk1/claude_d32/loco_d32b_1013_s1600.pt', map_location='cpu', weights_only=False)['model'], strict=True); loco.eval()
cache = Path('/mnt/nvme_disk1/t2_locoformer_1010/fusion_training_cache')
manifest = json.loads((cache/'manifest.json').read_text())
rows = [torch.load(cache/(i+'.pt'), map_location='cpu', weights_only=False) for i in manifest['ids']]
rows = [r for r in rows if r['state'] not in manifest['heldout_states'] and len(r['noisy']) <= 10*16000]
z = np.load('/mnt/nvme_disk1/t2_d32_bank.npz', allow_pickle=True)
D32 = [np.asarray(z[k], np.float32) for k in z.keys() if np.abs(z[k]).max() > 0]
rng = random.Random(1051); syn = [r for r in rows if not r['natural']]; nat = [r for r in rows if r['natural']]
print('rows syn/nat', len(syn), len(nat), flush=True)


def best_part(wave, br):
    mp_out = pcm16_roundtrip(br['mp'])
    bs = br['bs'].copy(); bs *= min(1.0, .9/max(float(np.max(np.abs(bs))), 1e-12)); bs = pcm16_roundtrip(bs)
    l_, r_ = mp_out.astype(np.float64)-mp_out.mean(), bs.astype(np.float64)-bs.mean()
    g = float(np.clip(np.dot(l_, r_)/max(float(np.dot(r_, r_)), 1e-12), .1, 10))
    base = .3*mp_out + .7*g*bs; base /= max(1.0, float(np.max(np.abs(base)))/.99)
    base = pcm16_roundtrip(base.astype(np.float32))
    pr = br['pr']/(np.max(np.abs(br['pr']))+1e-8)*np.max(np.abs(wave)); pr = pcm16_roundtrip(pr)
    best = .7*base + .3*pr; best /= max(1.0, float(np.max(np.abs(best)))/.99)
    return pcm16_roundtrip(best.astype(np.float32))


out = []
for k in range(1500):
    natural = rng.random() < .4
    r = rng.choice(nat if natural else syn)
    x = np.asarray(r['noisy'], np.float32).reshape(-1); clean = None
    if not natural:
        clean = np.asarray(r['clean'], np.float32).reshape(-1)
        if rng.random() < .7:
            n = len(clean); act = clean[np.abs(clean) > .02*np.abs(clean).max()]; srms = float(np.sqrt(np.mean(act**2)))+1e-6
            res = np.zeros(n, np.float32); placed = 0; guard = 0
            while placed < .34*n and guard < 60:
                guard += 1; a = D32[rng.randrange(len(D32))]; L = min(len(a), n); p = rng.randrange(0, max(1, n-L))
                res[p:p+L] += a[:L]*(srms*10**(rng.gauss(9., 3.)/20)/(float(np.sqrt(np.mean(a[:L]**2)))+1e-9)); placed += L
            x = clean + res; pk = max(1.0, float(np.abs(x).max())/.95); x = x/pk; clean = clean/pk
    x = pcm16_roundtrip(x.astype(np.float32))
    br = branches_raw(x, 0.10)
    lo = br['lo'].copy(); lo /= max(1, float(np.max(np.abs(lo)))/.99); lo = pcm16_roundtrip(lo)
    best = best_part(x, br)
    out.append(dict(x=torch.from_numpy(x), clean=None if clean is None else torch.from_numpy(clean),
                    best=torch.from_numpy(best), frozen_final=torch.from_numpy((.5*best + .5*lo).astype(np.float32)),
                    transcript=r['transcript'], natural=natural, id=r['id']))
    if k % 100 == 0: print(k, flush=True)
torch.save(out, '/mnt/nvme_disk1/claude_d32/e2e_cache_1013.pt')
print('E2E_CACHE_DONE', len(out))
