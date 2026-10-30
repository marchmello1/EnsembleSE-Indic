"""BSRNN (deployed bsrnn_extended_s1600) fine-tuned STANDALONE on d32 reused-pool remixes, input RMS .10
(as deployed). Same cache/state exclusions/d32 events/RNNT objective as the Loco and TFGrid d32 runs.
Saves a Lightning-compatible .ckpt (original dict with state_dict replaced) so the package loads it as is."""
import os
os.environ.setdefault('USE_TF', '0')
import json, random, sys, copy
from pathlib import Path
import numpy as np, torch
from transformers import AutoModel
PKG = '/mnt/nvme_disk1/t2_four_neural_codepkg_1011/marchmello01'
sys.path.insert(0, PKG); sys.path.insert(0, '/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from finetune_mpsenet_rnnt import clean_text, rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr
from baseline_code.d_model import SEModel

cache = Path('/mnt/nvme_disk1/t2_locoformer_1010/fusion_training_cache')
manifest = json.loads((cache/'manifest.json').read_text())
seed = 1034; rng = random.Random(seed); torch.manual_seed(seed)
torch.cuda.set_per_process_memory_fraction(.5)
CK = PKG+'/checkpoints/bsrnn_extended_s1600.ckpt'
raw = torch.load(CK, map_location='cpu', weights_only=False)
bsm = SEModel.load_from_checkpoint(CK, map_location='cpu').cuda().train()
params = [q for q in bsm.parameters() if q.requires_grad]
asr = AutoModel.from_pretrained('/mnt/nvme_disk1/sravaani_model', trust_remote_code=True).cuda().eval(); asr._ensure_loaded()
for m in (asr.encoder, asr.decoder_joint):
    for q in m.parameters(): q.requires_grad_(False)
tok = asr._get_tokenizer(); pools = [[], []]
_z = np.load('/mnt/nvme_disk1/t2_d32_bank.npz', allow_pickle=True)
D32 = [torch.from_numpy(np.asarray(_z[k], np.float32)) for k in _z.keys() if np.abs(_z[k]).max() > 0]
for ident in manifest['ids']:
    row = torch.load(cache/(ident+'.pt'), map_location='cpu', weights_only=False)
    assert row['state'] not in manifest['heldout_states']
    t = tok.encode(clean_text(row['transcript']))
    if t and len(row['noisy']) <= 10*16000: pools[int(row['natural'])].append((row, t))
print('POOLS', [len(p) for p in pools], 'TRAINABLE', sum(q.numel() for q in params), flush=True)
opt = torch.optim.AdamW(params, lr=3e-6, weight_decay=1e-6)
for step in range(1, 801):
    natural = rng.random() < .35
    row, tokens = rng.choice(pools[int(natural)])
    x = row['noisy'][None].cuda(); target = None
    if not natural:
        clean = row['clean'][None].cuda(); n = clean.shape[-1]
        if rng.random() < .7:
            act = clean[clean.abs() > .02*clean.abs().max()]; srms = act.square().mean().sqrt().clamp_min(1e-6)
            res = torch.zeros_like(clean); placed = 0; guard = 0
            while placed < .34*n and guard < 60:
                guard += 1; a = D32[rng.randrange(len(D32))].cuda(); L = min(len(a), n); p = rng.randrange(0, max(1, n-L))
                res[0, p:p+L] += a[:L]*(srms*10**(rng.gauss(9., 3.)/20)/a[:L].square().mean().sqrt().clamp_min(1e-9)); placed += L
            x = clean + res
            pk = (x.abs().max()/.95).clamp_min(1); x = x/pk; clean = clean/pk
        target = clean
    xin = x*(0.10/x.pow(2).mean().sqrt().clamp_min(1e-8))
    out, _ = bsm.se_model(xin, torch.tensor([x.shape[-1]], device='cuda'), 16000)
    out = out.reshape(1, -1)[:, :x.shape[-1]]
    if out.shape[-1] < x.shape[-1]: out = torch.nn.functional.pad(out, (0, x.shape[-1]-out.shape[-1]))
    out = out/out.abs().amax(-1, keepdim=True).clamp_min(1e-8)*.9
    recognition = rnnt_loss(asr, out, tokens)
    acoustic = out.new_zeros(()) if natural else neg_sisdr(out, target).clamp(min=-40)
    loss = 2*acoustic + 3*recognition
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward(); g = torch.nn.utils.clip_grad_norm_(params, 3.)
    assert torch.isfinite(g) and float(g) > 0; opt.step()
    if step == 1 or step % 10 == 0: print('STEP', step, 'LOSS', float(loss.detach()), 'RNNT', float(recognition.detach()), 'NAT', natural, 'GRAD', float(g), flush=True)
    if step % 200 == 0:
        ck = copy.copy(raw); ck['state_dict'] = {k: v.detach().cpu() for k, v in bsm.state_dict().items()}
        ck['claude_d32'] = dict(step=step, seed=seed, objective='SI2+RNNT3 standalone, d32 remix .7, input RMS .10')
        d = Path('/mnt/nvme_disk1/claude_d32')/f'bsrnn_d32_1013_s{step}.ckpt'; tmp = d.with_suffix('.partial')
        torch.save(ck, tmp); tmp.replace(d); print('SAVED', d, flush=True)
