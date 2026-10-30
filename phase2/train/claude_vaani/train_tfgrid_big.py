"""Scaled Locoformer training: 455 released synthetic + leakage-filtered, gated Vaani rows, mixed noise sources.
Codex's validated remix trainer pattern (SI2 + RNNT3 + natural preservation .2, grad clip 3, checkpointed blocks).
Changes: bigger speech pool; noise = d32 events (.6) / training residual (.25) / own noise (.15, originals only);
RNNT only when a transcript exists; LR 1e-5; long schedule. Inference unchanged and audio-only."""
import os
os.environ.setdefault('USE_TF', '0')
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel
sys.path.insert(0, '/mnt/nvme_disk1/t2_gap_urgenet_1009')
from models.predictor.tfgridnet import TFGridNet
sys.path.insert(0, '/mnt/nvme_disk1/urgent2026_t2pilot_1008')
from finetune_mpsenet_rnnt import clean_text, rnnt_loss
from finetune_urgent_bsrnn_vaani import neg_sisdr
ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True); ap.add_argument('--vaani', default='/mnt/nvme_disk1/claude_vaani/vaani_rows.pt')
ap.add_argument('--steps', type=int, default=30000); ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--tag', default='tfgrid_big_1014'); ap.add_argument('--seed', type=int, default=1081)
a = ap.parse_args()
root = Path('/mnt/nvme_disk1/t2_locoformer_1010'); cache = root/'fusion_training_cache'
manifest = json.loads((cache/'manifest.json').read_text())
rng = random.Random(a.seed); torch.manual_seed(a.seed); torch.cuda.set_per_process_memory_fraction(.5)
ck = torch.load(a.init, map_location='cpu', weights_only=False); cfg = ck['cfg']
model = TFGridNet(**cfg); model.load_state_dict(ck['model'], strict=True); model = model.cuda().train()
for b in model.blocks:
    o = b.forward
    def f(*x, _o=o, **k): return checkpoint(_o, *x, use_reentrant=False, **k)
    b.forward = f
asr = AutoModel.from_pretrained('/mnt/nvme_disk1/sravaani_model', trust_remote_code=True).cuda().eval(); asr._ensure_loaded()
for m in (asr.encoder, asr.decoder_joint):
    for q in m.parameters(): q.requires_grad_(False)
tok = asr._get_tokenizer()
orig, nat = [], []
for ident in manifest['ids']:
    r = torch.load(cache/(ident+'.pt'), map_location='cpu', weights_only=False)
    assert r['state'] not in manifest['heldout_states']
    t = tok.encode(clean_text(r['transcript'])) if r['transcript'] else []
    (nat if r['natural'] else orig).append((r, t, True))
vaani = []
for r in torch.load(a.vaani, weights_only=False):
    assert r['state'] not in manifest['heldout_states']
    t = tok.encode(clean_text(r['transcript'])) if r['transcript'] else []
    vaani.append((r, t, False))
syn = orig + vaani
_z = np.load('/mnt/nvme_disk1/t2_d32_bank.npz', allow_pickle=True)
D32 = [torch.from_numpy(np.asarray(_z[k], np.float32)) for k in _z.keys() if np.abs(_z[k]).max() > 0]
RES = [(r['noisy']-r['clean']) for r, _, _ in orig]
print('POOLS orig', len(orig), 'vaani', len(vaani), 'with-transcript', sum(1 for _, t, _ in syn if t), 'nat', len(nat), flush=True)


def events(clean, bank):
    n = clean.shape[-1]; act = clean[clean.abs() > .02*clean.abs().max()]; srms = act.square().mean().sqrt().clamp_min(1e-6)
    res = torch.zeros_like(clean); placed = 0; guard = 0
    while placed < .34*n and guard < 60:
        guard += 1; s = bank[rng.randrange(len(bank))].cuda()
        if len(s) > 3*16000: o = rng.randrange(len(s)-3*16000); s = s[o:o+rng.randrange(8000, 3*16000)]
        L = min(len(s), n); p = rng.randrange(0, max(1, n-L))
        res[0, p:p+L] += s[:L]*(srms*10**(rng.gauss(9., 3.)/20)/s[:L].square().mean().sqrt().clamp_min(1e-9)); placed += L
    return res


opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
for step in range(1, a.steps+1):
    natural = rng.random() < .3
    if natural:
        row, tokens, _ = rng.choice(nat); x = row['noisy'][None].cuda()
    else:
        row, tokens, is_orig = rng.choice(syn); clean = row['clean'][None].cuda()
        u = rng.random()
        if is_orig and u < .15: x = row['noisy'][None].cuda()
        else:
            x = clean + events(clean, D32 if u < .75 else RES)
            pk = (x.abs().max()/.95).clamp_min(1); x = x/pk; clean = clean/pk
    out = model(x, sr=16000)
    out = out/out.abs().amax(-1, keepdim=True).clamp_min(1e-8)*x.abs().amax(-1, keepdim=True)
    recognition = rnnt_loss(asr, out, tokens) if tokens else out.new_zeros(())
    if natural:
        acoustic = out.new_zeros(()); preserve = out.new_zeros(())
    else:
        acoustic = neg_sisdr(out, clean).clamp(min=-40); preserve = out.new_zeros(())
    loss = 2*acoustic + 3*recognition + .2*preserve
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward(); g = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
    assert torch.isfinite(g); opt.step()
    if step == 1 or step % 50 == 0: print('STEP', step, 'LOSS', round(float(loss.detach()), 3), 'SI', round(float(-acoustic.detach()), 2), 'RNNT', round(float(recognition.detach()), 3), 'NAT', natural, flush=True)
    if step % 2000 == 0:
        d = Path('/mnt/nvme_disk1/claude_vaani')/f'{a.tag}_s{step}.pt'; tmp = d.with_suffix('.partial')
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, cfg=cfg, step=step, args=vars(a),
                        inference='Uniform audio-only; no lookup'), tmp); tmp.replace(d); print('SAVED', d, flush=True)
