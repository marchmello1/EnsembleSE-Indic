import sys
p = '/mnt/nvme_disk1/claude_d32/train_loco_d32_1013.py'; s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, ('anchor', old[:60]); s = s.replace(old, new)
rep('"""Source/state-excluded stochastic residual-remix training; audio-only inference."""',
    '"""Codex remix trainer, ONE change: synthetic remix noise = d32 reused-pool events at Phase-2-like stats.\n'
    'Speech/state exclusions, objectives, LR, clipping unchanged. Inference stays audio-only (no lookup)."""\nimport numpy as np')
rep("seed=1027;", "seed=1031;")
rep("tokenizer=asr._get_tokenizer();pools=[[],[]]",
    "tokenizer=asr._get_tokenizer();pools=[[],[]]\n"
    "_z=np.load('/mnt/nvme_disk1/t2_d32_bank.npz',allow_pickle=True)\n"
    "D32=[torch.from_numpy(np.asarray(_z[k],np.float32)) for k in _z.keys() if np.abs(_z[k]).max()>0]")
rep("    remixed=not natural and rng.random()<.5", "    remixed=not natural and rng.random()<.7")
old_block = s[s.index("        noise_row,_=rng.choice(pools[0])"):s.index("        x=clean+residual")]
rep(old_block,
    "        clean=row['clean'][None].cuda();n=clean.shape[-1]\n"
    "        act=clean[clean.abs()>.02*clean.abs().max()];srms=act.square().mean().sqrt().clamp_min(1e-6)\n"
    "        residual=torch.zeros_like(clean);placed=0;guard=0\n"
    "        while placed<.34*n and guard<60:\n"
    "            guard+=1;a=D32[rng.randrange(len(D32))].cuda();L=min(len(a),n);p=rng.randrange(0,max(1,n-L))\n"
    "            seg=a[:L]*(srms*10**(rng.gauss(9.,3.)/20)/a[:L].square().mean().sqrt().clamp_min(1e-9))\n"
    "            residual[0,p:p+L]+=seg;placed+=L\n")
rep("for step in range(1,1601):", "for step in range(1,601):")
rep("dest=root/f'locoformer_remix_1011_s{step}.pt'", "dest=Path('/mnt/nvme_disk1/claude_d32')/f'loco_d32_1013_s{step}.pt'")
rep("synthetic_remix_prob=.5,snr_range=[0,20]", "synthetic_remix_prob=.7,noise='d32 reused-pool events, +9dB(3) over speech, coverage .34'")
open(p, 'w').write(s); print('PATCHED trainer')
