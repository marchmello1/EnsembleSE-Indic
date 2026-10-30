"""Learn complex-STFT corrections on released, state-disjoint training pairs."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Conv2d(7,32,3,padding=1),nn.GroupNorm(4,32),nn.SiLU()]
        for d in (1,2,4,8):
            layers += [nn.Conv2d(32,32,3,padding=(1,d),dilation=(1,d)),nn.GroupNorm(4,32),nn.SiLU()]
        layers += [nn.Conv2d(32,2,1)]
        self.net=nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('window',torch.hann_window(512))

    def stft(self,x):
        return torch.stft(x,512,128,window=self.window,return_complex=True)

    def forward(self,noisy,mp,bs):
        z=[self.stft(x) for x in (noisy,mp,bs)]
        magnitude=z[0].abs()
        scale=magnitude+.03*magnitude.mean((-2,-1),keepdim=True)+1e-5
        features=torch.stack([v/scale for s in z for v in (s.real,s.imag)]+
                             [torch.log1p(magnitude)],dim=1)
        correction=self.net(features).tanh()*.5*scale[:,None]
        base=.3*z[1]+.7*z[2]
        enhanced=base+torch.complex(correction[:,0],correction[:,1])
        return torch.istft(enhanced,512,128,window=self.window,length=noisy.shape[-1])


def si(pred,clean):
    pred=pred-pred.mean(-1,keepdim=True)
    clean=clean-clean.mean(-1,keepdim=True)
    target=clean*((pred*clean).sum(-1,keepdim=True)/(clean.square().sum(-1,keepdim=True)+1e-8))
    return 10*torch.log10((target.square().sum(-1)+1e-8)/((pred-target).square().sum(-1)+1e-8))


def evaluate(model,validation,mp_dir,bs_dir,output):
    output.mkdir(parents=True,exist_ok=True)
    scores=[]
    for path in sorted(bs_dir.glob('*.wav')):
        noisy,sr=sf.read(validation/'syntheticNoiseAudio'/path.name,dtype='float32')
        mp,mr=sf.read(mp_dir/path.name,dtype='float32')
        bs,br=sf.read(path,dtype='float32')
        clean,cr=sf.read(validation/'syntheticCleanRefAudio'/path.name,dtype='float32')
        assert sr==mr==br==cr==16000 and len(noisy)==len(mp)==len(bs)==len(clean)
        left=mp-mp.mean();right=bs-bs.mean()
        bs*=float(np.clip(np.dot(left,right)/(np.dot(right,right)+1e-12),.1,10))
        with torch.inference_mode():
            y=model(*[torch.from_numpy(x)[None].cuda() for x in (noisy,mp,bs)])[0]
            reference=torch.from_numpy(clean).cuda()[None]
            baseline=torch.from_numpy(.3*mp+.7*bs).cuda()[None]
            scores.append({'id':path.stem,'candidate':float(si(y[None],reference)),
                           'baseline':float(si(baseline,reference))})
        y=y.cpu().numpy()
        y/=max(1.,float(np.max(np.abs(y)))/.99)
        sf.write(output/path.name,y,sr,subtype='PCM_16')
    result={'clips':scores,'mean':float(np.mean([r['candidate'] for r in scores])),
            'baseline':float(np.mean([r['baseline'] for r in scores]))}
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print('EVAL',result['mean'],result['baseline'],flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('cache',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--validation',type=Path,required=True)
    p.add_argument('--mp-dir',type=Path,required=True)
    p.add_argument('--bs-dir',type=Path,required=True)
    a=p.parse_args()
    rows=torch.load(a.cache,map_location='cpu',weights_only=False)
    held={path.stem for path in a.bs_dir.glob('*.wav')}
    meta={r['segmentFileName'].removesuffix('.wav'):r for r in json.loads((a.validation/'validationMetadata.json').read_text())}
    held_states={meta[i]['state'] for i in held}
    assert len(held)==188 and not held.intersection(r['id'] for r in rows)
    assert not held_states.intersection(r['state'] for r in rows)
    assert len({r['id'] for r in rows})==len(rows)
    print('TRAIN_ROWS',len(rows),'HELD',len(held),'STATES',sorted(held_states),flush=True)
    torch.manual_seed(1011);rng=random.Random(1011)
    model=Fusion().cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    a.output.mkdir(parents=True,exist_ok=True)
    for step in range(1,a.steps+1):
        batch=[]
        for _ in range(4):
            row=rng.choice(rows)
            n=len(row['noisy']);start=rng.randrange(max(1,n-48000+1))
            batch.append([F.pad(row[k][start:start+48000],(0,max(0,48000-n)))
                          for k in ('noisy','mp','bs','clean')])
        noisy,mp,bs,clean=[torch.stack([r[i] for r in batch]).cuda() for i in range(4)]
        pred=model(noisy,mp,bs)
        # Align the training target's arbitrary gain to the processed baseline.
        base=.3*mp+.7*bs
        gain=(base*clean).sum(-1,keepdim=True)/clean.square().sum(-1,keepdim=True).clamp_min(1e-8)
        target=clean*gain.detach()
        spectral=(model.stft(pred)-model.stft(target)).abs().mean()/model.stft(target).abs().mean().clamp_min(1e-4)
        loss=-si(pred,clean).clamp(max=40).mean()+spectral
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
        if step%20==0:
            print('STEP',step,'LOSS',float(loss.detach()),flush=True)
        if step%500==0 or step==a.steps:
            torch.save({'state_dict':model.state_dict(),'step':step,'training_ids':[r['id'] for r in rows]},a.output/f's{step}.pt')
            model.eval();evaluate(model,a.validation,a.mp_dir,a.bs_dir,a.output/f's{step}_hold188');model.train()


if __name__=='__main__':
    main()
