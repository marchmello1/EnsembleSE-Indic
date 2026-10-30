#!/usr/bin/env python3
"""Fine-tune a tag-conditioned MP-SENet against SI-SDR and frozen SraVaani RNNT."""
import argparse, json, random, re, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
from finetune_category_mpsenet import TOP_TAGS, tag_masks, neg_sisdr

def clean_text(s):
    return ' '.join(re.sub(r'<[^>]*>|\[[^\]]*\]', ' ', s).split())

def acoustic_features(asr, wav):
    """SraVaani's exact differentiable preprocessor (its public wrapper uses no_grad)."""
    p=asr._p;n_fft=p['n_fft'];hop=p['hop_length'];win=p['win_length']
    x=torch.cat((wav[:,:1],wav[:,1:]-p['preemph']*wav[:,:-1]),1)
    spec=torch.stft(x,n_fft,hop_length=hop,win_length=win,window=asr._window.float(),
                    center=True,pad_mode='constant',return_complex=True)
    mag=spec.abs().pow(p['mag_power']);mel=torch.matmul(asr._fb.float(),mag)
    mel=torch.log(mel+p['log_zero_guard_value'])
    flen=torch.tensor([wav.shape[-1]//hop],dtype=torch.long,device=wav.device)
    valid=torch.arange(mel.shape[-1],device=wav.device)[None]<flen[:,None]
    count=valid.sum(1).clamp_min(2)
    mean=torch.where(valid[:,None],mel,0).sum(-1)/count[:,None]
    delta=torch.where(valid[:,None],mel-mean[:,:,None],0)
    std=torch.sqrt(delta.square().sum(-1)/(count[:,None]-1)).nan_to_num()+p['CONSTANT']
    feature=(mel-mean[:,:,None])/std[:,:,None]
    feature=feature.masked_fill(~valid[:,None],p['pad_value'])
    return feature,flen

def rnnt_loss(asr,wav,tokens,encoded=None):
    if encoded is None:
        feature,flen=acoustic_features(asr,wav)
        enc,elen=asr.encoder(feature.to(asr._io_dtype),flen)
    else:
        enc,elen=encoded
    tok=torch.tensor(tokens,dtype=torch.int32,device=wav.device)
    ids=torch.cat((tok.new_tensor([asr.config.blank_id]),tok))[None]
    h=torch.zeros(asr.config.pred_rnn_layers,1,asr.config.pred_hidden,
                  device=wav.device,dtype=asr._io_dtype)
    logits=asr.decoder_joint(enc,ids,torch.tensor([len(tokens)+1],dtype=torch.int32,device=wav.device),h,h.clone())[0]
    valid_time=int(elen.max().item())
    if valid_time > logits.shape[1]:
        raise RuntimeError(f'RNNT encoder length {valid_time} exceeds logits {tuple(logits.shape)}')
    # The exported joint can retain padded encoder frames. Torchaudio requires
    # the time dimension to equal the maximum supplied logit length.
    logits=logits[:,:valid_time,:len(tokens)+1,:asr.config.blank_id+1].float().contiguous()
    return torchaudio.functional.rnnt_loss(logits,tok[None],elen.int(),
        torch.tensor([len(tokens)],dtype=torch.int32,device=wav.device),
        blank=asr.config.blank_id)/len(tokens)

def main():
    p=argparse.ArgumentParser();p.add_argument('train_cache',type=Path);p.add_argument('metadata',type=Path)
    p.add_argument('mp_repo',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--asr-dir',type=Path,default=Path('/mnt/nvme_disk1/SraVaani_local'))
    p.add_argument('--steps',type=int,default=300);p.add_argument('--lr',type=float,default=1e-6)
    p.add_argument('--rnnt-weight',type=float,default=3.0);p.add_argument('--max-seconds',type=float,default=8.0)
    p.add_argument('--natural-cache',type=Path,
                   help='State-disjoint natural audio with predicted event spans')
    p.add_argument('--natural-prob',type=float,default=0.,
                   help='Fraction of updates using natural audio and transcript supervision')
    p.add_argument('--natural-identity-weight',type=float,default=3.,
                   help='Natural-audio distortion penalty relative to RNNT loss')
    p.add_argument('--natural-full-mask',action='store_true',
                   help='Supervise every natural clip through the same full-clip neural condition')
    p.add_argument('--synthetic-full-mask',action='store_true',
                   help='Train synthetic clips through the full-clip inference condition')
    p.add_argument('--seed',type=int,default=1008);a=p.parse_args()
    if not 0 <= a.natural_prob <= 1:
        p.error('--natural-prob must be in [0,1]')
    if a.natural_prob and not a.natural_cache:
        p.error('--natural-prob requires --natural-cache')
    random.seed(a.seed);torch.manual_seed(a.seed);np.random.seed(a.seed)
    sys.path.insert(0,str(a.mp_repo))
    from env import AttrDict
    from dataset import mag_pha_stft,mag_pha_istft
    from models.model import MPNet,DenseEncoder
    class TagMPNet(MPNet):
        def __init__(self,cfg):
            super().__init__(cfg);self.dense_encoder=DenseEncoder(cfg,in_channel=2+len(TOP_TAGS))
        def forward(self,amp,phase,mask):
            maps=mask[:,:,:,None].expand(-1,-1,-1,amp.shape[1])
            z=torch.stack((amp,phase),-1).permute(0,3,2,1)
            z=self.dense_encoder(torch.cat((z,maps),1))
            for block in self.TSTransformer:z=block(z)
            return amp*self.mask_decoder(z),self.phase_decoder(z)
    cfg=AttrDict(json.loads((a.mp_repo/'best_ckpt/config.json').read_text()))
    model=TagMPNet(cfg).cuda();state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert state.get('conditioning')=='tag';model.load_state_dict(state['generator']);model.train()
    asr=AutoModel.from_pretrained(str(a.asr_dir),trust_remote_code=True).cuda().eval();asr._ensure_loaded()
    for module in (asr.encoder,asr.decoder_joint):
        for param in module.parameters():param.requires_grad_(False)
    sp=asr._get_tokenizer();meta={r['segmentFileName'].removesuffix('.wav'):r
        for r in json.loads(a.metadata.read_text())}
    rows=torch.load(a.train_cache,map_location='cpu',weights_only=False);usable=[]
    for row in rows:
        if row['id'] not in meta or len(row['waveform'])>a.max_seconds*16000:continue
        tokens=sp.encode(clean_text(meta[row['id']]['transcript']))
        if tokens:usable.append((row,tokens))
    assert len(usable)>200,len(usable)
    natural_usable=[]
    if a.natural_cache:
        natural_rows=torch.load(a.natural_cache,map_location='cpu',weights_only=False)
        train_ids={row['id'] for row,_ in usable}
        for row in natural_rows:
            if row['id'] in train_ids:
                p.error('Natural cache overlaps paired synthetic training IDs')
            if row['id'] not in meta or (not a.natural_full_mask and not row.get('events')) or len(row['waveform'])>a.max_seconds*16000:
                continue
            tokens=sp.encode(clean_text(meta[row['id']]['transcript']))
            if tokens:natural_usable.append((row,tokens))
        print('NATURAL_POOL',len(natural_usable),flush=True)
        if a.natural_prob and not natural_usable:
            p.error('No usable natural rows')
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-6)
    for step in range(1,a.steps+1):
        is_natural=bool(natural_usable and random.random()<a.natural_prob)
        row,tokens=random.choice(natural_usable if is_natural else usable)
        x=torch.as_tensor(np.asarray(row['waveform'],np.float32),device='cuda')[None]
        c=torch.as_tensor(np.asarray(row['clean'],np.float32),device='cuda')[None]
        n=min(x.shape[-1],c.shape[-1]);x=x[:,:n];c=c[:,:n]
        sample_mask=tag_masks(row.get('events',[]),n)
        if (is_natural and a.natural_full_mask) or (not is_natural and a.synthetic_full_mask):
            sample_mask.fill(0);sample_mask[-1,:]=1
        mask=torch.from_numpy(sample_mask)[None].cuda()
        opt.zero_grad(set_to_none=True)
        norm=torch.sqrt(n/x.square().sum(-1).clamp_min(1e-8));xn=x*norm[:,None]
        amp,phase,_=mag_pha_stft(xn,cfg.n_fft,cfg.hop_size,cfg.win_size,cfg.compress_factor)
        fm=F.interpolate(mask,size=amp.shape[-1],mode='linear',align_corners=False)
        out_amp,out_phase=model(amp,phase,fm)
        enhanced=mag_pha_istft(out_amp,out_phase,cfg.n_fft,cfg.hop_size,cfg.win_size,cfg.compress_factor)/norm[:,None]
        radius=1600;gate=F.max_pool1d(mask.amax(1)[:,None],2*radius+1,stride=1,padding=radius).squeeze(1)
        gate=F.avg_pool1d(gate[:,None],801,stride=1,padding=400).squeeze(1).clamp(0,1)
        valid=min(n,enhanced.shape[-1]);x=x[:,:valid];c=c[:,:valid];gate=gate[:,:valid]
        y=x+gate*(enhanced[:,:valid]-x)
        si=(y.new_zeros(()) if is_natural else neg_sisdr(y,c))
        try:asr_loss=rnnt_loss(asr,y,tokens)
        except RuntimeError as e:
            if 'input length mismatch' in str(e):continue
            raise
        preserve=((y-x).abs()*(1-gate)).mean()/(x.abs().mean()+1e-4)
        natural_distortion=((y-x).abs().mean()/(x.abs().mean()+1e-4)
                            if is_natural else y.new_zeros(()))
        loss=si+a.rnnt_weight*asr_loss+.1*preserve+a.natural_identity_weight*natural_distortion
        if not torch.isfinite(loss):continue
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3.0);opt.step()
        if step%10==0:print('STEP',step,'loss',round(float(loss),4),'sisdr',round(float(-si),3),
                            'rnnt',round(float(asr_loss),3),'preserve',round(float(preserve),4),
                            'natural',is_natural,'distortion',round(float(natural_distortion),4),flush=True)
        if step%50==0 or step==a.steps:
            out=a.output.with_name(a.output.stem+f'_s{step}.pt');out.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'generator':model.state_dict(),'conditions':TOP_TAGS,'conditioning':'tag',
                        'step':step,'rnnt_weight':a.rnnt_weight},out)
    torch.save({'generator':model.state_dict(),'conditions':TOP_TAGS,'conditioning':'tag',
                'step':a.steps,'rnnt_weight':a.rnnt_weight},a.output)
    print('WROTE',a.output,flush=True)
if __name__=='__main__':main()
