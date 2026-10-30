"""Cache neural estimates for released state-excluded fusion training only."""
import json
from pathlib import Path
import torch
import yaml
from standalone.tflocoformer_separator import TFLocoformerSeparator
from espnet2.enh.encoder.stft_encoder import STFTEncoder
from espnet2.enh.decoder.stft_decoder import STFTDecoder
import sys
sys.path.insert(0,'/mnt/nvme_disk1/t2_gap_urgenet_1009')
from models.predictor.tfgridnet import TFGridNet

torch.cuda.set_per_process_memory_fraction(.35)
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
out=root/'fusion_training_cache';out.mkdir(exist_ok=True)
meta={r['segmentFileName'].removesuffix('.wav'):r for r in json.loads(Path('/mnt/nvme_disk1/official_validation_20260907/validationMetadata.json').read_text())}
held=torch.load('/mnt/nvme_disk1/t2_statehold_0924_holdout.pt',map_location='cpu',weights_only=False)
heldids={r['id'] for r in held};states={meta[i]['state'] for i in heldids}
cfg=yaml.safe_load((root/'egs2/dns_ins20/enh1/exp/enh_train_enh_tflocoformer_raw/config.yaml').read_text())
models=[]
for adapted in (False,True):
    net=TFLocoformerSeparator(**cfg['separator_conf'])
    ck=torch.load(root/('locoformer_acoustic_1010_s1600.pt' if adapted else 'dns_pretrained.pth'),map_location='cpu',weights_only=False)
    if adapted:
        assert not set(ck['training_ids'])&heldids
        assert not {meta[i]['state'] for i in ck['training_ids']}&states
        weights=ck['model']
    else:weights={k.removeprefix('separator.'):v for k,v in ck.items()}
    net.load_state_dict(weights,strict=True);models.append(net.cuda().eval())
    del ck,weights
encoder=STFTEncoder(**cfg['encoder_conf']).cuda().eval()
decoder=STFTDecoder(**cfg['decoder_conf']).cuda().eval()
ck=torch.load('/mnt/nvme_disk1/t2_gap_urgenet_1009/predictor_fulladapt_1010_s800.pt',map_location='cpu',weights_only=False)
assert not set(ck['training_ids'])&heldids
assert not {meta[i]['state'] for i in ck['training_ids']}&states
predictor=TFGridNet(**ck['cfg']).cuda().eval();predictor.load_state_dict(ck['model'],strict=True)
del ck
def cap(x):return x/(x.abs().amax(-1,keepdim=True)/.99).clamp_min(1)
def pcm(x):return torch.floor(x.clamp(-1,1-1/32768)*32768)/32768
ids=[]
for cache in ('neural_fusion_train507_1009.pt','neural_fusion_natural763_1009.pt'):
    rows=torch.load(Path('/mnt/nvme_disk1/urgent2026_t2pilot_1008')/cache,map_location='cpu',weights_only=False)
    for row in rows:
        if len(row['noisy'])>8*16000:continue
        assert row['id'] not in heldids and row['state'] not in states
        target=out/(row['id']+'.pt');ids.append(row['id'])
        if target.exists():continue
        x=row['noisy'][None].cuda()
        with torch.inference_mode():
            ls=cap((.3*row['mp']+.7*row['bs'])[None].cuda())
            pred=predictor(x,sr=16000)
            pred=pred/pred.abs().amax(-1,keepdim=True).clamp_min(1e-8)*x.abs().amax(-1,keepdim=True)
            base=pcm(cap(.7*pcm(ls)+.3*pcm(pred)))
            lengths=torch.tensor([x.shape[-1]],device=x.device)
            std=x.std(1,keepdim=True).clamp_min(1e-8)
            features,_=encoder(x/std,lengths)
            outputs=[]
            for model in models:
                spec=model(features);audio,_=decoder(spec[:,0],lengths)
                outputs.append(pcm(cap(audio*std))[0].cpu())
        result={'id':row['id'],'state':row['state'],'noisy':row['noisy'],
                'base':base[0].cpu(),'loco_frozen':outputs[0],'loco_acoustic':outputs[1],
                'transcript':meta[row['id']]['transcript'],'natural':'clean' not in row}
        if 'clean' in row:result['clean']=row['clean']
        temporary=target.with_suffix('.partial');torch.save(result,temporary);temporary.replace(target)
        print('CACHED',len(ids),row['id'],flush=True)
    del rows
manifest={'ids':ids,'count':len(ids),'heldout_states':sorted(states),'reference_use':'Released training supervision only; no held/test clips',
          'processed_branches':['MP+LSBS+TFGrid base','DNS Locoformer','state-excluded acoustic Locoformer']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',len(ids),flush=True)
