#!/usr/bin/env python3
"""Internal worker: transcribe the final serialized WAVs, never reference text."""
import os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('TRANSFORMERS_NO_TF','1')
import argparse,json
from pathlib import Path
import torch
from transformers import AutoModel
from huggingface_hub import get_token

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output_folder',type=Path)
    args=parser.parse_args()
    names=sorted(args.output_folder.glob('*.wav'))
    if not names:raise ValueError('No enhanced WAV files')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    asr=AutoModel.from_pretrained('ARTPARK-IISc/SraVaani-1.0',trust_remote_code=True,token=get_token()).to(device).eval()
    temporary=args.output_folder/'transcripts.jsonl.partial'
    with temporary.open('w',encoding='utf-8') as output:
        for start in range(0,len(names),8):
            batch=names[start:start+8]
            hypotheses=asr.transcribe([str(p) for p in batch],batch_size=len(batch),return_hypotheses=True,timestamps=False)
            if len(hypotheses)!=len(batch):raise RuntimeError('ASR batch coverage mismatch')
            for path,hypothesis in zip(batch,hypotheses):
                text=str(getattr(hypothesis,'text',hypothesis) or '').strip()
                output.write(json.dumps(dict(clip_id=path.stem,text=text),ensure_ascii=False)+'\n')
            output.flush()
            print('TRANSCRIBED',start+len(batch),len(names),flush=True)
    temporary.replace(args.output_folder/'transcripts.jsonl')
    print('COMPLETE',len(names),flush=True)
if __name__=='__main__':main()
