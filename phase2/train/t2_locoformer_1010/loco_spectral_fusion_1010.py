"""Audio-conditioned convex fusion of three processed neural estimates."""
import torch
from torch import nn

class SpectralFusion(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Conv2d(10,32,3,padding=1),nn.SiLU()]
        for dilation in (1,2,4):
            layers += [nn.Conv2d(32,32,3,padding=dilation,dilation=dilation),nn.SiLU()]
        self.context=nn.Sequential(*layers)
        self.head=nn.Conv2d(32,3,1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():self.head.bias.copy_(torch.tensor([.7,.299,.001]).log())
        self.register_buffer('window',torch.hann_window(512))
    def forward(self,noisy,estimates):
        assert estimates.ndim==3 and estimates.shape[1]==3
        batch,_,length=estimates.shape
        all_audio=torch.cat([noisy[:,None],estimates],1)
        spectrum=torch.stft(all_audio.reshape(-1,length),512,128,window=self.window,return_complex=True)
        spectrum=spectrum.reshape(batch,4,*spectrum.shape[-2:])
        scale=noisy.std(-1,keepdim=True).clamp_min(1e-5)[:,:,None,None]
        normalized=spectrum/scale
        magnitude=torch.log1p(normalized.abs())
        differences=normalized[:,1:]-normalized[:,:1]
        features=torch.cat([magnitude,torch.log1p(differences.abs()),
                            (normalized[:,1:]*normalized[:,:1].conj()).real/
                            (normalized[:,1:].abs()*normalized[:,:1].abs()+1e-5)],1)
        weights=self.head(self.context(features)).softmax(1)
        mixed=(weights*spectrum[:,1:]).sum(1)
        audio=torch.istft(mixed,512,128,window=self.window,length=length)
        audio=audio/(audio.abs().amax(-1,keepdim=True)/.99).clamp_min(1)
        return audio,weights
