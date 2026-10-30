"""Bounded complex-mask correction of an already enhanced waveform."""
import torch
from torch import nn

class ComplexCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[nn.Conv2d(10,32,3,padding=1),nn.SiLU()]
        for dilation in (1,2,4):
            layers.extend([nn.Conv2d(32,32,3,padding=dilation,dilation=dilation),nn.SiLU()])
        self.context=nn.Sequential(*layers)
        self.head=nn.Conv2d(32,2,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('window',torch.hann_window(512))
    def forward(self,noisy,estimates):
        batch,branches,length=estimates.shape;assert branches==3
        audio=torch.cat([noisy[:,None],estimates],1)
        spectrum=torch.stft(audio.reshape(-1,length),512,128,window=self.window,return_complex=True)
        spectrum=spectrum.reshape(batch,4,*spectrum.shape[-2:])
        scale=noisy.std(-1,keepdim=True).clamp_min(1e-5)[:,:,None,None]
        z=spectrum/scale
        features=torch.cat([torch.log1p(z.abs()),torch.log1p((z[:,1:]-z[:,:1]).abs()),
            (z[:,1:]*z[:,:1].conj()).real/(z[:,1:].abs()*z[:,:1].abs()+1e-5)],1)
        correction=.25*self.head(self.context(features)).tanh()
        # No raw/noisy bypass: correction acts on the fixed enhanced estimate.
        base=.7*spectrum[:,1]+.3*spectrum[:,2]
        output=base*torch.complex(1+correction[:,0],correction[:,1])
        waveform=torch.istft(output,512,128,window=self.window,length=length)
        waveform=waveform/(waveform.abs().amax(-1,keepdim=True)/.99).clamp_min(1)
        return waveform,correction
