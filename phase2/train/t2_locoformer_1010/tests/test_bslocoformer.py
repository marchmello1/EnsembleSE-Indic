# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from standalone.bslocoformer_separator import BSLocoformerSeparator


@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("emb_dim", [32])
@pytest.mark.parametrize("norm_type", ["rmsgroupnorm"])
@pytest.mark.parametrize("num_groups", [1])
@pytest.mark.parametrize("tf_order", ["tf", "ft"])
@pytest.mark.parametrize("n_heads", [1, 4])
@pytest.mark.parametrize("attention_dim", [32])
@pytest.mark.parametrize("pos_enc", ["nope", "rope"])
@pytest.mark.parametrize("ffn_type", [["swiglu_conv1d", "swiglu_conv1d"]])
@pytest.mark.parametrize("ffn_hidden_dim", [[32, 32]])
@pytest.mark.parametrize("conv1d_kernel", [1, 4])
@pytest.mark.parametrize("conv1d_shift", [1])
@pytest.mark.parametrize("dropout", [0.1])
@pytest.mark.parametrize("sample_rate", [44100, 48000])
@pytest.mark.parametrize("stft_size", [2048])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("masking", [False, True])
@pytest.mark.parametrize("stereo", [False, True])
def test_tuss_forward_backward(
    num_spk,
    n_layers,
    # general setup
    emb_dim,
    norm_type,
    num_groups,  # used only in RMSGroupNorm
    tf_order,
    # self-attention related
    n_heads,
    attention_dim,
    pos_enc,
    # ffn related
    ffn_type,
    ffn_hidden_dim,
    conv1d_kernel,
    conv1d_shift,
    dropout,
    # banmd-split related
    sample_rate,
    stft_size,
    # others
    eps,
    masking,
    stereo,
):

    model = BSLocoformerSeparator(
        num_spk=num_spk,
        n_layers=n_layers,
        emb_dim=emb_dim,
        norm_type=norm_type,
        num_groups=num_groups,
        tf_order=tf_order,
        n_heads=n_heads,
        attention_dim=attention_dim,
        pos_enc=pos_enc,
        ffn_type=ffn_type,
        ffn_hidden_dim=ffn_hidden_dim,
        conv1d_kernel=conv1d_kernel,
        conv1d_shift=conv1d_shift,
        dropout=dropout,
        sample_rate=sample_rate,
        stft_size=stft_size,
        eps=eps,
        masking=masking,
        stereo=stereo,
    )
    model.train()

    # Create dummy inputs
    n_batch = 2
    n_freqs = stft_size // 2 + 1
    n_frames = 50

    if stereo:
        real = torch.randn(n_batch, 2, n_frames, n_freqs)
        imag = torch.randn(n_batch, 2, n_frames, n_freqs)
    else:
        real = torch.randn(n_batch, n_frames, n_freqs)
        imag = torch.randn(n_batch, n_frames, n_freqs)
    x = torch.complex(real, imag)

    output = model(x)
    if stereo:
        assert output.shape == (n_batch, num_spk, 2, n_frames, n_freqs)
    else:
        assert output.shape == (n_batch, num_spk, n_frames, n_freqs)
    sum(output).abs().mean().backward()
