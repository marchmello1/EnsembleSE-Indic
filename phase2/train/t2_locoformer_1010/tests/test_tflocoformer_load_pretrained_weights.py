# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from standalone.tflocoformer_separator import TFLocoformerSeparator


@pytest.mark.parametrize("num_spk", [2])
@pytest.mark.parametrize("n_layers", [6])
@pytest.mark.parametrize("emb_dim", [128])
@pytest.mark.parametrize("norm_type", ["rmsgroupnorm"])
@pytest.mark.parametrize("num_groups", [4])
@pytest.mark.parametrize("tf_order", ["ft"])
@pytest.mark.parametrize("n_heads", [4])
@pytest.mark.parametrize("attention_dim", [128])
@pytest.mark.parametrize("pos_enc", ["rope"])
@pytest.mark.parametrize("ffn_type", [["swiglu_conv1d", "swiglu_conv1d"]])
@pytest.mark.parametrize("ffn_hidden_dim", [[192, 192]])
@pytest.mark.parametrize("conv1d_kernel", [8])
@pytest.mark.parametrize("conv1d_shift", [1])
@pytest.mark.parametrize("dropout", [0.0])
@pytest.mark.parametrize("eps", [1e-5])
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
    # others
    eps,
):

    model = TFLocoformerSeparator(
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
        eps=eps,
    )

    state_dict = torch.load("./egs2/whamr/enh1/exp/enh_train_enh_tflocoformer_raw/valid.loss.ave_5best.pth")

    # provided pre-trained weights have keys starting with 'separator.', but this has to be removed.
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = ".".join(k.split(".")[1:])
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)

    model.train()

    # Create dummy inputs
    n_batch = 2
    n_freqs = 128 // 2 + 1
    n_frames = 50

    real = torch.randn(n_batch, n_frames, n_freqs)
    imag = torch.randn(n_batch, n_frames, n_freqs)
    x = torch.complex(real, imag)

    output = model(x)
    assert output.shape == (n_batch, num_spk, n_frames, n_freqs)
    sum(output).abs().mean().backward()
