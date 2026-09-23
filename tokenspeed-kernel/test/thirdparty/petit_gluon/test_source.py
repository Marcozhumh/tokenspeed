# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.thirdparty.petit_gluon import petit_kernel
from tokenspeed_kernel.thirdparty.petit_gluon.lib.moe.rocm import mega_moe


def test_gluon_imports_use_tokenspeed_triton() -> None:
    assert mega_moe.g.__name__ == "tokenspeed_triton.experimental.gluon"
    assert mega_moe.l.__name__ == "tokenspeed_triton.experimental.gluon.language"


@pytest.mark.parametrize(
    (
        "num_experts",
        "topk",
        "model_dim",
        "inter_dim",
        "activation_function",
        "has_bias",
        "compute_model_dim",
    ),
    (
        (
            128,
            4,
            2880,
            3072,
            petit_kernel.MegaMoeActivationFunction.swiglu,
            True,
            3072,
        ),
        (
            384,
            6,
            7168,
            3072,
            petit_kernel.MegaMoeActivationFunction.silu,
            False,
            7168,
        ),
    ),
)
def test_supported_mega_moe_profiles(
    num_experts: int,
    topk: int,
    model_dim: int,
    inter_dim: int,
    activation_function: petit_kernel.MegaMoeActivationFunction,
    has_bias: bool,
    compute_model_dim: int,
) -> None:
    config = petit_kernel.MegaMoeConfig(
        world_size=8,
        num_experts=num_experts,
        topk=topk,
        model_dim=model_dim,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=activation_function,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=inter_dim,
        has_bias=has_bias,
    )

    assert config.compute_model_dim == compute_model_dim
    assert config.max_tokens_per_rank == 1024


def test_native_mxfp4_repack_preserves_shapes() -> None:
    weight = (
        torch.arange(256 * 128, dtype=torch.int64).to(torch.uint8).reshape(1, 256, 128)
    )
    scales = torch.arange(256 * 8, dtype=torch.int64).to(torch.uint8).reshape(1, 256, 8)

    packed_weight, packed_scales = petit_kernel.repack_moe_kernel_layout(
        weight,
        scales,
        layout=petit_kernel.MoeKernelLayout.native_mxfp4,
        petit_format=True,
    )

    assert packed_weight.shape == weight.shape
    assert packed_scales.shape == scales.shape
    assert packed_weight.is_contiguous()
    assert packed_scales.is_contiguous()
    assert not torch.equal(packed_weight, weight)
    assert not torch.equal(packed_scales, scales)
