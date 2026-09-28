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

import importlib

import pytest
import torch
from tokenspeed_kernel.thirdparty.gluon_petit import petit_kernel
from utils import assert_no_triton_compile

mega_moe = importlib.import_module("lib.moe.rocm.mega_moe")


@mega_moe.g.jit
def _petit_compiler_contract_kernel():
    storage = mega_moe.l.allocate_shared_memory(
        mega_moe.l.uint32,
        [68],
        mega_moe.l.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    shared = mega_moe.l.full((), 0, mega_moe.l.uint64).to(
        mega_moe.l.pointer_type(mega_moe.l.uint32, 3)
    )
    mega_moe.l.store(shared, 0)
    storage._keep_alive()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU compiler")
def test_gluon_petit_compiler_contract() -> None:
    compiled = _petit_compiler_contract_kernel.warmup(grid=(1,), num_warps=1)

    assert compiled.metadata.shared == 68 * 4
    assert "addrspace(3)" in compiled.asm["llir"]


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").startswith(
        "gfx950"
    ),
    reason="requires a GFX950 compiler",
)
@pytest.mark.parametrize(
    ("num_experts", "topk", "model_dim", "activation_function", "has_bias"),
    (
        (
            128,
            4,
            2880,
            petit_kernel.MegaMoeActivationFunction.swiglu,
            True,
        ),
        (
            384,
            6,
            7168,
            petit_kernel.MegaMoeActivationFunction.silu,
            False,
        ),
    ),
)
@pytest.mark.parametrize("num_tokens", (1, 256, 1024))
def test_supported_mega_moe_profiles_compile(
    num_tokens: int,
    num_experts: int,
    topk: int,
    model_dim: int,
    activation_function: petit_kernel.MegaMoeActivationFunction,
    has_bias: bool,
) -> None:
    config = petit_kernel.MegaMoeConfig(
        world_size=8,
        num_experts=num_experts,
        topk=topk,
        model_dim=model_dim,
        activation=petit_kernel.MegaMoeActivation.mxfp4,
        activation_function=activation_function,
        stages=petit_kernel.MegaMoeStages.two_stage,
        inter_dim=3072,
        has_bias=has_bias,
    )
    adapter = mega_moe._MegaMoESolutions()[config._solution_id_for_tokens(num_tokens)]
    stage1, stage2, combine = adapter.Kernels(
        True,
        num_tokens >= 256,
        num_tokens >= (1024 if num_experts == 128 else 256),
    )
    device = torch.device("cuda", 0)
    uint8 = torch.empty(1, dtype=torch.uint8, device=device)
    int32 = torch.empty(1, dtype=torch.int32, device=device)
    float32 = torch.empty(1, dtype=torch.float32, device=device)
    bfloat16 = torch.empty(1, dtype=torch.bfloat16, device=device)
    bias = bfloat16 if has_bias else None

    compiled = (
        mega_moe.MegaMoEStage1.warmup(
            uint8,
            uint8,
            num_tokens,
            bias,
            uint8,
            0,
            uint8,
            int32,
            float32,
            stage1,
            grid=(stage1.kNumSMs,),
            num_warps=stage1.kNumWarps,
            enable_fp_fusion=False,
        ),
        mega_moe.MegaMoEStage2.warmup(
            uint8,
            uint8,
            bias,
            uint8,
            0,
            stage2,
            grid=(stage2.kStage2GridBlocks,),
            num_warps=stage2.kNumWarps,
            enable_fp_fusion=False,
        ),
        mega_moe.MegaMoECombine.warmup(
            bfloat16,
            num_tokens,
            config.compute_model_dim,
            uint8,
            0,
            combine,
            grid=(combine.kNumSMs,),
            num_warps=combine.kNumWarps,
            enable_fp_fusion=False,
        ),
    )

    # Warm each runtime integer specialization class before varying row counts.
    def warm_rows(rows):
        mega_moe.MegaMoEStage1.warmup(
            uint8,
            uint8,
            rows,
            bias,
            uint8,
            0,
            uint8,
            int32,
            float32,
            stage1,
            grid=(stage1.kNumSMs,),
            num_warps=stage1.kNumWarps,
            enable_fp_fusion=False,
        )
        mega_moe.MegaMoECombine.warmup(
            bfloat16,
            rows,
            config.compute_model_dim,
            uint8,
            0,
            combine,
            grid=(combine.kNumSMs,),
            num_warps=combine.kNumWarps,
            enable_fp_fusion=False,
        )

    for rows in (0, 1, 2, 16):
        warm_rows(rows)
    with assert_no_triton_compile(mega_moe.MegaMoEStage1), assert_no_triton_compile(
        mega_moe.MegaMoECombine
    ):
        for rows in (3, 17, 32, 63, 127, 256, 1024):
            warm_rows(rows)

    assert [kernel.metadata.shared for kernel in compiled] == [
        65552 if num_tokens >= 256 else 32784,
        32784,
        0,
    ]
    assert all("thread_device.ll" not in kernel.asm["ttgir"] for kernel in compiled)
    assert all(kernel.metadata.triton_version == "3.8.0" for kernel in compiled)
