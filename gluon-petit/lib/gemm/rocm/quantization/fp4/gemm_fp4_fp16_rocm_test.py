# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import ctypes
from math import lcm

import pytest
import torch
from lib.gemm.rocm.quantization.fp4.gemm_fp4_fp16_grid import (
    GemmFp4Fp16Grid,
    GemmMxFp4Fp16Grid,
)
from lib.gemm.rocm.quantization.fp4.quantization_utils import (
    DequantPetitFp4,
    DequantPetitMxFp4,
    RepackMxFp4ToPetitFp4Scales,
    RepackNvFp4ToPetitFp4Weights,
)
from lib.gemm.rocm.quantization.gemm import (
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    MatmulWarpPartition,
    PetitSolutionHints,
    SolutionId,
)
from lib.gemm.rocm.quantization.types import DataType
from lib.tests.native import default_native
from lib.tests.quantization import GemmMPTestData


def CheckHipblasStatus(status):
    if status != 0:
        raise RuntimeError(f"HipBLAS Error: {status}")


def IsNearBf16(arg, ref):
    # Native matcher arguments contain the raw bfloat16 bits.
    a_f = arg.to(torch.int16).view(torch.bfloat16).float()
    b_f = ref.to(torch.int16).view(torch.bfloat16).float()
    return (a_f - b_f).abs() < torch.maximum(
        torch.full_like(b_f, 1e-2), b_f.abs() * 0.01
    )


def IsNearFp16(arg, ref):
    a_f, b_f = arg.float(), ref.float()
    return (a_f - b_f).abs() < torch.maximum(
        torch.full_like(b_f, 1e-2), b_f.abs() * 0.01
    )


class GemmFp4Fp16Test:
    kWorkspaceSize = 32 * 1024 * 1024

    def SetUp(self):
        self.d_workspace_ = torch.empty(
            self.kWorkspaceSize, device="cuda:0", dtype=torch.uint8
        )
        self.d_global_scale_ = torch.empty(1, device="cuda:0", dtype=torch.float32)
        self.native_ = default_native()
        self.handle_, self.matmul_desc_ = ctypes.c_void_p(), ctypes.c_void_p()
        CheckHipblasStatus(
            self.native_.lib.native_reference_create(
                ctypes.byref(self.handle_), ctypes.byref(self.matmul_desc_)
            )
        )
        self.dev_ = torch.device("cuda:0")

    def TearDown(self):
        del self.d_workspace_
        CheckHipblasStatus(
            self.native_.lib.native_reference_destroy(self.handle_, self.matmul_desc_)
        )
        del self.d_global_scale_

    def ComputeReference(self, ctx):
        # The bridge performs the native hipBLASLt layouts and C^T = B^T * A^T
        # operation, with alpha=1, beta=0 and the fixture's 32 MiB workspace.
        CheckHipblasStatus(
            self.native_.lib.native_reference_with_context(
                self.handle_,
                self.matmul_desc_,
                self.d_workspace_.data_ptr(),
                self.kWorkspaceSize,
                ctx.reference.data_ptr(),
                ctx.input.data_ptr(),
                ctx.weights.data_ptr(),
                ctx.m,
                ctx.n,
                ctx.k,
                int(self.dequant_type_),
            )
        )

    def CopyAndCompareOutput(self, ctx):
        h_output, h_reference = ctx.output.cpu(), ctx.reference.cpu()
        torch.cuda.synchronize(self.dev_)
        if self.dequant_type_ == DataType.Fp16:
            good = IsNearFp16(h_output, h_reference)
        elif self.dequant_type_ == DataType.Bf16:
            good = IsNearBf16(h_output.view(torch.int16), h_reference.view(torch.int16))
        else:
            return
        bad = (~good).flatten().nonzero().flatten()
        assert not bad.numel(), (
            f"Output and reference differ at index {bad[0].item()}: "
            f"{h_output.flatten()[bad[0]]} vs {h_reference.flatten()[bad[0]]}"
        )

    def TestGemmBySolutionId(
        self,
        m,
        n,
        k,
        global_scale,
        solution_id,
        data_type,
        b_type,
        require_high_precision,
        group_size,
    ):
        is_mxfp4 = b_type == DataType.MxFp4e2m1
        if is_mxfp4:
            assert (
                data_type == DataType.Bf16
            ), "MXFP4 only supports BF16 input/output type"
            self.dequant_type_ = DataType.Bf16
        else:
            self.dequant_type_ = data_type
        dtype = torch.bfloat16 if self.dequant_type_ == DataType.Bf16 else torch.float16
        # The shared FP4 fixture generates the native fixed group sizes.
        assert group_size == (32 if is_mxfp4 else 16)
        ctx = GemmMPTestData(m, n, k, dtype, is_mxfp4)
        ctx.input, ctx.weights_quant, ctx.scales = (
            x.to(self.dev_) for x in ctx.PrepareData(False)
        )
        ctx.weights = torch.empty((n, k), dtype=dtype, device=self.dev_)
        ctx.output = torch.empty((m, n), dtype=dtype, device=self.dev_)
        ctx.reference = torch.empty_like(ctx.output)
        self.d_global_scale_.copy_(torch.tensor([global_scale], dtype=torch.float32))
        weights_ptr, scales_ptr = ctx.weights_quant, ctx.scales
        stream = torch.cuda.default_stream(self.dev_)
        d_petit_weights = d_petit_scales = None
        if is_mxfp4:
            d_petit_weights = torch.empty_like(ctx.weights_quant)
            d_petit_scales = torch.empty_like(ctx.scales)
            RepackNvFp4ToPetitFp4Weights(
                d_petit_weights, ctx.weights_quant, k, n, stream
            )
            RepackMxFp4ToPetitFp4Scales(d_petit_scales, ctx.scales, k, n, stream)
            weights_ptr, scales_ptr = d_petit_weights, d_petit_scales
            assert (
                DequantPetitMxFp4(
                    ctx.weights,
                    d_petit_weights,
                    d_petit_scales,
                    global_scale,
                    self.dequant_type_,
                    k,
                    n,
                )
                == 0
            ), "DequantPetitMxFp4 failed"
        else:
            assert (
                DequantPetitFp4(
                    ctx.weights,
                    ctx.weights_quant,
                    ctx.scales,
                    global_scale,
                    self.dequant_type_,
                    k,
                    n,
                )
                == 0
            ), "DequantPetitFp4 failed"
        self.ComputeReference(ctx)
        hints = PetitSolutionHints(data_type, b_type, data_type, require_high_precision)
        if is_mxfp4:
            err = GemmMxFp4Fp16Grid(
                ctx.output,
                ctx.input,
                weights_ptr,
                scales_ptr,
                self.d_global_scale_,
                m,
                n,
                k,
                hints,
                solution_id,
                stream,
            )
        else:
            err = GemmFp4Fp16Grid(
                ctx.output,
                ctx.input,
                weights_ptr,
                scales_ptr,
                self.d_global_scale_,
                m,
                n,
                k,
                hints,
                solution_id,
                stream,
            )
        assert err == 0
        del weights_ptr, scales_ptr, d_petit_weights, d_petit_scales
        self.CopyAndCompareOutput(ctx)

    def TestGemm(self, m, n, k, global_scale, sol_id, b_type, group_size):
        if b_type == DataType.MxFp4e2m1:
            assert (
                sol_id.mfma_type == MatmulMfmaType.kMatmulMfmaTypeBf16
            ), "MXFP4 only supports BF16 accumulation"
        data_type = (
            DataType.Bf16
            if sol_id.mfma_type == MatmulMfmaType.kMatmulMfmaTypeBf16
            else DataType.Fp16
        )
        require_high_precision = bool(
            sol_id.features & MatmulFeatures.kMatmulFeatures_HighPrecision
        )
        self.TestGemmBySolutionId(
            m,
            n,
            k,
            global_scale,
            sol_id.Repr(),
            data_type,
            b_type,
            require_high_precision,
            group_size,
        )


def Fp4MNK(
    features,
    element_b,
    mfma_type,
    tile_m,
    tile_n,
    tile_k,
    partition_m,
    partition_n,
    partition_k,
):
    return SolutionId.MultiStage(
        features,
        element_b,
        mfma_type,
        tile_m,
        tile_n,
        tile_k,
        MatmulWarpPartition.kMatmulWarpPartition_NK,
        partition_m,
        partition_n,
        partition_k,
    )


def Fp4Bf16(element_b, tile_m, tile_n, tile_k, partition_m, partition_n, partition_k):
    return Fp4MNK(
        MatmulFeatures.kMatmulFeatures_Grid,
        element_b,
        MatmulMfmaType.kMatmulMfmaTypeBf16,
        tile_m,
        tile_n,
        tile_k,
        partition_m,
        partition_n,
        partition_k,
    )


def Fp4Hp(mfma_type, tile_m, tile_n, tile_k, partition_m, partition_n, partition_k):
    return Fp4MNK(
        MatmulFeatures.kMatmulFeatures_Grid
        | MatmulFeatures.kMatmulFeatures_HighPrecision,
        MatmulElementB.kMatmulTypeBNvFp4,
        mfma_type,
        tile_m,
        tile_n,
        tile_k,
        partition_m,
        partition_n,
        partition_k,
    )


def TEST_BF16(m, n, k, partition_m, partition_n, partition_k):
    # Python equivalent of native TEST_BF16: register one named pytest test.
    name = f"test_TestGemm_{m}x{n}x{k}_{partition_m}x{partition_n}x{partition_k}"

    def test(gemm_fp4_fp16):
        gemm_fp4_fp16.TestGemm(
            m,
            lcm(n, 32),
            lcm(k, 256),
            1.0,
            Fp4Bf16(
                MatmulElementB.kMatmulTypeBNvFp4,
                m // 16,
                n // 16,
                k // 16,
                partition_m,
                partition_n,
                partition_k,
            ),
            DataType.Fp4e2m1,
            16,
        )

    test.__name__ = test.__qualname__ = name
    globals()[name] = test


@pytest.fixture
def gemm_fp4_fp16():
    fixture = GemmFp4Fp16Test()
    with torch.cuda.device(0), torch.cuda.stream(torch.cuda.default_stream(0)):
        fixture.SetUp()
        try:
            yield fixture
        finally:
            fixture.TearDown()


# High precision avoids FP16 denormal flushing on MI210, as in native.
def test_TestGemm16x32x256Fp16HighPrecision(gemm_fp4_fp16):
    gemm_fp4_fp16.TestGemm(
        16,
        64,
        256,
        1.0,
        Fp4Hp(MatmulMfmaType.kMatmulMfmaTypeFp16, 1, 2, 16, 1, 1, 4),
        DataType.Fp4e2m1,
        16,
    )


def test_TestGemm16x32x256Bf16HighPrecision(gemm_fp4_fp16):
    gemm_fp4_fp16.TestGemm(
        16,
        64,
        256,
        1.0,
        Fp4Hp(MatmulMfmaType.kMatmulMfmaTypeBf16, 1, 2, 16, 1, 1, 4),
        DataType.Fp4e2m1,
        16,
    )


TEST_BF16(64, 32, 128, 4, 1, 1)
TEST_BF16(16, 32, 256, 1, 1, 4)
TEST_BF16(32, 32, 256, 2, 1, 2)
TEST_BF16(32, 32, 256, 1, 1, 4)
TEST_BF16(64, 32, 256, 2, 1, 2)
TEST_BF16(16, 32, 512, 1, 1, 4)
TEST_BF16(32, 32, 512, 2, 1, 2)
TEST_BF16(16, 64, 128, 1, 2, 2)
TEST_BF16(32, 64, 128, 2, 2, 1)
TEST_BF16(64, 64, 128, 2, 2, 1)
TEST_BF16(96, 64, 128, 2, 2, 1)
TEST_BF16(128, 64, 128, 2, 2, 1)
TEST_BF16(160, 64, 128, 2, 2, 1)
TEST_BF16(16, 64, 256, 1, 2, 2)
TEST_BF16(32, 64, 256, 2, 2, 1)
TEST_BF16(64, 64, 256, 2, 2, 1)
TEST_BF16(16, 64, 512, 1, 2, 2)
TEST_BF16(32, 64, 512, 2, 2, 1)
TEST_BF16(64, 96, 128, 2, 1, 2)
TEST_BF16(96, 96, 128, 2, 1, 2)
TEST_BF16(16, 128, 64, 1, 4, 1)
TEST_BF16(128, 128, 64, 2, 2, 1)
TEST_BF16(192, 128, 64, 2, 2, 1)
TEST_BF16(224, 128, 64, 2, 2, 1)
TEST_BF16(256, 128, 64, 2, 2, 1)
TEST_BF16(32, 128, 128, 2, 2, 1)
TEST_BF16(64, 128, 128, 2, 2, 1)
TEST_BF16(80, 128, 128, 1, 2, 2)
TEST_BF16(160, 128, 64, 2, 2, 1)
TEST_BF16(128, 192, 64, 2, 2, 1)
TEST_BF16(160, 192, 64, 2, 2, 1)
TEST_BF16(192, 192, 64, 2, 2, 1)
TEST_BF16(224, 192, 64, 2, 2, 1)
TEST_BF16(256, 192, 64, 2, 2, 1)
TEST_BF16(128, 256, 64, 2, 2, 1)
TEST_BF16(160, 256, 64, 2, 2, 1)
TEST_BF16(192, 256, 64, 2, 2, 1)
TEST_BF16(224, 256, 64, 2, 2, 1)
TEST_BF16(256, 256, 64, 2, 2, 1)


def test_TestGemm_32x32x256_2x1x2_Pipeline_DoubleBuffer(gemm_fp4_fp16):
    for k in (512, 768, 1024):
        gemm_fp4_fp16.TestGemm(
            32,
            32,
            k,
            1.0,
            Fp4MNK(
                MatmulFeatures.kMatmulFeatures_Grid,
                MatmulElementB.kMatmulTypeBNvFp4,
                MatmulMfmaType.kMatmulMfmaTypeBf16,
                2,
                2,
                16,
                2,
                1,
                2,
            ),
            DataType.Fp4e2m1,
            16,
        )


def test_TestGemm_256x256x64_2x2x1_Pipeline_SingleBuffer(gemm_fp4_fp16):
    for k in (512, 768, 1024):
        gemm_fp4_fp16.TestGemm(
            256,
            256,
            k,
            1.0,
            Fp4MNK(
                MatmulFeatures.kMatmulFeatures_Grid,
                MatmulElementB.kMatmulTypeBNvFp4,
                MatmulMfmaType.kMatmulMfmaTypeBf16,
                16,
                16,
                4,
                2,
                2,
                1,
            ),
            DataType.Fp4e2m1,
            16,
        )


def test_TestGemmMx_32x64x256_2x2x1(gemm_fp4_fp16):
    gemm_fp4_fp16.TestGemm(
        32,
        64,
        256,
        1.0,
        Fp4Bf16(MatmulElementB.kMatmulTypeBMxFp4, 2, 4, 16, 2, 2, 1),
        DataType.MxFp4e2m1,
        32,
    )


def test_TestGemmMx_64x64x256_2x2x1(gemm_fp4_fp16):
    gemm_fp4_fp16.TestGemm(
        64,
        64,
        256,
        1.0,
        Fp4Bf16(MatmulElementB.kMatmulTypeBMxFp4, 4, 4, 16, 2, 2, 1),
        DataType.MxFp4e2m1,
        32,
    )


def test_TestGemmMx_64x128x256_NormalPrecision(gemm_fp4_fp16):
    gemm_fp4_fp16.TestGemm(
        64,
        128,
        256,
        1.0,
        Fp4Bf16(MatmulElementB.kMatmulTypeBMxFp4, 2, 4, 16, 2, 2, 1),
        DataType.MxFp4e2m1,
        32,
    )


def test_TestGemmMx_64x128x256_HighPrecision(gemm_fp4_fp16):
    gemm_fp4_fp16.TestGemm(
        64,
        128,
        256,
        1.0,
        Fp4Hp(MatmulMfmaType.kMatmulMfmaTypeBf16, 2, 4, 16, 2, 2, 1),
        DataType.MxFp4e2m1,
        32,
    )
