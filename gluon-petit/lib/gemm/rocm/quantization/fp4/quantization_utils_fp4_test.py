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

import numpy as np
import torch
from lib.gemm.rocm.quantization.fp4.quantization_utils import (
    DequantMxFp4,
    DequantNvFp4,
    DequantPetitFp4,
    DequantPetitMxFp4,
    RepackMxFp4ToPetitFp4Scales,
    RepackNvFp4ToPetitFp4Scales,
    RepackNvFp4ToPetitFp4Weights,
)
from lib.gemm.rocm.quantization.types import DataType
from lib.tests.floating_points import bf16_t, fp8_e4m3_t, fp16_t
from lib.tests.quantization import Mt19937

kVecSize = 16
kPackFactor = 8
kQuantVecSize = 4
kNvRowGroupSize = 16
kMxRowGroupSize = 32
kQweightBitMin = 0
kQweightBitMax = 15
kDefaultGlobalScale = 1.0


class DeviceContext:
    ScaleType = fp8_e4m3_t

    def __init__(self, Element, kM, kN):
        self.Element, self.kM, self.kN = Element, kM, kN
        self.kOutVecSize = 16 // 2
        self.d_weights_quant = torch.empty(
            kM * kN // kPackFactor, dtype=torch.int32, device="cuda"
        )
        self.d_scales = torch.empty(
            kM * kN // kNvRowGroupSize, dtype=torch.uint8, device="cuda"
        )
        self.d_reference = torch.empty(
            (kN, kM), dtype=Element.torch_dtype, device="cuda"
        )
        self.d_petit_weights = torch.empty_like(self.d_weights_quant)
        self.d_petit_scales = torch.empty_like(self.d_scales)
        self.d_output = torch.empty_like(self.d_reference)

    @staticmethod
    def PrepareDevice(Element, kM, kN):
        d_ctx = DeviceContext(Element, kM, kN)
        gen = Mt19937(42)
        # Native generates weights first, then positive E4M3 scale bytes.
        h_qweights = gen.uniform_int(0, 0xFFFFFFFF, kM * kN // kPackFactor)
        h_scales = gen.uniform_int(1, 0x7E, kM * kN // kNvRowGroupSize)
        d_ctx.d_weights_quant.copy_(torch.from_numpy(h_qweights.view(np.int32)))
        d_ctx.d_scales.copy_(torch.from_numpy(h_scales.astype(np.uint8)))
        return d_ctx

    def CompareOutputsFromDevice(self):
        h_reference = self.d_reference.cpu().flatten()
        h_petit_output = self.d_output.cpu().flatten()
        torch.cuda.synchronize()
        # Native Element equality considers +0 and -0 equal, and NaNs unequal.
        bad = (h_reference != h_petit_output).nonzero().flatten()
        assert not bad.numel(), f"Output and reference differ at index {bad[0].item()}"


class NvFp4ToPetitFp4Test:
    def TestConvert(self, Element, kM, kN, global_scale, out_type):
        d_ctx = DeviceContext.PrepareDevice(Element, kM, kN)
        DequantNvFp4(
            d_ctx.d_reference,
            d_ctx.d_weights_quant,
            d_ctx.d_scales,
            global_scale,
            out_type,
            kM,
            kN,
        )
        RepackNvFp4ToPetitFp4Weights(
            d_ctx.d_petit_weights,
            d_ctx.d_weights_quant,
            kM,
            kN,
            torch.cuda.default_stream(),
        )
        RepackNvFp4ToPetitFp4Scales(
            d_ctx.d_petit_scales, d_ctx.d_scales, kM, kN, torch.cuda.default_stream()
        )
        DequantPetitFp4(
            d_ctx.d_output,
            d_ctx.d_petit_weights,
            d_ctx.d_petit_scales,
            global_scale,
            out_type,
            kM,
            kN,
        )
        d_ctx.CompareOutputsFromDevice()
        del d_ctx


class ExhaustiveDeviceContext:
    kM = 256

    def __init__(self, Element):
        self.Element = Element
        self.kOutVecSize = 16 // 2
        self.d_weights_quant = torch.empty(
            self.kM * self.kN // kPackFactor, dtype=torch.int32, device="cuda"
        )
        self.d_scales = torch.empty(
            self.kM * self.kN // self.kScaleRowGroupSize,
            dtype=torch.uint8,
            device="cuda",
        )
        self.d_reference = torch.empty(
            (self.kN, self.kM), dtype=Element.torch_dtype, device="cuda"
        )
        self.d_petit_weights = torch.empty_like(self.d_weights_quant)
        self.d_petit_scales = torch.empty_like(self.d_scales)
        self.d_output = torch.empty_like(self.d_reference)

    @classmethod
    def PrepareDevice(cls, Element, generate_q, generate_scale):
        d_ctx = cls(Element)
        kM, kN, group_size = cls.kM, cls.kN, cls.kScaleRowGroupSize
        h_qweights = np.empty(kM * kN // kPackFactor, dtype=np.uint32)
        h_scales = np.empty(kM * kN // group_size, dtype=np.uint8)
        for i in range(h_qweights.size):
            row, col = i % (kM // kPackFactor), i // (kM // kPackFactor)
            qweight = min(max(generate_q(row, col), kQweightBitMin), kQweightBitMax)
            h_qweights[i] = qweight * 0x11111111
        for i in range(h_scales.size):
            row, col = i % (kM // group_size), i // (kM // group_size)
            h_scales[i] = generate_scale(row, col) & 0xFF
        d_ctx.d_weights_quant.copy_(torch.from_numpy(h_qweights.view(np.int32)))
        d_ctx.d_scales.copy_(torch.from_numpy(h_scales))
        return d_ctx

    def CompareReferenceAndOutputFromDevice(self):
        h_reference = self.d_reference.cpu().view(torch.int16).flatten()
        h_petit_output = self.d_output.cpu().view(torch.int16).flatten()
        torch.cuda.synchronize()
        bad = (h_reference != h_petit_output).nonzero().flatten()
        assert not bad.numel(), f"Output mismatch at index {bad[0].item()}"

    def ValidateOutputsFromDevice(self, generate_output):
        h_petit_output = self.d_output.cpu().view(torch.uint16).flatten()
        torch.cuda.synchronize()
        # Native validates one element from each group of identical qweights.
        for i in range(self.kM * self.kN // self.kScaleRowGroupSize):
            row = i % (self.kM // self.kScaleRowGroupSize)
            col = i // (self.kM // self.kScaleRowGroupSize)
            reference = self.Element.from_fp32(generate_output(row, col))
            actual = self.Element.from_bits(
                h_petit_output[i * self.kScaleRowGroupSize].item()
            )
            assert reference == actual, (
                f"Output and reference differ at ({row}, {col}): "
                f"expected 0x{reference.to_bits():04x}, got 0x{actual.to_bits():04x}"
            )


Fp8ScaleType = fp8_e4m3_t
kNvScaleBitMin = 1
kNvScaleBitMax = 0x7E


def GenerateQweight(i):
    return min(max(i, kQweightBitMin), kQweightBitMax)


def GenerateNvScale(i):
    i = min(max(i, kNvScaleBitMin), kNvScaleBitMax)
    return Fp8ScaleType.from_bits(i)


def GenerateNvOutput(i_qweight, i_scale):
    qweight = GenerateQweight(i_qweight)
    scale = GenerateNvScale(i_scale)
    fp4_values = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )
    return scale.to_fp32() * fp4_values[qweight]


kMxScaleMin = 1
# Native range avoids BF16 overflow after multiplying by the largest FP4 value.
kMxScaleNoOverflowMax = 237
kMxScaleCount = kMxScaleNoOverflowMax - kMxScaleMin + 1


def GenerateMxScale(row, col):
    mixed = (col + 29 * row) % kMxScaleCount
    return kMxScaleMin + mixed


class NvExhaustiveDeviceContext(ExhaustiveDeviceContext):
    kN = 128
    kScaleRowGroupSize = kNvRowGroupSize


class MxExhaustiveDeviceContext(ExhaustiveDeviceContext):
    kN = 256
    kScaleRowGroupSize = kMxRowGroupSize


def RunPetitDequant(d_ctx, global_scale, out_type, repack_scales, dequant_petit):
    kM, kN = d_ctx.kM, d_ctx.kN
    RepackNvFp4ToPetitFp4Weights(
        d_ctx.d_petit_weights,
        d_ctx.d_weights_quant,
        kM,
        kN,
        torch.cuda.default_stream(),
    )
    repack_scales(
        d_ctx.d_petit_scales, d_ctx.d_scales, kM, kN, torch.cuda.default_stream()
    )
    assert (
        dequant_petit(
            d_ctx.d_output,
            d_ctx.d_petit_weights,
            d_ctx.d_petit_scales,
            global_scale,
            out_type,
            kM,
            kN,
        )
        == 0
    )


class MxFp4DequantTest:
    def TestAllCombinations(self, Element, out_type):
        Context = MxExhaustiveDeviceContext
        assert Context.kN >= kMxScaleCount, "kN must cover all MX scale values"
        d_ctx = Context.PrepareDevice(
            Element,
            lambda row, col: GenerateQweight(
                row * kPackFactor // kMxRowGroupSize + col
            ),
            lambda row, col: GenerateMxScale(row, col),
        )
        assert (
            DequantMxFp4(
                d_ctx.d_reference,
                d_ctx.d_weights_quant,
                d_ctx.d_scales,
                kDefaultGlobalScale,
                out_type,
                Context.kM,
                Context.kN,
            )
            == 0
        )
        RunPetitDequant(
            d_ctx,
            kDefaultGlobalScale,
            out_type,
            RepackMxFp4ToPetitFp4Scales,
            DequantPetitMxFp4,
        )
        d_ctx.CompareReferenceAndOutputFromDevice()
        del d_ctx


class ExhaustiveFp4DequantTest:
    def TestAllCombinations(self, Element, out_type):
        d_ctx = NvExhaustiveDeviceContext.PrepareDevice(
            Element,
            lambda row, col: GenerateQweight(row * kPackFactor // kNvRowGroupSize),
            lambda row, col: GenerateNvScale(col).to_bits(),
        )
        RunPetitDequant(
            d_ctx,
            kDefaultGlobalScale,
            out_type,
            RepackNvFp4ToPetitFp4Scales,
            DequantPetitFp4,
        )
        d_ctx.ValidateOutputsFromDevice(lambda row, col: GenerateNvOutput(row, col))
        del d_ctx


def test_TestLayout128x16Bf16():
    NvFp4ToPetitFp4Test().TestConvert(bf16_t, 512, 512, 1.0, DataType.Bf16)


def test_TestLayout128x16Fp16():
    NvFp4ToPetitFp4Test().TestConvert(fp16_t, 512, 512, 1.0, DataType.Fp16)


def test_AllMxFp4ScaleCombinationsBf16():
    MxFp4DequantTest().TestAllCombinations(bf16_t, DataType.Bf16)


def test_MxFp4ScaleCoverageBf16():
    MxFp4DequantTest().TestAllCombinations(bf16_t, DataType.Bf16)


def test_AllFp4ScaleCombinationsBf16():
    ExhaustiveFp4DequantTest().TestAllCombinations(bf16_t, DataType.Bf16)


def test_AllFp4ScaleCombinationsFp16():
    ExhaustiveFp4DequantTest().TestAllCombinations(fp16_t, DataType.Fp16)
