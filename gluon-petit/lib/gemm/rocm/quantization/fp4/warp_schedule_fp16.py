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

from dataclasses import dataclass

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import mma_m16n16k16_bf16, mma_m16n16k16_fp16
from lib.gemm.rocm.quantization.dequant import (
    UnifiedDequantizerForFp4Fp16,
    UnifiedDequantizerForMxFp4Bf16,
    UnifiedDequantizerForNvFp4Bf16,
)
from lib.gemm.rocm.quantization.memory_ops import MatrixALayout, MatrixBLayout
from lib.gemm.rocm.quantization.types import DataType
from triton.experimental.gluon import language as l


@dataclass(frozen=True)
class MmaSelector:
    """Native ElementA, weight type, and precision specializations."""

    ElementA: object
    kElementBTypeId: DataType
    kHighPrecision: bool

    @property
    def UDQ(self):
        if self.ElementA == l.float16:
            return UnifiedDequantizerForFp4Fp16
        return (
            UnifiedDequantizerForMxFp4Bf16
            if self.kElementBTypeId == DataType.MxFp4e2m1
            else UnifiedDequantizerForNvFp4Bf16
        )

    __triton_builtin__ = True

    @g.jit
    def Mma(fa, fb, c, BF16: l.constexpr):
        if BF16:
            return mma_m16n16k16_bf16(fa, fb, c)
        else:
            return mma_m16n16k16_fp16(fa, fb, c)


@g.jit
def FetchScalesRegs(
    shm_scales, group_k, group_n, wtid, GN: l.constexpr, MX: l.constexpr
):
    if MX:
        lane_scale = (wtid // 32) * 16 + wtid % 16
        idx = group_k * GN + group_n * 32 + lane_scale
    else:
        idx = group_k * (GN // 32 * 64) + group_n * 64 + wtid
    return l.load(shm_scales.to(l.pointer_type(l.uint16, 3)) + idx).to(l.uint32)


class WarpPartitionMatmul:
    __triton_builtin__ = True

    @g.jit
    def ReadShmA(
        sa,
        tile_idx_k,
        PM: l.constexpr,
        PN: l.constexpr,
        AM: l.constexpr,
        AK: l.constexpr,
        GK: l.constexpr,
        Layout: l.constexpr,
    ):
        V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
        tid = l.arange(0, 256, layout=l.SliceLayout(1, V))
        wid = tid // 64
        return MatrixALayout.FetchRegisters(
            sa,
            (wid // PN % PM) * AM,
            (wid // (PM * PN)) * AK + tile_idx_k,
            tid % 64,
            AM,
            Layout.WarpMatmulLayout.kReadBatchA,
            GK,
            Layout,
        )

    @g.jit
    def ReadShmB(
        sb,
        ss,
        tile_idx_k,
        tile_idx_n,
        PM: l.constexpr,
        PN: l.constexpr,
        BN: l.constexpr,
        AK: l.constexpr,
        GN: l.constexpr,
        MX: l.constexpr,
    ):
        V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
        tid = l.arange(0, 256, layout=l.SliceLayout(1, V))
        wid = tid // 64
        group_k = (wid // (PM * PN)) * AK + tile_idx_k
        group_n = (wid % PN) * BN + tile_idx_n
        packed_scales = FetchScalesRegs(ss, group_k, group_n, tid % 64, GN, MX)
        packed_scales = l.convert_layout(
            packed_scales, l.BlockedLayout([1], [64], [4], [0])
        )
        qw = MatrixBLayout.FetchRegisters(sb, group_n, group_k, tid % 64, GN)
        return qw + (packed_scales,)

    @g.jit
    def Prefetch(
        sa,
        sb,
        ss,
        PM: l.constexpr,
        PN: l.constexpr,
        AM: l.constexpr,
        BN: l.constexpr,
        AK: l.constexpr,
        MX: l.constexpr,
        GK: l.constexpr,
        GN: l.constexpr,
        Layout: l.constexpr,
    ):
        a = WarpPartitionMatmul.ReadShmA(sa, 0, PM, PN, AM, AK, GK, Layout)
        b = WarpPartitionMatmul.ReadShmB(sb, ss, 0, 0, PM, PN, BN, AK, GN, MX)
        return a, b

    @g.jit
    def PipelineCompute(
        sa,
        sb,
        ss,
        a,
        b,
        acc,
        PM: l.constexpr,
        PN: l.constexpr,
        AM: l.constexpr,
        BN: l.constexpr,
        AK: l.constexpr,
        MX: l.constexpr,
        HP: l.constexpr,
        BF16: l.constexpr,
        GK: l.constexpr,
        GN: l.constexpr,
        Layout: l.constexpr,
    ):
        for ki in l.static_range(AK):
            for ni in l.static_range(BN):
                acc = WarpPartitionMatmul.Matmul(
                    a, b, acc, MX, HP, BF16, AM, BN, ni, Layout
                )
                if ni + 1 < BN:
                    b = WarpPartitionMatmul.ReadShmB(
                        sb, ss, ki, ni + 1, PM, PN, BN, AK, GN, MX
                    )
            if ki + 1 < AK:
                a = WarpPartitionMatmul.ReadShmA(sa, ki + 1, PM, PN, AM, AK, GK, Layout)
                b = WarpPartitionMatmul.ReadShmB(
                    sb, ss, ki + 1, 0, PM, PN, BN, AK, GN, MX
                )
        return acc

    @g.jit
    def Matmul(
        va,
        b,
        acc,
        MX: l.constexpr,
        HP: l.constexpr,
        BF16: l.constexpr,
        AM: l.constexpr,
        BN: l.constexpr,
        warp_idx_n: l.constexpr,
        Layout: l.constexpr,
    ):
        """Apply native j -> m -> uint2-fragment accumulation, returning SSA state."""
        accum_layout: l.constexpr = Layout.WarpAccumLayout[warp_idx_n]
        reg_a_layout: l.constexpr = Layout.WarpMatmulRegALayout
        UDQ: l.constexpr = (
            UnifiedDequantizerForMxFp4Bf16
            if MX
            else (
                UnifiedDequantizerForNvFp4Bf16 if BF16 else UnifiedDequantizerForFp4Fp16
            )
        )
        ST: l.constexpr = l.bfloat16 if MX else l.float16
        ds = UDQ.DequantScales(b[4], HP)
        scales = (
            (ds & 65535).to(l.uint16).to(ST, bitcast=True),
            (ds >> 16).to(l.uint16).to(ST, bitcast=True),
        )
        for j in l.static_range(4):
            q = b[j]
            dq = UDQ.DequantWithScale(q, scales[0] if j < 2 else scales[1], HP)
            # uint4 -> two uint2 fragments; these tuple views do not move data.
            frag_b = (dq[:2], dq[2:])
            for m in l.static_range(AM):
                va_ptr = (va[m][reg_a_layout[j]][:2], va[m][reg_a_layout[j]][2:])
                for fragment in l.static_range(2):
                    # C^T = B^T * A, as in ArchMma::Mma(frag_b[l], va_ptr[l], ...).
                    value = MmaSelector.Mma(
                        frag_b[fragment],
                        va_ptr[fragment],
                        acc[m * BN * 2 + accum_layout[j]],
                        BF16,
                    )
                    # Gluon's tuple is the flattened native acc[m][acc_idx].
                    acc = (
                        acc[: m * BN * 2 + accum_layout[j]]
                        + (value,)
                        + acc[m * BN * 2 + accum_layout[j] + 1 :]
                    )
        return acc
