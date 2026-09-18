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

"""Native quantization_utils.cuh/.cu; layout methods return flattened strides."""

from dataclasses import dataclass

import torch
import triton
import triton.experimental.gluon as g
from lib.gemm.rocm.amd_fastmath import hmul2
from lib.gemm.rocm.amd_intrinsics import _amdgcn_dequant_library, bitreverse
from lib.gemm.rocm.quantization.dequant import (
    UnifiedDequantizerForFp4Fp16,
    UnifiedDequantizerForMxFp4Bf16,
    UnifiedDequantizerForNvFp4Bf16,
)
from lib.gemm.rocm.quantization.types import DataType
from triton.experimental.gluon import language as l

kBits = 4
kPackFactor = 32 // kBits
kQuantVecSize = 4
kRowGroupSize = 16
kMxRowGroupSize = 32


@dataclass(frozen=True)
class TileShmLayout:
    __triton_builtin__ = True
    kLayoutM: int
    kLayoutN: int
    kTileM: int
    kTileN: int
    kPackSize: int
    kOutputVecBatch: int
    kBlockGroupM: int
    kBlockGroupN: int

    @property
    def kGroupM(self):
        return self.kBlockGroupM * self.kLayoutM

    @property
    def kGroupN(self):
        return self.kBlockGroupN * self.kLayoutN

    @triton.constexpr_function
    def GetShmLayout(self):
        # ((tile_m,tile_n),(block_m,block_n),(lane%16,lane//16)).
        return (
            1,
            16 * self.kGroupM // self.kPackSize,
            self.kLayoutM // self.kPackSize,
            self.kGroupM * self.kLayoutN // self.kPackSize,
            self.kGroupM // self.kPackSize,
            self.kTileM,
        )

    @triton.constexpr_function
    def GetOnDiskLayout(self, n):
        # (id_m,id_n,wtid,(wid%block_m,wid//block_m)), in output vectors.
        return (
            n * self.kGroupM // self.kPackSize // self.kOutputVecBatch,
            self.kLayoutM * self.kGroupN // self.kPackSize // self.kOutputVecBatch,
            1,
            n * self.kLayoutM // self.kPackSize // self.kOutputVecBatch,
            64,
        )


@dataclass(frozen=True)
class RepackQWeightLayout(TileShmLayout):
    def __post_init__(self):
        assert self.kTileM * self.kTileN == 4, "The weight tile must be 4"

    @property
    def kNumWarps(self):
        return self.kBlockGroupM * self.kBlockGroupN

    kDequantOutputBatch = 4


@dataclass(frozen=True)
class RepackScaleLayoutBase:
    __triton_builtin__ = True
    kLayoutM: int
    kLayoutN: int
    kPackSize: int
    kBlockGroupM: int
    kBlockGroupN: int
    kNumWarps = 1
    kOutputVecBatch = 4

    @property
    def kGroupM(self):
        return self.kBlockGroupM * self.kLayoutM

    @property
    def kGroupN(self):
        return self.kBlockGroupN * self.kLayoutN

    @property
    def kSubWarpSize(self):
        return self.kLayoutM * self.kLayoutN // self.kPackSize // 4

    @triton.constexpr_function
    def GetShmLayout(self):
        # ((i%2,i//2),((idx%sub%8,idx%sub//8),(idx//sub%block_m,idx//sub//block_m))).
        return (
            16 * self.kGroupM // self.kPackSize,
            self.kGroupM // self.kPackSize,
            2 * self.kGroupM // self.kPackSize,
            1,
            self.kLayoutM // self.kPackSize,
            self.kGroupM * self.kLayoutN // self.kPackSize,
        )

    @triton.constexpr_function
    def GetOnDiskLayout(self, n):
        return (
            n * self.kGroupM // self.kPackSize // 4,
            self.kLayoutM * self.kGroupN // self.kPackSize // 4,
            1,
            n * self.kLayoutM // self.kPackSize // 4,
            self.kSubWarpSize,
        )


@dataclass(frozen=True)
class RepackScaleLayout(RepackScaleLayoutBase):
    kExpBias: int

    @staticmethod
    @g.jit
    def Transform(v, Layout: l.constexpr):
        if Layout.kExpBias == 0:
            return v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24)
        else:
            r = l.full(v[0].shape, 0, l.uint32, v[0].type.layout)
            for i in l.static_range(2):
                x = v[i * 2].to(l.uint8).to(l.float8e4nv, bitcast=True).to(l.float16)
                y = (
                    v[i * 2 + 1]
                    .to(l.uint8)
                    .to(l.float8e4nv, bitcast=True)
                    .to(l.float16)
                )
                h2 = x.to(l.uint16, bitcast=True).to(l.uint32) | (
                    y.to(l.uint16, bitcast=True).to(l.uint32) << 16
                )
                multiple = (
                    l.full((), 1 << Layout.kExpBias, l.float16)
                    .to(l.uint16, bitcast=True)
                    .to(l.uint32)
                )
                scaled = hmul2(h2, multiple | (multiple << 16), False)
                s = ((scaled & 65535) >> 7) | ((scaled >> 23) << 8)
                r |= s << (i * 16)
            return r


class RepackMxScaleLayout(RepackScaleLayoutBase):
    @staticmethod
    @g.jit
    def Transform(v, Layout: l.constexpr):
        return v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24)


RepackQWeightLayout128x16 = RepackQWeightLayout(128, 16, 4, 1, 8, 4, 2, 2)
RepackQWeightLayout64x32 = RepackQWeightLayout(64, 32, 2, 2, 8, 4, 4, 1)
RepackScaleLayout64x32 = RepackScaleLayout(64, 32, 16, 1, 2, 7)
RepackMxScaleLayout64x32 = RepackMxScaleLayout(64, 32, 32, 4, 1)


@g.jit
def PetitFormat(v):
    r = l.full(v.shape, 0, l.uint32, v.type.layout)
    for i in l.static_range(8):
        off_s = 15 - (i % 4 // 2) * 8 + (i % 2) * 16
        off_d = off_s - 6
        if i >= 4:
            off_s = 31 - off_s
            off_d = off_s + 4
        u = (v >> (i * 4)) & 15
        val = u & 7
        sign = l.where(val == 0, 0, u >> 3)
        if i >= 4:
            val = bitreverse(val) >> 29
        r |= (sign << off_s) | (val << off_d)
    return r


@g.jit
def RepackNvFp4ToPetitFp4WeightsKernel(
    Process: l.constexpr, Out, In, K: l.constexpr, N: l.constexpr, Layout: l.constexpr
):
    GM: l.constexpr = Layout.kGroupM
    GN: l.constexpr = Layout.kGroupN
    THREADS: l.constexpr = Layout.kNumWarps * 64
    INTS: l.constexpr = GM * GN // 8
    V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [Layout.kNumWarps, 1], [1, 0])
    tid = l.arange(0, THREADS, layout=l.SliceLayout(1, V))
    word = l.arange(0, 4, layout=l.SliceLayout(0, V))
    wid, wtid = tid // 64, tid % 64
    id_m, id_n = l.program_id(0), l.program_id(1)
    in_ptr = (
        In.to(l.pointer_type(l.uint32)) + (id_m * GM // 32 + id_n * K * GN // 32) * 4
    )
    shm = l.allocate_shared_memory(
        l.uint32, [INTS], l.SwizzledSharedLayout(1, 1, 1, [0])
    )
    ptr = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
    for i in l.static_range(triton.cdiv(INTS // 4, THREADS)):
        idx = tid + i * THREADS
        row, col = idx // (GM // 32), idx % (GM // 32)
        value = l.load(
            in_ptr + (row * K // 32 + col)[:, None] * 4 + word[None, :],
            (idx < INTS // 4)[:, None],
            other=0,
        )
        l.store(
            ptr + idx[:, None] * 4 + word[None, :], value, (idx < INTS // 4)[:, None]
        )
    l.thread_barrier()
    s: l.constexpr = Layout.GetShmLayout()
    ret = ()
    for i in l.static_range(4):
        coord = (
            (i % Layout.kTileM) * s[0]
            + (i // Layout.kTileM) * s[1]
            + (wid % Layout.kBlockGroupM) * s[2]
            + (wid // Layout.kBlockGroupM) * s[3]
            + (wtid % 16) * s[4]
            + (wtid // 16) * s[5]
        )
        ret += (Process(l.load(ptr + coord)),)
    o: l.constexpr = Layout.GetOnDiskLayout(N)
    out_idx = (
        id_m * o[0]
        + id_n * o[1]
        + wtid * o[2]
        + (wid % Layout.kBlockGroupM) * o[3]
        + (wid // Layout.kBlockGroupM) * o[4]
    )
    values = l.join(l.join(ret[0], ret[2]), l.join(ret[1], ret[3])).reshape(
        [THREADS, 4]
    )
    values = l.convert_layout(values, V)
    l.store(
        Out.to(l.pointer_type(l.uint32)) + out_idx[:, None] * 4 + word[None, :], values
    )
    shm._keep_alive()


@g.jit
def RepackFp4ScalesKernel(Out, In, K: l.constexpr, N: l.constexpr, Layout: l.constexpr):
    GM: l.constexpr = Layout.kGroupM
    GN: l.constexpr = Layout.kGroupN
    GS: l.constexpr = Layout.kPackSize
    THREADS: l.constexpr = Layout.kNumWarps * 64
    TOTAL: l.constexpr = GM * GN // GS // 4
    tid = l.arange(
        0, THREADS, layout=l.BlockedLayout([1], [64], [Layout.kNumWarps], [0])
    )
    id_m, id_n = l.program_id(0), l.program_id(1)
    in_ptr = (
        In.to(l.pointer_type(l.uint32))
        + id_m * GM // GS // 4
        + id_n * K * GN // GS // 4
    )
    shm = l.allocate_shared_memory(
        l.uint32, [TOTAL], l.SwizzledSharedLayout(1, 1, 1, [0])
    )
    ptr = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
    for i in l.static_range(triton.cdiv(TOTAL, THREADS)):
        idx = tid + i * THREADS
        row, col = idx // (GM // GS // 4), idx % (GM // GS // 4)
        value = l.load(in_ptr + row * K // GS // 4 + col, idx < TOTAL, other=0)
        l.store(ptr + idx, value, idx < TOTAL)
    l.thread_barrier()
    s: l.constexpr = Layout.GetShmLayout()
    o: l.constexpr = Layout.GetOnDiskLayout(N)
    SUB: l.constexpr = Layout.kSubWarpSize
    for batch in l.static_range(triton.cdiv(TOTAL, THREADS)):
        idx = tid + batch * THREADS
        v = ()
        for i in l.static_range(4):
            coord = (
                i % 2 * s[0]
                + i // 2 * s[1]
                + (idx % SUB % 8) * s[2]
                + (idx % SUB // 8) * s[3]
                + (idx // SUB % Layout.kBlockGroupM) * s[4]
                + (idx // SUB // Layout.kBlockGroupM) * s[5]
            )
            v += (
                l.load(
                    ptr.to(l.pointer_type(l.uint8, 3)) + coord, idx < TOTAL, other=0
                ).to(l.uint32),
            )
        out_idx = (
            id_m * o[0]
            + id_n * o[1]
            + (idx % SUB) * o[2]
            + (idx // SUB % Layout.kBlockGroupM) * o[3]
            + (idx // SUB // Layout.kBlockGroupM) * o[4]
        )
        l.store(
            Out.to(l.pointer_type(l.uint32)) + out_idx,
            Layout.Transform(v, Layout),
            idx < TOTAL,
        )
    shm._keep_alive()


class DequantTraitNativeMixIn:
    __triton_builtin__ = True
    kLayoutM = 64
    kLayoutN = 32
    kIsNativeQWFormat = True

    @property
    def kNumWarps(self):
        return self.kBlockGroupK * self.kBlockGroupN

    @property
    def kThreads(self):
        return self.kNumWarps * 64

    @property
    def kGroupK(self):
        return self.kBlockGroupK * self.kLayoutM

    @property
    def kGroupN(self):
        return self.kBlockGroupN * self.kLayoutN

    @property
    def kThreadPerRow(self):
        return self.kGroupK // 32

    @triton.constexpr_function
    def GetFetchQWLayout(self, size_k, size_n):
        return (
            self.kGroupK // 32,
            size_k * self.kGroupN // 32,
            1,
            size_k // 32,
            size_k // 32 * (64 // self.kThreadPerRow),
        )

    @triton.constexpr_function
    def GetFetchScaleLayout(self, size_k, size_n):
        return (
            self.kGroupN * size_k // self.kRowGroupSize // 4,
            self.kGroupK // self.kRowGroupSize // 4,
            1,
            size_k // self.kRowGroupSize // 4,
        )

    @triton.constexpr_function
    def GetOutputLayout(self, size_k, size_n):
        return (self.kGroupK // 8, size_k * self.kGroupN // 8, 4, size_k // 8, 1)


@dataclass(frozen=True)
class DequantTraitNvFp4(DequantTraitNativeMixIn):
    kOutputType: DataType
    kBlockGroupK = 2
    kBlockGroupN = 1
    kRowGroupSize = 16
    Scale = l.float16

    @property
    def UDQ(self):
        return (
            UnifiedDequantizerForFp4Fp16
            if self.kOutputType == DataType.Fp16
            else UnifiedDequantizerForNvFp4Bf16
        )

    @staticmethod
    @g.jit
    def GetScale(shm_scales, tid, Trait: l.constexpr):
        idx = tid % Trait.kThreadPerRow + tid // Trait.kThreadPerRow * (
            Trait.kGroupK // Trait.kRowGroupSize // 2
        )
        packed = l.load(shm_scales.to(l.pointer_type(l.uint16, 3)) + idx).to(l.uint32)
        x = (packed & 255).to(l.uint8).to(l.float8e4nv, bitcast=True).to(l.float16)
        y = (packed >> 8).to(l.uint8).to(l.float8e4nv, bitcast=True).to(l.float16)
        h2 = x.to(l.uint16, bitcast=True).to(l.uint32) | (
            y.to(l.uint16, bitcast=True).to(l.uint32) << 16
        )
        multiple = l.full((), 128, l.float16).to(l.uint16, bitcast=True).to(l.uint32)
        ds = hmul2(h2, multiple | (multiple << 16), False)
        lo, hi = ds & 65535, ds >> 16
        return lo | (lo << 16), hi | (hi << 16)


@dataclass(frozen=True)
class DequantTraitMxFp4(DequantTraitNativeMixIn):
    kOutputType = DataType.Bf16
    kBlockGroupK = 2
    kBlockGroupN = 1
    kRowGroupSize = 32
    UDQ = UnifiedDequantizerForMxFp4Bf16
    Scale = l.bfloat16

    @staticmethod
    @g.jit
    def GetScale(shm_scales, tid, Trait: l.constexpr):
        idx = tid % Trait.kThreadPerRow + tid // Trait.kThreadPerRow * (
            Trait.kGroupK // Trait.kRowGroupSize
        )
        s = l.load(shm_scales.to(l.pointer_type(l.uint8, 3)) + idx).to(l.uint32)
        ds = Trait.UDQ.DequantScales(s | (s << 8), True)
        return ds, ds


class DequantTraitPetitMixIn:
    __triton_builtin__ = True
    kIsNativeQWFormat = False

    @property
    def kBlockGroupK(self):
        return self.QWLayout.kBlockGroupM

    @property
    def kBlockGroupN(self):
        return self.QWLayout.kBlockGroupN

    @property
    def kLayoutM(self):
        return self.QWLayout.kLayoutM

    @property
    def kLayoutN(self):
        return self.QWLayout.kLayoutN

    @property
    def kGroupK(self):
        return self.QWLayout.kGroupM

    @property
    def kGroupN(self):
        return self.QWLayout.kGroupN

    @property
    def kNumWarps(self):
        return self.QWLayout.kNumWarps

    @property
    def kThreads(self):
        return self.kNumWarps * 64

    @property
    def kThreadPerRow(self):
        return self.kGroupK // 32

    @triton.constexpr_function
    def GetFetchQWLayout(self, size_k, size_n):
        return self.QWLayout.GetOnDiskLayout(size_n)

    @triton.constexpr_function
    def GetFetchScaleLayout(self, size_k, size_n):
        return (
            self.kLayoutM * self.kGroupN // self.kRowGroupSize // 4,
            size_n * self.kGroupK // self.kRowGroupSize // 4,
            1,
            size_n * self.kLayoutM // self.kRowGroupSize // 4,
        )

    @triton.constexpr_function
    def GetOutputLayout(self, size_k, size_n):
        return (
            self.kGroupK // 8,
            size_k * self.kGroupN // 8,
            size_k // 8,
            self.QWLayout.kTileM,
            self.kLayoutM // 8,
            size_k * self.kLayoutN // 8,
            1,
            size_k * 16 // 8,
        )


@dataclass(frozen=True)
class DequantTraitPetitNvFp4(DequantTraitPetitMixIn):
    QWLayout: RepackQWeightLayout
    kOutputType: DataType
    kRowGroupSize = 16
    Scale = l.float16

    @property
    def UDQ(self):
        return (
            UnifiedDequantizerForFp4Fp16
            if self.kOutputType == DataType.Fp16
            else UnifiedDequantizerForNvFp4Bf16
        )

    @staticmethod
    @g.jit
    def GetScale(shm_scales, tid, Trait: l.constexpr):
        packed = l.load(shm_scales.to(l.pointer_type(l.uint16, 3)) + tid).to(l.uint32)
        ds = Trait.UDQ.DequantScales(packed, True)
        lo, hi = ds & 65535, ds >> 16
        return lo | (lo << 16), hi | (hi << 16)


@dataclass(frozen=True)
class DequantTraitPetitMxFp4(DequantTraitPetitMixIn):
    QWLayout: RepackQWeightLayout
    kOutputType = DataType.Bf16
    kRowGroupSize = 32
    UDQ = UnifiedDequantizerForMxFp4Bf16
    Scale = l.bfloat16

    @staticmethod
    @g.jit
    def GetScale(shm_scales, tid, Trait: l.constexpr):
        l.static_assert(Trait.kLayoutM == 64 and Trait.kLayoutN == 32)
        wid, wtid = tid // 64, tid % 64
        t = wid * 32 + wtid // 32 * 16 + wtid % 16
        packed = l.load(shm_scales.to(l.pointer_type(l.uint16, 3)) + t).to(l.uint32)
        ds = Trait.UDQ.DequantScales(packed, True)
        lo, hi = ds & 65535, ds >> 16
        return lo | (lo << 16), hi | (hi << 16)


@g.jit
def DequantizeFp4Kernel(
    Out, In, Scales, global_scale, K: l.constexpr, N: l.constexpr, Trait: l.constexpr
):
    THREADS: l.constexpr = Trait.kThreads
    SCALE_U32: l.constexpr = Trait.kGroupK * Trait.kGroupN // Trait.kRowGroupSize // 4
    V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [Trait.kNumWarps, 1], [1, 0])
    tid = l.arange(0, THREADS, layout=l.SliceLayout(1, V))
    word = l.arange(0, 4, layout=l.SliceLayout(0, V))
    wid, wtid = tid // 64, tid % 64
    id_k, id_n = l.program_id(0), l.program_id(1)
    shm = l.allocate_shared_memory(
        l.uint32, [SCALE_U32], l.SwizzledSharedLayout(1, 1, 1, [0])
    )
    ptr = l.full((), 0, l.uint64).to(l.pointer_type(l.uint32, 3))
    s: l.constexpr = Trait.GetFetchScaleLayout(K, N)
    SUB: l.constexpr = (
        Trait.kGroupK // Trait.kRowGroupSize // 4
        if Trait.kIsNativeQWFormat
        else Trait.kLayoutM * Trait.kLayoutN // Trait.kRowGroupSize // 4
    )
    for i in l.static_range(triton.cdiv(SCALE_U32, THREADS)):
        idx = tid + i * THREADS
        coord = id_n * s[0] + id_k * s[1] + (idx % SUB) * s[2] + (idx // SUB) * s[3]
        value = l.load(
            Scales.to(l.pointer_type(l.uint32)) + coord, idx < SCALE_U32, other=0
        )
        l.store(ptr + idx, value, idx < SCALE_U32)
    l.thread_barrier()
    qw_layout: l.constexpr = Trait.GetFetchQWLayout(K, N)
    if Trait.kIsNativeQWFormat:
        coord = (
            id_k * qw_layout[0]
            + id_n * qw_layout[1]
            + (wtid % Trait.kThreadPerRow) * qw_layout[2]
            + (wtid // Trait.kThreadPerRow) * qw_layout[3]
            + wid * qw_layout[4]
        )
    else:
        coord = (
            id_k * qw_layout[0]
            + id_n * qw_layout[1]
            + wtid * qw_layout[2]
            + (wid % Trait.kBlockGroupK) * qw_layout[3]
            + (wid // Trait.kBlockGroupK) * qw_layout[4]
        )
    qw = l.load(In.to(l.pointer_type(l.uint32)) + coord[:, None] * 4 + word[None, :])
    even, odd = l.split(qw.reshape([THREADS, 2, 2]))
    q0, q2 = l.split(even)
    q1, q3 = l.split(odd)
    qw = (q0, q1, q2, q3)
    scale = Trait.GetScale(ptr, tid, Trait)
    DT: l.constexpr = l.bfloat16 if Trait.kOutputType == 5 else l.float16
    global_scale_f16 = global_scale.to(DT)
    gs = global_scale_f16.to(l.uint16, bitcast=True).to(l.uint32)
    gs2 = gs | (gs << 16)
    ret = ()
    for i in l.static_range(4):
        q = l.convert_layout(qw[i], l.SliceLayout(1, V))
        s_bits = (scale[i // 2] >> (i % 2 * 16)).to(l.uint16)
        value_scale = s_bits.to(Trait.Scale, bitcast=True)
        if Trait.kIsNativeQWFormat:
            q_shifted = PetitFormat(q)
        else:
            q_shifted = q
        dq = Trait.UDQ.DequantWithScale(q_shifted, value_scale, True)
        for j in l.static_range(4):
            if Trait.kOutputType == 5:
                # Native HIP __hmul2 rounds BF16 products; fastmath::hmul2 truncates.
                x = (dq[j] & 65535).to(l.uint16).to(l.bfloat16, bitcast=True)
                y = (dq[j] >> 16).to(l.uint16).to(l.bfloat16, bitcast=True)
                x = (x.to(l.float32) * global_scale_f16.to(l.float32)).to(l.bfloat16)
                y = (y.to(l.float32) * global_scale_f16.to(l.float32)).to(l.bfloat16)
                product = x.to(l.uint16, bitcast=True).to(l.uint32) | (
                    y.to(l.uint16, bitcast=True).to(l.uint32) << 16
                )
            else:
                product = hmul2(dq[j], gs2, False)
            ret += (product,)
    o: l.constexpr = Trait.GetOutputLayout(K, N)
    for i in l.static_range(4):
        if Trait.kIsNativeQWFormat:
            out_idx = (
                id_k * o[0]
                + id_n * o[1]
                + (tid % Trait.kThreadPerRow) * o[2]
                + (tid // Trait.kThreadPerRow) * o[3]
                + i * o[4]
            )
        else:
            out_idx = (
                id_k * o[0]
                + id_n * o[1]
                + (wtid % 16) * o[2]
                + (wtid // 16) * o[3]
                + (wid % Trait.kBlockGroupK) * o[4]
                + (wid // Trait.kBlockGroupK) * o[5]
                + (i % Trait.QWLayout.kTileM) * o[6]
                + (i // Trait.QWLayout.kTileM) * o[7]
            )
        values = l.join(
            l.join(ret[i * 4], ret[i * 4 + 2]), l.join(ret[i * 4 + 1], ret[i * 4 + 3])
        ).reshape([THREADS, 4])
        values = l.convert_layout(values, V)
        l.store(
            Out.to(l.pointer_type(l.uint32)) + out_idx[:, None] * 4 + word[None, :],
            values,
        )
    shm._keep_alive()


def DequantNvFp4(output, input, scales, global_scale, out_type, k, n):
    """Decode native NVFP4 weights/scales into N×K FP16/BF16; return 0 or -1."""
    if out_type not in (DataType.Fp16, DataType.Bf16):
        return -1
    trait = DequantTraitNvFp4(out_type)
    if k % trait.kGroupK or n % trait.kGroupN:
        return -1
    DequantizeFp4Kernel[(k // trait.kGroupK, n // trait.kGroupN)](
        output,
        input,
        scales,
        float(global_scale),
        k,
        n,
        trait,
        num_warps=trait.kNumWarps,
        extern_libs={"petit_dequant": _amdgcn_dequant_library()},
    )
    return 0


def DequantMxFp4(output, input, scales, global_scale, out_type, k, n):
    """Decode native MXFP4 weights/scales into N×K BF16; return 0 or -1."""
    if out_type != DataType.Bf16:
        return -1
    trait = DequantTraitMxFp4()
    if k % trait.kGroupK or n % trait.kGroupN:
        return -1
    global_scale *= trait.UDQ.GlobalScaleFactor(True)
    DequantizeFp4Kernel[(k // trait.kGroupK, n // trait.kGroupN)](
        output,
        input,
        scales,
        float(global_scale),
        k,
        n,
        trait,
        num_warps=trait.kNumWarps,
        extern_libs={"petit_dequant": _amdgcn_dequant_library()},
    )
    return 0


def DequantPetitFp4(output, input, scales, global_scale, out_type, k, n):
    """Decode Petit NVFP4 weights/scales into N×K FP16/BF16; return 0 or -1."""
    layout = RepackQWeightLayout64x32
    if k % layout.kGroupM or n % layout.kGroupN:
        return -1
    if out_type not in (DataType.Fp16, DataType.Bf16):
        return -1
    trait = DequantTraitPetitNvFp4(layout, out_type)
    global_scale *= trait.UDQ.GlobalScaleFactor(True)
    DequantizeFp4Kernel[(k // trait.kGroupK, n // trait.kGroupN)](
        output,
        input,
        scales,
        float(global_scale),
        k,
        n,
        trait,
        num_warps=trait.kNumWarps,
        extern_libs={"petit_dequant": _amdgcn_dequant_library()},
    )
    return 0


def DequantPetitMxFp4(output, input, scales, global_scale, out_type, k, n):
    """Decode Petit MXFP4 weights/scales into N×K BF16; return 0 or -1."""
    layout = RepackQWeightLayout64x32
    if k % layout.kGroupM or n % layout.kGroupN:
        return -1
    if out_type != DataType.Bf16:
        return -1
    trait = DequantTraitPetitMxFp4(layout)
    global_scale *= trait.UDQ.GlobalScaleFactor(True)
    DequantizeFp4Kernel[(k // trait.kGroupK, n // trait.kGroupN)](
        output,
        input,
        scales,
        float(global_scale),
        k,
        n,
        trait,
        num_warps=trait.kNumWarps,
        extern_libs={"petit_dequant": _amdgcn_dequant_library()},
    )
    return 0


def RepackNvFp4ToPetitFp4Weights(output, input, k, n, stream):
    """Repack native FP4 weights into caller-owned Petit storage; return None."""
    layout = RepackQWeightLayout64x32
    with torch.cuda.stream(stream):
        RepackNvFp4ToPetitFp4WeightsKernel[(k // layout.kGroupM, n // layout.kGroupN)](
            PetitFormat,
            output,
            input,
            k,
            n,
            layout,
            num_warps=layout.kNumWarps,
        )


def RepackNvFp4ToPetitFp4Scales(output, input, k, n, stream):
    """Repack E4M3 scales with native bias +7 into output; return None."""
    layout = RepackScaleLayout64x32
    with torch.cuda.stream(stream):
        RepackFp4ScalesKernel[(k // layout.kGroupM, n // layout.kGroupN)](
            output,
            input,
            k,
            n,
            layout,
            num_warps=layout.kNumWarps,
            extern_libs={"petit_dequant": _amdgcn_dequant_library()},
        )


def RepackMxFp4ToPetitFp4Scales(output, input, k, n, stream):
    """Repack E8M0 scale bytes unchanged into Petit output; return None."""
    layout = RepackMxScaleLayout64x32
    with torch.cuda.stream(stream):
        RepackFp4ScalesKernel[(k // layout.kGroupM, n // layout.kGroupN)](
            output,
            input,
            k,
            n,
            layout,
            num_warps=layout.kNumWarps,
        )
