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

import triton
import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import BufferResource, GetConditionShmPtr
from triton.experimental.gluon import language as l


class MatrixALayout:
    __triton_builtin__ = True

    @g.jit
    def InitializeBufferResource(tile_m, k, a_ptr, GK: l.constexpr):
        return (
            a_ptr.to(l.pointer_type(l.uint32)),
            ((tile_m - 1) * k + GK) * 2,
            BufferResource.kDataFormatU32Config,
        )

    @g.jit
    def FetchGlobal(r_a, n, k, tid, GM: l.constexpr, GK: l.constexpr):
        reg_a = ()
        for i in l.static_range(triton.cdiv(GM * GK // 8, 256)):
            idx = tid + i * 256
            row = idx // (GK // 8)
            col = idx % (GK // 8)
            reg_a += (
                BufferResource.Load(
                    r_a, (row * k // 8 + col) * 16, 0, BufferResource.kNone
                ),
            )
        return reg_a

    @g.jit
    def StoreShared(
        reg_a, tid, shm, GM: l.constexpr, GK: l.constexpr, Layout: l.constexpr
    ):
        V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
        word = l.arange(0, 4, layout=l.SliceLayout(0, V))
        for i in l.static_range(triton.cdiv(GM * GK // 8, 256)):
            idx = tid + i * 256
            coord = Layout.WriteShmCoordA(tid, i, Layout)
            ptr = GetConditionShmPtr(shm + coord * 4, idx < GM * GK // 8)
            l.store(ptr[:, None] + word[None, :], reg_a[i])

    @g.jit
    def FetchRegisters(
        shm,
        tile_idx_m,
        tile_idx_k,
        wtid,
        AM: l.constexpr,
        BATCH: l.constexpr,
        GK: l.constexpr,
        Layout: l.constexpr,
    ):
        V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
        L: l.constexpr = l.BlockedLayout([1], [64], [4], [0])
        word = l.arange(0, 4, layout=l.SliceLayout(0, V))
        va = ()
        for m in l.static_range(AM):
            va += ((),)
        # Native traversal: batch first, then every M tile in DataA.
        for j in l.static_range(BATCH):
            for m in l.static_range(AM):
                coord = Layout.ReadShmCoordA(
                    tile_idx_m + m, tile_idx_k, j, wtid, Layout
                )
                v = l.load(shm + coord[:, None] * 4 + word[None, :])
                even, odd = l.split(v.reshape([256, 2, 2]))
                v0, v2 = l.split(even)
                v1, v3 = l.split(odd)
                batch = (
                    l.convert_layout(v0, L),
                    l.convert_layout(v1, L),
                    l.convert_layout(v2, L),
                    l.convert_layout(v3, L),
                )
                va = va[:m] + (va[m] + (batch,),) + va[m + 1 :]
        return va


class MatrixBLayout:
    __triton_builtin__ = True

    @g.jit
    def InitializeBufferResource(n, b_ptr, GN: l.constexpr, GK: l.constexpr):
        return (
            b_ptr.to(l.pointer_type(l.uint32)),
            ((GK // 64 - 1) * n + GN) * 32,
            BufferResource.kDataFormatU32Config,
        )

    @g.jit
    def FetchGlobal(r_b, n, k, tid, GN: l.constexpr, GK: l.constexpr):
        reg_b = ()
        for i in l.static_range(triton.cdiv(GK * GN // 32, 256)):
            idx = tid + i * 256
            row = idx // (2 * GN)
            col = idx % (2 * GN)
            reg_b += (
                BufferResource.Load(
                    r_b, (row * n * 2 + col) * 16, 0, BufferResource.kNone
                ),
            )
        return reg_b

    @g.jit
    def StoreShared(reg_b, tid, shm, GN: l.constexpr, GK: l.constexpr):
        V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
        word = l.arange(0, 4, layout=l.SliceLayout(0, V))
        for i in l.static_range(triton.cdiv(GK * GN // 32, 256)):
            idx = tid + i * 256
            ptr = GetConditionShmPtr(shm + idx * 4, idx < GK * GN // 32)
            l.store(ptr[:, None] + word[None, :], reg_b[i])

    @g.jit
    def FetchRegisters(shm, tile_idx_n, tile_idx_k, wtid, GN: l.constexpr):
        V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
        L: l.constexpr = l.BlockedLayout([1], [64], [4], [0])
        word = l.arange(0, 4, layout=l.SliceLayout(0, V))
        coord = tile_idx_k * (2 * GN) + tile_idx_n * 64 + wtid
        v = l.load(shm + coord[:, None] * 4 + word[None, :])
        even, odd = l.split(v.reshape([256, 2, 2]))
        v0, v2 = l.split(even)
        v1, v3 = l.split(odd)
        return (
            l.convert_layout(v0, L),
            l.convert_layout(v1, L),
            l.convert_layout(v2, L),
            l.convert_layout(v3, L),
        )


class ScaleLayout:
    __triton_builtin__ = True

    @g.jit
    def InitializeBufferResource(
        n, scale_ptr, GN: l.constexpr, GK: l.constexpr, GROUP_SIZE: l.constexpr
    ):
        return (
            scale_ptr.to(l.pointer_type(l.uint32)),
            (GK // GROUP_SIZE - 1) * n + GN,
            BufferResource.kDataFormatU32Config,
        )

    @g.jit
    def FetchScale(
        r_scale, n, tid, GN: l.constexpr, GK: l.constexpr, GROUP_SIZE: l.constexpr
    ):
        reg_scale = ()
        for i in l.static_range(triton.cdiv(GN * (GK // GROUP_SIZE) // 16, 256)):
            idx = tid + i * 256
            row = idx // (64 // GROUP_SIZE * GN // 16)
            col = idx % (64 // GROUP_SIZE * GN // 16)
            reg_scale += (
                BufferResource.Load(
                    r_scale,
                    (row * n * 64 // GROUP_SIZE // 16 + col) * 16,
                    0,
                    BufferResource.kNone,
                ),
            )
        return reg_scale

    @g.jit
    def StoreScaleShm(
        reg_scale,
        tid,
        shm_scales,
        GN: l.constexpr,
        GK: l.constexpr,
        GROUP_SIZE: l.constexpr,
    ):
        V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
        word = l.arange(0, 4, layout=l.SliceLayout(0, V))
        for i in l.static_range(triton.cdiv(GN * (GK // GROUP_SIZE) // 16, 256)):
            idx = tid + i * 256
            ptr = GetConditionShmPtr(
                shm_scales + idx * 4, idx < GN * (GK // GROUP_SIZE) // 16
            )
            l.store(ptr[:, None] + word[None, :], reg_scale[i])


@dataclass(frozen=True)
class ShmBuf:
    """Native FP4 shared-union dimensions, in bytes, for GEMMFp4Fp16Config."""

    __triton_builtin__ = True
    config: object
    kMaxShmSize = 64 * 1024

    @property
    def kScaleZpBatch(self):
        return triton.cdiv(self.config.kGroupK, self.config.kGroupSize)

    @property
    def data_bytes(self):
        config = self.config
        return (
            config.kGroupM * config.kGroupK * config.ElementA.primitive_bitwidth // 8
            + config.kGroupN * config.kGroupK // 2
            + self.kScaleZpBatch * config.kGroupN
        )

    @property
    def kSingleBuffer(self):
        return self.data_bytes * self.config.kStages > self.kMaxShmSize

    @property
    def kMaxResultMTilePerPartition(self):
        config = self.config
        return min(
            config.kWarpTileM,
            self.kMaxShmSize // (config.kGroupN * 2 * 16 * config.WP.kPartitionM),
        )

    @property
    def kResultMTilePerPartition(self):
        am = self.config.kWarpTileM
        return am // triton.cdiv(am, self.kMaxResultMTilePerPartition)

    @property
    def reduction_bytes(self):
        return self.config.kThreadAccumTileNRegs * 2 * 64 * 16

    @property
    def result_bytes(self):
        return (
            self.kResultMTilePerPartition
            * self.config.WP.kPartitionM
            * 16
            * self.config.kGroupN
            * 2
        )

    @property
    def layout_bytes(self):
        return max(
            self.data_bytes * (1 if self.kSingleBuffer else self.config.kStages),
            self.reduction_bytes,
            self.result_bytes,
        )
