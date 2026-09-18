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

import triton
import triton.experimental.gluon as g
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language.amd.cdna3 import buffer_store


class WriteResultLayoutBase:
    __triton_builtin__ = True

    kTile = 16
    kThreads = 256
    kResultVecSize = 8
    kShmVecSize = 4


class WriteResultLayoutEven(WriteResultLayoutBase):
    @staticmethod
    @triton.constexpr_function
    def GetShmLayout(RM, PM, PN, GN):
        # Native strides in uint2 units: loc, j, WarpM, WarpN, (lane%16,lane/16).
        return (16 * GN // 4, 4, RM * 16 * GN // 4, GN // PN // 4, GN // 4, 1)

    @staticmethod
    @g.jit
    def WriteBackIdxToCoord(
        idx,
        m_tile_start,
        AM: l.constexpr,
        RM: l.constexpr,
        PM: l.constexpr,
        GN: l.constexpr,
    ):
        partition = idx // (RM * 16 * GN // 8)
        partition_idx = idx % (RM * 16 * GN // 8)
        row = partition_idx // (GN // 8) + (partition * AM + m_tile_start) * 16
        col = partition_idx % (GN // 8)
        return row, col


class WriteResultLayoutUneven(WriteResultLayoutBase):
    @staticmethod
    @triton.constexpr_function
    def GetShmLayout(RM, PM, PN, GN):
        return (PM * 16 * GN // 4, 4, 16 * GN // 4, GN // PN // 4, GN // 4, 1)

    @staticmethod
    @g.jit
    def WriteBackIdxToCoord(
        idx,
        m_tile_start,
        AM: l.constexpr,
        RM: l.constexpr,
        PM: l.constexpr,
        GN: l.constexpr,
    ):
        stripe = idx // (PM * 16 * GN // 8)
        stripe_idx = idx % (PM * 16 * GN // 8)
        partition = stripe_idx // (16 * GN // 8)
        partition_idx = stripe_idx % (16 * GN // 8)
        row = partition_idx // (GN // 8) + (partition * AM + m_tile_start + stripe) * 16
        col = partition_idx % (GN // 8)
        return row, col


@g.jit
def WriteResult(
    alpha,
    shm_buf,
    acc,
    C,
    tile_m,
    N: l.constexpr,
    GN: l.constexpr,
    PM: l.constexpr,
    PN: l.constexpr,
    AM: l.constexpr,
    BN: l.constexpr,
    HAS_ALPHA: l.constexpr,
    Op: l.constexpr,
):
    MAX_RM: l.constexpr = min(AM, 65536 // (GN * 2 * 16 * PM))
    RM: l.constexpr = AM // triton.cdiv(AM, MAX_RM)
    WRITE_LOOP: l.constexpr = triton.cdiv(AM, RM)
    # Capacity reservation for the native ShmBuf union's result member. Direct
    # LDS accesses use exact native strides, without padding physical storage.
    shm_result = l.allocate_shared_memory(
        C.dtype.element_ty, [RM * PM * 16 * GN], l.SwizzledSharedLayout(1, 1, 1, [0])
    )
    ptr = shm_buf.to(l.pointer_type(C.dtype.element_ty, 3))
    strides: l.constexpr = Op.GetShmLayout(RM, PM, PN, GN)
    FL: l.constexpr = (
        acc[0]
        .reshape([4, 4, 4, 16])
        .permute([0, 1, 3, 2])
        .reshape([4, 64, 4])
        .type.layout
    )
    wid = l.arange(0, 4, layout=l.SliceLayout(1, l.SliceLayout(2, FL)))
    wtid = l.arange(0, 64, layout=l.SliceLayout(0, l.SliceLayout(2, FL)))
    reg = l.arange(0, 4, layout=l.SliceLayout(0, l.SliceLayout(1, FL)))
    warp_m = wid[:, None, None] // PN % PM
    warp_n = wid[:, None, None] % PN
    acc_row = wid[:, None, None] // (PM * PN)
    lane = wtid[None, :, None]
    VL: l.constexpr = l.BlockedLayout([1, 8], [64, 1], [4, 1], [0, 1])
    ITEMS: l.constexpr = RM * PM * 16 * GN // 8
    for START in l.static_range(0, WRITE_LOOP * RM, RM):
        for loc in l.static_range(RM):
            if START + loc < AM:
                for j in l.static_range(BN * 2):
                    value = acc[(START + loc) * BN * 2 + j]
                    if HAS_ALPHA:
                        value = value * alpha
                    value = value.to(C.dtype.element_ty).reshape([4, 4, 4, 16])
                    value = value.permute([0, 1, 3, 2]).reshape([4, 64, 4])
                    coord = (
                        loc * strides[0]
                        + j * strides[1]
                        + warp_m * strides[2]
                        + warp_n * strides[3]
                        + lane % 16 * strides[4]
                        + lane // 16 * strides[5]
                    )
                    l.store(ptr + coord * 4 + reg[None, None, :], value, acc_row == 0)
        l.thread_barrier()
        pending_items = min(RM, AM - START) * PM * 16 * GN // 8
        # Native thread-strided uint4 reads/stores, including its final tail.
        for i in l.static_range(triton.cdiv(ITEMS, 256)):
            idx = i * 256 + l.arange(0, 256, layout=l.SliceLayout(1, VL))
            element = l.arange(0, 8, layout=l.SliceLayout(0, VL))
            output = l.load(
                ptr + idx[:, None] * 8 + element[None, :],
                (idx < pending_items)[:, None],
                other=0,
            )
            rm, cn = Op.WriteBackIdxToCoord(idx, START, AM, RM, PM, GN)
            buffer_store(
                output,
                C,
                (rm * N + cn * 8)[:, None] + element[None, :],
                ((idx < pending_items) & (rm < tile_m))[:, None],
                cache="",
            )
        if START + RM < AM:
            l.thread_barrier()
    shm_result._keep_alive()
