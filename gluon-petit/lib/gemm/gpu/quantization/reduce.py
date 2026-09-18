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

import triton.experimental.gluon as g
from triton.experimental.gluon import language as l


@g.jit
def BlockReduce(
    shm_buf,
    acc,
    PM: l.constexpr,
    PN: l.constexpr,
    PK: l.constexpr,
    AM: l.constexpr,
    BN: l.constexpr,
):
    # Native ReductionStorage: float4 acc[N register tile][2 warps][64 lanes].
    # The grid's pipeline LDS is dead here. Reserve capacity and reuse LDS
    # offset zero, as the native ShmBuf union does. No descriptor load/store
    # or layout conversion participates in this reduction.
    if PK <= 1:
        return acc
    else:
        shm_red = l.allocate_shared_memory(
            l.float32, [BN * 2, 2, 64, 4], l.SwizzledSharedLayout(1, 1, 1, [3, 2, 1, 0])
        )
        ptr = shm_buf.to(l.pointer_type(l.float32, 3))
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
        acc_row = wid[:, None, None] // (PM * PN)
        acc_col = wid[:, None, None] % (PM * PN)
        lane_reg = wtid[None, :, None] * 4 + reg[None, None, :]
        result = ()
        for m in l.static_range(AM):
            row_acc = ()
            for j in l.static_range(BN * 2):
                # MFMA (N,M) -> each physical lane's float4; only a view.
                row_acc += (
                    acc[m * BN * 2 + j]
                    .reshape([4, 4, 4, 16])
                    .permute([0, 1, 3, 2])
                    .reshape([4, 64, 4]),
                )
            for shift in l.static_range(PK.bit_length() - 2, -1, -1):
                red_off = 1 << shift
                active = (red_off <= acc_row) & (acc_row < 2 * red_off)
                next_acc = ()
                for j in l.static_range(BN * 2):
                    wr = (
                        ptr
                        + (j * 2 + (acc_row - red_off) * PM * PN + acc_col) * 256
                        + lane_reg
                    )
                    value = row_acc[j]
                    if (1 << shift) < PK // 2:
                        rd = l.load(
                            ptr
                            + (j * 2 + acc_row * PM * PN + acc_col) * 256
                            + lane_reg,
                            active,
                            other=0,
                        )
                        previous = l.load(wr, active, other=0)
                        value = l.where(active, value + (rd + previous), value)
                    l.store(wr, value, active)
                    next_acc += (value,)
                row_acc = next_acc
                l.thread_barrier()
            for j in l.static_range(BN * 2):
                rd = l.load(
                    ptr + (j * 2 + acc_col) * 256 + lane_reg, acc_row == 0, other=0
                )
                value = l.where(acc_row == 0, row_acc[j] + rd, row_acc[j])
                result += (
                    value.reshape([4, 4, 16, 4])
                    .permute([0, 1, 3, 2])
                    .reshape([4, 16, 16]),
                )
            l.thread_barrier()
        shm_red._keep_alive()
        return result
