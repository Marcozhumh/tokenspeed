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

from dataclasses import dataclass, replace
from threading import Lock

import torch
import triton
import triton.experimental.gluon as g
from lib.gemm.gpu.quantization.reduce import BlockReduce
from lib.gemm.rocm.amd_intrinsics import (
    _amdgcn_dequant_library,
    _amdgcn_intrinsics_library,
    amdgcn_s_waitcnt,
    amdgcn_sched_barrier,
    amdgcn_sched_group_barrier,
)
from lib.gemm.rocm.quantization.fp4.algo_chooser import ChooseDefaultFp4Fp16Solution
from lib.gemm.rocm.quantization.fp4.solution_map import SolutionMap
from lib.gemm.rocm.quantization.fp4.warp_schedule_fp16 import (
    MmaSelector,
    WarpPartitionMatmul,
)
from lib.gemm.rocm.quantization.gemm import (
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
    SolutionId,
    kErrorKernelShape,
    kErrorProblemShape,
)
from lib.gemm.rocm.quantization.kernel_config import TileShape, WarpPartition
from lib.gemm.rocm.quantization.memory_ops import (
    MatrixALayout,
    MatrixBLayout,
    ScaleLayout,
    ShmBuf,
)
from lib.gemm.rocm.quantization.qgemm import (
    WriteResult,
    WriteResultLayoutEven,
    WriteResultLayoutUneven,
)
from lib.gemm.rocm.quantization.types import DataType
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language.amd import AMDMFMALayout

kTile = 16


class QWeightOnDiskLayout128x16:
    __triton_builtin__ = True
    kLayoutM = 128
    kLayoutN = 16
    kLayoutElementsM = 4
    kLayoutElementsN = 1


class QWeightOnDiskLayout64x32:
    __triton_builtin__ = True
    kLayoutM = 64
    kLayoutN = 32
    kLayoutElementsM = 2
    kLayoutElementsN = 2


@dataclass(frozen=True)
class WarpMatmulLayoutTrait:
    __triton_builtin__ = True
    TS: TileShape
    WP: WarpPartition
    DiskLayout = QWeightOnDiskLayout64x32
    kMmaM = 16

    def __post_init__(self):
        assert (
            self.kGroupK % (self.kLayoutM * self.WP.kPartitionK) == 0
            and self.kWarpAtomK > 0
        )
        assert (
            self.TS.kGroupN % (self.kLayoutN * self.WP.kPartitionN) == 0
            and self.kWarpAtomN > 0
        )
        assert self.kThreadAccumTileNRegs > 0
        assert self.kGroupK % self.kLayoutM == 0
        assert self.TS.kNumTileN % self.WP.kPartitionN == 0
        assert self.TS.kNumTileK % self.WP.kPartitionK == 0

    @property
    def ElementA(self):
        return self.TS.ElementA

    @property
    def kLayoutM(self):
        return self.DiskLayout.kLayoutM

    @property
    def kLayoutN(self):
        return self.DiskLayout.kLayoutN

    @property
    def kGroupK(self):
        return self.TS.kGroupK

    @property
    def kVecSize(self):
        return 128 // self.ElementA.primitive_bitwidth

    @property
    def kMmaTileK(self):
        return self.kVecSize * 64 // self.kMmaM

    @property
    def kWarpTileM(self):
        return self.TS.kNumTileM // self.WP.kPartitionM

    @property
    def kWarpAtomK(self):
        return self.kGroupK // self.kLayoutM // self.WP.kPartitionK

    @property
    def kWarpAtomN(self):
        return self.TS.kGroupN // self.kLayoutN // self.WP.kPartitionN

    @property
    def kAccTilePerThread(self):
        return self.kLayoutN // kTile

    @property
    def kThreadAccumTileNRegs(self):
        return self.kWarpAtomN * self.kAccTilePerThread

    @property
    def ThreadAccum(self):
        # Native float4 array dimensions; each Gluon tensor batches the four waves.
        return self.kWarpTileM, self.kThreadAccumTileNRegs

    @property
    def WarpAccumLayout(self):
        # Native TAL shape (atom_n, (elements_m, 4/elements_m)), strides (acc_tiles, (0, 1)).
        return tuple(
            tuple(
                n * self.kAccTilePerThread + j // self.kLayoutElementsM
                for j in range(4)
            )
            for n in range(self.kWarpAtomN)
        )

    @property
    def WarpMatmulRegALayout(self):
        # Native TAL shape ((elements_m, 4/elements_m)), strides ((1, 0)).
        return tuple(j % self.kLayoutElementsM for j in range(4))

    @property
    def kLayoutElementsM(self):
        return self.DiskLayout.kLayoutElementsM

    @property
    def kReadBatchA(self):
        return self.DiskLayout.kLayoutElementsM * 8 // self.kVecSize

    @staticmethod
    @g.jit
    def XorShmLayout(tile_idx_m, tile_idx_k, batch_id, row, col, Trait: l.constexpr):
        base = (
            tile_idx_m * Trait.TS.kTile * Trait.kGroupK // Trait.kVecSize
            + tile_idx_k * Trait.TS.kTile * Trait.kLayoutM // Trait.kVecSize
            + batch_id * 64
        )
        xor_stride = col * Trait.kLayoutElementsM + batch_id
        return (base + col * Trait.kMmaM + row) ^ xor_stride

    @staticmethod
    @g.jit
    def WriteShmCoordA(tid, steps, Trait: l.constexpr):
        l.static_assert(
            Trait.WP.kThreads % (Trait.kGroupK // Trait.kVecSize) == 0,
            "Unaligned GroupK",
        )
        ROW_BLOCK: l.constexpr = Trait.kGroupK // Trait.kVecSize
        LOOP_INC: l.constexpr = Trait.WP.kThreads // ROW_BLOCK
        row, col = tid // ROW_BLOCK + LOOP_INC * steps, tid % ROW_BLOCK
        tile_idx_m, tile_idx_k = row // Trait.TS.kTile, col // (
            Trait.kLayoutM // Trait.kVecSize
        )
        row_in_tile, col_in_tile = row % Trait.TS.kTile, col % (
            Trait.kLayoutM // Trait.kVecSize
        )
        batch_id = col_in_tile % Trait.kLayoutElementsM
        return Trait.XorShmLayout(
            tile_idx_m,
            tile_idx_k,
            batch_id,
            row_in_tile,
            col_in_tile // Trait.kLayoutElementsM,
            Trait,
        )

    @staticmethod
    @g.jit
    def ReadShmCoordA(tile_idx_m, tile_idx_k, batch_id, wtid, Trait: l.constexpr):
        return Trait.XorShmLayout(
            tile_idx_m,
            tile_idx_k,
            batch_id,
            wtid % Trait.kMmaM,
            wtid // Trait.kMmaM,
            Trait,
        )


@dataclass(frozen=True)
class GEMMFp4Fp16Config:
    __triton_builtin__ = True
    TS: TileShape
    WP: WarpPartition
    kStages: int
    kHighPrecision: bool
    kUseZeroPoints = False
    kZpInShm = False
    kScaleVecSize = 16

    def __post_init__(self):
        layout = WarpMatmulLayoutTrait(self.TS, self.WP)
        object.__setattr__(self, "WarpMatmulLayout", layout)
        for name in (
            "ElementA",
            "kElementBTypeId",
            "kTile",
            "kNumTileM",
            "kNumTileN",
            "kNumTileK",
            "kGroupSize",
            "kGroupM",
            "kGroupN",
            "kGroupK",
        ):
            object.__setattr__(self, name, getattr(self.TS, name))
        for name in ("kNumWarps", "kThreads"):
            object.__setattr__(self, name, getattr(self.WP, name))
        for name in (
            "kLayoutM",
            "kLayoutN",
            "kVecSize",
            "kWarpTileM",
            "kThreadAccumTileNRegs",
            "ThreadAccum",
            "WarpAccumLayout",
            "WarpMatmulRegALayout",
        ):
            object.__setattr__(self, name, getattr(layout, name))

    @staticmethod
    @g.jit
    def WriteShmCoordA(tid, steps, Config: l.constexpr):
        Layout: l.constexpr = Config.WarpMatmulLayout
        return Layout.WriteShmCoordA(tid, steps, Layout)

    @staticmethod
    @g.jit
    def ReadShmCoordA(tile_idx_m, tile_idx_k, batch_id, wtid, Config: l.constexpr):
        Layout: l.constexpr = Config.WarpMatmulLayout
        return Layout.ReadShmCoordA(tile_idx_m, tile_idx_k, batch_id, wtid, Layout)


WPMatmul = WarpPartitionMatmul


class PipelineContext:
    """Native resource descriptors and two register stages, represented as SSA tuples."""

    __triton_builtin__ = True

    @g.jit
    def Initialize(A, B, S, tile_m, N, K, Config: l.constexpr):
        l.static_assert(Config.kStages == 2, "Only supports 2 stages")
        V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [Config.kNumWarps, 1], [1, 0])
        zero = l.full([Config.kThreads, 4], 0, l.uint32, V)
        reg_a, reg_b, reg_s = (), (), ()
        for i in l.static_range(
            triton.cdiv(
                Config.kGroupM * Config.kGroupK // Config.kVecSize, Config.kThreads
            )
        ):
            reg_a += (zero,)
        for i in l.static_range(
            triton.cdiv(Config.kGroupN * Config.kGroupK // 32, Config.kThreads)
        ):
            reg_b += (zero,)
        for i in l.static_range(
            triton.cdiv(
                Config.kGroupN
                * (Config.kGroupK // Config.kGroupSize)
                // Config.kScaleVecSize,
                Config.kThreads,
            )
        ):
            reg_s += (zero,)
        regs = (reg_a, reg_b, reg_s)
        return (
            MatrixALayout.InitializeBufferResource(tile_m, K, A, Config.kGroupK),
            MatrixBLayout.InitializeBufferResource(
                N, B, Config.kGroupN, Config.kGroupK
            ),
            ScaleLayout.InitializeBufferResource(
                N, S, Config.kGroupN, Config.kGroupK, Config.kGroupSize
            ),
            regs,
            regs,
            N,
        )

    @g.jit
    def AdvanceGlobalPtr(ctx, Config: l.constexpr):
        return (
            (ctx[0][0] + Config.kGroupK // Config.kVecSize * 4, ctx[0][1], ctx[0][2]),
            (ctx[1][0] + ctx[5] * Config.kGroupK // 8, ctx[1][1], ctx[1][2]),
            (
                ctx[2][0] + ctx[5] * (Config.kGroupK // Config.kGroupSize) // 4,
                ctx[2][1],
                ctx[2][2],
            ),
            ctx[3],
            ctx[4],
            ctx[5],
        )


@g.jit
def LoadGlobal(ctx, n, k, stage: l.constexpr, Config: l.constexpr):
    V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [Config.kNumWarps, 1], [1, 0])
    tid = l.arange(0, Config.kThreads, layout=l.SliceLayout(1, V))
    regs = (
        MatrixALayout.FetchGlobal(ctx[0], n, k, tid, Config.kGroupM, Config.kGroupK),
        MatrixBLayout.FetchGlobal(ctx[1], n, k, tid, Config.kGroupN, Config.kGroupK),
        ScaleLayout.FetchScale(
            ctx[2], n, tid, Config.kGroupN, Config.kGroupK, Config.kGroupSize
        ),
    )
    if stage == 0:
        return ctx[:3] + (regs, ctx[4], ctx[5])
    else:
        return ctx[:3] + (ctx[3], regs, ctx[5])


@g.jit
def StoreShm(ctx, sa, sb, ss, stage_reg: l.constexpr, Config: l.constexpr):
    V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [Config.kNumWarps, 1], [1, 0])
    tid = l.arange(0, Config.kThreads, layout=l.SliceLayout(1, V))
    regs = ctx[3 + stage_reg]
    MatrixALayout.StoreShared(regs[0], tid, sa, Config.kGroupM, Config.kGroupK, Config)
    MatrixBLayout.StoreShared(regs[1], tid, sb, Config.kGroupN, Config.kGroupK)
    ScaleLayout.StoreScaleShm(
        regs[2], tid, ss, Config.kGroupN, Config.kGroupK, Config.kGroupSize
    )


class MultiStagePipeline:
    __triton_builtin__ = True

    @g.jit
    def HotLoopScheduler(Config: l.constexpr):
        GM: l.constexpr = Config.kGroupM
        GN: l.constexpr = Config.kGroupN
        GK: l.constexpr = Config.kGroupK
        PM: l.constexpr = Config.WP.kPartitionM
        PN: l.constexpr = Config.WP.kPartitionN
        PK: l.constexpr = Config.WP.kPartitionK
        MX: l.constexpr = Config.kElementBTypeId == 7
        LOAD_A: l.constexpr = triton.cdiv(GM * GK // 8, 256)
        LOAD_B: l.constexpr = triton.cdiv(GN * GK // 32, 256)
        LOAD_S: l.constexpr = triton.cdiv(GN * (GK // (32 if MX else 16)) // 16, 256)
        GLOBALS: l.constexpr = LOAD_A + LOAD_B + LOAD_S
        MFMA: l.constexpr = (GM // 16 // PM) * (GN // 16 // PN) * (GK // 16 // PK)
        PER_LOAD: l.constexpr = (
            4 if MFMA // GLOBALS > 12 else (2 if MFMA // GLOBALS > 6 else 1)
        )
        l.static_assert(PER_LOAD * GLOBALS <= MFMA)
        amdgcn_sched_group_barrier(0x100, LOAD_A + 2, 0)
        for i in l.static_range(GLOBALS):
            amdgcn_sched_group_barrier(0x200, 1, 0)
            amdgcn_sched_group_barrier(0x8, PER_LOAD, 0)
        for i in l.static_range(GLOBALS):
            amdgcn_sched_group_barrier(0x20, 1, 0)
            amdgcn_sched_group_barrier(0x8, PER_LOAD * 2, 0)
        amdgcn_sched_barrier(0)

    @g.jit
    def Run(
        acc,
        sa,
        sb,
        ss,
        ctx,
        N: l.constexpr,
        K: l.constexpr,
        Config: l.constexpr,
        SINGLE: l.constexpr,
    ):
        l.static_assert(Config.kStages == 2, "Only supports 2 stages")
        GM: l.constexpr = Config.kGroupM
        GN: l.constexpr = Config.kGroupN
        GK: l.constexpr = Config.kGroupK
        PM: l.constexpr = Config.WP.kPartitionM
        PN: l.constexpr = Config.WP.kPartitionN
        PK: l.constexpr = Config.WP.kPartitionK
        AM: l.constexpr = Config.kWarpTileM
        BN: l.constexpr = Config.WarpMatmulLayout.kWarpAtomN
        AK: l.constexpr = Config.WarpMatmulLayout.kWarpAtomK
        MX: l.constexpr = Config.kElementBTypeId == 7
        HP: l.constexpr = Config.kHighPrecision
        BF16: l.constexpr = Config.ElementA == l.bfloat16
        Layout: l.constexpr = Config
        K_TOTAL: l.constexpr = K // GK
        ctx = LoadGlobal(ctx, N, K, 0, Config)
        amdgcn_sched_barrier(0x7DF)
        amdgcn_s_waitcnt(0, -1, -1)
        StoreShm(ctx, sa[0], sb[0], ss[0], 0, Config)
        if K_TOTAL > 1:
            ctx = PipelineContext.AdvanceGlobalPtr(ctx, Config)
            l.thread_barrier()
            ctx = LoadGlobal(ctx, N, K, 1, Config)

        # Native paired hot loop: k_idx + 3 < k_total, k_idx += 2.
        for k_idx in range(0, K_TOTAL - 3, 2):
            amdgcn_sched_barrier(0)
            ctx = PipelineContext.AdvanceGlobalPtr(ctx, Config)
            l.thread_barrier()
            a, b = WPMatmul.Prefetch(
                sa[0], sb[0], ss[0], PM, PN, AM, BN, AK, MX, GK, GN, Layout
            )
            ctx = LoadGlobal(ctx, N, K, 0, Config)
            if SINGLE:
                acc = WPMatmul.PipelineCompute(
                    sa[0],
                    sb[0],
                    ss[0],
                    a,
                    b,
                    acc,
                    PM,
                    PN,
                    AM,
                    BN,
                    AK,
                    MX,
                    HP,
                    BF16,
                    GK,
                    GN,
                    Layout,
                )
                l.thread_barrier()
                StoreShm(ctx, sa[0], sb[0], ss[0], 1, Config)
            else:
                StoreShm(ctx, sa[1], sb[1], ss[1], 1, Config)
                acc = WPMatmul.PipelineCompute(
                    sa[0],
                    sb[0],
                    ss[0],
                    a,
                    b,
                    acc,
                    PM,
                    PN,
                    AM,
                    BN,
                    AK,
                    MX,
                    HP,
                    BF16,
                    GK,
                    GN,
                    Layout,
                )
            MultiStagePipeline.HotLoopScheduler(Config)

            ctx = PipelineContext.AdvanceGlobalPtr(ctx, Config)
            l.thread_barrier()
            STAGE: l.constexpr = 0 if SINGLE else 1
            a, b = WPMatmul.Prefetch(
                sa[STAGE], sb[STAGE], ss[STAGE], PM, PN, AM, BN, AK, MX, GK, GN, Layout
            )
            ctx = LoadGlobal(ctx, N, K, 1, Config)
            if SINGLE:
                acc = WPMatmul.PipelineCompute(
                    sa[STAGE],
                    sb[STAGE],
                    ss[STAGE],
                    a,
                    b,
                    acc,
                    PM,
                    PN,
                    AM,
                    BN,
                    AK,
                    MX,
                    HP,
                    BF16,
                    GK,
                    GN,
                    Layout,
                )
                l.thread_barrier()
                StoreShm(ctx, sa[0], sb[0], ss[0], 0, Config)
            else:
                StoreShm(ctx, sa[0], sb[0], ss[0], 0, Config)
                acc = WPMatmul.PipelineCompute(
                    sa[STAGE],
                    sb[STAGE],
                    ss[STAGE],
                    a,
                    b,
                    acc,
                    PM,
                    PN,
                    AM,
                    BN,
                    AK,
                    MX,
                    HP,
                    BF16,
                    GK,
                    GN,
                    Layout,
                )
            MultiStagePipeline.HotLoopScheduler(Config)

        # k_idx is even at loop exit; the first drain stage is always zero.
        TAIL_START: l.constexpr = max(0, (K_TOTAL - 2) // 2 * 2)
        for i in l.static_range(3):
            if TAIL_START + i < K_TOTAL:
                ctx = PipelineContext.AdvanceGlobalPtr(ctx, Config)
                l.thread_barrier()
                a, b = WPMatmul.Prefetch(
                    sa[0 if SINGLE else i % 2],
                    sb[0 if SINGLE else i % 2],
                    ss[0 if SINGLE else i % 2],
                    PM,
                    PN,
                    AM,
                    BN,
                    AK,
                    MX,
                    GK,
                    GN,
                    Layout,
                )
                if TAIL_START + i + 2 < K_TOTAL:
                    ctx = LoadGlobal(ctx, N, K, i % 2, Config)
                if SINGLE:
                    acc = WPMatmul.PipelineCompute(
                        sa[0 if SINGLE else i % 2],
                        sb[0 if SINGLE else i % 2],
                        ss[0 if SINGLE else i % 2],
                        a,
                        b,
                        acc,
                        PM,
                        PN,
                        AM,
                        BN,
                        AK,
                        MX,
                        HP,
                        BF16,
                        GK,
                        GN,
                        Layout,
                    )
                    if TAIL_START + i + 1 < K_TOTAL:
                        l.thread_barrier()
                        StoreShm(ctx, sa[0], sb[0], ss[0], 1 - i % 2, Config)
                else:
                    if TAIL_START + i + 1 < K_TOTAL:
                        StoreShm(
                            ctx,
                            sa[1 - i % 2],
                            sb[1 - i % 2],
                            ss[1 - i % 2],
                            1 - i % 2,
                            Config,
                        )
                    acc = WPMatmul.PipelineCompute(
                        sa[0 if SINGLE else i % 2],
                        sb[0 if SINGLE else i % 2],
                        ss[0 if SINGLE else i % 2],
                        a,
                        b,
                        acc,
                        PM,
                        PN,
                        AM,
                        BN,
                        AK,
                        MX,
                        HP,
                        BF16,
                        GK,
                        GN,
                        Layout,
                    )
        return acc


@g.jit
def GemmFp4Fp16KernelGrid(
    C,
    A,
    B,
    S,
    G,
    M: l.constexpr,
    N: l.constexpr,
    K: l.constexpr,
    Config: l.constexpr,
    Shm: l.constexpr,
    FACTOR: l.constexpr,
):
    GM: l.constexpr = Config.kGroupM
    GN: l.constexpr = Config.kGroupN
    GK: l.constexpr = Config.kGroupK
    PM: l.constexpr = Config.WP.kPartitionM
    PN: l.constexpr = Config.WP.kPartitionN
    PK: l.constexpr = Config.WP.kPartitionK
    AM: l.constexpr = Config.kWarpTileM
    BN: l.constexpr = Config.WarpMatmulLayout.kWarpAtomN
    A_BYTES: l.constexpr = GM * GK * Config.ElementA.primitive_bitwidth // 8
    B_BYTES: l.constexpr = GN * GK // 2
    S_BYTES: l.constexpr = GN * GK // Config.kGroupSize
    DATA_BYTES: l.constexpr = A_BYTES + B_BYTES + S_BYTES
    STAGES: l.constexpr = 1 if Shm.kSingleBuffer else Config.kStages
    # Reserve sizeof(ShmBuf<Config>::Layout). All union members use LDS offset 0.
    storage = l.allocate_shared_memory(
        l.uint8, [Shm.layout_bytes], l.SwizzledSharedLayout(1, 1, 1, [0])
    )
    shm_buf = l.full((), 0, l.uint64).to(l.pointer_type(l.uint8, 3))
    global_scale = l.load(G) * FACTOR
    ML: l.constexpr = AMDMFMALayout(2, [16, 16, 16], False, [Config.kNumWarps, 1, 1])
    acc = ()
    for i in l.static_range(Config.ThreadAccum[0] * Config.ThreadAccum[1]):
        acc += (l.full([Config.kNumWarps, 16, 16], 0, l.float32, ML),)

    a_ptr = A + l.program_id(0) * K * (GM // Config.kVecSize) * 4
    b_ptr = B + l.program_id(1) * Config.kLayoutM * GN // 8
    scale_ptr = S + l.program_id(1) * (Config.kLayoutM // Config.kGroupSize) * GN // 4
    tile_m = l.minimum(GM, M - l.program_id(0) * GM)
    ctx = PipelineContext.Initialize(a_ptr, b_ptr, scale_ptr, tile_m, N, K, Config)
    sa, sb, ss = (), (), ()
    for stage in l.static_range(STAGES):
        base = shm_buf + stage * DATA_BYTES
        sa += (base.to(l.pointer_type(l.uint32, 3)),)
        sb += ((base + A_BYTES).to(l.pointer_type(l.uint32, 3)),)
        ss += ((base + A_BYTES + B_BYTES).to(l.pointer_type(l.uint32, 3)),)
    acc = MultiStagePipeline.Run(acc, sa, sb, ss, ctx, N, K, Config, Shm.kSingleBuffer)
    storage._keep_alive()
    acc = BlockReduce(shm_buf, acc, PM, PN, PK, AM, BN)
    c_ptr = C + l.program_id(0) * N * GM + l.program_id(1) * GN
    WRITE_LAYOUT: l.constexpr = (
        WriteResultLayoutEven
        if AM % Shm.kResultMTilePerPartition == 0
        else WriteResultLayoutUneven
    )
    WriteResult(
        global_scale,
        shm_buf,
        acc,
        c_ptr,
        tile_m,
        N,
        GN,
        PM,
        PN,
        AM,
        BN,
        True,
        WRITE_LAYOUT,
    )


@dataclass(frozen=True)
class MfmaElementTypes:
    kMfmaType: MatmulMfmaType

    @property
    def ElementA(self):
        return {
            MatmulMfmaType.kMatmulMfmaTypeFp16: l.float16,
            MatmulMfmaType.kMatmulMfmaTypeBf16: l.bfloat16,
            MatmulMfmaType.kMatmulMfmaTypeFp8: l.uint8,
        }[self.kMfmaType]


class ConfigSelector:
    """Native solution specialization; Invoke validates N/K and launches the grid."""

    def __init__(self, id):
        self.kElementBTypeId = (
            DataType.MxFp4e2m1
            if id.element_b == MatmulElementB.kMatmulTypeBMxFp4
            else DataType.Fp4e2m1
        )
        self.kGroupSize = 32 if self.kElementBTypeId == DataType.MxFp4e2m1 else 16
        self.kNumTilesM, self.kNumTilesN, self.kNumTilesK = (
            id.tile_m,
            id.tile_n,
            id.tile_k * 4,
        )
        self.kPipelineStages = 2
        self.kPartitionM, self.kPartitionN, self.kPartitionK = (
            id.warp_partition_m,
            id.warp_partition_n,
            id.warp_partition_k,
        )
        self.kMfmaType = id.mfma_type
        self.kHighPrecision = bool(
            id.features & MatmulFeatures.kMatmulFeatures_HighPrecision
        )
        self.ElementA = MfmaElementTypes(self.kMfmaType).ElementA
        self.TS = TileShape(
            self.ElementA,
            self.kElementBTypeId,
            self.kGroupSize,
            self.kNumTilesM,
            self.kNumTilesN,
            self.kNumTilesK,
        )
        self.WP = WarpPartition(self.kPartitionM, self.kPartitionN, self.kPartitionK)
        self.Config = GEMMFp4Fp16Config(
            self.TS, self.WP, self.kPipelineStages, self.kHighPrecision
        )
        self.ArchMma = MmaSelector(
            self.ElementA, self.kElementBTypeId, self.kHighPrecision
        )
        self.kNumWarps = self.WP.kNumWarps

    def Invoke(self, c, a, b, scales, global_scale, m, n, k, stream):
        """Write GEMM output C from explicit buffers/shape/stream; return native status."""
        config = self.Config
        if n % config.kGroupN or k % config.kGroupK:
            return kErrorProblemShape
        with torch.cuda.stream(stream):
            GemmFp4Fp16KernelGrid[
                (
                    triton.cdiv(m, kTile * self.kNumTilesM),
                    triton.cdiv(n, kTile * self.kNumTilesN),
                )
            ](
                c,
                a.view(torch.uint32),
                b.view(torch.uint32),
                scales.view(torch.uint32),
                global_scale,
                m,
                n,
                k,
                config,
                ShmBuf(config),
                self.ArchMma.UDQ.GlobalScaleFactor(self.kHighPrecision),
                extern_libs={
                    "petit_dequant": _amdgcn_dequant_library(),
                    "petit": _amdgcn_intrinsics_library(
                        triton.cdiv(
                            config.kGroupM * config.kGroupK // 8, config.kThreads
                        )
                    ),
                },
                num_warps=self.kNumWarps,
                waves_per_eu=1,
                enable_fp_fusion=False,
            )
        return 0


class Dispatcher:
    _instance = None
    _instance_lock = Lock()

    def __init__(self):
        self.solution_id_to_call_ = SolutionMap.GetDispatchEntries().copy()

    @staticmethod
    def GetInstance():
        """Return the lazily initialized native-style dispatch singleton."""
        if Dispatcher._instance is None:
            with Dispatcher._instance_lock:
                if Dispatcher._instance is None:
                    Dispatcher._instance = Dispatcher()
        return Dispatcher._instance

    def Dispatch(self, c, a, b, scales, global_scale, m, n, k, solution_id, stream):
        """Invoke the registered solution or return the native kernel-shape error."""
        call = self.solution_id_to_call_.get(solution_id)
        if call is None:
            return kErrorKernelShape
        return call(c, a, b, scales, global_scale, m, n, k, stream)


def GemmFp4Fp16GridImpl(
    c, a, b, scales, global_scale, m, n, k, hints, solution_id, stream
):
    """Apply native validation, in order, then dispatch an unsigned-dimension GEMM."""
    if m == 0 or n == 0 or k == 0:
        return 0

    # Python -1 and its unsigned-long representation name the same native ID.
    if solution_id in (-1, 2**64 - 1):
        solution_id = ChooseDefaultFp4Fp16Solution(m, n, k, hints)

    if solution_id in (-1, 2**64 - 1):
        return kErrorProblemShape

    sol = SolutionId.FromRepr(solution_id)
    if sol.element_b == MatmulElementB.kMatmulTypeBMxFp4:
        if k % 32 != 0:
            return kErrorProblemShape
        if hints.a_type != DataType.Bf16 or hints.c_type != DataType.Bf16:
            return kErrorKernelShape

    return Dispatcher.GetInstance().Dispatch(
        c, a, b, scales, global_scale, m, n, k, solution_id, stream
    )


def GemmFp4Fp16Grid(c, a, b, scales, global_scale, m, n, k, hints, solution_id, stream):
    """Forward buffers, unsigned dimensions, hints, ID, and stream to the native flow."""
    return GemmFp4Fp16GridImpl(
        c, a, b, scales, global_scale, m, n, k, hints, solution_id, stream
    )


def GemmMxFp4Fp16Grid(
    c, a, b, scales, global_scale, m, n, k, hints, solution_id, stream
):
    """Copy hints, force MXFP4 in hints/explicit IDs, and invoke the shared implementation."""
    mx_hints = replace(hints, b_type=DataType.MxFp4e2m1)
    if solution_id in (-1, 2**64 - 1):
        return GemmFp4Fp16GridImpl(
            c, a, b, scales, global_scale, m, n, k, mx_hints, solution_id, stream
        )
    mx_solution = replace(
        SolutionId.FromRepr(solution_id), element_b=MatmulElementB.kMatmulTypeBMxFp4
    )
    return GemmFp4Fp16GridImpl(
        c, a, b, scales, global_scale, m, n, k, mx_hints, mx_solution.Repr(), stream
    )
