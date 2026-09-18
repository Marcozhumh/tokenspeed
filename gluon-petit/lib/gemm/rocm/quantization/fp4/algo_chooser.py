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

from dataclasses import replace

from lib.gemm.rocm.quantization.fp4.solution_map import SOLUTIONS
from lib.gemm.rocm.quantization.gemm import (
    MatmulElementB,
    MatmulFeatures,
    MatmulMfmaType,
)
from lib.gemm.rocm.quantization.types import DataType


def GemmGetSolutions(hints, m, n, k, sols, n_sols):
    """Fill at most n_sols[0] solutions and return native status.

    hints and m/n/k describe the problem. sols is caller-owned storage, or None
    for a count query. n_sols is a mutable one-element sequence: its input value
    is capacity, and its output is the full matching count even when truncated.
    Unsupported weight types return -1 without modifying either output.
    """
    sol_count = 0
    if hints.b_type not in (DataType.Fp4e2m1, DataType.MxFp4e2m1):
        return -1
    for sol in SOLUTIONS.values():
        if sol.element_b != MatmulElementB.kMatmulTypeBNvFp4:
            continue
        is_high_precision = bool(
            sol.features & MatmulFeatures.kMatmulFeatures_HighPrecision
        )
        if hints.require_high_precision ^ is_high_precision:
            continue
        if (
            sol.mfma_type == MatmulMfmaType.kMatmulMfmaTypeFp16
            and hints.a_type == DataType.Fp16
        ) or (
            sol.mfma_type == MatmulMfmaType.kMatmulMfmaTypeBf16
            and hints.a_type == DataType.Bf16
        ):
            if (
                hints.b_type == DataType.MxFp4e2m1
                and sol.mfma_type != MatmulMfmaType.kMatmulMfmaTypeBf16
            ):
                continue
            group_n, group_k = sol.tile_n * 16, sol.tile_k * 64
            if n % group_n == 0 and k % group_k == 0:
                if sols is not None and sol_count < n_sols[0]:
                    sols[sol_count] = replace(
                        sol,
                        element_b=(
                            MatmulElementB.kMatmulTypeBMxFp4
                            if hints.b_type == DataType.MxFp4e2m1
                            else MatmulElementB.kMatmulTypeBNvFp4
                        ),
                    )
                sol_count += 1
    n_sols[0] = sol_count
    return 0


def ChooseDefaultFp4Fp16Solution(m, n, k, hints):
    n_sols = [0]
    if GemmGetSolutions(hints, m, n, k, None, n_sols) != 0:
        return -1
    candidates = [None] * n_sols[0]
    GemmGetSolutions(hints, m, n, k, candidates, n_sols)

    def better(a, b):
        if m <= 64:
            if a.tile_m != b.tile_m:
                return abs(m % (a.tile_m * 16) - a.tile_m * 8) < abs(
                    m % (b.tile_m * 16) - b.tile_m * 8
                )
            if a.warp_partition_k != b.warp_partition_k:
                return a.warp_partition_k > b.warp_partition_k
            if a.tile_n != b.tile_n:
                return a.tile_n < b.tile_n
        return (a.tile_m + a.tile_n, a.tile_m, a.tile_k, a.Repr()) > (
            b.tile_m + b.tile_n,
            b.tile_m,
            b.tile_k,
            b.Repr(),
        )

    best = None
    for s in candidates:
        if best is None or better(s, best):
            best = s
    return -1 if best is None else best.Repr()
