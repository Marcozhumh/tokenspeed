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

"""Capture compiled kernels from the production Invoke path for tests/profiling."""

from dataclasses import dataclass
from unittest.mock import patch

import torch
from lib.gemm.rocm.quantization.fp4 import gemm_fp4_fp16_rocm_test as native_tests
from lib.gemm.rocm.quantization.fp4.gemm_fp4_fp16_grid import (
    ConfigSelector,
    GemmFp4Fp16KernelGrid,
)
from lib.gemm.rocm.quantization.fp4.quantization_utils import (
    RepackMxFp4ToPetitFp4Scales,
    RepackNvFp4ToPetitFp4Weights,
)
from lib.gemm.rocm.quantization.gemm import PetitSolutionHints, SolutionId
from lib.gemm.rocm.quantization.types import DataType
from lib.tests.quantization import GemmMPTestData


def launch(c, a, b, scales, global_scale, m, n, k, sol, stream):
    """Launch through ConfigSelector.Invoke and return its compiled kernel artifact."""
    captured = []
    run = GemmFp4Fp16KernelGrid.run

    def record(*args, **kwargs):
        kernel = run(*args, **kwargs)
        captured.append(kernel)
        return kernel

    with patch.object(GemmFp4Fp16KernelGrid, "run", side_effect=record):
        status = ConfigSelector(sol).Invoke(
            c, a, b, scales, global_scale, m, n, k, stream
        )
    if status:
        raise ValueError(f"Native-style configuration rejected launch: {status}")
    assert len(captured) == 1
    return captured[0]


@dataclass(frozen=True)
class Case:
    name: str
    m: int
    n: int
    k: int
    solution: SolutionId
    mx: bool


def Cases():
    """Expand the native test declarations, including each pipeline K iteration."""
    result = []
    for name, test in vars(native_tests).items():
        if not name.startswith("test_"):
            continue
        calls = []
        fixture = native_tests.GemmFp4Fp16Test()
        with patch.object(
            fixture, "TestGemm", side_effect=lambda *args: calls.append(args)
        ):
            test(fixture)
        for m, n, k, global_scale, sol, b_type, group_size in calls:
            mx = b_type == DataType.MxFp4e2m1
            assert global_scale == 1.0 and group_size == (32 if mx else 16)
            case_name = name.removeprefix("test_")
            if "_Pipeline_" in case_name:
                case_name = "Pipeline_" + case_name.split("_Pipeline_")[1] + f"_K{k}"
            result.append(Case(case_name, m, n, k, sol, mx))
    return tuple(result)


CASES = Cases()


def prepare(case, global_scale):
    dtype = torch.bfloat16 if case.solution.mfma_type == 1 else torch.float16
    a, b, s = (
        x.cuda()
        for x in GemmMPTestData(case.m, case.n, case.k, dtype, case.mx).PrepareData(
            False
        )
    )
    if case.mx:
        rb = torch.empty_like(b)
        rs = torch.empty_like(s)
        stream = torch.cuda.current_stream()
        RepackNvFp4ToPetitFp4Weights(rb, b, case.k, case.n, stream)
        RepackMxFp4ToPetitFp4Scales(rs, s, case.k, case.n, stream)
        b, s = rb, rs
    out = torch.empty((case.m, case.n), dtype=dtype, device="cuda")
    global_tensor = torch.tensor([global_scale], device="cuda", dtype=torch.float32)
    hints = PetitSolutionHints(
        DataType.Bf16 if dtype == torch.bfloat16 else DataType.Fp16,
        DataType.MxFp4e2m1 if case.mx else DataType.Fp4e2m1,
        DataType.Bf16 if dtype == torch.bfloat16 else DataType.Fp16,
        bool(case.solution.features & 2),
    )
    return out, a, b, s, global_tensor, hints


def assert_near(actual, expected):
    actual, expected = actual.float(), expected.float()
    tolerance = torch.maximum(torch.full_like(expected, 0.01), expected.abs() * 0.01)
    good = (actual - expected).abs() < tolerance
    assert (
        good.all()
    ), f"{(~good).sum().item()} mismatches; max absolute error {(actual-expected).abs().max().item()}"
