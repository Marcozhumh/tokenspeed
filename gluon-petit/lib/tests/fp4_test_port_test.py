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

"""Check the native test inventory and Python matcher adaptations."""

import re
from pathlib import Path

import pytest
import torch
from lib.gemm.rocm.quantization.fp4 import gemm_fp4_fp16_rocm_test as gemm_tests
from lib.gemm.rocm.quantization.fp4 import quantization_utils_fp4_test as utility_tests
from lib.tests.floating_points import bf16_t, fp16_t
from lib.tests.gemm import CASES


@pytest.mark.parametrize(
    "module,stem",
    [
        (gemm_tests, "gemm_fp4_fp16_rocm_test"),
        (utility_tests, "quantization_utils_fp4_test"),
    ],
)
def test_native_test_inventory(module, stem):
    source = (
        Path(__file__).resolve().parents[3]
        / "petit-kernel/lib/gemm/rocm/quantization/fp4"
        / (stem + ".cc")
    ).read_text()
    expected = set(re.findall(r"TEST_F\(\w+,\s*(\w+)\)", source))
    for args in re.findall(r"^TEST_BF16\(([0-9, ]+)\)", source, re.MULTILINE):
        m, n, k, pm, pn, pk = (int(v) for v in args.split(","))
        expected.add(f"TestGemm_{m}x{n}x{k}_{pm}x{pn}x{pk}")
    actual = {
        name.removeprefix("test_") for name in vars(module) if name.startswith("test_")
    }
    assert actual == expected
    if module is gemm_tests:
        assert len(actual) == 47
        assert len(CASES) == len({case.name for case in CASES}) == 51
        for mode in ("SingleBuffer", "DoubleBuffer"):
            cases = [case for case in CASES if f"Pipeline_{mode}" in case.name]
            assert [case.k for case in cases] == [512, 768, 1024]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matcher_strict_tolerance_and_nonfinite(dtype):
    # 100 versus 101 lies exactly on the relative tolerance, so it must fail.
    actual = torch.tensor(
        [101, 100.5, 0.005, 0.02, float("nan"), float("inf")], dtype=dtype
    )
    reference = torch.tensor([100, 100, 0, 0, 1, float("inf")], dtype=dtype)
    if dtype == torch.bfloat16:
        good = gemm_tests.IsNearBf16(
            actual.view(torch.int16), reference.view(torch.int16)
        )
    else:
        good = gemm_tests.IsNearFp16(actual, reference)
    assert good.tolist() == [False, True, True, False, False, False]


@pytest.mark.parametrize("Element,nan", [(fp16_t, 0x7E00), (bf16_t, 0x7FC0)])
def test_native_numeric_equality(Element, nan):
    assert Element.from_bits(0) == Element.from_bits(0x8000)
    assert not Element.from_bits(nan) == Element.from_bits(nan)
    assert Element.from_fp32(1.0) == Element.from_bits(
        0x3C00 if Element is fp16_t else 0x3F80
    )
