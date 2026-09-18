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

"""Profile the ported utility kernels separately from GEMM."""

import argparse
import json
from pathlib import Path

import torch
from lib.gemm.rocm.amd_intrinsics import _amdgcn_dequant_library
from lib.gemm.rocm.quantization.fp4.quantization_utils import (
    DequantizeFp4Kernel,
    DequantTraitMxFp4,
    DequantTraitNvFp4,
    DequantTraitPetitMxFp4,
    DequantTraitPetitNvFp4,
    PetitFormat,
    RepackFp4ScalesKernel,
    RepackMxScaleLayout64x32,
    RepackNvFp4ToPetitFp4WeightsKernel,
    RepackQWeightLayout64x32,
    RepackScaleLayout64x32,
)
from lib.gemm.rocm.quantization.types import DataType
from lib.tests.native import default_native


def run(manifest, warmup, repeats, rounds):
    native = default_native()
    stream = torch.cuda.current_stream()
    k, n = 256, 64
    cases = []
    initial_launches = 0
    extern = {"petit_dequant": _amdgcn_dequant_library()}
    gen = torch.Generator().manual_seed(71)
    sources = [
        torch.randint(
            -(2**31), 2**31, (n * k // 8,), dtype=torch.int32, generator=gen
        ).cuda(),
        torch.randint(0, 127, (n * k // 16,), dtype=torch.uint8, generator=gen).cuda(),
        torch.randint(96, 150, (n * k // 32,), dtype=torch.uint8, generator=gen).cuda(),
    ]
    for kind, layout in enumerate(
        (RepackQWeightLayout64x32, RepackScaleLayout64x32, RepackMxScaleLayout64x32)
    ):
        source = sources[kind]
        out, expected = torch.empty_like(source), torch.empty_like(source)
        grid = (k // layout.kGroupM, n // layout.kGroupN, 1)
        if kind == 0:
            kernel = RepackNvFp4ToPetitFp4WeightsKernel[grid](
                PetitFormat,
                out,
                source,
                k,
                n,
                layout,
                num_warps=layout.kNumWarps,
            )
            args = (None, out.data_ptr(), source.data_ptr(), None, None, None)
        else:
            kernel = RepackFp4ScalesKernel[grid](
                out,
                source,
                k,
                n,
                layout,
                num_warps=layout.kNumWarps,
                extern_libs=extern,
            )
            args = (out.data_ptr(), source.data_ptr(), None, None, None)
        native.repack(expected, source, k, n, kind, stream)
        initial_launches += 2
        assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))
        cases.append(
            (
                f"repack_{kind}",
                kernel[grid],
                args,
                native.repack,
                (out, source, k, n, kind, stream),
            )
        )
    for petit in (True, False):
        for dtype, mx in (
            (torch.float16, False),
            (torch.bfloat16, False),
            (torch.bfloat16, True),
        ):
            b, s = sources[0], sources[2 if mx else 1]
            out_type = DataType.Bf16 if dtype == torch.bfloat16 else DataType.Fp16
            if petit:
                rb, rs = torch.empty_like(b), torch.empty_like(s)
                native.repack(rb, b, k, n, 0, stream)
                native.repack(rs, s, k, n, 2 if mx else 1, stream)
                initial_launches += 2
                b, s = rb, rs
                trait = (
                    DequantTraitPetitMxFp4(RepackQWeightLayout64x32)
                    if mx
                    else DequantTraitPetitNvFp4(RepackQWeightLayout64x32, out_type)
                )
                native_fn = native.dequant
            else:
                trait = DequantTraitMxFp4() if mx else DequantTraitNvFp4(out_type)
                native_fn = native.dequant_format
            out = torch.empty((n, k), dtype=dtype, device="cuda")
            expected = torch.empty_like(out)
            grid = (k // trait.kGroupK, n // trait.kGroupN, 1)
            kernel = DequantizeFp4Kernel[grid](
                out,
                b,
                s,
                1.0,
                k,
                n,
                trait,
                num_warps=trait.kNumWarps,
                extern_libs=extern,
            )
            assert native_fn(expected, b, s, 1.0, mx) == 0
            initial_launches += 2
            assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
            cases.append(
                (
                    f"dequant_{'petit' if petit else 'raw'}_{'mx' if mx else 'nv'}_{str(dtype)}",
                    kernel[grid],
                    (out.data_ptr(), b.data_ptr(), s.data_ptr(), 1.0, None, None, None),
                    native_fn,
                    (out, b, s, 1.0, mx),
                )
            )
    phases = []
    torch.cuda.synchronize()
    for round_id in range(rounds):
        for name, runner, args, native_fn, native_args in cases:
            for backend in (
                ("native", "gluon") if round_id % 2 == 0 else ("gluon", "native")
            ):
                for _ in range(warmup + repeats):
                    if backend == "native":
                        status = native_fn(*native_args)
                        if status is not None and status != 0:
                            raise RuntimeError(status)
                    else:
                        runner(*args, stream=stream.cuda_stream)
                torch.cuda.synchronize()
                phases.append(
                    dict(
                        case=name,
                        backend=backend,
                        round=round_id,
                        count=warmup + repeats,
                    )
                )
    manifest.write_text(
        json.dumps(
            dict(
                initial_launches=initial_launches,
                warmup=warmup,
                repeats=repeats,
                rounds=rounds,
                gpu=torch.cuda.get_device_properties(0).gcnArchName,
                phases=phases,
                kernel_filters=[
                    "RepackNvFp4ToPetitFp4WeightsKernel",
                    "RepackFp4ScalesKernel",
                    "DequantizeFp4Kernel",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--rounds", type=int, required=True)
    a = p.parse_args()
    run(a.manifest, a.warmup, a.repeats, a.rounds)
