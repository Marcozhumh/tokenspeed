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

"""Run matched native/Gluon launches under rocprofv3 --kernel-trace."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import triton
from lib.tests.gemm import (
    CASES,
    assert_near,
    launch,
    prepare,
)
from lib.tests.native import default_native


def run(output, warmup, repeats, rounds, case_filter):
    native = default_native()
    stream = torch.cuda.current_stream()
    cases = [c for c in CASES if case_filter in c.name]
    data = []
    for case in cases:
        out, a, b, s, g, hints = prepare(case, 1.0)
        sol = replace(case.solution, element_b=2) if case.mx else case.solution
        compiled = launch(out, a, b, s, g, case.m, case.n, case.k, sol, stream)
        expected = torch.empty_like(out)
        if native.gemm(expected, a, b, s, g, sol, case.mx, stream) != 0:
            raise RuntimeError(f"Native preflight failed: {case.name}")
        assert_near(out, expected)
        grid = (triton.cdiv(case.m, sol.tile_m * 16), case.n // (sol.tile_n * 16), 1)
        runner = compiled[grid]
        packed = (
            out.data_ptr(),
            a.data_ptr(),
            b.data_ptr(),
            s.data_ptr(),
            g.data_ptr(),
        ) + (None,) * (len(compiled.src.signature) - 5)
        native_args = native.gemm_args(out, a, b, s, g, sol, case.mx, stream)
        data.append((out, a, b, s, g, sol, runner, packed, native_args))
    torch.cuda.synchronize()
    phases = []
    for round_id in range(rounds):
        order = ("native", "gluon") if round_id % 2 == 0 else ("gluon", "native")
        for case, buffers in zip(cases, data, strict=True):
            out, a, b, s, g, sol, runner, packed, native_args = buffers
            for backend in order:
                torch.cuda.synchronize()
                for _ in range(warmup + repeats):
                    if backend == "native":
                        status = native.lib.native_gemm(*native_args)
                        if status:
                            raise RuntimeError(f"Native status {status}: {case.name}")
                    else:
                        runner(*packed, stream=stream.cuda_stream)
                torch.cuda.synchronize()
                phases.append(
                    dict(
                        case=case.name,
                        backend=backend,
                        round=round_id,
                        count=warmup + repeats,
                        m=case.m,
                        n=case.n,
                        k=case.k,
                        solution_id=hex(sol.Repr()),
                    )
                )
        print(f"Completed profiling round {round_id+1}/{rounds}", flush=True)
    output.write_text(
        json.dumps(
            dict(
                initial_launches=2 * len(cases),
                warmup=warmup,
                repeats=repeats,
                rounds=rounds,
                gpu=torch.cuda.get_device_properties(0).gcnArchName,
                triton=__import__("triton").__version__,
                torch=torch.__version__,
                phases=phases,
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
    p.add_argument("--case-filter", required=True)
    args = p.parse_args()
    run(args.manifest, args.warmup, args.repeats, args.rounds, args.case_filter)
