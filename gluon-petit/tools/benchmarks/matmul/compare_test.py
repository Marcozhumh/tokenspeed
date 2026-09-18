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

import csv
import json

import pytest
from tools.benchmarks.matmul.compare import compare


def trace_files(tmp_path, missing, wrong_backend):
    phases = []
    rows = []
    dispatch = 0
    for round_id in range(2):
        for case, gluon_ns in (("fast", 900), ("slow", 1100)):
            for backend in (
                ("native", "gluon") if round_id == 0 else ("gluon", "native")
            ):
                phases.append(dict(case=case, backend=backend, round=round_id, count=3))
                for sample in range(3):
                    dispatch += 1
                    # Deliberately reverse timestamp order across dispatches.
                    start = (100 - dispatch) * 10000
                    duration = (
                        100000
                        if sample == 0
                        else (1000 if backend == "native" else gluon_ns)
                    )
                    name = (
                        "causalflow::GemmFp4Fp16KernelGrid"
                        if backend == "native"
                        else "GemmFp4Fp16KernelGrid"
                    )
                    rows.append(
                        dict(
                            Dispatch_Id=dispatch,
                            Kernel_Name=name,
                            Start_Timestamp=start,
                            End_Timestamp=start + duration,
                        )
                    )
    if missing:
        rows.pop()
    if wrong_backend:
        rows[0]["Kernel_Name"] = "GemmFp4Fp16KernelGrid"
    trace = tmp_path / "trace.csv"
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.md"
    with trace.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest.write_text(
        json.dumps(
            dict(
                phases=phases,
                initial_launches=0,
                warmup=1,
                rounds=2,
                repeats=2,
                gpu="test",
            )
        )
    )
    return trace, manifest, report


def test_per_case_limit_and_dispatch_order(tmp_path):
    trace, manifest, report = trace_files(tmp_path, False, False)
    assert compare(trace, manifest, report, 1.05) == ["slow"]
    assert "| fast | 1.000 | 0.900 | 0.900 | yes |" in report.read_text()


@pytest.mark.parametrize("missing,wrong", [(True, False), (False, True)])
def test_reject_incomplete_or_mismatched_trace(tmp_path, missing, wrong):
    paths = trace_files(tmp_path, missing, wrong)
    with pytest.raises(ValueError):
        compare(*paths, 1.05)
