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

"""Validate trace completeness and report per-case GPU latency ratios."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median


def compare(trace, manifest, report, limit):
    meta = json.loads(manifest.read_text())
    with trace.open() as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if any(
                name in r["Kernel_Name"]
                for name in meta.get("kernel_filters", ["GemmFp4Fp16KernelGrid"])
            )
        ]
    rows.sort(key=lambda r: int(r["Dispatch_Id"]))
    rows = rows[meta["initial_launches"] :]
    expected = sum(p["count"] for p in meta["phases"])
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} measured/warmup GEMMs, found {len(rows)}"
        )
    values = {}
    offset = 0
    for phase in meta["phases"]:
        block = rows[offset : offset + phase["count"]]
        offset += phase["count"]
        for r in block:
            is_native = "causalflow" in r["Kernel_Name"]
            if is_native != (phase["backend"] == "native"):
                raise ValueError(f"Trace/manifest backend mismatch: {phase}")
        measured = block[meta["warmup"] :]
        durations = [
            (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000
            for r in measured
        ]
        values.setdefault(phase["case"], {}).setdefault(phase["backend"], []).append(
            median(durations)
        )
    shapes = {
        p["case"]: f"{p['m']}×{p['n']}×{p['k']}"
        for p in meta["phases"]
        if all(key in p for key in ("m", "n", "k"))
    }
    shape_header = " M×N×K |" if shapes else ""
    shape_separator = "---|" if shapes else ""
    lines = [
        "# FP4 kernel latency",
        "",
        f"GPU: `{meta['gpu']}`. Times are microseconds; median of {meta['rounds']} round medians.",
        f"Each phase has {meta['warmup']} warmups and {meta['repeats']} measured launches. Acceptance ratio: ≤ {limit:.2f}.",
        "",
        f"| Case |{shape_header} Native µs | Gluon µs | Ratio | Pass |",
        f"|---|{shape_separator}---:|---:|---:|:---:|",
    ]
    failures = []
    for name, v in values.items():
        native = median(v["native"])
        gluon = median(v["gluon"])
        ratio = gluon / native
        passed = ratio <= limit
        if not passed:
            failures.append(name)
        shape_column = f" {shapes[name]} |" if shapes else ""
        lines.append(
            f"| {name} |{shape_column} {native:.3f} | {gluon:.3f} | {ratio:.3f} | {'yes' if passed else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"{len(values)-len(failures)}/{len(values)} configurations meet the latency limit.",
            "",
        ]
    )
    report.write_text("\n".join(lines))
    return failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--limit", type=float, required=True)
    a = p.parse_args()
    failures = compare(a.trace, a.manifest, a.report, a.limit)
    print(f"Latency failures: {len(failures)}")
    raise SystemExit(bool(failures))
