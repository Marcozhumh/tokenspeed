# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Run the reference-scope MegaMoE benchmark with a registered EP8 profile.

For trace replay, skew routing, and asymmetric workloads, use bench_megamoe.py.
"""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

_PROFILE_ARGS = {
    "gpt_oss_120b": (
        "--model-name",
        "GPT-OSS-120B",
        "--global-experts",
        "128",
        "--topk",
        "4",
        "--hidden-size",
        "2880",
        "--padded-hidden-size",
        "3072",
        "--intermediate-size",
        "3072",
        "--activation-function",
        "swiglu",
        "--bias",
    ),
    "dsv4": (
        "--model-name",
        "DSV4",
        "--global-experts",
        "384",
        "--topk",
        "6",
        "--hidden-size",
        "7168",
        "--padded-hidden-size",
        "7168",
        "--intermediate-size",
        "3072",
        "--activation-function",
        "silu",
        "--no-bias",
    ),
}


def reference_argv(argv: list[str] | None = None) -> list[str]:
    """Translate profile CLI arguments into the shared benchmark's arguments."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--profile", choices=tuple(_PROFILE_ARGS), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--mode", choices=("graph",), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--graph-iters", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--stage-breakdown", action="store_true")
    args = parser.parse_args(argv)
    translated = [
        "--dp-size",
        "8",
        "--tp-size",
        "1",
        "--ep-size",
        "8",
        *_PROFILE_ARGS[args.profile],
        "--m",
        *(str(m) for m in args.tokens),
        "--batch-size",
        *(str(m) for m in args.tokens),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--graph-iters",
        str(args.graph_iters),
        "--seed",
        str(args.seed),
    ]
    if args.jsonl is not None:
        translated.extend(("--jsonl", str(args.jsonl)))
    if args.csv is not None:
        translated.extend(("--csv", str(args.csv)))
    if args.stage_breakdown:
        translated.append("--stage-breakdown")
    return translated


def main() -> int:
    argv = reference_argv()
    benchmark = runpy.run_path(str(Path(__file__).with_name("bench_megamoe.py")))
    return benchmark["main"](argv)


if __name__ == "__main__":
    raise SystemExit(main())
