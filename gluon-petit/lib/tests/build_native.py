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

"""Build the test-only oracle against the Clang23 native build."""

import json
import re
import shlex
import subprocess
from pathlib import Path


def build_native(source_root, build_root):
    """Return the shared oracle path built from source_root and build_root."""
    commands = json.loads((build_root / "compile_commands.json").read_text())
    entry = next(
        c for c in commands if c["file"].endswith("gemm_fp4_fp16_rocm_test.cc")
    )
    expected = source_root / "lib/gemm/rocm/quantization/fp4/gemm_fp4_fp16_rocm_test.cc"
    if not expected.samefile(entry["file"]):
        raise ValueError("The native build belongs to a different source checkout")
    args = shlex.split(entry["command"])
    version = subprocess.check_output([args[0], "--version"], text=True)
    if not re.search(r"clang version 23\.", version):
        raise ValueError("The native baseline must use Clang23")
    if not re.search(
        r"CMAKE_BUILD_TYPE:[^=]+=Release", (build_root / "CMakeCache.txt").read_text()
    ):
        raise ValueError("The native baseline must use a Release build")
    args = args[: args.index("-o")]
    args = [arg for arg in args if arg not in ("-fPIE",)]
    device_entry = next(
        c for c in commands if c["file"].endswith("/fp4/quantization_utils.cu")
    )
    device_flags = [
        arg
        for arg in shlex.split(device_entry["command"])
        if arg.startswith("--offload-arch=")
    ]
    source = Path(__file__).with_name("native_bridge.cc")
    output = build_root / "gluon_native_bridge.so"
    temporary = output.with_suffix(".tmp.so")
    link = subprocess.check_output(
        ["ninja", "-C", str(build_root), "-t", "commands", "gemm_fp4_fp16_rocm_test"],
        text=True,
    ).splitlines()[-1]
    tokens = shlex.split(link)
    # Reuse exactly the native target's libraries and their order.
    libs = tokens[tokens.index("-o") + 2 :]
    libs = [
        x
        for x in libs[: libs.index("&&")]
        if "libgtest" not in x and "libgmock" not in x
    ]
    subprocess.run(
        args
        + device_flags
        + [
            "-shared",
            "-fPIC",
            "-x",
            "hip",
            str(source),
            "-x",
            "none",
            "-o",
            str(temporary),
        ]
        + libs,
        cwd=build_root,
        check=True,
    )
    temporary.replace(output)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_native(args.source_root.resolve(), args.build_root.resolve()))
