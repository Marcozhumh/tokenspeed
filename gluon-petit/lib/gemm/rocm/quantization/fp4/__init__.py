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


# Public declarations in quantization/gemm.h. Lazy resolution avoids the
# header/implementation import cycle without adding another execution path.
__all__ = [
    "GemmFp4Fp16Grid",
    "GemmMxFp4Fp16Grid",
    "GemmGetSolutions",
    "RepackNvFp4ToPetitFp4Weights",
    "RepackNvFp4ToPetitFp4Scales",
    "RepackMxFp4ToPetitFp4Scales",
]


def __getattr__(name):
    from importlib import import_module

    if name in ("GemmFp4Fp16Grid", "GemmMxFp4Fp16Grid"):
        module = "gemm_fp4_fp16_grid"
    elif name == "GemmGetSolutions":
        module = "algo_chooser"
    elif name in __all__:
        module = "quantization_utils"
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"lib.gemm.rocm.quantization.fp4.{module}"), name)
