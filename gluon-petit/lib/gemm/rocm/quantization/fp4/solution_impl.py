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

from lib.gemm.rocm.quantization.gemm import SolutionId, kErrorKernelShape


class SolutionAdapter:
    """Bind a native solution ID to its configured Gluon invocation."""

    def __init__(self, representation):
        self.solution = SolutionId.FromRepr(representation)

    def Invoke(self, c, a, b, scales, global_scale, m, n, k, stream):
        """Write C for the bound solution and explicit buffers/shape/stream; return status."""
        from lib.gemm.rocm.quantization.fp4.gemm_fp4_fp16_grid import ConfigSelector

        if self.solution.element_b == 2 and self.solution.mfma_type != 1:
            return kErrorKernelShape
        return ConfigSelector(self.solution).Invoke(
            c, a, b, scales, global_scale, m, n, k, stream
        )
