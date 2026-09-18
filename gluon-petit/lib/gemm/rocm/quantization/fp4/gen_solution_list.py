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

from lib.gemm.rocm.quantization.gemm import SolutionId

TSWP = (
    (4, 2, 8, 4, 1, 1),
    (1, 2, 16, 1, 1, 4),
    (2, 2, 16, 2, 1, 2),
    (2, 2, 16, 1, 1, 4),
    (4, 2, 16, 2, 1, 2),
    (1, 2, 32, 1, 1, 4),
    (2, 2, 32, 2, 1, 2),
    (1, 4, 8, 1, 2, 2),
    (2, 4, 8, 2, 2, 1),
    (4, 4, 8, 2, 2, 1),
    (6, 4, 8, 2, 2, 1),
    (8, 4, 8, 2, 2, 1),
    (10, 4, 8, 2, 2, 1),
    (1, 4, 16, 1, 2, 2),
    (2, 4, 16, 2, 2, 1),
    (4, 4, 16, 2, 2, 1),
    (1, 4, 32, 1, 2, 2),
    (2, 4, 32, 2, 2, 1),
    (4, 6, 8, 2, 1, 2),
    (6, 6, 8, 2, 1, 2),
    (1, 8, 4, 1, 4, 1),
    (8, 8, 4, 2, 2, 1),
    (12, 8, 4, 2, 2, 1),
    (14, 8, 4, 2, 2, 1),
    (16, 8, 4, 2, 2, 1),
    (2, 8, 8, 2, 2, 1),
    (4, 8, 8, 2, 2, 1),
    (5, 8, 8, 1, 2, 2),
    (10, 8, 4, 2, 2, 1),
    (8, 12, 4, 2, 2, 1),
    (10, 12, 4, 2, 2, 1),
    (12, 12, 4, 2, 2, 1),
    (14, 12, 4, 2, 2, 1),
    (16, 12, 4, 2, 2, 1),
    (8, 16, 4, 2, 2, 1),
    (10, 16, 4, 2, 2, 1),
    (12, 16, 4, 2, 2, 1),
    (14, 16, 4, 2, 2, 1),
    (16, 16, 4, 2, 2, 1),
)


def FromTSWPList():
    return tuple(
        SolutionId.MultiStage(features, element, mfma, *shape[:3], 0, *shape[3:])
        for features in (1, 3)
        for element, mfma in ((1, 0), (1, 1), (2, 1))
        for shape in TSWP
    )
