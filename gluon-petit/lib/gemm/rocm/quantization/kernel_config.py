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

from dataclasses import dataclass

import triton.experimental.gluon as g
from triton.experimental.gluon import language as l


@dataclass(frozen=True)
class TileShape:
    """Native shared tile type: element types, scale group, and M/N/K tile counts."""

    __triton_builtin__ = True
    ElementA: object
    kElementBTypeId: object
    kGroupSize: int
    kNumTileM: int
    kNumTileN: int
    kNumTileK: int
    kTile = 16

    @property
    def kGroupM(self):
        return self.kNumTileM * self.kTile

    @property
    def kGroupN(self):
        return self.kNumTileN * self.kTile

    @property
    def kGroupK(self):
        return self.kNumTileK * self.kTile


@dataclass(frozen=True)
class WarpPartition:
    __triton_builtin__ = True
    kPartitionM: int
    kPartitionN: int
    kPartitionK: int

    @property
    def kNumWarps(self):
        return self.kPartitionM * self.kPartitionN * self.kPartitionK

    @property
    def kThreads(self):
        return self.kNumWarps * 64

    @staticmethod
    @g.jit
    def WarpK(wid, WP: l.constexpr):
        return wid // WP.kPartitionN // WP.kPartitionM

    @staticmethod
    @g.jit
    def WarpN(wid, WP: l.constexpr):
        return wid % WP.kPartitionN

    @staticmethod
    @g.jit
    def WarpM(wid, WP: l.constexpr):
        return wid // WP.kPartitionN % WP.kPartitionM
