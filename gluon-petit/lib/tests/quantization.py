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

"""Native seed-42 distributions, including libstdc++ integer downscaling."""

import numpy as np
import torch


class Mt19937:
    def __init__(self, seed):
        self.gen = np.random.RandomState(seed)

    def uint32(self, size):
        return self.gen.randint(0, 2**32, size=size, dtype=np.uint32)

    def uniform_int(self, low, high, size):
        count = high - low + 1
        threshold = (2**32 - count) % count
        chunks = []
        remaining = size
        while remaining:
            product = self.uint32(remaining).astype(np.uint64) * count
            accepted = product[(product & 0xFFFFFFFF) >= threshold]
            chunks.append((accepted >> 32).astype(np.uint32) + low)
            remaining -= len(accepted)
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint32)

    def uniform_float(self, low, high, size):
        unit = self.uint32(size).astype(np.float32) * np.float32(2**-32)
        unit = np.minimum(unit, np.nextafter(np.float32(1), np.float32(0)))
        return (unit * np.float32(high - low) + np.float32(low)).astype(np.float32)


def MaskNegativeZeroOnPetitFp4Format(v):
    v = v.copy()
    for i in range(4):
        shift = i * 8
        for mask, nan in ((0x8E, 0x80), (0x71, 0x01)):
            full = np.uint32(mask << shift)
            v = np.where((v & full) == np.uint32(nan << shift), v & ~full, v)
    return v


def UpscaleFp8e4m3ToE5m3(v):
    exponent = (v >> 3).astype(np.int32)
    mantissa = (v & 7).astype(np.float32)
    value = np.where(
        exponent == 0, mantissa * 2**-9, (1 + mantissa / 8) * np.exp2(exponent - 7)
    )
    return ((value * 128).astype(np.float16).view(np.uint16) >> 7).astype(np.uint8)


class GemmMPTestData:
    def __init__(self, m, n, k, dtype, mx):
        self.m, self.n, self.k, self.dtype, self.mx = m, n, k, dtype, mx

    def GenerateInputs(self, gen):
        low, high = (-2, 2) if self.dtype == torch.bfloat16 else (-1, 1)
        v = gen.uniform_float(low, high, self.m * self.k)
        if self.dtype == torch.bfloat16:
            return (
                torch.from_numpy((v.view(np.uint32) >> 16).astype(np.uint16))
                .view(torch.bfloat16)
                .reshape(self.m, self.k)
            )
        return torch.from_numpy(v.astype(np.float16)).reshape(self.m, self.k)

    def GenerateScales(self, gen):
        size = self.k * self.n // (32 if self.mx else 16)
        v = gen.uniform_int(1, 237 if self.mx else 126, size)
        return torch.from_numpy(
            v.astype(np.uint8) if self.mx else UpscaleFp8e4m3ToE5m3(v)
        )

    def GenerateQWeights(self, gen):
        return torch.from_numpy(
            MaskNegativeZeroOnPetitFp4Format(gen.uint32(self.k * self.n // 8)).view(
                np.int32
            )
        )

    def PrepareData(self, use_zeros):
        if use_zeros:
            raise ValueError("The FP4 test does not use zero points")
        gen = Mt19937(42)
        a = self.GenerateInputs(gen)
        scales = self.GenerateScales(gen)
        weights = self.GenerateQWeights(gen)
        return a, weights, scales
