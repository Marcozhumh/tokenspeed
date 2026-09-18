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

"""CPU numeric types used by the native FP4 utility tests."""

import math
import struct

import numpy as np
import torch


class fp8_e4m3_t:
    def __init__(self, storage):
        self.storage_ = storage & 0xFF

    @staticmethod
    def from_bits(bits):
        return fp8_e4m3_t(bits)

    def to_bits(self):
        return self.storage_

    def to_fp32(self):
        x = self.storage_
        if x & 0x7F == 0x7F:
            return math.nan
        exponent, mantissa = (x & 0x7F) >> 3, x & 7
        value = (
            mantissa * 2**-9
            if exponent == 0
            else (1 + mantissa / 8) * 2 ** (exponent - 7)
        )
        return -value if x & 0x80 else value


class fp16_t:
    torch_dtype = torch.float16

    def __init__(self, storage):
        self.storage_ = storage & 0xFFFF

    @staticmethod
    def from_bits(bits):
        return fp16_t(bits)

    @staticmethod
    def from_fp32(val):
        return fp16_t.from_bits(int(np.float16(np.float32(val)).view(np.uint16)))

    def to_bits(self):
        return self.storage_

    def is_nan(self):
        return self.storage_ & 0x7FFF > 0x7C00

    def is_inf(self):
        return self.storage_ & 0x7FFF == 0x7C00

    def is_zero(self):
        return self.storage_ & 0x7FFF == 0

    def __eq__(self, other):
        if self.is_nan() or other.is_nan():
            return False
        if self.is_zero() and other.is_zero():
            return True
        return self.to_bits() == other.to_bits()


class bf16_t:
    torch_dtype = torch.bfloat16

    def __init__(self, storage):
        self.storage_ = storage & 0xFFFF

    @staticmethod
    def from_bits(bits):
        return bf16_t(bits)

    @staticmethod
    def from_fp32(val):
        bits = struct.unpack("<I", struct.pack("<f", val))[0]
        # Preserve the native unsigned bitwise-complement condition.
        if (~(bits & 0x7F800000)) & 0xFFFFFFFF:
            bits += 0x7FFF + ((bits >> 16) & 1)
        elif bits & 0xFFFF:
            bits |= 0x10000
        return bf16_t.from_bits(bits >> 16)

    def to_bits(self):
        return self.storage_

    def is_nan(self):
        return self.storage_ & 0x7FFF > 0x7F80

    def is_inf(self):
        return self.storage_ & 0x7FFF == 0x7F80

    def is_zero(self):
        return self.storage_ & 0x7FFF == 0

    def __eq__(self, other):
        if self.is_nan() or other.is_nan():
            return False
        if self.is_zero() and other.is_zero():
            return True
        return self.to_bits() == other.to_bits()
