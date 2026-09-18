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

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    _amdgcn_dequant_library,
    amdgcn_perm_b32,
    amdgcn_pk_mul_f32,
)
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language._core import builtin


class Fp16Trait:
    __triton_builtin__ = True

    @g.jit
    def ToFloat(value):
        return value.to(l.float32)

    @g.jit
    def ToFloat2(scale):
        s = Fp16Trait.ToFloat(scale).to(l.uint32, bitcast=True).to(l.uint64)
        return s | (s << 32)


@g.jit
def hmul2(a, b, BF16: l.constexpr):
    if BF16:
        a0 = amdgcn_perm_b32(a, l.full((), 0, l.uint32), 0x05040C0C)
        a1 = amdgcn_perm_b32(a, l.full((), 0, l.uint32), 0x07060C0C)
        b0 = amdgcn_perm_b32(b, l.full((), 0, l.uint32), 0x05040C0C)
        b1 = amdgcn_perm_b32(b, l.full((), 0, l.uint32), 0x07060C0C)
        r = amdgcn_pk_mul_f32(
            a0.to(l.uint64) | (a1.to(l.uint64) << 32),
            b0.to(l.uint64) | (b1.to(l.uint64) << 32),
        )
        return amdgcn_perm_b32((r >> 32).to(l.uint32), r.to(l.uint32), 0x07060302)
    else:
        return llvm_hmul2(a, b)


@builtin
def llvm_hmul2(a, b, _semantic):
    a, b = _semantic.broadcast_impl_value(a, b)
    handle = _semantic.builder.create_extern_elementwise(
        "petit_dequant",
        _amdgcn_dequant_library(),
        "petit_dequant_hmul2",
        [a.handle, b.handle],
        a.type.to_ir(_semantic.builder),
        True,
    )
    return l.tensor(handle, a.type)
