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

"""FP4 dependency subset of native amd_intrinsics.cuh, in native declaration order.

Packed float2 values use uint64 bits; MFMA uint2 arguments use pairs of tensors.
Compiler adaptation helpers are collected after the native-facing definitions.
"""

from functools import cache
from hashlib import sha256

import triton.experimental.gluon as g
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language._core import _unwrap_if_constexpr, builtin
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna3 import mfma
from triton.runtime.cache import get_cache_manager

# Native capability macros for this port's supported gfx950 target.
HAS_AMD_BF8_PACK_CONVERSION = True
HAS_AMD_SCHED_BARRIER = True
HAS_AMD_SCHED_GROUP_BARRIER = True
kWarpSize = l.constexpr(64)


@g.jit
def llvm_amdgcn_raw_buffer_load_v4i32(rsrc, voffset, soffset, AUX: l.constexpr):
    # Gluon represents native uint4 as the last distributed tensor axis.
    V: l.constexpr = l.BlockedLayout([1, 4], [64, 1], [4, 1], [1, 0])
    word = l.arange(0, 4, layout=l.SliceLayout(0, V)).to(l.uint32)
    zero = l.full([256, 4], 0, l.uint32, V)
    return _raw_buffer_load_word(
        rsrc[0].to(l.uint64) + zero.to(l.uint64),
        (rsrc[1] + zero).to(l.uint32),
        l.full([256, 4], rsrc[2], l.uint32, V),
        voffset[:, None].to(l.uint32) + zero,
        (soffset + zero).to(l.uint32),
        word[None, :] + zero,
        AUX,
    )


@builtin
def amdgcn_pk_mul_f32(a, b, _semantic):
    a, b = _semantic.broadcast_impl_value(a, b)
    handle = _semantic.builder.create_extern_elementwise(
        "petit_dequant",
        _amdgcn_dequant_library(),
        "petit_dequant_pk_mul_f32",
        [a.handle, b.handle],
        a.type.to_ir(_semantic.builder),
        True,
    )
    return l.tensor(handle, a.type)


@builtin
def amdgcn_perm_b32(hi, lo, selector, _semantic):
    hi, lo = _semantic.broadcast_impl_value(hi, lo)
    selector = _semantic.to_tensor(_unwrap_if_constexpr(selector))
    selector = _semantic.cast(selector, l.uint32)
    selector, hi = _semantic.broadcast_impl_value(selector, hi)
    handle = _semantic.builder.create_extern_elementwise(
        "petit_dequant",
        _amdgcn_dequant_library(),
        "petit_dequant_perm",
        [hi.handle, lo.handle, selector.handle],
        hi.type.to_ir(_semantic.builder),
        True,
    )
    return l.tensor(handle, hi.type)


@builtin
def amdgcn_cvt_pk_f32_bf8(src, WORD_HI, _semantic):
    word_hi = _unwrap_if_constexpr(WORD_HI)
    ty = (
        l.distributed_type(l.uint64, src.type.shape, src.type.layout)
        if src.type.is_block()
        else l.uint64
    )
    handle = _semantic.builder.create_extern_elementwise(
        "petit_dequant",
        _amdgcn_dequant_library(),
        "petit_dequant_cvt_bf8_hi" if word_hi else "petit_dequant_cvt_bf8_lo",
        [src.handle],
        ty.to_ir(_semantic.builder),
        True,
    )
    return l.tensor(handle, ty)


@builtin
def amdgcn_sched_barrier(mask, _semantic):
    mask = _unwrap_if_constexpr(mask)
    _semantic.builder.create_extern_elementwise(
        "petit",
        "",
        f"petit_sched_barrier_{mask}",
        [],
        l.int32.to_ir(_semantic.builder),
        False,
    )


@builtin
def amdgcn_sched_group_barrier(mask, size, sync_id, _semantic):
    mask, size, sync_id = map(_unwrap_if_constexpr, (mask, size, sync_id))
    _semantic.builder.create_extern_elementwise(
        "petit",
        "",
        f"petit_sched_group_barrier_{mask}_{size}_{sync_id}",
        [],
        l.int32.to_ir(_semantic.builder),
        False,
    )


@g.jit
def mma_m16n16k16_fp16(fa, fb, c):
    ML: l.constexpr = AMDMFMALayout(2, [16, 16, 16], False, [4, 1, 1])
    DT: l.constexpr = l.float16
    a0, a1 = fa
    b0, b1 = fb
    a = (
        l.join(l.join(a0, a1).to(l.uint16), (l.join(a0, a1) >> 16).to(l.uint16))
        .to(DT, bitcast=True)
        .reshape([4, 4, 16, 4])
        .permute([0, 2, 1, 3])
        .reshape([4, 16, 16])
    )
    b = (
        l.join(l.join(b0, b1).to(l.uint16), (l.join(b0, b1) >> 16).to(l.uint16))
        .to(DT, bitcast=True)
        .reshape([4, 4, 16, 4])
        .permute([0, 2, 1, 3])
        .reshape([4, 16, 16])
        .permute([0, 2, 1])
    )
    a = l.convert_layout(a, l.DotOperandLayout(0, ML, 4))
    b = l.convert_layout(b, l.DotOperandLayout(1, ML, 4))
    return mfma(a, b, c)


@g.jit
def mma_m16n16k16_bf16(fa, fb, c):
    ML: l.constexpr = AMDMFMALayout(2, [16, 16, 16], False, [4, 1, 1])
    DT: l.constexpr = l.bfloat16
    a0, a1 = fa
    b0, b1 = fb
    a = (
        l.join(l.join(a0, a1).to(l.uint16), (l.join(a0, a1) >> 16).to(l.uint16))
        .to(DT, bitcast=True)
        .reshape([4, 4, 16, 4])
        .permute([0, 2, 1, 3])
        .reshape([4, 16, 16])
    )
    b = (
        l.join(l.join(b0, b1).to(l.uint16), (l.join(b0, b1) >> 16).to(l.uint16))
        .to(DT, bitcast=True)
        .reshape([4, 4, 16, 4])
        .permute([0, 2, 1, 3])
        .reshape([4, 16, 16])
        .permute([0, 2, 1])
    )
    a = l.convert_layout(a, l.DotOperandLayout(0, ML, 4))
    b = l.convert_layout(b, l.DotOperandLayout(1, ML, 4))
    return mfma(a, b, c)


class BufferResource:
    """Native resource fields (pointer, byte range, config) represented by a tuple."""

    __triton_builtin__ = True
    kDataFormatU32Config = 4 << 15
    kNone = 0
    kGLCBit = 1 << 0
    kSLCBit = 1 << 1

    @g.jit
    def Load(resource, voffset, soffset, AUX: l.constexpr):
        return llvm_amdgcn_raw_buffer_load_v4i32(resource, voffset, soffset, AUX)


@g.jit
def GetConditionShmPtr(ptr, cond):
    # Native uses the CDNA out-of-range LDS address to discard inactive stores.
    return l.where(cond, ptr, l.full((), 65536, l.uint64).to(ptr.dtype))


@builtin
def amdgcn_s_waitcnt(vm_cnt, exp_cnt, lgkm_cnt, _semantic):
    vm_cnt, exp_cnt, lgkm_cnt = map(_unwrap_if_constexpr, (vm_cnt, exp_cnt, lgkm_cnt))
    assert vm_cnt < 64, "vm_cnt must be less than 64"
    assert exp_cnt < 8, "exp_cnt must be less than 8"
    assert lgkm_cnt < 16, "lgkm_cnt must be less than 16"
    vm = vm_cnt & 63
    mask = ((vm & 48) << 10) | ((lgkm_cnt & 15) << 8) | ((exp_cnt & 7) << 4) | (vm & 15)
    _semantic.builder.create_extern_elementwise(
        "petit",
        "",
        f"petit_s_waitcnt_{mask}",
        [],
        l.int32.to_ir(_semantic.builder),
        False,
    )


# Gluon compiler support. These are not additional native algorithms.
# Void/vector LLVM operations need always-inline adapters for Gluon's scalar
# extern interface. Host launchers explicitly link the generated libraries.


@g.jit
def bitreverse(v):
    return l.inline_asm_elementwise(
        "v_bfrev_b32 $0, $1;", "=v,v", [v], l.uint32, True, 1
    )


@builtin
def _raw_buffer_load_word(
    ptr, byte_range, config, voffset, soffset, word, aux, _semantic
):
    handle = _semantic.builder.create_extern_elementwise(
        "petit",
        "",
        f"petit_buffer_load_{_unwrap_if_constexpr(aux)}",
        [
            ptr.handle,
            byte_range.handle,
            config.handle,
            voffset.handle,
            soffset.handle,
            word.handle,
        ],
        word.type.to_ir(_semantic.builder),
        False,
    )
    return l.tensor(handle, word.type)


@cache
def _amdgcn_intrinsics_library(load_global_a):
    """Return an LLVM library exposing native scheduling, wait, and buffer loads.

    ``load_global_a`` is the native per-thread uint4 load count. Gluon's
    external calls require a result, so always-inline wrappers return an unused
    i32 for void scheduling/wait operations. The uint4-load wrapper returns
    one extracted word per Gluon element. After inlining, only native LLVM
    intrinsics remain; the four word extractions share one uint4 buffer load.
    """
    calls = [
        ("sched.barrier", (0,)),
        ("sched.barrier", (0x7DF,)),
        ("s.waitcnt", (0xF70,)),
    ]
    calls += [
        ("sched.group.barrier", args)
        for args in (
            (0x100, load_global_a + 2, 0),
            (0x200, 1, 0),
            (0x20, 1, 0),
            (0x8, 1, 0),
            (0x8, 2, 0),
            (0x8, 4, 0),
            (0x8, 8, 0),
        )
    ]
    declarations = set()
    definitions = []
    for intrinsic, args in calls:
        name = "petit_" + intrinsic.replace(".", "_") + "_" + "_".join(map(str, args))
        params = ", ".join(f"i32 {value}" for value in args)
        declarations.add(
            f"declare void @llvm.amdgcn.{intrinsic}("
            + ", ".join("i32 immarg" for _ in args)
            + ")"
        )
        definitions.append(
            f"define i32 @{name}() alwaysinline {{\n"
            f"  call void @llvm.amdgcn.{intrinsic}({params})\n"
            "  ret i32 0\n}"
        )
    # Gluon's extern interface is elementwise. Four extracted words share one
    # inlined native uint4 load after LLVM CSE; the resource retains native bounds.
    declarations.add(
        "declare <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32>, i32, i32, i32 immarg)"
    )
    for aux in range(4):
        definitions.append(
            f"""define i32 @petit_buffer_load_{aux}(i64 %ptr, i32 %range, i32 %config, i32 %voffset, i32 %soffset, i32 %word) alwaysinline {{
      %lo = trunc i64 %ptr to i32
      %shift = lshr i64 %ptr, 32
      %hi = trunc i64 %shift to i32
      %r0 = insertelement <4 x i32> poison, i32 %lo, i32 0
      %r1 = insertelement <4 x i32> %r0, i32 %hi, i32 1
      %r2 = insertelement <4 x i32> %r1, i32 %range, i32 2
      %r3 = insertelement <4 x i32> %r2, i32 %config, i32 3
      %v = call <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32(<4 x i32> %r3, i32 %voffset, i32 %soffset, i32 {aux})
      %result = extractelement <4 x i32> %v, i32 %word
      ret i32 %result
    }}"""
        )
    source = "\n".join(
        ['target triple = "amdgcn-amd-amdhsa"', *sorted(declarations), *definitions]
    )
    manager = get_cache_manager(sha256(source.encode()).hexdigest())
    path = manager.get_file("petit_intrinsics.ll")
    if path is None:
        path = manager.put(source, "petit_intrinsics.ll", binary=False)
    return path


@cache
def _amdgcn_dequant_library():
    """Expose native vector operations to LLVM's instruction and hazard passes."""
    source = """target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.perm(i32, i32, i32)
declare <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32, i1 immarg)
define i32 @petit_dequant_perm(i32 %hi, i32 %lo, i32 %selector) alwaysinline {
  %r = call i32 @llvm.amdgcn.perm(i32 %hi, i32 %lo, i32 %selector)
  ret i32 %r
}
define i64 @petit_dequant_cvt_bf8_lo(i32 %src) alwaysinline {
  %r = call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %src, i1 false)
  %bits = bitcast <2 x float> %r to i64
  ret i64 %bits
}
define i64 @petit_dequant_cvt_bf8_hi(i32 %src) alwaysinline {
  %r = call <2 x float> @llvm.amdgcn.cvt.pk.f32.bf8(i32 %src, i1 true)
  %bits = bitcast <2 x float> %r to i64
  ret i64 %bits
}
define i64 @petit_dequant_pk_mul_f32(i64 %a, i64 %b) alwaysinline {
  %av = bitcast i64 %a to <2 x float>
  %bv = bitcast i64 %b to <2 x float>
  %r = fmul <2 x float> %av, %bv
  %bits = bitcast <2 x float> %r to i64
  ret i64 %bits
}
define i32 @petit_dequant_hmul2(i32 %a, i32 %b) alwaysinline {
  %av = bitcast i32 %a to <2 x half>
  %bv = bitcast i32 %b to <2 x half>
  %r = fmul <2 x half> %av, %bv
  %bits = bitcast <2 x half> %r to i32
  ret i32 %bits
}
"""
    manager = get_cache_manager(sha256(source.encode()).hexdigest())
    path = manager.get_file("petit_dequant.ll")
    if path is None:
        path = manager.put(source, "petit_dequant.ll", binary=False)
    return path
