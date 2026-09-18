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

"""ctypes adapter for the test-only native oracle."""

import ctypes as c
from pathlib import Path

import torch


class Native:
    def __init__(self, path):
        self.lib = c.CDLL(str(path))
        p = c.c_void_p
        u = c.c_uint
        i = c.c_int
        self.lib.native_generate.argtypes = [u, u, u, i, i, p, p, p]
        self.lib.native_uniform_int.argtypes = [u, u, u, p]
        self.lib.native_uniform_int.restype = None
        self.lib.native_gemm.argtypes = [p, p, p, p, p, u, u, u, i, i, i, c.c_ulong, p]
        self.lib.native_dequant.argtypes = [p, p, p, c.c_float, i, u, u, i]
        self.lib.native_dequant_format.argtypes = [p, p, p, c.c_float, i, u, u, i]
        self.lib.native_repack.argtypes = [p, p, u, u, i, p]
        self.lib.native_repack.restype = None
        self.lib.native_dequant_scale_probe.argtypes = [p, p, p, u, i, i, p]
        self.lib.native_dequant_probe.argtypes = [p, p, p, p, u, i, i, i, p]
        self.lib.native_buffer_load_probe.argtypes = [p, p, p, i, p]
        self.lib.native_reference.argtypes = [p, p, p, u, u, u, i]
        self.lib.native_reference_create.argtypes = [c.POINTER(p), c.POINTER(p)]
        self.lib.native_reference_destroy.argtypes = [p, p]
        self.lib.native_reference_with_context.argtypes = [
            p,
            p,
            p,
            c.c_size_t,
            p,
            p,
            p,
            u,
            u,
            u,
            i,
        ]
        self.lib.native_solution_id.argtypes = [u, c.c_ulong, p, p]
        self.lib.native_solution_id.restype = c.c_ulong
        self.lib.native_gemm_enums.argtypes = [p]
        self.lib.native_gemm_enums.restype = None
        self.lib.native_gemm_get_solutions.argtypes = [i, i, i, i, u, u, u, p, p]

    def generate(self, m, n, k, dtype, mx):
        a = torch.empty((m, k), dtype=dtype)
        b = torch.empty(n * k // 8, dtype=torch.int32)
        s = torch.empty(n * k // (32 if mx else 16), dtype=torch.uint8)
        assert (
            self.lib.native_generate(
                m,
                n,
                k,
                5 if dtype == torch.bfloat16 else 4,
                mx,
                a.data_ptr(),
                b.data_ptr(),
                s.data_ptr(),
            )
            == 0
        )
        return a, b, s

    def gemm(self, out, a, b, s, g, sol, mx, stream):
        return self.lib.native_gemm(*self.gemm_args(out, a, b, s, g, sol, mx, stream))

    def gemm_args(self, out, a, b, s, g, sol, mx, stream):
        """Prepare native launch arguments once, outside a profiled launch loop."""
        m, k = a.shape
        n = out.shape[1]
        return (
            out.data_ptr(),
            a.data_ptr(),
            b.data_ptr(),
            s.data_ptr(),
            g.data_ptr(),
            m,
            n,
            k,
            5 if a.dtype == torch.bfloat16 else 4,
            mx,
            bool(sol.features & 2),
            sol.Repr(),
            stream.cuda_stream,
        )

    def dequant(self, out, b, s, global_scale, mx):
        n, k = out.shape
        return self.lib.native_dequant(
            out.data_ptr(),
            b.data_ptr(),
            s.data_ptr(),
            global_scale,
            5 if out.dtype == torch.bfloat16 else 4,
            k,
            n,
            mx,
        )

    def dequant_format(self, out, b, s, global_scale, mx):
        n, k = out.shape
        return self.lib.native_dequant_format(
            out.data_ptr(),
            b.data_ptr(),
            s.data_ptr(),
            global_scale,
            5 if out.dtype == torch.bfloat16 else 4,
            k,
            n,
            mx,
        )

    def repack(self, out, input, k, n, kind, stream):
        self.lib.native_repack(
            out.data_ptr(), input.data_ptr(), k, n, kind, stream.cuda_stream
        )

    def dequant_probe(self, q, scales, out, decoded, kind, hp, fallback, stream):
        return self.lib.native_dequant_probe(
            q.data_ptr(),
            scales.data_ptr(),
            out.data_ptr(),
            decoded.data_ptr(),
            q.numel(),
            kind,
            hp,
            fallback,
            stream.cuda_stream,
        )

    def dequant_scale_probe(self, q, scales, out, kind, upscale, stream):
        return self.lib.native_dequant_scale_probe(
            q.data_ptr(),
            scales.data_ptr(),
            out.data_ptr(),
            q.numel(),
            kind,
            upscale,
            stream.cuda_stream,
        )

    def buffer_load_probe(self, data, out, offsets, aux, stream):
        return self.lib.native_buffer_load_probe(
            data.data_ptr(), out.data_ptr(), offsets.data_ptr(), aux, stream.cuda_stream
        )

    def reference(self, out, a, weights):
        m, k = a.shape
        n = out.shape[1]
        return self.lib.native_reference(
            out.data_ptr(),
            a.data_ptr(),
            weights.data_ptr(),
            m,
            n,
            k,
            5 if a.dtype == torch.bfloat16 else 4,
        )


def default_native():
    return Native(
        Path(__file__).resolve().parents[3]
        / "petit-kernel/build/gluon_native_bridge.so"
    )
