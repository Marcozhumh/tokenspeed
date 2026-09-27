"""Native per-wave matrix multiply policies and register layouts."""

import triton.experimental.gluon as g
from lib.gemm.rocm.amd_intrinsics import (
    mma_scale_m16n16k128_fp4_fp4_f32,
)
from lib.tal.device import DeviceTemplate, device_method
from triton.experimental.gluon import language as l


@g.jit
def ScaledMxFp4Mfma(
    opsel_a: l.constexpr, opsel_b: l.constexpr, a, scale_a, b, scale_b, acc
):
    # The native dispatch lambdas specialize both immediate selectors.
    if opsel_a == 0:
        kOpSelA: l.constexpr = 0
    elif opsel_a == 1:
        kOpSelA: l.constexpr = 1
    elif opsel_a == 2:
        kOpSelA: l.constexpr = 2
    else:
        kOpSelA: l.constexpr = 3
    if opsel_b == 0:
        kOpSelB: l.constexpr = 0
    elif opsel_b == 1:
        kOpSelB: l.constexpr = 1
    elif opsel_b == 2:
        kOpSelB: l.constexpr = 2
    else:
        kOpSelB: l.constexpr = 3
    return mma_scale_m16n16k128_fp4_fp4_f32(
        a, scale_a, b, scale_b, acc, kOpSelA, kOpSelB
    )


class NativeMxFp4Matmul(DeviceTemplate):
    def __init__(self, kTileM, kTileN):
        assert kTileM in (32, 64) and kTileN % 32 == 0
        self._key = (kTileM, kTileN)
        self.kMRepeats, self.kNRepeats = kTileM // 16, kTileN // 16
        self.kKStages = 2
        self.kActivationFragments = self.kMRepeats * self.kKStages
        self.kAccumFragments = self.kMRepeats * self.kNRepeats
        self.kWeightFragments = self.kNRepeats
        self.kScaleFragments = self.kMRepeats // 2

    @device_method
    def Matmul(self, t, w, x, scale_x, scale_w):
        for k128 in l.static_range(2):
            for n_fragment in l.static_range(self.kWeightFragments):
                for m16 in l.static_range(self.kMRepeats):
                    t_idx: l.constexpr
                    t_idx = n_fragment * self.kMRepeats + m16
                    value = ScaledMxFp4Mfma(
                        2 * k128 + (n_fragment & 1),
                        2 * k128 + (m16 & 1),
                        w[k128][n_fragment],
                        scale_w[n_fragment // 2],
                        x[m16 * 2 + k128],
                        scale_x[m16 // 2],
                        t[t_idx],
                    )
                    t = t[:t_idx] + (value,) + t[t_idx + 1 :]
        return t
