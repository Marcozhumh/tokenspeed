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

"""Native gemm.h declarations; dataclass metadata models unsigned C++ bitfields."""

from dataclasses import dataclass, field, fields
from enum import IntEnum

from lib.gemm.rocm.quantization.types import DataType


class MatmulFeatures(IntEnum):
    kMatmulFeatures_Global = 0
    kMatmulFeatures_Grid = 1
    kMatmulFeatures_HighPrecision = 1 << 1


kMatmulFeatures_Global = MatmulFeatures.kMatmulFeatures_Global
kMatmulFeatures_Grid = MatmulFeatures.kMatmulFeatures_Grid
kMatmulFeatures_HighPrecision = MatmulFeatures.kMatmulFeatures_HighPrecision


class MatmulElementB(IntEnum):
    kMatmulTypeBInt4 = 0
    kMatmulTypeBNvFp4 = 1
    kMatmulTypeBMxFp4 = 2


kMatmulTypeBInt4 = MatmulElementB.kMatmulTypeBInt4
kMatmulTypeBNvFp4 = MatmulElementB.kMatmulTypeBNvFp4
kMatmulTypeBMxFp4 = MatmulElementB.kMatmulTypeBMxFp4


class MatmulMfmaType(IntEnum):
    kMatmulMfmaTypeFp16 = 0
    kMatmulMfmaTypeBf16 = 1
    kMatmulMfmaTypeFp8 = 2


kMatmulMfmaTypeFp16 = MatmulMfmaType.kMatmulMfmaTypeFp16
kMatmulMfmaTypeBf16 = MatmulMfmaType.kMatmulMfmaTypeBf16
kMatmulMfmaTypeFp8 = MatmulMfmaType.kMatmulMfmaTypeFp8


class MatmulWarpPartition(IntEnum):
    # Each warp handles the full tile.
    kMatmulWarpPartition_NK = 0
    # Warps handle the tile cooperatively.
    kMatmulWarpPartition_Cooperative = 1


kMatmulWarpPartition_NK = MatmulWarpPartition.kMatmulWarpPartition_NK
kMatmulWarpPartition_Cooperative = MatmulWarpPartition.kMatmulWarpPartition_Cooperative


@dataclass(slots=True)
class SolutionId:
    tile_m: int = field(metadata={"bits": 8})
    tile_n: int = field(metadata={"bits": 8})
    # Number of K tiles / 4; the stored unit is 64 elements.
    tile_k: int = field(metadata={"bits": 8})
    features: MatmulFeatures = field(metadata={"bits": 4})
    element_b: MatmulElementB = field(metadata={"bits": 4})
    mfma_type: MatmulMfmaType = field(metadata={"bits": 4})
    warp_partition_m: int = field(metadata={"bits": 4})
    warp_partition_n: int = field(metadata={"bits": 4})
    warp_partition_k: int = field(metadata={"bits": 4})
    warp_partition: MatmulWarpPartition = field(metadata={"bits": 4})
    padding: int = field(metadata={"bits": 12})

    def __setattr__(self, name, value):
        # Python has no native bitfields. Truncate on construction AND assignment,
        # preserving unnamed enum bit patterns accepted by native FromRepr.
        spec = self.__dataclass_fields__.get(name)
        if spec is not None:
            value = int(value) & ((1 << spec.metadata["bits"]) - 1)
        object.__setattr__(self, name, value)

    def Repr(self):
        """Return the native 64-bit field representation, including padding."""
        representation = 0
        shift = 0
        for spec in fields(self):
            representation |= getattr(self, spec.name) << shift
            shift += spec.metadata["bits"]
        return representation

    @staticmethod
    def FromRepr(repr):
        """Decode a native unsigned-long ID and clear its padding bits."""
        return SolutionId(
            tile_m=(repr >> 0) & 0xFF,
            tile_n=(repr >> 8) & 0xFF,
            tile_k=(repr >> 16) & 0xFF,
            features=(repr >> 24) & 0xF,
            element_b=(repr >> 28) & 0xF,
            mfma_type=(repr >> 32) & 0xF,
            warp_partition_m=(repr >> 36) & 0xF,
            warp_partition_n=(repr >> 40) & 0xF,
            warp_partition_k=(repr >> 44) & 0xF,
            warp_partition=(repr >> 48) & 0xF,
            padding=0,
        )

    @staticmethod
    def MultiStage(
        features,
        element_b,
        mfma_type,
        tile_m,
        tile_n,
        tile_k,
        warp_partition,
        warp_partition_m,
        warp_partition_n,
        warp_partition_k,
    ):
        """Build a native multistage ID; tile_k is a count of 16-element tiles."""
        return SolutionId(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=(int(tile_k) & 0xFFFFFFFF) // 4,
            features=features,
            element_b=element_b,
            mfma_type=mfma_type,
            warp_partition_m=warp_partition_m,
            warp_partition_n=warp_partition_n,
            warp_partition_k=warp_partition_k,
            warp_partition=warp_partition,
            padding=0,
        )

    @staticmethod
    def Default():
        """Return the native default configuration, distinct from the -1 chooser sentinel."""
        return SolutionId(
            tile_m=1,
            tile_n=4,
            tile_k=2,
            features=kMatmulFeatures_Grid,
            element_b=kMatmulTypeBNvFp4,
            mfma_type=kMatmulMfmaTypeFp16,
            warp_partition_m=1,
            warp_partition_n=2,
            warp_partition_k=2,
            warp_partition=kMatmulWarpPartition_NK,
            padding=0,
        )


assert sum(spec.metadata["bits"] for spec in fields(SolutionId)) == 64

kErrorProblemShape = 1
kErrorKernelShape = 2


@dataclass(slots=True)
class PetitSolutionHints:
    """Algorithm-selection hints; explicit solution IDs ignore these hints."""

    a_type: DataType
    b_type: DataType
    c_type: DataType
    require_high_precision: bool

    def __setattr__(self, name, value):
        if name == "require_high_precision":
            value = bool(value)
        object.__setattr__(self, name, value)


# Native namespace fp4 declarations resolve to their implementation modules.
from lib.gemm.rocm.quantization import fp4
