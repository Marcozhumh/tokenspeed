/*
MIT License

Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
// Test-only oracle; never imported by the Gluon implementation.
#include "gemm/rocm/quantization/fp4/gemm_fp4.h"
#include "gemm/rocm/quantization/gemm.h"
#include "gemm/rocm/quantization/dequant.cuh"
#include "tests/quantization.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <hipblaslt/hipblaslt.h>

using namespace causalflow::petit::rocm::quantization;
namespace fp4 = causalflow::petit::rocm::quantization::fp4;
namespace causalflow::petit::rocm::quantization::fp4 {
int DequantNvFp4(unsigned *, const unsigned *, const unsigned *, float,
                  DataType, unsigned, unsigned);
int DequantPetitMxFp4(unsigned *, const unsigned *, const unsigned *, float,
                      DataType, unsigned, unsigned);
}

extern "C" int native_generate(unsigned m, unsigned n, unsigned k, int dtype,
                               int mx, void *a, void *b, void *s) {
  causalflow::petit::tests::quantization::GemmMPTestData ctx(
      nullptr, static_cast<DataType>(dtype),
      mx ? DataType::kDataTypeMxFp4e2m1 : DataType::kDataTypeFp4e2m1, m, n, k,
      mx ? 32 : 16);
  std::mt19937 gen(42);
  ctx.h_qweights_.resize(k * n / 8);
  if (!ctx.GenerateInputs(&gen).ok() || !ctx.GenerateScales(&gen).ok() ||
      !ctx.GenerateQWeights(&gen).ok())
    return 1;
  std::memcpy(a, ctx.h_input_.data(), ctx.h_input_.size());
  std::memcpy(b, ctx.h_qweights_.data(), ctx.h_qweights_.size() * 4);
  std::memcpy(s, ctx.h_scales_.data(), ctx.h_scales_.size());
  return 0;
}
extern "C" int native_gemm(void *c, void *a, void *b, void *s, void *g,
                           unsigned m, unsigned n, unsigned k, int dtype,
                           int mx, int hp, unsigned long id, void *stream) {
  PetitSolutionHints hints{static_cast<DataType>(dtype),
                           mx ? DataType::kDataTypeMxFp4e2m1
                              : DataType::kDataTypeFp4e2m1,
                           static_cast<DataType>(dtype), static_cast<bool>(hp)};
  auto fn = mx ? fp4::GemmMxFp4Fp16Grid : fp4::GemmFp4Fp16Grid;
  return fn(static_cast<unsigned *>(c), static_cast<unsigned *>(a),
            static_cast<unsigned *>(b), static_cast<unsigned *>(s),
            static_cast<float *>(g), m, n, k, hints, id,
            static_cast<hipStream_t>(stream));
}
extern "C" int native_dequant(void *out, void *b, void *s, float g, int dtype,
                              unsigned k, unsigned n, int mx) {
  auto fn = mx ? fp4::DequantPetitMxFp4 : fp4::DequantPetitFp4;
  return fn(static_cast<unsigned *>(out), static_cast<unsigned *>(b),
            static_cast<unsigned *>(s), g, static_cast<DataType>(dtype), k, n);
}
extern "C" int native_dequant_format(void *out, void *b, void *s, float g, int dtype,
                                     unsigned k, unsigned n, int mx) {
  auto fn = mx ? fp4::DequantMxFp4 : fp4::DequantNvFp4;
  return fn(static_cast<unsigned *>(out), static_cast<unsigned *>(b),
            static_cast<unsigned *>(s), g, static_cast<DataType>(dtype), k, n);
}
extern "C" void native_repack(void *out, void *in, unsigned k, unsigned n,
                              int kind, void *stream) {
  auto fn = kind == 0   ? fp4::RepackNvFp4ToPetitFp4Weights
            : kind == 1 ? fp4::RepackNvFp4ToPetitFp4Scales
                        : fp4::RepackMxFp4ToPetitFp4Scales;
  fn(static_cast<unsigned *>(out), static_cast<unsigned *>(in), k, n,
     static_cast<hipStream_t>(stream));
}
// Fixture-owned hipBLASLt resources for the native test procedure.
extern "C" int native_reference_create(void **handle, void **desc) {
  auto *h = reinterpret_cast<hipblasLtHandle_t *>(handle);
  auto *d = reinterpret_cast<hipblasLtMatmulDesc_t *>(desc);
  auto status = hipblasLtCreate(h);
  if (status != HIPBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = hipblasLtMatmulDescCreate(d, HIPBLAS_COMPUTE_32F, HIP_R_32F);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    hipblasLtDestroy(*h);
    return static_cast<int>(status);
  }
  const auto transpose = HIPBLAS_OP_T;
  status = hipblasLtMatmulDescSetAttribute(
      *d, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
  if (status != HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatmulDescDestroy(*d);
    hipblasLtDestroy(*h);
  }
  return static_cast<int>(status);
}

extern "C" int native_reference_destroy(void *handle, void *desc) {
  auto desc_status = hipblasLtMatmulDescDestroy(
      static_cast<hipblasLtMatmulDesc_t>(desc));
  auto handle_status = hipblasLtDestroy(static_cast<hipblasLtHandle_t>(handle));
  return static_cast<int>(desc_status != HIPBLAS_STATUS_SUCCESS
                              ? desc_status : handle_status);
}

extern "C" int native_reference_with_context(
    void *handle, void *desc, void *workspace, size_t workspace_size,
    void *out, void *a, void *weights, unsigned m, unsigned n, unsigned k,
    int dtype) {
  hipDataType type_a, type_c;
  switch (static_cast<DataType>(dtype)) {
  case DataType::kDataTypeFp16: type_a = type_c = HIP_R_16F; break;
  case DataType::kDataTypeBf16: type_a = type_c = HIP_R_16BF; break;
  case DataType::kDataTypeFp8e4m3:
    type_a = HIP_R_8F_E4M3_FNUZ; type_c = HIP_R_16F; break;
  default: return static_cast<int>(HIPBLAS_STATUS_INVALID_VALUE);
  }
  hipblasLtMatrixLayout_t la, lb, lc;
  auto status = hipblasLtMatrixLayoutCreate(&la, type_a, k, m, k);
  if (status != HIPBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = hipblasLtMatrixLayoutCreate(&lb, type_a, k, n, k);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatrixLayoutDestroy(la);
    return static_cast<int>(status);
  }
  status = hipblasLtMatrixLayoutCreate(&lc, type_c, n, m, n);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    hipblasLtMatrixLayoutDestroy(la);
    hipblasLtMatrixLayoutDestroy(lb);
    return static_cast<int>(status);
  }
  const float alpha = 1.0f, beta = 0.0f;
  if (hipMemset(out, 0, static_cast<size_t>(m) * n * sizeof(unsigned short))
      != hipSuccess) {
    status = HIPBLAS_STATUS_EXECUTION_FAILED;
  } else {
    status = hipblasLtMatmul(
        static_cast<hipblasLtHandle_t>(handle),
        static_cast<hipblasLtMatmulDesc_t>(desc), &alpha, weights, lb, a, la,
        &beta, out, lc, out, lc, nullptr, workspace, workspace_size, nullptr);
  }
  auto sa = hipblasLtMatrixLayoutDestroy(la);
  auto sb = hipblasLtMatrixLayoutDestroy(lb);
  auto sc = hipblasLtMatrixLayoutDestroy(lc);
  for (auto result : {status, sa, sb, sc})
    if (result != HIPBLAS_STATUS_SUCCESS) return static_cast<int>(result);
  return 0;
}

extern "C" int native_reference(void *out, void *a, void *weights, unsigned m,
                                unsigned n, unsigned k, int dtype) {
  void *handle, *desc, *workspace;
  constexpr size_t workspace_size = 32 * 1024 * 1024;
  if (hipMalloc(&workspace, workspace_size) != hipSuccess) return 1;
  auto status = native_reference_create(&handle, &desc);
  if (status == 0) {
    status = native_reference_with_context(handle, desc, workspace, workspace_size,
                                           out, a, weights, m, n, k, dtype);
    auto destroy_status = native_reference_destroy(handle, desc);
    if (status == 0) status = destroy_status;
  }
  hipFree(workspace);
  return status;
}

extern "C" void native_uniform_int(unsigned low, unsigned high, unsigned size,
                                   unsigned *output) {
  std::mt19937 generator(42);
  std::uniform_int_distribution<unsigned> distribution(low, high);
  for (unsigned i = 0; i < size; ++i)
    output[i] = distribution(generator);
}

// Expose the actual native primitive, including the protected fallback, to tests.
template <bool HP> struct DequantProbe : UnifiedDequantizerForFp4Bf16<HP> {
  using UnifiedDequantizerForFp4Bf16<HP>::DequantWithScaleImplFp16;
};

template <class UDQ, bool HP, bool Fallback>
__global__ void DequantProbeKernel(const unsigned *q, const unsigned short *s,
                                  unsigned *out, unsigned *decoded, unsigned size) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  auto ds = UDQ::DequantScales(s[i]);
  decoded[i] = *reinterpret_cast<unsigned *>(&ds);
  typename UDQ::UnpackedData data;
  for (unsigned j = 0; j < 2; ++j) {
    auto scale = j == 0 ? ds.x : ds.y;
    if constexpr (Fallback) {
      auto s2 = causalflow::petit::rocm::fastmath::Fp16Trait<decltype(scale)>::ToFloat2(scale);
      DequantProbe<HP>::DequantWithScaleImplFp16(data, q[i], s2);
    } else {
      UDQ::DequantWithScale(data, q[i], scale);
    }
    for (unsigned w = 0; w < 4; ++w)
      out[(i * 2 + j) * 4 + w] = reinterpret_cast<unsigned *>(&data)[w];
  }
}

template <bool HP, bool Fallback>
int LaunchDequantProbe(const unsigned *q, const unsigned short *s, unsigned *out,
                       unsigned *decoded, unsigned size, int kind, hipStream_t stream) {
  if (kind == 0) {
    DequantProbeKernel<UnifiedDequantizerForFp4Fp16<HP>, HP, false>
        <<<(size + 255) / 256, 256, 0, stream>>>(q, s, out, decoded, size);
  } else if (kind == 1) {
    DequantProbeKernel<UnifiedDequantizerForNvFp4Bf16<HP>, HP, Fallback>
        <<<(size + 255) / 256, 256, 0, stream>>>(q, s, out, decoded, size);
  } else {
    DequantProbeKernel<UnifiedDequantizerForMxFp4Bf16<HP>, HP, Fallback>
        <<<(size + 255) / 256, 256, 0, stream>>>(q, s, out, decoded, size);
  }
  return hipGetLastError();
}

extern "C" int native_dequant_probe(const unsigned *q, const unsigned short *s,
                                    unsigned *out, unsigned *decoded, unsigned size,
                                    int kind, int hp, int fallback, void *stream) {
  auto fn = hp ? (fallback ? LaunchDequantProbe<true, true> : LaunchDequantProbe<true, false>)
               : (fallback ? LaunchDequantProbe<false, true> : LaunchDequantProbe<false, false>);
  return fn(q, s, out, decoded, size, kind, static_cast<hipStream_t>(stream));
}

template <bool BF16, DataType Intermediate, bool Upscale>
__global__ void DequantScaleProbeKernel(const unsigned *q, const unsigned short *s,
                                       unsigned *out, unsigned size) {
  using T = std::conditional_t<BF16, __hip_bfloat162, half2>;
  using DS = DequantizerForFp8Scale<T, Intermediate, Upscale>;
  using DQ = Dequantizer<T, kDataTypeFp4e2m1>;
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= size) return;
  T raw, full, data[2];
  DS::Dequant(&raw, s[i]);
  DS::DequantFullScale(&full, s[i]);
  DQ::Dequant(data, q[i]);
  auto bias = DQ::Bias(Upscale);
  out[i * 7] = *reinterpret_cast<unsigned *>(&raw);
  out[i * 7 + 1] = *reinterpret_cast<unsigned *>(&full);
  out[i * 7 + 2] = *reinterpret_cast<unsigned short *>(&bias);
  out[i * 7 + 3] = reinterpret_cast<unsigned *>(data)[0];
  out[i * 7 + 4] = reinterpret_cast<unsigned *>(data)[1];
  out[i * 7 + 5] = std::bit_cast<unsigned>(DS::GlobalScaleFactor());
  auto product = causalflow::petit::rocm::fastmath::hmul2(*reinterpret_cast<const T *>(&q[i]), raw);
  out[i * 7 + 6] = *reinterpret_cast<unsigned *>(&product);
}

template <bool Upscale>
int LaunchDequantScaleProbe(const unsigned *q, const unsigned short *s, unsigned *out,
                            unsigned size, int kind, hipStream_t stream) {
  if (kind == 0) {
    DequantScaleProbeKernel<false, kDataTypeFp16, Upscale>
        <<<(size + 255) / 256, 256, 0, stream>>>(q, s, out, size);
  } else if (kind == 1) {
    DequantScaleProbeKernel<true, kDataTypeFp16, Upscale>
        <<<(size + 255) / 256, 256, 0, stream>>>(q, s, out, size);
  } else if (kind == 2) {
    DequantScaleProbeKernel<true, kDataTypeFp8e5m2Fnuz, Upscale>
        <<<(size + 255) / 256, 256, 0, stream>>>(q, s, out, size);
  } else {
    DequantScaleProbeKernel<true, kDataTypeFp8e8m0, Upscale>
        <<<(size + 255) / 256, 256, 0, stream>>>(q, s, out, size);
  }
  return hipGetLastError();
}

extern "C" int native_dequant_scale_probe(const unsigned *q, const unsigned short *s,
                                          unsigned *out, unsigned size, int kind,
                                          int upscale, void *stream) {
  auto fn = upscale ? LaunchDequantScaleProbe<true> : LaunchDequantScaleProbe<false>;
  return fn(q, s, out, size, kind, static_cast<hipStream_t>(stream));
}

template <int Aux>
__global__ void BufferLoadProbeKernel(const unsigned *input, unsigned *output,
                                      const int *scalar_offsets) {
  causalflow::petit::rocm::BufferResource resource;
  resource.v = {
      .ptr = reinterpret_cast<uintptr_t>(input + 7),
      .range = 128 * 16,
      .config = causalflow::petit::rocm::BufferResource::kDataFormatU32Config,
  };
  uint4 value = resource.Load<Aux>(threadIdx.x * 16, scalar_offsets[blockIdx.x]);
  reinterpret_cast<uint4 *>(output)[blockIdx.x * 256 + threadIdx.x] = value;
}

extern "C" int native_buffer_load_probe(const unsigned *input, unsigned *output,
                                         const int *scalar_offsets, int aux, void *stream) {
  auto s = static_cast<hipStream_t>(stream);
  switch (aux) {
  case 0: BufferLoadProbeKernel<0><<<3, 256, 0, s>>>(input, output, scalar_offsets); break;
  case 1: BufferLoadProbeKernel<1><<<3, 256, 0, s>>>(input, output, scalar_offsets); break;
  case 2: BufferLoadProbeKernel<2><<<3, 256, 0, s>>>(input, output, scalar_offsets); break;
  case 3: BufferLoadProbeKernel<3><<<3, 256, 0, s>>>(input, output, scalar_offsets); break;
  default: return -1;
  }
  return hipGetLastError();
}


extern "C" unsigned long native_solution_id(unsigned op, unsigned long repr,
                                             const unsigned *v, unsigned *out) {
  SolutionId sol{};
  if (op == 1) {
    sol = SolutionId::FromRepr(repr);
  } else if (op == 2) {
    sol = SolutionId::MultiStage(static_cast<MatmulFeatures>(v[3]),
        static_cast<MatmulElementB>(v[4]), static_cast<MatmulMfmaType>(v[5]),
        v[0], v[1], v[2], static_cast<MatmulWarpPartition>(v[9]), v[6], v[7], v[8]);
  } else if (op == 3) {
    sol = SolutionId::Default();
  } else {
    sol.tile_m = v[0];
    sol.tile_n = v[1];
    sol.tile_k = v[2];
    sol.features = static_cast<MatmulFeatures>(v[3]);
    sol.element_b = static_cast<MatmulElementB>(v[4]);
    sol.mfma_type = static_cast<MatmulMfmaType>(v[5]);
    sol.warp_partition_m = v[6];
    sol.warp_partition_n = v[7];
    sol.warp_partition_k = v[8];
    sol.warp_partition = static_cast<MatmulWarpPartition>(v[9]);
    sol.padding = v[10];
  }
  const unsigned fields[] = {sol.tile_m, sol.tile_n, sol.tile_k,
      static_cast<unsigned>(sol.features), static_cast<unsigned>(sol.element_b),
      static_cast<unsigned>(sol.mfma_type), sol.warp_partition_m,
      sol.warp_partition_n, sol.warp_partition_k,
      static_cast<unsigned>(sol.warp_partition), sol.padding};
  std::memcpy(out, fields, sizeof(fields));
  return sol.Repr();
}

extern "C" void native_gemm_enums(unsigned *out) {
  const unsigned values[] = {kMatmulFeatures_Global, kMatmulFeatures_Grid,
      kMatmulFeatures_HighPrecision, kMatmulTypeBInt4, kMatmulTypeBNvFp4,
      kMatmulTypeBMxFp4, kMatmulMfmaTypeFp16, kMatmulMfmaTypeBf16,
      kMatmulMfmaTypeFp8, kMatmulWarpPartition_NK, kMatmulWarpPartition_Cooperative};
  std::memcpy(out, values, sizeof(values));
}

extern "C" int native_gemm_get_solutions(int a_type, int b_type, int c_type,
                                         int hp, unsigned m, unsigned n,
                                         unsigned k, unsigned long *reprs,
                                         unsigned *count) {
  PetitSolutionHints hints{static_cast<DataType>(a_type), static_cast<DataType>(b_type),
                          static_cast<DataType>(c_type), static_cast<bool>(hp)};
  const unsigned capacity = *count;
  std::vector<SolutionId> sols(reprs ? capacity : 0);
  const int status = fp4::GemmGetSolutions(hints, m, n, k,
                                         reprs ? sols.data() : nullptr, count);
  if (status == 0 && reprs) {
    for (unsigned i = 0; i < std::min(capacity, *count); ++i)
      reprs[i] = sols[i].Repr();
  }
  return status;
}
