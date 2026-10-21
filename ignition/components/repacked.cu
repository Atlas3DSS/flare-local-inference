// Counter-guided finite-weight and conflict-free activation path.
// Standalone Volta weight-only projections. No Flare or third-party kernel code.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#ifndef SOL_STAGE
#define SOL_STAGE 0
#endif
constexpr int SOL_PAD=SOL_STAGE>=4?4:8;

__device__ __forceinline__ uint32_t h2mul(uint32_t a, uint32_t b) {
  uint32_t c;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(c) : "r"(a), "r"(b));
  return c;
}
// A rebias multiply also normalizes encoded E4M3 subnormals. 0x7f is NaN.
__device__ __forceinline__ uint32_t fp8pair(uint32_t bytes) {
  uint32_t v = ((bytes & 0x7f) << 7) | ((bytes & 0x80) << 8)
             | ((bytes & 0x7f00) << 15) | ((bytes & 0x8000) << 16);
  uint32_t result = h2mul(v, 0x5c005c00u); // 2^8, bias 7 -> 15
  if ((bytes & 0x7f) == 0x7f) result = (result & 0xffff0000u) | 0x7e00;
  if ((bytes & 0x7f00) == 0x7f00) result = (result & 0xffffu) | 0x7e000000u;
  return result;
}
template<bool PRMT>
__device__ __forceinline__ uint32_t fp4pair_impl(uint32_t byte,uint32_t scale) {
  if constexpr(PRMT) {
    uint32_t selector=((byte&7)<<4)|((byte&0x70)<<8);
    uint32_t v=__byte_perm(0x3e3c3800u,0x46444240u,selector)
               |((byte&8)<<12)|((byte&0x80)<<24);
    return h2mul(v,scale);
  } else {
    uint32_t v=((byte&7)<<9)|((byte&8)<<12)|((byte&0x70)<<21)|((byte&0x80)<<24);
    return h2mul(h2mul(v,0x74007400u),scale);
  }
}
__device__ __forceinline__ uint32_t fp4pair(uint32_t byte,uint32_t scale) {
  return fp4pair_impl<(SOL_STAGE>=3)>(byte,scale);
}
__device__ __forceinline__ void mma(float (&c)[8], uint32_t a0, uint32_t a1,
                                       uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
      "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
    : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float round_output(float x, int dtype) {
  if (dtype == 1) return __half2float(__float2half_rn(x));
  if (dtype == 2) {
    uint32_t b=__float_as_uint(x);
    b = (b & 0x7fffffffu) > 0x7f800000u ? b | 0x00400000u
        : b + 0x7fff + ((b >> 16) & 1);
    return __uint_as_float(b & 0xffff0000u);
  }
  return x;
}
// Storage is physical; FP32 consumers continue to pass dtype=0.
__device__ __forceinline__ void store_output(float* y, int64_t i, float v, int dtype) {
  if (dtype == 1) reinterpret_cast<half*>(y)[i] = __float2half_rn(v);
  else if (dtype == 2) reinterpret_cast<uint16_t*>(y)[i] = __float_as_uint(v) >> 16;
  else y[i] = v;
}
__device__ __forceinline__ float epilogue(float v, float inverse, float gain,
                                         int reciprocal, float bias, int dtype, bool has_bias) {
  v *= inverse;
  v = reciprocal ? __fdiv_rn(v,gain) : v*gain;
  v=round_output(v,dtype);
  return has_bias ? round_output(v+bias,dtype) : v;
}

#include "repacked_kernel.cuh"
