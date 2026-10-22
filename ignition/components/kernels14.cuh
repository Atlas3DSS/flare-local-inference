// Sol-14 common device code. The decoders, producers, repack and native-scale
// kernels below are copied verbatim from sol-10/selected.cu (lines 12-100 and
// 105-229); the only edit is the exported symbol sol7_native_scales ->
// sol14_native_scales. No decoder bit changes.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#ifndef SOL_STAGE
#define SOL_STAGE 0
#endif
// Activation row padding in halves: 4 (sol-10, 64-bit fragment loads) or 8 (SOL14_LDS128: 128-bit loads).
#ifndef SOL14_LDS128
#define SOL14_LDS128 0
#endif
constexpr int SOL14_PAD=SOL14_LDS128?8:4;
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

template<bool F>
__device__ __forceinline__ uint4 fetch_weight(const uint8_t* w,const uint8_t* scales,
    int n,int k,int N,int end,int PK,uint32_t& sb) {
  uint4 raw={0,0,0,0}; sb=0;
  if(n<N && k<end) {
    raw=__ldg(reinterpret_cast<const uint4*>(w+(int64_t)n*(F?PK/2:PK)+(F?k/2:k)));
    if(F) {
      int g=k/16;
      int64_t off=((int64_t)(n/128)*((PK/16+3)/4)+g/4)*512
                  +(n%32)*16+((n/32)%4)*4+g%4;
      sb=__ldg(reinterpret_cast<const uint16_t*>(scales+off));
    }
  }
  return raw;
}



__device__ __forceinline__ uint32_t scale_fold(uint32_t b) {
  uint32_t v=((b&0x7f)<<7)|((b&0x80)<<8)|((b&0x7f00)<<15)|((b&0x8000)<<16);
  return h2mul(v,0x78007800u); // finite E4M3 * 128, <= 57344
}
__device__ __forceinline__ void decode4_fp8(uint32_t w,uint32_t& b0,uint32_t& b1) {
  uint32_t v0=w&0xbf80bf80u,v1,sign=(w<<1)&0x80008000u;
  asm("lop3.b32 %0, %1, %2, 0x3f803f80, 0xec;":"=r"(v1):"r"(w<<7),"r"(sign));
  b0=h2mul(v0,0x5c005c00u);b1=h2mul(v1,0x5c005c00u);
}


__global__ void finish_deep(const float*p,const float*inv,const float*bias,float*y,
    int M,int N,int split,float gain,int rec,int dtype) {
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<(int64_t)M*N) {
    float v=0;for(int s=0;s<split;s++)v+=p[(int64_t)s*M*N+i];
    store_output(y,i,epilogue(v,inv[i/N],gain,rec,bias?bias[i%N]:0,dtype,bias!=nullptr),dtype);
  }
}
__global__ void produce(const float* x,half* y,float* inverse,int M,int K) {
  __shared__ float maxima[256];
  int row=blockIdx.x,t=threadIdx.x;
  float a=0;
  for(int k=t;k<K;k+=256) a=fmaxf(a,fabsf(x[(int64_t)row*K+k]));
  maxima[t]=a;__syncthreads();
  for(int s=128;s;s>>=1) {if(t<s) maxima[t]=fmaxf(maxima[t],maxima[t+s]);__syncthreads();}
  int e=maxima[0]>0 && isfinite(maxima[0]) ? ilogbf(maxima[0]) : 14;
  // Cap the exponent at +/-126 so both the scale and inverse are normal FP32.
  int shift=max(-126,min(126,14-e));
  if(t==0) inverse[row]=scalbnf(1.f,-shift);
  for(int k=t;k<K;k+=256) y[(int64_t)row*K+k]=__float2half_rn(scalbnf(x[(int64_t)row*K+k],shift));
}
// E4M3 rounding by integer significand arithmetic, ties-to-even, finite clamp.
__device__ __forceinline__ uint8_t encode_fp8(float x) {
  uint32_t b=__float_as_uint(x), sign=(b>>24)&128;
  if((b&0x7fffffffu)>0x7f800000u) return sign|127;
  float v=fminf(fabsf(x),448.f);
  if(v==0) return sign;
  int e=ilogbf(v),step=max(-9,e-3);
  int q=__float2int_rn(scalbnf(v,-step));
  float rounded=scalbnf((float)q,step);
  if(rounded<0x1p-6f) return sign|q;
  e=ilogbf(rounded);
  int m=__float2int_rn(scalbnf(rounded,3-e))-8;
  return sign|((e+7)<<3)|m;
}
__global__ void produce_fp8(const float* x,half* y,float* inverse,int M,int K,float input_scale) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<M) inverse[i]=1;
  if(i<M*K) {
    uint32_t h=fp8pair(encode_fp8(__fdiv_rn(x[i],input_scale)));
    y[i]=__ushort_as_half(h&65535);
  }
}
__device__ __forceinline__ uint32_t interleave8(uint32_t v) {
  uint32_t w=0;
  #pragma unroll
  for(int j=0;j<4;j++) {
    uint32_t b=(v>>(j*8))&255u;
    int half=(j&1)*16, mag=j<2?7:0, sign=j<2?15:14;
    w|=(b&127u)<<(half+mag);
    w|=(b>>7)<<(half+sign);
  }
  return w;
}
__device__ __forceinline__ uint32_t deinterleave8(uint32_t w) {
  uint32_t v=0;
  #pragma unroll
  for(int j=0;j<4;j++) {
    int half=(j&1)*16, mag=j<2?7:0, sign=j<2?15:14;
    v|=(((w>>(half+mag))&127u)|(((w>>(half+sign))&1u)<<7))<<(j*8);
  }
  return v;
}
__device__ __forceinline__ uint32_t interleave(uint32_t v) {
  uint32_t w=0;
  #pragma unroll
  for(int j=0;j<4;j++) {
    int mag=9-3*j,sign=j==0?15:j==1?12:j==2?14:13;
    uint32_t a=(v>>(8*j))&15,b=(v>>(8*j+4))&15;
    w|=((a&7)<<mag)|((a&8)<<(sign-3))|((b&7)<<(mag+16))|((b&8)<<(sign+13));
  }
  return w;
}
__device__ __forceinline__ uint32_t deinterleave(uint32_t w) {
  uint32_t v=0;
  #pragma unroll
  for(int j=0;j<4;j++) {
    int mag=9-3*j,sign=j==0?15:j==1?12:j==2?14:13;
    v|=(((w>>mag)&7)|(((w>>sign)&1)<<3))<<(8*j);
    v|=(((w>>(mag+16))&7)|(((w>>(sign+16))&1)<<3))<<(8*j+4);
  }
  return v;
}
__device__ __forceinline__ int lane_n(int lane) {
  return ((lane>>2)&3)*8+(lane&3)+(lane>>4)*4;
}
template<bool F,bool INVERSE> __global__ void repack_weights(const uint8_t*src,uint8_t*dst,int N,int PK) {
  constexpr int Q=F?32:16;
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=(int64_t)((N+31)/32)*(PK/Q)*32)return;
  int lane=i%32,step=i/32%(PK/Q),n=i/(32*(PK/Q))*32+lane_n(lane);
  if constexpr(INVERSE) {
    if(n<N) {
      uint4 r=reinterpret_cast<const uint4*>(src)[i];
      if constexpr(F){r.x=deinterleave(r.x);r.y=deinterleave(r.y);r.z=deinterleave(r.z);r.w=deinterleave(r.w);}
      else {r.x=deinterleave8(r.x);r.y=deinterleave8(r.y);r.z=deinterleave8(r.z);r.w=deinterleave8(r.w);}
      reinterpret_cast<uint4*>(dst+(int64_t)n*(F?PK/2:PK)+step*16)[0]=r;
    }
  } else {
    uint4 r={0,0,0,0};
    if(n<N)r=reinterpret_cast<const uint4*>(src+(int64_t)n*(F?PK/2:PK)+step*16)[0];
    if constexpr(F){r.x=interleave(r.x);r.y=interleave(r.y);r.z=interleave(r.z);r.w=interleave(r.w);}
    else {r.x=interleave8(r.x);r.y=interleave8(r.y);r.z=interleave8(r.z);r.w=interleave8(r.w);}
    reinterpret_cast<uint4*>(dst)[i]=r;
  }
}
template<bool INVERSE>__global__ void native_scales(const uint8_t*src,uint8_t*dst,int N,int PK) {
  int i=blockIdx.x*blockDim.x+threadIdx.x,count=((N+31)/32)*(PK/64)*32;
  if(i>=count)return;
  int lane=i%32,step=i/32%(PK/64),n=i/(32*(PK/64))*32+lane_n(lane);
  uint32_t word=INVERSE?reinterpret_cast<const uint32_t*>(src)[i]:0;
  #pragma unroll
  for(int j=0;j<4;j++)if(n<N) {
    int offset=n*(PK/16)+step*4+j;
    if constexpr(INVERSE)dst[offset]=word>>(8*j);else word|=(uint32_t)src[offset]<<(8*j);
  }
  if constexpr(!INVERSE)reinterpret_cast<uint32_t*>(dst)[i]=word;
}
extern "C" int sol14_native_scales(const uint8_t*src,uint8_t*dst,int N,int PK,int inv) {
  if(N<1||N>2147483616||PK<64||PK%64||(int64_t)N*(PK/16)>2147483647LL||(int64_t)((N+31)/32)*(PK/64)*32>2147483647LL)return cudaErrorInvalidValue;
  int blocks=(((N+31)/32)*(PK/64)*32+255)/256;
  if(inv)native_scales<true><<<blocks,256>>>(src,dst,N,PK);else native_scales<false><<<blocks,256>>>(src,dst,N,PK);
  return cudaGetLastError();
}

