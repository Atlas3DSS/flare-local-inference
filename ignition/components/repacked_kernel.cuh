// Argon layout: four independent 8-column quadpairs, consecutive K per lane.
// Storage is a bijective load-time transform; no decoded weights are retained.
__device__ __forceinline__ uint32_t scale_fold(uint32_t b) {
  uint32_t v=((b&0x7f)<<7)|((b&0x80)<<8)|((b&0x7f00)<<15)|((b&0x8000)<<16);
  return h2mul(v,0x78007800u); // finite E4M3 * 128, <= 57344
}
__device__ __forceinline__ uint32_t finite8(uint32_t b) {
  uint32_t v=((b&0x7f)<<7)|((b&0x80)<<8)|((b&0x7f00)<<15)|((b&0x8000)<<16);
  return h2mul(v,0x5c005c00u);
}
// Place the first FP8 pair at its final half2 bit positions. The second
// pair occupies the holes, preserving every packed bit and the byte count.
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
__device__ __forceinline__ void decode4_fp8(uint32_t w,uint32_t& b0,uint32_t& b1) {
  uint32_t v0=w&0xbf80bf80u,v1,sign=(w<<1)&0x80008000u;
  asm("lop3.b32 %0, %1, %2, 0x3f803f80, 0xec;":"=r"(v1):"r"(w<<7),"r"(sign));
  b0=h2mul(v0,0x5c005c00u);b1=h2mul(v1,0x5c005c00u);
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
__device__ __forceinline__ void decode8(uint32_t w,uint32_t sc,uint32_t (&b)[4]) {
  uint32_t v0=w&0x8e008e00u,v1=(w<<3)&0x8e008e00u;
  uint32_t s2=(w<<1)&0x80008000u,s3=(w<<2)&0x80008000u,v2,v3;
  asm("lop3.b32 %0, %1, %2, 0x0e000e00, 0xec;":"=r"(v2):"r"(w<<6),"r"(s2));
  asm("lop3.b32 %0, %1, %2, 0x0e000e00, 0xec;":"=r"(v3):"r"(w<<9),"r"(s3));
  b[0]=h2mul(v0,sc);b[1]=h2mul(v1,sc);b[2]=h2mul(v2,sc);b[3]=h2mul(v3,sc);
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
template<bool INVERSE> __global__ void repack_scales(const uint8_t*src,uint8_t*dst,int N,int PK) {
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=(int64_t)((N+31)/32)*(PK/64)*32)return;
  int lane=i%32,step=i/32%(PK/64),n=i/(32*(PK/64))*32+lane_n(lane);
  uint32_t v=INVERSE?reinterpret_cast<const uint32_t*>(src)[i]:0;
  #pragma unroll
  for(int j=0;j<4;j++) {
    int g=step*4+j;
    int64_t off=((int64_t)(n/128)*((PK/16+3)/4)+g/4)*512+(n%32)*16+((n/32)%4)*4+g%4;
    if(n<N) {
      if constexpr(INVERSE)dst[off]=v>>(j*8);
      else v|=(uint32_t)src[off]<<(j*8);
    }
  }
  if constexpr(!INVERSE)reinterpret_cast<uint32_t*>(dst)[i]=v;
}
extern "C" int sol6_repack(const uint8_t*w,const uint8_t*s,uint8_t*rw,uint8_t*rs,int n,int pk,int fp4,int inv) {
  if(n<1||pk%64)return cudaErrorInvalidValue;
  int64_t count=(int64_t)((n+31)/32)*(pk/(fp4?32:16))*32;
  #define R(F,I) repack_weights<F,I><<<(count+255)/256,256>>>(w,rw,n,pk)
  if(fp4){if(inv){R(true,true);}else{R(true,false);}}
  else {if(inv){R(false,true);}else{R(false,false);}}
  #undef R
  if(fp4) {count=(int64_t)((n+31)/32)*(pk/64)*32;
    if(inv)repack_scales<true><<<(count+255)/256,256>>>(s,rs,n,pk);
    else repack_scales<false><<<(count+255)/256,256>>>(s,rs,n,pk);
  }
  return cudaGetLastError();
}
template<bool F> __device__ __forceinline__ uint4 load_raw(const uint4*w,int nw,int k,int PK,int end,int lane) {
  constexpr int Q=F?32:16;
  uint4 r={0,0,0,0};
  if(k<end)r=__ldg(w+(nw*(PK/Q)+k/Q)*32+lane);
  return r;
}
__device__ __forceinline__ uint32_t load_scale(const uint32_t*s,int nw,int k,int PK,int end,int lane) {
  return k<end?__ldg(s+(nw*(PK/64)+k/64)*32+lane):0;
}
__global__ void finish_repacked(const float*p,const float*inv,const float*bias,float*y,
    int M,int N,int split,float gain,int rec,int dtype) {
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<(int64_t)M*N) {
    float v=0;for(int s=0;s<split;s++)v+=p[(int64_t)s*M*N+i];
    store_output(y,i,epilogue(v,inv[i/N],gain,rec,bias?bias[i%N]:0,dtype,bias!=nullptr),dtype);
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
#include "fast_kernel.cuh"
#include "deep_kernel.cuh"
