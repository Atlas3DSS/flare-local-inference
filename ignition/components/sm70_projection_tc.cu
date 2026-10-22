// Reviewed Sol-14 projections (M <= 24, one pinned tuple per shape, padded row
// count R in {8,16,20,24} selects the kernel instance, real M at run time) behind
// the stream-aware atlas_* ABI. kernels14.cuh, local14_kernel.cuh,
// deep14_kernel.cuh, fast14_kernel.cuh, step14.cuh and stage14.cuh are verbatim
// sol-14 sources (Argon-approved); sm70_projection_tuples.inc is sol-14's
// tuples-selected.inc, generated from plan14.json (sm70_projection_plan14.json).
// Everything below is Flare-side wrapping: admission (a host replica of
// sol14_project), the dispatch table, stream placement of the reviewed launches,
// the one-time dynamic shared-memory opt-in, and the C3/C7 layout probes.
#include "kernels14.cuh"
#include "local14_kernel.cuh"
#include "deep14_kernel.cuh"
#include "fast14_kernel.cuh"
#ifndef SOL14_TUPLES
#define SOL14_TUPLES "sm70_projection_tuples.inc"
#endif
#define ATLAS_MAX_M 24

// The producers are row-independent (one CTA per row / one thread per element) and also feed the prefill GEMM
// (sm70_prefill.py) at up to 512 rows, so m is bounded by the int32 product only, as in the Sol-10 component; the
// projection kernels below admit m <= 24 (sol14_project).
extern "C" int atlas_produce(const float*x,half*h,float*i,int m,int k,int fp8,float scale,cudaStream_t stream) {
  if(!x||!h||!i||m<1||k<1||k%64||(int64_t)m*k>2147483647LL)return cudaErrorInvalidValue;
  if(fp8)produce_fp8<<<(m*k+255)/256,256,0,stream>>>(x,h,i,m,k,scale);
  else produce<<<m,256,0,stream>>>(x,h,i,m,k);
  return cudaGetLastError();
}
extern "C" int atlas_repack(const uint8_t*w,const uint8_t*s,uint8_t*rw,uint8_t*rs,int n,int pk,int fp4,int inv,cudaStream_t stream) {
  if(!w||!rw||(fp4&&(!s||!rs))||n<1||n%128||pk<64||pk%64||(int64_t)n*(pk/16)>2147483647LL||
     (int64_t)((n+31)/32)*(pk/(fp4?32:16))*32>2147483647LL)return cudaErrorInvalidValue;
  int64_t count=(int64_t)((n+31)/32)*(pk/(fp4?32:16))*32;
  #define R(F,I) repack_weights<F,I><<<(count+255)/256,256,0,stream>>>(w,rw,n,pk)
  if(fp4){if(inv){R(true,true);}else{R(true,false);}}
  else {if(inv){R(false,true);}else{R(false,false);}}
  #undef R
  if(fp4) {count=(int64_t)((n+31)/32)*(pk/64)*32;
    if(inv)native_scales<true><<<(count+255)/256,256,0,stream>>>(s,rs,n,pk);
    else native_scales<false><<<(count+255)/256,256,0,stream>>>(s,rs,n,pk);
  }
  return cudaGetLastError();
}

// Host replica of LocalShape::smem_bytes (local14_kernel.cuh), as in sol-14's api14.cuh.
static int local_smem_bytes(int R,int bk,int warps,int split) {
  int rows=R+((R%8)?1:0);
  int act=split*2*rows*(bk+SOL14_PAD)*2, part=split*R*warps*32*4;
  return act>part?act:part;
}
static int padded_rows(int m){return m<=8?8:m<=16?16:m<=20?20:24;}

struct Call {
  const half*x;const uint8_t*w;const uint8_t*s;const float*inverse;float gain;int rec;const float*bias;
  float*y;float*partial;int m,n,k,pk,split,bk,warps,path;const float*column_gains;cudaStream_t stream;
};
// The reviewed kernels, launched on the caller's stream (sol-14's launch_* helpers
// use the per-thread default stream, so they are not used here). Dynamic shared
// memory above 48 KiB (R=24 instances only) is opted in once by atlas_init.
template<int R,bool F,int BK,int W,int S> int run_local(const Call&c) {
  using Sh=LocalShape<R,F,BK,W,S>;
  if constexpr(Sh::feasible) {
    projection_local<R,F,BK,W,S><<<c.n/(W*32),W*S*32,Sh::smem_bytes,c.stream>>>(
      c.x,(const uint4*)c.w,(const uint32_t*)c.s,c.inverse,c.gain,c.rec,c.bias,c.y,c.m,c.n,c.k,c.pk,0,c.column_gains);
    return 0;
  } else return 2;
}
template<int R,bool F,int BK,int W> void run_deep(const Call&c) {
  projection_deep<R,F,BK,W,(W==4?4:2)><<<dim3(c.n/(W*32),c.split),W*32,0,c.stream>>>(
    c.x,(const uint4*)c.w,(const uint32_t*)c.s,c.inverse,c.gain,c.rec,c.bias,c.y,c.partial,c.m,c.n,c.k,c.pk,c.split,0);
}
template<int R,bool F,int BK,int W> void run_fast(const Call&c) {
  projection_fast<R,F,BK,W,(W==4?4:2)><<<dim3(c.n/(W*32),c.split),W*32,0,c.stream>>>(
    c.x,(const uint4*)c.w,(const uint32_t*)c.s,c.inverse,c.gain,c.rec,c.bias,c.y,c.partial,c.m,c.n,c.k,c.pk,c.split,0);
}
// 0 launched, 1 tuple not compiled, 2 compiled but infeasible at this R (sol-10 C3: never a silent non-launch).
template<int R,bool F> int dispatch(const Call&c) {
  #define T60(FF,BK,W,S) if constexpr(F==(FF!=0)){if(c.path==60&&c.bk==BK&&c.warps==W&&c.split==S)return run_local<R,F,BK,W,S>(c);}
  #define T51(FF,BK,W) if constexpr(F==(FF!=0)){if(c.path==51&&c.bk==BK&&c.warps==W){run_deep<R,F,BK,W>(c);return 0;}}
  #define T47(FF,BK,W) if constexpr(F==(FF!=0)){if(c.path==47&&c.bk==BK&&c.warps==W){run_fast<R,F,BK,W>(c);return 0;}}
  #include SOL14_TUPLES
  #undef T60
  #undef T51
  #undef T47
  return 1;
}
template<int R> int prepare() {
  int e=0;
  #define T60(FF,BK,W,S) if(!e)e=prepare_local<R,(FF!=0),BK,W,S>();
  #define T51(FF,BK,W)
  #define T47(FF,BK,W)
  #include SOL14_TUPLES
  #undef T60
  #undef T51
  #undef T47
  return e;
}
static bool initialized=false;
// Once per process, before any stream capture: opts every compiled >48 KiB
// path-60 instance into dynamic shared memory (sol-14 integration step 2).
extern "C" int atlas_init() {
  int e=prepare<8>();
  if(!e)e=prepare<16>();
  if(!e)e=prepare<20>();
  if(!e)e=prepare<24>();
  if(!e)initialized=true;
  return e;
}
extern "C" int atlas_project(const half*x,const uint8_t*w,const uint8_t*s,const float*i,float g,int rec,
 const float*b,float*y,float*p,int m,int n,int k,int pk,int split,int fp4,int bk,int warps,int path,const float*cg,cudaStream_t stream) {
 if(!initialized)return cudaErrorInitializationError;
 // Admission: sol-14's sol14_project checks, plus non-null operands, k % 64, column gains only on path 60 and
 // partial planes required wherever the fixed-order finish reads them (paths 51/47 with split > 1).
 if(!x||!w||!i||!y||(fp4&&!s))return cudaErrorInvalidValue;
 if(m<1||m>ATLAS_MAX_M||n<1||n>2147483616||k<64||k%64||pk<k||pk%64||
    split<1||split>32||(int64_t)m*n*split>2147483647LL||
    (int64_t)((n+31)/32)*(pk/(fp4?32:16))*32>2147483647LL||
    (fp4&&(int64_t)n*(pk/16)>2147483647LL)||(bk!=64&&bk!=128&&bk!=256)||
    (warps!=1&&warps!=2&&warps!=4&&warps!=8)||n%(warps*32)||k%(split*bk)||
    (path!=47&&path!=51&&path!=60)||(path==60&&(split!=2&&split!=4&&split!=8))||(path==60&&warps>4)||
    (path!=60&&(warps<4||bk<128))||(path==47&&split!=1))return cudaErrorInvalidValue;
 if(k/split<4*(fp4?32:16))return cudaErrorInvalidValue;
 if(cg&&path!=60)return cudaErrorInvalidValue;
 if(path!=60&&split>1&&!p)return cudaErrorInvalidValue;
 int R=padded_rows(m);
 if(path==60&&(local_smem_bytes(R,bk,warps,split)>98304||warps*split>32))return cudaErrorInvalidValue;
 Call c{x,w,s,i,g,rec,b,y,p,m,n,k,pk,split,bk,warps,path,cg,stream};
 int status;
 #define RT(RR) status=fp4?dispatch<RR,true>(c):dispatch<RR,false>(c)
 switch(R){case 8:{RT(8);break;}case 16:{RT(16);break;}case 20:{RT(20);break;}case 24:{RT(24);break;}default:return cudaErrorInvalidValue;}
 #undef RT
 if(status)return cudaErrorInvalidValue;
 if(split>1&&path!=60)finish_deep<<<(m*n+255)/256,256,0,stream>>>(p,i,b,y,m,n,split,g,rec,0);
 return cudaGetLastError();
}
// Feasibility/compile probe (sol-14's sol14_tuple_status): 0 compiled and feasible at this m, 1 not compiled,
// 2 infeasible, 3 m out of range.
template<int R> int probe(int fp4,int path,int bk,int warps,int split) {
  #define T60(FF,BK,W,S) if(fp4==FF&&path==60&&bk==BK&&warps==W&&split==S)return LocalShape<R,(FF!=0),BK,W,S>::feasible?0:2;
  #define T51(FF,BK,W) if(fp4==FF&&path==51&&bk==BK&&warps==W)return 0;
  #define T47(FF,BK,W) if(fp4==FF&&path==47&&bk==BK&&warps==W)return 0;
  #include SOL14_TUPLES
  #undef T60
  #undef T51
  #undef T47
  return 1;
}
extern "C" int atlas_tuple_status(int fp4,int path,int bk,int warps,int split,int m) {
  if(m<1||m>ATLAS_MAX_M)return 3;
  int R=padded_rows(m);fp4=fp4?1:0;
  if(path==60&&(local_smem_bytes(R,bk,warps,split)>98304||warps*split>32))return 2;
  switch(R){case 8:return probe<8>(fp4,path,bk,warps,split);case 16:return probe<16>(fp4,path,bk,warps,split);
    case 20:return probe<20>(fp4,path,bk,warps,split);default:return probe<24>(fp4,path,bk,warps,split);}
}

// Fable sol-10 C3/C7 layout probes (unchanged from the sol-10 component). decode8 is the
// reviewed four-code NVFP4 decoder from repacked_kernel.cuh, copied verbatim because the
// sol-14 kernels fold the same bit operations into step14.cuh/fast14_kernel.cuh instead.
__device__ __forceinline__ void decode8(uint32_t w,uint32_t sc,uint32_t (&b)[4]) {
  uint32_t v0=w&0x8e008e00u,v1=(w<<3)&0x8e008e00u;
  uint32_t s2=(w<<1)&0x80008000u,s3=(w<<2)&0x80008000u,v2,v3;
  asm("lop3.b32 %0, %1, %2, 0x0e000e00, 0xec;":"=r"(v2):"r"(w<<6),"r"(s2));
  asm("lop3.b32 %0, %1, %2, 0x0e000e00, 0xec;":"=r"(v3):"r"(w<<9),"r"(s3));
  b[0]=h2mul(v0,sc);b[1]=h2mul(v1,sc);b[2]=h2mul(v2,sc);b[3]=h2mul(v3,sc);
}
// C3: independent low/high halves, all interleaved patterns and finite scales.
__global__ void atlas_position_decode(half*out,int first) {
 int pattern=blockIdx.x*blockDim.x+threadIdx.x,scale=first+blockIdx.y;
 uint32_t w=pattern|((pattern^0xa5a5u)<<16),b[4];
 decode8(w,scale_fold(scale|(scale<<8)),b);
 for(int j=0;j<4;j++)reinterpret_cast<uint32_t*>(out)[(blockIdx.y*65536+pattern)*4+j]=b[j];
}
extern "C" int atlas_position_exact(half*out,int first,int count,cudaStream_t stream) {
 atlas_position_decode<<<dim3(256,count),256,0,stream>>>(out,first);return cudaGetLastError();
}
// C7: independently mixed halves for both FP8 placement and inverse proof.
__global__ void atlas_fp8_decode(half*out,uint32_t*inverse) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 uint32_t original=i|((i^0xa5a5u)<<16),w=interleave8(original),b0,b1;
 inverse[i]=deinterleave8(w);decode4_fp8(w,b0,b1);
 reinterpret_cast<uint32_t*>(out)[2*i]=b0;
 reinterpret_cast<uint32_t*>(out)[2*i+1]=b1;
}
extern "C" int atlas_fp8_exact(half*out,uint32_t*inverse,cudaStream_t stream) {
 atlas_fp8_decode<<<256,256,0,stream>>>(out,inverse);return cudaGetLastError();
}
