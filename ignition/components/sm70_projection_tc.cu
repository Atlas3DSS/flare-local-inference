// Reviewed Sol-7 paths 47/51 share one helper translation unit.
// Stream-aware launch wrappers and the reviewed Sol-5 row-local producer.
#include "repacked.cu"
#include "local_kernel.cuh"
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

extern "C" int atlas_produce(const float*x,half*h,float*i,int m,int k,int fp8,float scale,cudaStream_t stream) {
  if(fp8)produce_fp8<<<(m*k+255)/256,256,0,stream>>>(x,h,i,m,k,scale);
  else produce<<<m,256,0,stream>>>(x,h,i,m,k);
  return cudaGetLastError();
}
extern "C" int atlas_repack(const uint8_t*w,const uint8_t*s,uint8_t*rw,uint8_t*rs,int n,int pk,int fp4,int inv,cudaStream_t stream) {
  if(n<1||n%128||pk%64||(int64_t)n*(pk/16)>2147483647LL)return cudaErrorInvalidValue;
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
template<int M,bool F,int BK,int W> void atlas_launch(const half*x,const uint8_t*w,const uint8_t*s,const float*i,
 float g,int rec,const float*b,float*y,float*p,int n,int k,int pk,int split,int path,cudaStream_t stream) {
  if(path==47)projection_fast<M,F,BK,W,(W==4?4:2)><<<dim3(n/(W*32),split),W*32,0,stream>>>(
    x,(const uint4*)w,(const uint32_t*)s,i,g,rec,b,y,p,n,k,pk,split,0);
  else projection_deep<M,F,BK,W,(W==4?4:2)><<<dim3(n/(W*32),split),W*32,0,stream>>>(
    x,(const uint4*)w,(const uint32_t*)s,i,g,rec,b,y,p,n,k,pk,split,0);
}
template<int M,bool F>void atlas_dispatch(const half*x,const uint8_t*w,const uint8_t*s,const float*i,
 float g,int rec,const float*b,float*y,float*p,int n,int k,int pk,int split,int bk,int warps,int path,cudaStream_t stream) {
 #define GO(B,W) atlas_launch<M,F,B,W>(x,w,s,i,g,rec,b,y,p,n,k,pk,split,path,stream)
 if(warps==4){if(bk==128)GO(128,4);else GO(256,4);}
 else {if(bk==128)GO(128,8);else GO(256,8);}
 #undef GO
}
template<int M,bool F,int BK,int W,int S> void atlas_local(const half*x,const uint8_t*w,const uint8_t*s,const float*i,
 float g,int rec,const float*b,float*y,int n,int k,int pk,const float*cg,cudaStream_t stream) {
 static_assert(S*2*((M+7)/8)*8*(BK+4)*2<=49152 && S*((M+7)/8)*8*W*32*4<=49152 && W*S<=32);
 projection_local<M,F,BK,W,S><<<n/(W*32),W*S*32,0,stream>>>(x,(const uint4*)w,(const uint32_t*)s,i,g,rec,b,y,nullptr,n,k,pk,S,0,cg);
}
template<int M> void atlas_dispatch_local(const half*x,const uint8_t*w,const uint8_t*s,const float*i,
 float g,int rec,const float*b,float*y,int n,int k,int pk,int fp4,int split,const float*cg,cudaStream_t stream) {
 #define LOCAL(F,B,W,S) atlas_local<M,F,B,W,S>(x,w,s,i,g,rec,b,y,n,k,pk,cg,stream)
 if(fp4){LOCAL(true,256,4,2);}
 else if(split==8){LOCAL(false,64,1,8);}
 else if(split==2){LOCAL(false,64,2,2);}
 else {LOCAL(false,64,2,4);}
 #undef LOCAL
}
extern "C" int atlas_project(const half*x,const uint8_t*w,const uint8_t*s,const float*i,float g,int rec,
 const float*b,float*y,float*p,int m,int n,int k,int pk,int split,int fp4,int bk,int warps,int path,const float*cg,cudaStream_t stream) {
 if(path==60) {
  bool tuple=fp4 ? bk==256&&warps==4&&split==2 : bk==64&&((warps==1&&split==8)||(warps==2&&(split==2||split==4)));
  int padded=(m+7)/8*8;
  if(!tuple||m<1||m>16||n<1||k<64||pk<k||pk%64||n%(warps*32)||k%(split*bk)||
     (int64_t)n*(pk/(fp4?2:1))>2147483647LL||(int64_t)n*(pk/16)>2147483647LL||
     (int64_t)m*n*split>2147483647LL||split*2*padded*(bk+4)*2>49152||split*padded*warps*32*4>49152) return cudaErrorInvalidValue;
  #define LOCAL_RUN(M) atlas_dispatch_local<M>(x,w,s,i,g,rec,b,y,n,k,pk,fp4,split,cg,stream)
  switch(m){case 1:{LOCAL_RUN(1);break;}case 2:{LOCAL_RUN(2);break;}case 4:{LOCAL_RUN(4);break;}case 8:{LOCAL_RUN(8);break;}case 12:{LOCAL_RUN(12);break;}case 16:{LOCAL_RUN(16);break;}default:return cudaErrorInvalidValue;}
  #undef LOCAL_RUN
  return cudaGetLastError();
 }
 if(cg)return cudaErrorInvalidValue;
 if(n<1||n%128||k<1||pk<k||k%64||pk%64||(bk!=128&&bk!=256)||(warps!=4&&warps!=8)||split<1||n%(warps*32)||k%(split*bk)||(path!=47&&path!=51)||
   (int64_t)((n+31)/32)*(pk/(fp4?32:16))*32>2147483647LL||(int64_t)m*n*split>2147483647LL)return cudaErrorInvalidValue;
 #define RUN(M) if(fp4)atlas_dispatch<M,true>(x,w,s,i,g,rec,b,y,p,n,k,pk,split,bk,warps,path,stream);else atlas_dispatch<M,false>(x,w,s,i,g,rec,b,y,p,n,k,pk,split,bk,warps,path,stream)
 switch(m){case 1:{RUN(1);break;}case 2:{RUN(2);break;}case 4:{RUN(4);break;}case 8:{RUN(8);break;}case 12:{RUN(12);break;}case 16:{RUN(16);break;}default:return cudaErrorInvalidValue;}
 #undef RUN
 if(split>1)finish_repacked<<<(m*n+255)/256,256,0,stream>>>(p,i,b,y,m,n,split,g,rec,0);
 return cudaGetLastError();
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
