#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>

// All tactics use the same row-local tree at every M. BF16 storage is decoded
// by a bit shift; no BF16 instructions or FP16 weight narrowing are used.
__device__ __forceinline__ float bf16(uint16_t b) { return __uint_as_float(uint32_t(b)<<16); }
__device__ __forceinline__ float load_x(const void* x, int i, int dtype) {
    if (dtype==1) return bf16(static_cast<const uint16_t*>(x)[i]);
    if (dtype==2) return __half2float(static_cast<const __half*>(x)[i]);
    return static_cast<const float*>(x)[i];
}
__device__ __forceinline__ void store_y(void* y, int i, float v, int dtype) {
    if (dtype==1) {
        uint32_t b=__float_as_uint(v);
        static_cast<uint16_t*>(y)[i]=((b&0x7fffffff)>0x7f800000) ? (b>>16)|0x40 : (b+0x7fff+((b>>16)&1))>>16;
    } else if (dtype==2) static_cast<__half*>(y)[i]=__float2half_rn(v);
    else static_cast<float*>(y)[i]=v;
}

// Vectorize activations as well as weights: scalar stride-V loads otherwise
// saturate the L1 request pipe long before DRAM reaches its bandwidth.
template<int V,int R,int WARPS,int CG=0,int SPLIT=1,int UNROLL=1>
__global__ void dense_vec(const float* __restrict__ x,const uint16_t* __restrict__ w,
 const float* __restrict__ bias,void* __restrict__ y,int m,int n,int k,int xs,int ys,int yd) {
 int lane=threadIdx.x&31,row=blockIdx.x*WARPS+threadIdx.x/32,mr=blockIdx.y*R;
 if(row>=n)return;
 float a[R][4]={};
 int ks=(k/(32*V)*blockIdx.z/SPLIT)*(32*V), ke=(k/(32*V)*(blockIdx.z+1)/SPLIT)*(32*V);
 #pragma unroll UNROLL
 for(int base=ks+lane*V;base<ke;base+=32*V) {
  uint32_t bits[V/2];
  if constexpr(V>=8) {
   #pragma unroll
   for(int j=0;j<V/2;j+=4) {
    uint4 z;
    if constexpr(CG) asm("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(z.x),"=r"(z.y),"=r"(z.z),"=r"(z.w) : "l"(w+size_t(row)*k+base+2*j));
    else z=*reinterpret_cast<const uint4*>(w+size_t(row)*k+base+2*j);
    bits[j]=z.x;bits[j+1]=z.y;bits[j+2]=z.z;bits[j+3]=z.w;
   }
  } else {
   uint2 z;
   if constexpr(CG) asm("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(z.x),"=r"(z.y) : "l"(w+size_t(row)*k+base));
   else z=*reinterpret_cast<const uint2*>(w+size_t(row)*k+base);
   bits[0]=z.x;bits[1]=z.y;
  }
  float wf[V];
  #pragma unroll
  for(int j=0;j<V;j++)wf[j]=bf16(uint16_t(bits[j/2]>>((j&1)*16)));
  #pragma unroll
  for(int r=0;r<R;r++) {
   if(mr+r<m) {
    #pragma unroll
    for(int j=0;j<V;j+=4) {
     float4 z=*reinterpret_cast<const float4*>(x+(mr+r)*xs+base+j);
     a[r][0]=fmaf(z.x,wf[j],a[r][0]);a[r][1]=fmaf(z.y,wf[j+1],a[r][1]);
     a[r][2]=fmaf(z.z,wf[j+2],a[r][2]);a[r][3]=fmaf(z.w,wf[j+3],a[r][3]);
    }
   }
  }
 }
 #pragma unroll
 for(int r=0;r<R;r++) {
  float v=__fadd_rn(__fadd_rn(a[r][0],a[r][2]),__fadd_rn(a[r][1],a[r][3]));
  #pragma unroll
  for(int q=16;q>0;q/=2)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,q));
  if(lane==0 && mr+r<m)store_y(y,(blockIdx.z*m+mr+r)*ys+row,bias?__fadd_rn(v,bias[row]):v,yd);
 }
}
// A CTA cooperates on one output channel. This raises parallelism for N=96
// without a second launch or an inter-CTA reduction.
template<int R,int WARPS>
__global__ void dense_cta(const float* __restrict__ x,const uint16_t* __restrict__ w,
 const float* __restrict__ bias,void* __restrict__ y,int m,int n,int k,int xs,int ys,int yd) {
 int t=threadIdx.x,lane=t&31,warp=t/32,row=blockIdx.x,mr=blockIdx.y*R;
 float a[R][4]={};
 for(int base=t*4;base<k;base+=WARPS*32*4) {
  uint2 z=*reinterpret_cast<const uint2*>(w+size_t(row)*k+base);
  float v0=bf16(z.x),v1=bf16(z.x>>16),v2=bf16(z.y),v3=bf16(z.y>>16);
  #pragma unroll
  for(int r=0;r<R;r++)if(mr+r<m) {
   float4 q=*reinterpret_cast<const float4*>(x+(mr+r)*xs+base);
   a[r][0]=fmaf(q.x,v0,a[r][0]);a[r][1]=fmaf(q.y,v1,a[r][1]);
   a[r][2]=fmaf(q.z,v2,a[r][2]);a[r][3]=fmaf(q.w,v3,a[r][3]);
  }
 }
 __shared__ float partial[R][WARPS];
 float v[R];
 #pragma unroll
 for(int r=0;r<R;r++) {
  v[r]=__fadd_rn(__fadd_rn(a[r][0],a[r][2]),__fadd_rn(a[r][1],a[r][3]));
  #pragma unroll
  for(int s=16;s>0;s/=2)v[r]=__fadd_rn(v[r],__shfl_down_sync(0xffffffff,v[r],s));
  if(lane==0)partial[r][warp]=v[r];
 }
 __syncthreads();
 if(warp==0) {
  #pragma unroll
  for(int r=0;r<R;r++) {
   float q=lane<WARPS?partial[r][lane]:0;
   #pragma unroll
   for(int s=WARPS/2;s>0;s/=2)q=__fadd_rn(q,__shfl_down_sync(0xffffffff,q,s));
   if(lane==0 && mr+r<m)store_y(y,(mr+r)*ys+row,bias?__fadd_rn(q,bias[row]):q,yd);
  }
 }
}
// Only the independently qualified FP32-in/out, bias-free non-split tactics.
// The launch arithmetic and reductions above are unchanged from Sol-9.
#define BCASE(ID,R,W) case ID: dense_cta<R,W><<<dim3(n,(m+R-1)/R),W*32,0,stream>>>(static_cast<const float*>(x),w,bias,y,m,n,k,xs,ys,yd); break
#define UCASE(ID,V,R,W,C,U) case ID: dense_vec<V,R,W,C,1,U><<<dim3((n+W-1)/W,(m+R-1)/R),W*32,0,stream>>>(static_cast<const float*>(x),w,bias,y,m,n,k,xs,ys,yd); break
#define VCASE(ID,V,R,W) case ID: dense_vec<V,R,W><<<dim3((n+W-1)/W,(m+R-1)/R),W*32,0,stream>>>(static_cast<const float*>(x),w,bias,y,m,n,k,xs,ys,yd); break
extern "C" int sol9_dense(const void* x,const uint16_t* w,const float* bias,void* y,float* scratch,
                         int m,int n,int k,int xs,int ys,int xd,int yd,int tactic,cudaStream_t stream) {
    if(!x || !w || !y || bias || scratch || xd!=0 || yd!=0 ||
       reinterpret_cast<uintptr_t>(w)%16 || reinterpret_cast<uintptr_t>(x)%16 ||
       reinterpret_cast<uintptr_t>(y)%alignof(float) || m<1 || m>20 || n<1 ||
       k<1 || k%256 || xs<k || xs%4 || ys<n ||
       int64_t(m-1)*xs+k>INT32_MAX || int64_t(m-1)*ys+n>INT32_MAX)
        return cudaErrorInvalidValue;
    // dense_cta<4,4> performs complete 512-element iterations.
    if(tactic==54 && k%512) return cudaErrorInvalidValue;
    switch(tactic) {
        VCASE(22,8,1,4);
        UCASE(36,8,4,4,1,4);
        UCASE(42,4,4,4,0,8);
        UCASE(43,8,4,4,1,8);
        BCASE(54,4,4);
        default:return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
