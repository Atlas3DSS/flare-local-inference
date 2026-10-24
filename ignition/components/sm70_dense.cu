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

// astra-21: single-pass multi-row launches for prefill graphs and MTP absorption (m up to 512 rows in ONE launch, so a
// projection's weights stream once per R-row block instead of once per 20-row call). Each multi-row instantiation keeps
// its decode tactic's per-element arithmetic: dense_vec with the same V, CG, SPLIT=1 and UNROLL (lane l accumulates
// k = l*V + i*32*V, i ascending, j ascending into a[r][j%4], then (a0+a2)+(a1+a3) and the lane shuffle tree 16..1),
// dense_cta with the same WARPS=4 (thread t accumulates k = 4t + i*512, warp shuffle tree, then the WARPS/2..1 tree over
// the shared partials). R (rows per CTA) and W (warps = output channels per CTA of dense_vec) only choose which CTA
// computes an element and how often a weight word is reloaded, never the element's operation sequence.
// DENSE_ROWS_CONFIGS(X) lists the compiled instantiations as X(tactic, R, W) (dense_cta: W is its fixed WARPS=4).
// Reviewed template space: tactics 22 and 42 (dense_vec) R in {4,8,16}, W in {4,8}; tactic 54 (dense_cta) R in
// {8,16,32}. Other tactics keep the 20-row path.
#ifndef DENSE_ROWS_CONFIGS
#define DENSE_ROWS_CONFIGS(X)
#endif
template<int TACTIC,int R,int W> int dense_rows_launch(const float* x,const uint16_t* w,void* y,int m,int n,int k,
                                                       int xs,int ys,cudaStream_t stream) {
    static_assert((TACTIC==22||TACTIC==42) ? ((R==4||R==8||R==16)&&(W==4||W==8))
                  : (TACTIC==54&&W==4&&(R==8||R==16||R==32)), "reviewed astra-21 template space");
    dim3 grid(TACTIC==54 ? n : (n+W-1)/W,(m+R-1)/R);
    if constexpr(TACTIC==22) dense_vec<8,R,W><<<grid,W*32,0,stream>>>(x,w,nullptr,y,m,n,k,xs,ys,0);         // VCASE(22,8,1,4)
    else if constexpr(TACTIC==42) dense_vec<4,R,W,0,1,8><<<grid,W*32,0,stream>>>(x,w,nullptr,y,m,n,k,xs,ys,0); // UCASE(42,4,4,4,0,8)
    else dense_cta<R,4><<<grid,128,0,stream>>>(x,w,nullptr,y,m,n,k,xs,ys,0);                              // BCASE(54,4,4)
    return cudaGetLastError();
}

// astra-21 R3: channel-blocked multi-row variants for prefill / absorption (one launch per <= 512 rows). One warp
// (dense_vec_cb) or one CTA (dense_cta_cb) computes C consecutive output channels x R rows, so an activation float4
// loaded once feeds C channels and a weight word feeds R rows. Row blocks are the FASTEST grid dimension: the ceil(m/R)
// CTAs that share a channel block's weights run back to back and re-read them from L2, not DRAM. Per element the
// operation sequence is the decode instantiation's:
//  * dense_vec_cb<V,R,C,W,CG,UNROLL> = dense_vec<V,*,*,CG,SPLIT=1,UNROLL>: lane l accumulates k = l*V + i*32*V
//    (i ascending), j ascending into a[c][r][j%4] with fmaf, then (a0+a2)+(a1+a3) and the lane shuffle tree 16..1.
//  * dense_cta_cb<R,C> = dense_cta<*,WARPS=4>: thread t accumulates k = 4t + i*512 with fmaf into a[c][r][0..3], then
//    (a0+a2)+(a1+a3), the warp shuffle tree 16..1, and warp 0's tree 2,1 over the four per-warp partials.
template<int V,int R,int C,int WARPS,int CG,int UNROLL>
__global__ void dense_vec_cb(const float* __restrict__ x,const uint16_t* __restrict__ w,void* __restrict__ y,
                             int m,int n,int k,int xs,int ys) {
 const int lane=threadIdx.x&31,channel=(blockIdx.y*WARPS+threadIdx.x/32)*C,mr=blockIdx.x*R;
 if(channel>=n)return;                                  // host: n % C == 0, so channel < n covers all C channels
 float a[C][R][4]={};
 const int ke=(k/(32*V))*(32*V);
 #pragma unroll UNROLL
 for(int base=lane*V;base<ke;base+=32*V) {
  float wf[C][V];
  #pragma unroll
  for(int c=0;c<C;c++) {
   uint32_t bits[V/2];
   const uint16_t* row=w+size_t(channel+c)*k+base;
   if constexpr(V>=8) {
    #pragma unroll
    for(int j=0;j<V/2;j+=4) {
     uint4 z;
     if constexpr(CG) asm("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(z.x),"=r"(z.y),"=r"(z.z),"=r"(z.w) : "l"(row+2*j));
     else z=*reinterpret_cast<const uint4*>(row+2*j);
     bits[j]=z.x;bits[j+1]=z.y;bits[j+2]=z.z;bits[j+3]=z.w;
    }
   } else {
    uint2 z;
    if constexpr(CG) asm("ld.global.cg.v2.u32 {%0,%1}, [%2];" : "=r"(z.x),"=r"(z.y) : "l"(row));
    else z=*reinterpret_cast<const uint2*>(row);
    bits[0]=z.x;bits[1]=z.y;
   }
   #pragma unroll
   for(int j=0;j<V;j++)wf[c][j]=bf16(uint16_t(bits[j/2]>>((j&1)*16)));
  }
  #pragma unroll
  for(int r=0;r<R;r++) {
   if(mr+r<m) {
    #pragma unroll
    for(int j=0;j<V;j+=4) {
     float4 z=*reinterpret_cast<const float4*>(x+(mr+r)*xs+base+j);
     #pragma unroll
     for(int c=0;c<C;c++) {
      a[c][r][0]=fmaf(z.x,wf[c][j],a[c][r][0]);a[c][r][1]=fmaf(z.y,wf[c][j+1],a[c][r][1]);
      a[c][r][2]=fmaf(z.z,wf[c][j+2],a[c][r][2]);a[c][r][3]=fmaf(z.w,wf[c][j+3],a[c][r][3]);
     }
    }
   }
  }
 }
 #pragma unroll
 for(int c=0;c<C;c++) {
  #pragma unroll
  for(int r=0;r<R;r++) {
   float v=__fadd_rn(__fadd_rn(a[c][r][0],a[c][r][2]),__fadd_rn(a[c][r][1],a[c][r][3]));
   #pragma unroll
   for(int q=16;q>0;q/=2)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,q));
   if(lane==0 && mr+r<m)store_y(y,(mr+r)*ys+channel+c,v,0);
  }
 }
}
template<int R,int C>
__global__ void dense_cta_cb(const float* __restrict__ x,const uint16_t* __restrict__ w,void* __restrict__ y,
                             int m,int n,int k,int xs,int ys) {
 constexpr int WARPS=4;
 const int t=threadIdx.x,lane=t&31,warp=t/32,channel=blockIdx.y*C,mr=blockIdx.x*R;
 float a[C][R][4]={};
 for(int base=t*4;base<k;base+=WARPS*32*4) {
  float v0[C],v1[C],v2[C],v3[C];
  #pragma unroll
  for(int c=0;c<C;c++) {
   uint2 z=*reinterpret_cast<const uint2*>(w+size_t(channel+c)*k+base);
   v0[c]=bf16(z.x);v1[c]=bf16(z.x>>16);v2[c]=bf16(z.y);v3[c]=bf16(z.y>>16);
  }
  #pragma unroll
  for(int r=0;r<R;r++)if(mr+r<m) {
   float4 q=*reinterpret_cast<const float4*>(x+(mr+r)*xs+base);
   #pragma unroll
   for(int c=0;c<C;c++) {
    a[c][r][0]=fmaf(q.x,v0[c],a[c][r][0]);a[c][r][1]=fmaf(q.y,v1[c],a[c][r][1]);
    a[c][r][2]=fmaf(q.z,v2[c],a[c][r][2]);a[c][r][3]=fmaf(q.w,v3[c],a[c][r][3]);
   }
  }
 }
 __shared__ float partial[C][R][WARPS];
 #pragma unroll
 for(int c=0;c<C;c++) {
  #pragma unroll
  for(int r=0;r<R;r++) {
   float v=__fadd_rn(__fadd_rn(a[c][r][0],a[c][r][2]),__fadd_rn(a[c][r][1],a[c][r][3]));
   #pragma unroll
   for(int s=16;s>0;s/=2)v=__fadd_rn(v,__shfl_down_sync(0xffffffff,v,s));
   if(lane==0)partial[c][r][warp]=v;
  }
 }
 __syncthreads();
 if(warp==0) {
  #pragma unroll
  for(int c=0;c<C;c++) {
   #pragma unroll
   for(int r=0;r<R;r++) {
    float q=lane<WARPS?partial[c][r][lane]:0;
    #pragma unroll
    for(int s=WARPS/2;s>0;s/=2)q=__fadd_rn(q,__shfl_down_sync(0xffffffff,q,s));
    if(lane==0 && mr+r<m)store_y(y,(mr+r)*ys+channel+c,q,0);
   }
  }
 }
}
// DENSE_CB_CONFIGS(X) lists the compiled channel-blocked instantiations as X(tactic, R, W, C) (dense_cta_cb: W = 4).
// Reviewed template space (R3): tactics 22 / 42 (dense_vec_cb) R, C in {2,4,8}, W in {2,4,8}; tactic 54 (dense_cta_cb)
// R, C in {2,4,8}. The launch needs n % C == 0.
#ifndef DENSE_CB_CONFIGS
#define DENSE_CB_CONFIGS(X) X(22,4,4,2) X(42,8,4,4) X(54,4,4,8)
#endif
template<int TACTIC,int R,int W,int C> int dense_cb_launch(const float* x,const uint16_t* w,void* y,int m,int n,int k,
                                                           int xs,int ys,cudaStream_t stream) {
    static_assert((R==2||R==4||R==8)&&(C==2||C==4||C==8)&&
                  ((TACTIC==22||TACTIC==42) ? (W==2||W==4||W==8) : (TACTIC==54&&W==4)), "reviewed astra-21 R3 space");
    if(n%C)return cudaErrorInvalidValue;
    if constexpr(TACTIC==54) {
        dense_cta_cb<R,C><<<dim3((m+R-1)/R,n/C),128,0,stream>>>(x,w,y,m,n,k,xs,ys);                    // BCASE(54,4,4)
    } else {
        dim3 grid((m+R-1)/R,(n/C+W-1)/W);
        if constexpr(TACTIC==22) dense_vec_cb<8,R,C,W,0,1><<<grid,W*32,0,stream>>>(x,w,y,m,n,k,xs,ys);  // VCASE(22,8,1,4)
        else dense_vec_cb<4,R,C,W,0,8><<<grid,W*32,0,stream>>>(x,w,y,m,n,k,xs,ys);                      // UCASE(42,4,4,4,0,8)
    }
    return cudaGetLastError();
}

// Same operand contract as sol9_dense (FP32 in/out, bias-free, no scratch) for 1 <= m <= 512; `config` = R | W << 8
// | C << 16 selects a compiled instantiation of `tactic` (C = 0: dense_vec/dense_cta with channel-major grid; C >= 2:
// the R3 channel-blocked kernels with row-major grid); no effect on bits.
extern "C" int sol9_dense_rows(const void* x,const uint16_t* w,void* y,int m,int n,int k,int xs,int ys,int tactic,
                               int config,cudaStream_t stream) {
    if(!x || !w || !y || reinterpret_cast<uintptr_t>(w)%16 || reinterpret_cast<uintptr_t>(x)%16 ||
       reinterpret_cast<uintptr_t>(y)%alignof(float) || m<1 || m>512 || n<1 ||
       k<1 || k%256 || xs<k || xs%4 || ys<n || int64_t(m-1)*xs+k>INT32_MAX || int64_t(m-1)*ys+n>INT32_MAX)
        return cudaErrorInvalidValue;
    if(tactic==54 && k%512) return cudaErrorInvalidValue;
    #define DENSE_ROWS_TRY(T_,R_,W_) if(tactic==T_ && config==(R_|(W_<<8))) \
        return dense_rows_launch<T_,R_,W_>(static_cast<const float*>(x),w,y,m,n,k,xs,ys,stream);
    DENSE_ROWS_CONFIGS(DENSE_ROWS_TRY)
    #undef DENSE_ROWS_TRY
    #define DENSE_CB_TRY(T_,R_,W_,C_) if(tactic==T_ && config==(R_|(W_<<8)|(C_<<16))) \
        return dense_cb_launch<T_,R_,W_,C_>(static_cast<const float*>(x),w,y,m,n,k,xs,ys,stream);
    DENSE_CB_CONFIGS(DENSE_CB_TRY)
    #undef DENSE_CB_TRY
    return cudaErrorInvalidValue;
}
