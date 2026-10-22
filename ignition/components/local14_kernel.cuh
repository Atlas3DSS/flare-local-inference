// Sol-14 path 60: CTA-local split-K with fixed partitions, generalised from
// sol-10/local_kernel.cuh (template parameter M) to a padded row count R with
// the real row count M at run time.
//
//   R in {8,16,20,24}; MT=(R+7)/8 row tiles of the m8n8k4 HMMA.
//   R=20 (MT=3, 4 padded rows in the last tile): lanes whose fragment row
//   would be >= R read one shared ZERO ROW (index R) instead, so the activation
//   buffer holds R+1 rows, not MT*8.  Every real row's HMMA sequence (K
//   partition, ascending packed-step order, fixed-order join) is identical for
//   every R and every M, which is the row-invariance property.
//
//   Shared memory is dynamic (extern) so one tuple can exceed the 48 KiB static
//   limit at R=24 (correctness only) while staying under it at R<=20.
//   The host (sol14_project) replicates the compile-time bound and returns an
//   error instead of launching nothing (Fable sol-10 C3).
#ifndef SOL14_PIPE2
#define SOL14_PIPE2 0
#endif
#include "step14.cuh"
#include "stage14.cuh"
template<int R,bool F,int BK,int W,int S>
struct LocalShape {
  static constexpr int MT=(R+7)/8, ZR=(R%8)?1:0, ROWS=R+ZR;
  static constexpr int activation_bytes=S*2*ROWS*(BK+SOL14_PAD)*2;
  static constexpr int partial_bytes=S*R*(W*32)*4;
  static constexpr int smem_bytes=activation_bytes>partial_bytes?activation_bytes:partial_bytes;
  static constexpr bool feasible=smem_bytes<=98304 && W*S<=32 && BK%(2*(F?32:16))==0;
};

template<int R,bool F,int BK,int W,int S>
__global__ __launch_bounds__(W*S*32,1) void projection_local(
    const half*__restrict__ x,const uint4*__restrict__ w,const uint32_t*__restrict__ scales,
    const float*__restrict__ inverse,float gain,int reciprocal,const float*__restrict__ bias,
    float*__restrict__ y,int M,int N,int K,int PK,int dtype,const float*column_gains) {
  using Sh=LocalShape<R,F,BK,W,S>;
  constexpr int MT=Sh::MT,ZR=Sh::ZR,ROWS=Sh::ROWS,Q=F?32:16;
  extern __shared__ __align__(16) unsigned char sm_raw[];
  typedef half Act[S][2][ROWS][BK+SOL14_PAD];
  typedef float Part[S][R][W*32];
  Act&sa=*reinterpret_cast<Act*>(sm_raw);
  Part&sp_=*reinterpret_cast<Part*>(sm_raw);
  int group=(threadIdx.x/32)/W;
  half (*sx)[ROWS][BK+SOL14_PAD]=sa[group];
  int tid=threadIdx.x%(W*32),lane=tid&31,warp=tid>>5,quad=(lane>>2)&3,local=(lane&3)+(lane>>4)*4;
  int nw=blockIdx.x*W+warp,chunk=K/S,begin=group*chunk,end=begin+chunk,buffer=0;
  // Fragment rows: tile r lane row r*8+local, clamped to the zero row when padded.
  int frow[MT];
  #pragma unroll
  for(int r=0;r<MT;r++){int rr=r*8+local;frow[r]=(ZR && r==MT-1 && rr>=R)?R:rr;}
  float c[MT][8]={};
  Stager<R,BK,W> stager(x,K,M,tid);
  stager.stage(sx[0],begin,x,K);
  if constexpr(ZR) {
    // The zero row is written once, in both buffers, and never rewritten.
    #pragma unroll
    for(int i=tid;i<BK/4;i+=W*32) {
      uint2 zero={0,0};
      *reinterpret_cast<uint2*>(&sx[0][R][i*4])=zero;
      *reinterpret_cast<uint2*>(&sx[1][R][i*4])=zero;
    }
  }
  const uint4*wp=w+(nw*(PK/Q)+begin/Q)*32+lane;
  const uint32_t*sp=scales;
  if constexpr(F)sp+=(nw*(PK/64)+begin/64)*32+lane;
  uint4 next0=__ldg(wp),next1=__ldg(wp+32),next2=__ldg(wp+64),next3=__ldg(wp+96);uint32_t nextscale=0,scale1=0;
  if constexpr(F){nextscale=__ldg(sp);scale1=__ldg(sp+32);}
  __syncthreads();
  for(int base=begin;base<end;base+=BK) {
    if(base+BK<end)stager.stage(sx[buffer^1],base+BK,x,K);
    block_steps<MT,F,BK,SOL14_PIPE2!=0>(c,sx[buffer],frow,wp,sp,base,end,next0,next1,next2,next3,nextscale,scale1);
    __syncthreads();buffer^=1;
  }
  // Activation storage (including the zero row) is dead after the final
  // barrier. Reuse it for deterministic K partials; join in ascending order.
  #pragma unroll
  for(int r=0;r<MT;r++) {
    #pragma unroll
    for(int j=0;j<8;j++) {
      int row=r*8+(lane&1)+(j&2)+(lane>>4)*4;
      int col=warp*32+quad*8+(j&4)+(lane&2)+(j&1);
      if(row<M) {
        float v=c[r][j];if constexpr(F)v*=128.f;
        sp_[group][row][col]=v;
      }
    }
  }
  __syncthreads();
  for(int t=threadIdx.x;t<M*W*32;t+=W*S*32) {
    int row=t/(W*32),localcol=t%(W*32),col=blockIdx.x*W*32+localcol;
    float v=0;
    #pragma unroll
    for(int s=0;s<S;s++)v=__fadd_rn(v,sp_[s][row][localcol]);
    float g=column_gains?column_gains[col]:gain;
    store_output(y,(int64_t)row*N+col,epilogue(v,inverse[row],g,reciprocal,bias?bias[col]:0,dtype,bias!=nullptr),dtype);
  }
}

// Returns false when the tuple is infeasible at this R (the host guard makes
// that unreachable; sol14_project turns false into cudaErrorInvalidValue).
template<int R,bool F,int BK,int W,int S> bool launch_local(const half*x,const uint8_t*w,const uint8_t*s,const float*i,
    float g,int rec,const float*b,float*y,int m,int n,int k,int pk,int dtype,const float*column_gains) {
  using Sh=LocalShape<R,F,BK,W,S>;
  if constexpr(Sh::feasible) {
    if constexpr(Sh::smem_bytes>49152) {
      static bool prepared=false;
      if(!prepared) {
        if(cudaFuncSetAttribute(projection_local<R,F,BK,W,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sh::smem_bytes))return false;
        prepared=true;
      }
    }
    projection_local<R,F,BK,W,S><<<n/(W*32),W*S*32,Sh::smem_bytes>>>(x,(const uint4*)w,(const uint32_t*)s,i,g,rec,b,y,m,n,k,pk,dtype,column_gains);
    return true;
  } else return false;
}
template<int R,bool F,int BK,int W,int S> int prepare_local() {
  using Sh=LocalShape<R,F,BK,W,S>;
  if constexpr(Sh::feasible && Sh::smem_bytes>49152)
    return cudaFuncSetAttribute(projection_local<R,F,BK,W,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,Sh::smem_bytes);
  return 0;
}
