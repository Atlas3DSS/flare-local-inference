// Activation block staging shared by the sol-14 kernels: rows [0,R) x BK halves
// of x at K offset kbase -> smem rows [0,R); rows >= M are stored as zeros.
// Same loads, same stores and same zero fill as sol-10's inline loops
// (local_kernel.cuh lines 18-23 and 33-39); the per-thread row index, source
// pointer and row<M predicates are computed once instead of once per block,
// which removes most of the per-block address arithmetic (see REPORT.md, SASS).
#ifndef SOL14_STAGER
#define SOL14_STAGER 1
#endif
#if SOL14_STAGER
template<int R,int BK,int W>
struct Stager {
  static constexpr int PER_ROW=BK/4, TOTAL=R*PER_ROW, STRIDE=W*32, U=(TOTAL+STRIDE-1)/STRIDE;
  static constexpr bool ALIGNED=(STRIDE%PER_ROW)==0;   // every pass advances whole rows
  static constexpr int ROWSTEP=ALIGNED?STRIDE/PER_ROW:0;
  const half* src; int64_t rowbytes_step; int kk0; int r0; uint32_t mask;
  __device__ __forceinline__ Stager(const half* x,int K,int M,int tid) {
    r0=tid/PER_ROW;kk0=tid%PER_ROW*4;src=x+(int64_t)r0*K+kk0;rowbytes_step=(int64_t)ROWSTEP*K;mask=0;
    #pragma unroll
    for(int u=0;u<U;u++) {
      int i=tid+u*STRIDE,r=ALIGNED?r0+u*ROWSTEP:i/PER_ROW;
      if(i<TOTAL && r<M)mask|=1u<<u;
    }
  }
  __device__ __forceinline__ void stage(half (*dst)[BK+SOL14_PAD],int kbase,const half* x,int K) const {
    #pragma unroll
    for(int u=0;u<U;u++) {
      int i=tid_free_index(u),r,kk;
      if constexpr(ALIGNED){r=r0+u*ROWSTEP;kk=kk0;}else{r=i/PER_ROW;kk=i%PER_ROW*4;}
      uint2 a={0,0};
      if(mask>>u&1) {
        if constexpr(ALIGNED)a=__ldg(reinterpret_cast<const uint2*>(src+kbase+u*rowbytes_step));
        else a=__ldg(reinterpret_cast<const uint2*>(x+(int64_t)r*K+kbase+kk));
      }
      if(U*STRIDE==TOTAL || i<TOTAL)*reinterpret_cast<uint2*>(&dst[r][kk])=a;
    }
  }
  // i = tid + u*STRIDE, reconstructed from r0/kk0 so no extra register is kept.
  __device__ __forceinline__ int tid_free_index(int u) const {return r0*PER_ROW+kk0/4+u*STRIDE;}
};
#else
// SOL14_STAGER=0: the sol-10 inline staging loops, same interface (A/B reference).
template<int R,int BK,int W>
struct Stager {
  int tid,M;
  __device__ __forceinline__ Stager(const half* x,int K,int M_,int tid_):tid(tid_),M(M_){}
  __device__ __forceinline__ void stage(half (*dst)[BK+SOL14_PAD],int kbase,const half* x,int K) const {
    #pragma unroll
    for(int i=tid;i<R*(BK/4);i+=W*32) {
      int r=i/(BK/4),kk=i%(BK/4)*4;uint2 a={0,0};
      if(r<M)a=__ldg(reinterpret_cast<const uint2*>(x+r*K+kbase+kk));
      *reinterpret_cast<uint2*>(&dst[r][kk])=a;
    }
  }
};
#endif
