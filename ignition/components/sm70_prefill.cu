// Sol-8 tiled/finish arithmetic retained; FP8 loads decode the Sol-7/10 bit layout.
#include "repacked.cu"
#include <cutlass/gemm/warp/mma_tensor_op_sm70.h>
#include <cutlass/gemm/warp/mma_tensor_op_policy.h>
using Policy=cutlass::gemm::warp::MmaTensorOpPolicy<cutlass::arch::Mma<
  cutlass::gemm::GemmShape<16,16,4>,32,cutlass::half_t,cutlass::layout::RowMajor,
  cutlass::half_t,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,
  cutlass::arch::OpMultiplyAdd>,cutlass::MatrixShape<1,1>>;
using LA=cutlass::layout::RowMajorVoltaTensorOpMultiplicandCrosswise<16,32>;
using LB=cutlass::layout::ColumnMajorVoltaTensorOpMultiplicandCrosswise<16,32>;

template<bool F,int BM,int BN,int BK,int THREADS> struct Tile {
  static constexpr int AC=BM*BK/(THREADS*8), BC=(BN*(BK/(F?32:16))+THREADS-1)/THREADS;
  uint4 a[AC],b[BC];uint32_t s[BC];
  __device__ __forceinline__ void fetch(const half*x,const uint4*w,const uint32_t*sc,
      int mb,int nb,int kb,int M,int N,int K,int PK,int end) {
    int tid=threadIdx.x;
    #pragma unroll
    for(int j=0;j<AC;j++) {
      int i=tid+j*THREADS,rm=i/(BK/8),kk=(i%(BK/8))*8;
      a[j]={0,0,0,0};
      if(mb+rm<M&&kb+kk<end)a[j]=__ldg(reinterpret_cast<const uint4*>(x+(mb+rm)*K+kb+kk));
    }
    #pragma unroll
    for(int j=0;j<BC;j++) {
      constexpr int Q=F?32:16;
      int i=tid+j*THREADS, group=i/32, nw=nb/32+group%(BN/32), kk=group/(BN/32)*Q;
      int ln=tid&31,packed_lane=(ln&3)|((ln&4)<<2)|((ln&8)>>1)|((ln&16)>>1);
      b[j]={0,0,0,0};s[j]=0;
      if(i<BN*(BK/Q)&&nw<(N+31)/32&&kb+kk<end) {
        b[j]=__ldg(w+(nw*(PK/Q)+(kb+kk)/Q)*32+packed_lane);
        if constexpr(F)s[j]=__ldg(sc+(nw*(PK/64)+(kb+kk)/64)*32+packed_lane)>>(((kb+kk)%64)/32*16);
      }
    }
  }
  __device__ __forceinline__ void store(half*sa,half*sb) {
    LA la=LA::packed({BM,BK});LB lb=LB::packed({BK,BN});int tid=threadIdx.x;
    #pragma unroll
    for(int j=0;j<AC;j++) {
      int i=tid+j*THREADS,rm=i/(BK/8),kk=i%(BK/8)*8;
      *reinterpret_cast<uint2*>(sa+la({rm,kk}))=make_uint2(a[j].x,a[j].y);
      *reinterpret_cast<uint2*>(sa+la({rm,kk+4}))=make_uint2(a[j].z,a[j].w);
    }
    #pragma unroll
    for(int j=0;j<BC;j++) {
      constexpr int Q=F?32:16;
      int i=tid+j*THREADS,group=i/32,n=group%(BN/32)*32+(tid&31),kk=group/(BN/32)*Q;
      if(i<BN*(BK/Q)) {
      uint4 r=b[j];
      if constexpr(F) {
        uint32_t ss=scale_fold(s[j]);
        #pragma unroll
        for(int v=0;v<4;v++) {
          uint32_t bits=v==0?r.x:v==1?r.y:v==2?r.z:r.w, out[4];
          uint32_t sc=(v<2?ss&65535:ss>>16)*0x10001u;
          decode8(bits,sc,out);
          *reinterpret_cast<uint2*>(sb+lb({kk+v*8,n}))=make_uint2(out[0],out[1]);
          *reinterpret_cast<uint2*>(sb+lb({kk+v*8+4,n}))=make_uint2(out[2],out[3]);
        }
      } else {
        #pragma unroll
        for(int v=0;v<2;v++) {
          uint32_t lo=v==0?r.x:r.z,hi=v==0?r.y:r.w;
          uint32_t lo0,lo1,hi0,hi1;
          decode4_fp8(lo,lo0,lo1);decode4_fp8(hi,hi0,hi1);
          *reinterpret_cast<uint2*>(sb+lb({kk+v*8,n}))=make_uint2(lo0,lo1);
          *reinterpret_cast<uint2*>(sb+lb({kk+v*8+4,n}))=make_uint2(hi0,hi1);
        }
      }
      }
    }
  }
};

template<bool F,int BM,int BN,int BK,bool DOUBLE,int W=4>
__global__ __launch_bounds__(W*32) void tiled(const half*x,const uint4*w,const uint32_t*s,
    const float*inv,float gain,int rec,const float*bias,float*y,float*partial,
    int M,int N,int K,int PK,int split,int dtype) {
  constexpr int WM=BM/(W/2),WN=BN/2,SZ=(BM+BN)*BK;
  using Mma=cutlass::gemm::warp::MmaVoltaTensorOp<cutlass::gemm::GemmShape<WM,WN,4>,
    cutlass::half_t,LA,cutlass::half_t,LB,float,cutlass::layout::RowMajor,Policy>;
  extern __shared__ __align__(16) half sm[];
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,mb=blockIdx.y*BM,nb=blockIdx.x*BN;
  int chunk=((K+split*BK-1)/(split*BK))*BK,begin=blockIdx.z*chunk,end=min(K,begin+chunk),buffer=0;
  Mma mma;typename Mma::FragmentC c;c.clear();
  Tile<F,BM,BN,BK,W*32> tile;
  if constexpr(DOUBLE) {tile.fetch(x,w,s,mb,nb,begin,M,N,K,PK,end);tile.store(sm,sm+BM*BK);__syncthreads();}
  for(int base=begin;base<end;base+=BK) {
    if constexpr(!DOUBLE) {tile.fetch(x,w,s,mb,nb,base,M,N,K,PK,end);tile.store(sm,sm+BM*BK);__syncthreads();}
    if constexpr(DOUBLE)tile.fetch(x,w,s,mb,nb,base+BK,M,N,K,PK,end);
    half*sa=sm+buffer*SZ;half*sb=sa+BM*BK;
    typename Mma::IteratorA ia({reinterpret_cast<cutlass::half_t*>(sa),LA::packed({BM,BK})},lane);
    typename Mma::IteratorB ib({reinterpret_cast<cutlass::half_t*>(sb),LB::packed({BK,BN})},lane);
    ia.add_tile_offset({warp/2,0});ib.add_tile_offset({0,warp%2});
    typename Mma::FragmentA a[2];typename Mma::FragmentB b[2];
    ia.load(a[0]);ib.load(b[0]);++ia;++ib;
    #pragma unroll
    for(int kk=0;kk<BK/4;kk++) {
      if(kk+1<BK/4) {ia.load(a[(kk+1)&1]);ib.load(b[(kk+1)&1]);++ia;++ib;}
      mma(c,a[kk&1],b[kk&1],c);
    }
    if constexpr(DOUBLE) {tile.store(sm+(buffer^1)*SZ,sm+(buffer^1)*SZ+BM*BK);buffer^=1;}
    __syncthreads();
  }
  // CUTLASS Sm70 accumulator coordinate mapping, with logical predication.
  int quad=lane>>2,rm=(((quad&4)>>1)+(quad&1))*8+(lane&1),cn=((quad>>1)&1)*8+(lane&2);
  #pragma unroll
  for(int tn=0;tn<WN/32;tn++) {
    #pragma unroll
    for(int tm=0;tm<WM/32;tm++) {
      #pragma unroll
      for(int nc=0;nc<2;nc++) {
        #pragma unroll
        for(int mr=0;mr<2;mr++) {
          #pragma unroll
          for(int p=0;p<2;p++) {
            #pragma unroll
            for(int r=0;r<2;r++) {
              #pragma unroll
              for(int col=0;col<2;col++) {
                int row=mb+(warp/2)*WM+rm+tm*32+mr*4+r*2,n=nb+(warp%2)*WN+cn+tn*32+nc*4+p*16+col;
                int ci=(((tn*(WM/32)+tm)*2+nc)*2+mr)*8+p*4+r*2+col;
                if(row<M&&n<N) {
                  int64_t idx=(int64_t)row*N+n;float v=c[ci];if constexpr(F)v*=128.f;
                  if(split>1)partial[(int64_t)blockIdx.z*M*N+idx]=v;
                  else store_output(y,idx,epilogue(v,inv[row],gain,rec,bias?bias[n]:0,dtype,bias!=nullptr),dtype);
                }
              }
            }
          }
        }
      }
    }
  }
}

__global__ void finish(const float*p,const float*inv,const float*bias,float*y,int M,int N,int split,float gain,int rec,int dtype) {
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(int64_t)M*N)return;
  float v=0;for(int s=0;s<split;s++)v+=p[(int64_t)s*M*N+i];
  store_output(y,i,epilogue(v,inv[i/N],gain,rec,bias?bias[i%N]:0,dtype,bias!=nullptr),dtype);
}


template<bool F,int BM,int BN,int BK,bool DB,int W> int launch_selected(const half*x,const uint4*w,const uint32_t*s,
 const float*inv,float g,int rec,const float*b,float*y,float*p,int m,int n,int k,int pk,int split,cudaStream_t stream) {
 constexpr int bytes=(BM+BN)*BK*2*(DB?2:1);
 tiled<F,BM,BN,BK,DB,W><<<dim3((n+BN-1)/BN,(m+BM-1)/BM,split),W*32,bytes,stream>>>(x,w,s,inv,g,rec,b,y,p,m,n,k,pk,split,0);
 if(split>1)finish<<<((int64_t)m*n+255)/256,256,0,stream>>>(p,inv,b,y,m,n,split,g,rec,0);
 return cudaGetLastError();
}
extern "C" int atlas_prefill_init() {
 return cudaFuncSetAttribute(tiled<false,128,128,64,true,8>,cudaFuncAttributeMaxDynamicSharedMemorySize,65536);
}
extern "C" int atlas_prefill(const half*x,const uint4*w,const uint32_t*s,const float*inv,float g,int rec,
 const float*b,float*y,float*p,int m,int n,int k,int pk,int split,int fp4,int tactic,cudaStream_t stream) {
 if(m<1||m>512||n<1||n%32||k<64||k%64||pk<k||pk%64||!x||!w||!inv||!y||
    (uintptr_t)x%16||(uintptr_t)w%16||split<1||split>4||(split>1&&!p)||
    (int64_t)n*pk>2147483647LL||(int64_t)m*n*split>2147483647LL) return cudaErrorInvalidValue;
 bool selected=fp4 ? ((k==5120&&(n==17408||n==248320)&&split==1&&tactic==14)||
                         (k==17408&&n==5120&&split==2&&tactic==23)) :
   ((k==6144&&n==5120&&split==1&&tactic==23)||(k==5120&&n==10240&&split==1&&tactic==15)||
    (k==5120&&(n==6144||n==12288)&&split==1&&tactic==9)||(k==5120&&n==1024&&split==4&&tactic==23));
 if(!selected|| (fp4&&!s))return cudaErrorInvalidValue;
 #define CALL(F,BM,BN,BK,DB,W) return launch_selected<F,BM,BN,BK,DB,W>(x,w,s,inv,g,rec,b,y,p,m,n,k,pk,split,stream)
 if(fp4){if(tactic==14){CALL(true,128,128,64,false,8);}else {CALL(true,128,64,64,true,8);}}
 if(tactic==9){CALL(false,64,64,32,true,4);}
 if(tactic==15){CALL(false,128,128,64,true,8);}
 CALL(false,128,64,64,true,8);
 #undef CALL
}
