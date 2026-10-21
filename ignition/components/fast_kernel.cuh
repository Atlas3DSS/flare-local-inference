// Model-aligned path: two consecutive packed steps per loop, pointer increments,
// no per-load N/K admission, and activation copies outside the packed-step loop.
template<int M,bool F,int BK,int W,int MINBLOCKS>
__global__ __launch_bounds__(W*32,MINBLOCKS) void projection_fast(
    const half*__restrict__ x,const uint4*__restrict__ w,const uint32_t*__restrict__ scales,
    const float*__restrict__ inverse,float gain,int reciprocal,const float*__restrict__ bias,
    float*__restrict__ y,float*__restrict__ partial,int N,int K,int PK,int split,int dtype) {
  constexpr int MT=(M+7)/8,Q=F?32:16;
  __shared__ __align__(16) half sx[2][MT*8][BK+4];
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,quad=(lane>>2)&3,local=(lane&3)+(lane>>4)*4;
  int nw=blockIdx.x*W+warp,chunk=K/split,begin=blockIdx.y*chunk,end=begin+chunk,buffer=0;
  float c[MT][8]={};volatile float*published=partial;
  #pragma unroll
  for(int i=tid;i<MT*8*(BK/4);i+=W*32) {
    int r=i/(BK/4),kk=i%(BK/4)*4;uint2 a={0,0};
    if(r<M)a=__ldg(reinterpret_cast<const uint2*>(x+r*K+begin+kk));
    *reinterpret_cast<uint2*>(&sx[0][r][kk])=a;
  }
  const uint4*wp=w+(nw*(PK/Q)+begin/Q)*32+lane;
  const uint32_t*sp=scales;
  if constexpr(F)sp+=(nw*(PK/64)+begin/64)*32+lane;
  uint4 next0=__ldg(wp),next1=__ldg(wp+32);uint32_t nextscale=0;
  if constexpr(F)nextscale=__ldg(sp);
  __syncthreads();
  for(int base=begin;base<end;base+=BK) {
    // No activation-copy loop or predicate is replicated per packed step.
    if(base+BK<end) {
      #pragma unroll
      for(int i=tid;i<MT*8*(BK/4);i+=W*32) {
        int r=i/(BK/4),kk=i%(BK/4)*4;uint2 a={0,0};
        if(r<M)a=__ldg(reinterpret_cast<const uint2*>(x+r*K+base+BK+kk));
        *reinterpret_cast<uint2*>(&sx[buffer^1][r][kk])=a;
      }
    }
    #pragma unroll 1
    for(int tile=0;tile<BK;tile+=2*Q) {
      uint4 raw0=next0,raw1=next1;uint32_t sb=nextscale;wp+=64;
      if constexpr(F)sp+=32;
      if(base+tile+2*Q<end) {
        next0=__ldg(wp);next1=__ldg(wp+32);
        if constexpr(F)nextscale=__ldg(sp);
      }
      #pragma unroll
      for(int z=0;z<2;z++) {
        uint4 raw=z==0?raw0:raw1;
        if constexpr(F) {
          uint32_t ss=scale_fold(sb>>(z*16)),sc0=(ss&65535)*0x10001u,sc1=(ss>>16)*0x10001u;
          #pragma unroll
          for(int j=0;j<4;j++) {
            uint32_t word=j==0?raw.x:j==1?raw.y:j==2?raw.z:raw.w,sc=j<2?sc0:sc1;
            #pragma unroll
            for(int t=0;t<2;t++) {
              uint32_t v0,v1;
              if(t==0){v0=word&0x8e008e00u;v1=(word<<3)&0x8e008e00u;}
              else {
                uint32_t sign0=(word<<1)&0x80008000u,sign1=(word<<2)&0x80008000u;
                asm("lop3.b32 %0, %1, %2, 0x0e000e00, 0xec;":"=r"(v0):"r"(word<<6),"r"(sign0));
                asm("lop3.b32 %0, %1, %2, 0x0e000e00, 0xec;":"=r"(v1):"r"(word<<9),"r"(sign1));
              }
              uint32_t b0=h2mul(v0,sc),b1=h2mul(v1,sc);
              #pragma unroll
              for(int r=0;r<MT;r++) {
                const uint32_t*a=reinterpret_cast<const uint32_t*>(&sx[buffer][r*8+local][tile+z*Q+j*8+t*4]);
                mma(c[r],a[0],a[1],b0,b1);
              }
            }
          }
        }else{
          #pragma unroll
          for(int t=0;t<4;t++) {
            uint32_t word=t==0?raw.x:t==1?raw.y:t==2?raw.z:raw.w,b0,b1;decode4_fp8(word,b0,b1);
            #pragma unroll
            for(int r=0;r<MT;r++) {
              const uint32_t*a=reinterpret_cast<const uint32_t*>(&sx[buffer][r*8+local][tile+z*Q+t*4]);
              mma(c[r],a[0],a[1],b0,b1);
            }
          }
        }
      }
    }
    __syncthreads();buffer^=1;
  }
  #pragma unroll
  for(int r=0;r<MT;r++) {
    #pragma unroll
    for(int j=0;j<8;j++) {
      int row=r*8+(lane&1)+(j&2)+(lane>>4)*4,col=nw*32+quad*8+(j&4)+(lane&2)+(j&1);
      if(row<M) {
        int idx=row*N+col;float v=c[r][j];if constexpr(F)v*=128.f;
        if(split>1)published[blockIdx.y*M*N+idx]=v;
        else store_output(y,idx,epilogue(v,inverse[row],gain,reciprocal,bias?bias[col]:0,dtype,bias!=nullptr),dtype);
      }
    }
  }
}
