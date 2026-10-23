// Sol-14 paths 51 (split-K grid, separate fixed-order finish) and 47 (split 1),
// from the Sol-10 deep_kernel.cuh and fast_kernel.cuh with the padded row count R
// as the template parameter and the real row count M at run time. Same packed
// step sequence, same prefetch structure, same epilogue; R=20 clamps padded
// fragment rows to a shared zero row exactly as in local14_kernel.cuh.
template<int R,bool F,int BK,int W,int MINBLOCKS>
__global__ __launch_bounds__(W*32,MINBLOCKS) void projection_deep(
    const half*__restrict__ x,const uint4*__restrict__ w,const uint32_t*__restrict__ scales,
    const float*__restrict__ inverse,float gain,int reciprocal,const float*__restrict__ bias,
    float*__restrict__ y,float*__restrict__ partial,int M,int N,int K,int PK,int split,int dtype) {
  constexpr int MT=(R+7)/8,ZR=(R%8)?1:0,ROWS=R+ZR,Q=F?32:16;
  __shared__ __align__(16) half sx[2][ROWS][BK+SOL14_PAD];
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,quad=(lane>>2)&3,local=(lane&3)+(lane>>4)*4;
  int nw=blockIdx.x*W+warp,chunk=K/split,begin=blockIdx.y*chunk,end=begin+chunk,buffer=0;
  int frow[MT];
  #pragma unroll
  for(int r=0;r<MT;r++){int rr=r*8+local;frow[r]=(ZR && r==MT-1 && rr>=R)?R:rr;}
  float c[MT][8]={};volatile float*published=partial;
  Stager<R,BK,W> stager(x,K,M,tid);
  stager.stage(sx[0],begin,x,K);
  if constexpr(ZR) {
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
  #pragma unroll
  for(int r=0;r<MT;r++) {
    #pragma unroll
    for(int j=0;j<8;j++) {
      int row=r*8+(lane&1)+(j&2)+(lane>>4)*4,col=nw*32+quad*8+(j&4)+(lane&2)+(j&1);
      if(row<M) {
        int64_t idx=(int64_t)row*N+col;float v=c[r][j];if constexpr(F)v*=128.f;
        if(split>1)published[(int64_t)blockIdx.y*M*N+idx]=v;
        else store_output(y,idx,epilogue(v,inverse[row],gain,reciprocal,bias?bias[col]:0,dtype,bias!=nullptr),dtype);
      }
    }
  }
}

template<int R,bool F,int BK,int W>void launch_deep(const half*x,const uint8_t*w,const uint8_t*s,const float*i,
    float g,int rec,const float*b,float*y,float*p,int m,int n,int k,int pk,int split,int dtype) {
  dim3 grid(n/(W*32),split);
  projection_deep<R,F,BK,W,(W==4?4:2)><<<grid,W*32>>>(x,(const uint4*)w,(const uint32_t*)s,i,g,rec,b,y,p,m,n,k,pk,split,dtype);
}
