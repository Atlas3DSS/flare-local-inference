// Sol-20 register-B prefill projection for sm_70 (author: Claude Opus 5.5, 2026-10-22). Include after repacked.cu
// and sm70_prefill.cu (uses h2mul/decode8/decode4_fp8/scale_fold/mma/epilogue/store_output and Sol-8's `finish`).
//
// Weights stay in the Sol-6/7/10 packed layout (one copy, no new preparation): for a 32-column group nw and a packed
// step of Q k-values, lane L of a warp reads uint4 (nw*(PK/Q)+k/Q)*32+L, which holds column 32*nw+lane_n(L) for Q
// consecutive k. That is exactly the mma.m8n8k4 .col B fragment of quad-pair (L>>2)&3, thread (L&3)+4*(L>>4), so a warp
// loads B straight into registers and decodes it in place (no shared-memory B tile, no B STS/LDS, no B bank conflicts).
// A (FP16 rows from the producer) is a double-buffered shared tile; each lane reads its row r*8+local with LDS.128
// (two k-steps), as in the integrated Sol-14 decode kernels.
//
// Exactness contract (DESIGN.md): every output element is c=0, then one mma.sync.m8n8k4.row.col.f32.f16.f16.f32 per
// k-group {4g..4g+3} of the pinned split chunk [z*K/split, (z+1)*K/split), g ascending, k-slot s = k 4g+s, with B halves
// from the same decode8/decode4_fp8/scale_fold calls on the same bytes as sm70_prefill.cu Tile::store, then x128 for
// NVFP4, then Sol-8's epilogue (split 1) or the FP32 partial + Sol-8's `finish` (split > 1). Tile sizes (MT, NG, W, BK)
// change which warp/lane computes an element, never its operand sequence (ordering is enforced by the accumulator data
// dependency: two mma on the same c[r][g] can not be reordered by the compiler).
//
// Template space submitted for review: F in {0,1}; MT (8-row m-blocks per warp) 1..8; NG (32-column groups per warp)
// 1..2; W (warps per CTA, all along N, sharing one A tile) 1..8; BK (A tile depth) in {32, 64, 128}.
// r2 adds MT2 (tail-segment tile height) 1..MT, MINB (__launch_bounds__ min blocks) 1..8 and the runtime split ya.

template<bool F,int MT,int NG,int W,int BK>
struct PgemmCfg {
  static constexpr int Q=F?32:16;            // k per packed uint4 word group
  static constexpr int U=F?1:2;              // uint4 per lane per 32-k step and group
  static constexpr int ROWS=MT*8;            // CTA rows (one M tile; all W warps share it)
  static constexpr int LD=BK+8;              // padded A row (halves): 16-byte slot stride odd -> LDS.128/STS.128 conflict-free
  static constexpr int TILE=ROWS*LD;         // halves per A buffer
  static constexpr int AV=ROWS*BK/8;         // uint4 per A tile
  static constexpr int AC=(AV+W*32-1)/(W*32);
  static constexpr int SMEM=2*TILE*2;        // bytes, double buffer
  static_assert(MT>=1&&MT<=8&&(NG==1||NG==2)&&W>=1&&W<=8&&(BK==32||BK==64||BK==128),"reviewed template space");
};

template<bool F,int NG>
struct BStep {                               // one 32-k packed step of NG column groups (registers)
  uint4 r[NG][F?1:2];
  uint32_t sc[NG];                           // NVFP4: scale word already shifted to this 32-k half (Sol-8 fetch)
};

template<bool F,int NG,int Q>
__device__ __forceinline__ void pgemm_load_b(BStep<F,NG>&b,const uint4*const(&wb)[NG],const uint32_t*const(&sb)[NG],
    int k,int begin,int end) {
  #pragma unroll
  for(int g=0;g<NG;g++) {
    #pragma unroll
    for(int u=0;u<(F?1:2);u++)b.r[g][u]=make_uint4(0,0,0,0);
    b.sc[g]=0;
    if(k<end) {
      #pragma unroll
      for(int u=0;u<(F?1:2);u++)b.r[g][u]=__ldg(wb[g]+((k-begin)/Q+u)*32);
      // Sol-8: __ldg(sc+(nw*(PK/64)+k/64)*32+lane)>>((k%64)/32*16); sb[g] points at word begin/64 (begin%64==0).
      if constexpr(F)b.sc[g]=__ldg(sb[g]+((k-begin)/64)*32)>>(((k%64)/32)*16);
    }
  }
}

// One 32-k packed step: for every 8-k chunk j (ascending) decode the two k-steps 8j and 8j+4 of each group, then for
// every m-block r issue mma(k-step 8j) and mma(k-step 8j+4) on each c[r][g]. Per element: k-step order 8j, 8j+4.
template<bool F,int MT,int NG,int LD>
__device__ __forceinline__ void pgemm_step(float (&c)[MT][NG][8],const BStep<F,NG>&b,const half*cur,int col,int local) {
  uint32_t sc0[NG],sc1[NG];
  if constexpr(F) {
    #pragma unroll
    for(int g=0;g<NG;g++){uint32_t ss=scale_fold(b.sc[g]);sc0[g]=(ss&65535)*0x10001u;sc1[g]=(ss>>16)*0x10001u;}
  }
  #pragma unroll
  for(int j=0;j<4;j++) {
    uint32_t d[NG][4];
    #pragma unroll
    for(int g=0;g<NG;g++) {
      if constexpr(F) {
        uint32_t word=j==0?b.r[g][0].x:j==1?b.r[g][0].y:j==2?b.r[g][0].z:b.r[g][0].w;
        decode8(word,j<2?sc0[g]:sc1[g],d[g]);                  // d[0],d[1]: k 8j..8j+3; d[2],d[3]: 8j+4..8j+7
      } else {
        const uint4&q=b.r[g][j>>1];                            // uint4 j>>1 covers k 16(j>>1)..+15
        uint32_t lo=(j&1)?q.z:q.x,hi=(j&1)?q.w:q.y;            // Sol-8 store: lo -> k+0..3, hi -> k+4..7
        decode4_fp8(lo,d[g][0],d[g][1]);decode4_fp8(hi,d[g][2],d[g][3]);
      }
    }
    uint4 af[MT];
    #pragma unroll
    for(int r=0;r<MT;r++)af[r]=*reinterpret_cast<const uint4*>(cur+(r*8+local)*LD+col+j*8);
    #pragma unroll
    for(int r=0;r<MT;r++) {
      #pragma unroll
      for(int g=0;g<NG;g++)mma(c[r][g],af[r].x,af[r].y,d[g][0],d[g][1]);
    }
    #pragma unroll
    for(int r=0;r<MT;r++) {
      #pragma unroll
      for(int g=0;g<NG;g++)mma(c[r][g],af[r].z,af[r].w,d[g][2],d[g][3]);
    }
  }
}

// r2 (delta): the r1 kernel body, unchanged except that the M-tile origin `mb` is a parameter, so one launch can mix
// two tile heights (MT for column tiles y < ya, MT2 for y >= ya) to balance the last wave; see pgemm below.
template<bool F,int MT,int NG,int W,int BK>
__device__ __forceinline__ void pgemm_body(half*sa,const int mb,const half*__restrict__ x,const uint4*__restrict__ w,
    const uint32_t*__restrict__ s,const float*__restrict__ inv,float gain,int rec,const float*__restrict__ bias,
    float*__restrict__ y,float*__restrict__ partial,int M,int N,int K,int PK,int split) {
  using C_=PgemmCfg<F,MT,NG,W,BK>;
  constexpr int Q=C_::Q,LD=C_::LD,TILE=C_::TILE,AV=C_::AV,AC=C_::AC;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,quad=(lane>>2)&3,local=(lane&3)+(lane>>4)*4;
  const int nw0=(blockIdx.y*W+warp)*NG;
  const int chunk=K/split,begin=blockIdx.z*chunk,end=begin+chunk;   // host: K%(split*BK)==0 -> Sol-8's chunk
  const bool active=nw0<N/32;                                         // host: (N/32)%NG==0 (warp-uniform)
  const uint4*wb[NG];const uint32_t*sb[NG];
  #pragma unroll
  for(int g=0;g<NG;g++) {
    int nw=active?nw0+g:0;
    wb[g]=w+((int64_t)nw*(PK/Q)+begin/Q)*32+lane;
    sb[g]=F?s+((int64_t)nw*(PK/64)+begin/64)*32+lane:nullptr;
  }
  float c[MT][NG][8];
  #pragma unroll
  for(int r=0;r<MT;r++)
    #pragma unroll
    for(int g=0;g<NG;g++)
      #pragma unroll
      for(int e=0;e<8;e++)c[r][g][e]=0.f;
  uint4 ar[AC];
  auto load_a=[&](int k) {
    #pragma unroll
    for(int j=0;j<AC;j++) {
      int i=tid+j*W*32,row=i/(BK/8),kc=(i%(BK/8))*8;ar[j]=make_uint4(0,0,0,0);
      if(i<AV&&mb+row<M)ar[j]=__ldg(reinterpret_cast<const uint4*>(x+(int64_t)(mb+row)*K+k+kc));
    }
  };
  auto store_a=[&](half*dst) {
    #pragma unroll
    for(int j=0;j<AC;j++) {
      int i=tid+j*W*32,row=i/(BK/8),kc=(i%(BK/8))*8;
      if(i<AV)*reinterpret_cast<uint4*>(dst+row*LD+kc)=ar[j];
    }
  };
  BStep<F,NG> bn;
  load_a(begin);store_a(sa);
  if(active)pgemm_load_b<F,NG,Q>(bn,wb,sb,begin,begin,end);
  __syncthreads();
  int buffer=0;
  for(int base=begin;base<end;base+=BK) {
    const bool more=base+BK<end;
    if(more)load_a(base+BK);                       // next A tile -> registers, stored after this tile's MMA
    const half*cur=sa+buffer*TILE;
    #pragma unroll
    for(int st=0;st<BK;st+=32) {
      BStep<F,NG> bc=bn;
      if(active) {
        pgemm_load_b<F,NG,Q>(bn,wb,sb,base+st+32,begin,end);   // next 32-k step (guarded at end)
        pgemm_step<F,MT,NG,LD>(c,bc,cur,st,local);
      }
    }
    if(more)store_a(sa+(buffer^1)*TILE);           // buffer^1 was last read before the previous barrier
    __syncthreads();
    buffer^=1;
  }
  if(!active)return;
  // mma.m8n8k4 f32 C layout (as Sol-6 projection_fast): row (lane&1)+(e&2)+4*(lane>>4), column quad*8+(e&4)+(lane&2)+(e&1).
  #pragma unroll
  for(int r=0;r<MT;r++) {
    #pragma unroll
    for(int g=0;g<NG;g++) {
      #pragma unroll
      for(int e=0;e<8;e++) {
        int row=mb+r*8+(lane&1)+(e&2)+(lane>>4)*4,n=(nw0+g)*32+quad*8+(e&4)+(lane&2)+(e&1);
        if(row<M) {
          int64_t idx=(int64_t)row*N+n;float v=c[r][g][e];if constexpr(F)v*=128.f;
          if(split>1)partial[(int64_t)blockIdx.z*M*N+idx]=v;
          else store_output(y,idx,epilogue(v,inv[row],gain,rec,bias?bias[n]:0,0,bias!=nullptr),0);
        }
      }
    }
  }
}

// Column tiles y < ya use M tiles of 8*MT rows, y >= ya use 8*MT2 rows (MT2 <= MT); grid.x covers the larger M-tile
// count and CTAs past their segment's last M tile exit before any barrier (CTA-uniform). MINB caps registers via
// __launch_bounds__ (spill-free configs only). MT2 == MT reproduces r1 exactly.
template<bool F,int MT,int NG,int W,int BK,int MT2,int MINB>
__global__ __launch_bounds__(W*32,MINB) void pgemm(const half*__restrict__ x,const uint4*__restrict__ w,
    const uint32_t*__restrict__ s,const float*__restrict__ inv,float gain,int rec,const float*__restrict__ bias,
    float*__restrict__ y,float*__restrict__ partial,int M,int N,int K,int PK,int split,int ya) {
  extern __shared__ __align__(16) half sa[];
  if(MT2!=MT&&(int)blockIdx.y>=ya) {
    const int mb=blockIdx.x*MT2*8;
    if(mb>=M)return;
    pgemm_body<F,MT2,NG,W,BK>(sa,mb,x,w,s,inv,gain,rec,bias,y,partial,M,N,K,PK,split);
  } else {
    const int mb=blockIdx.x*MT*8;
    if(mb>=M)return;
    pgemm_body<F,MT,NG,W,BK>(sa,mb,x,w,s,inv,gain,rec,bias,y,partial,M,N,K,PK,split);
  }
}

template<bool F,int MT,int NG,int W,int BK,int MT2=MT,int MINB=1>
int pgemm_launch(const half*x,const uint4*w,const uint32_t*s,const float*inv,float g,int rec,const float*b,float*y,
    float*p,int m,int n,int k,int pk,int split,cudaStream_t stream,int ya=0) {
  using C_=PgemmCfg<F,MT,NG,W,BK>;
  static_assert(MT2>=1&&MT2<=MT&&MINB>=1&&MINB<=8,"reviewed template space");
  // chunk = K/split must be a multiple of BK (no partial A tile) and of 64 (scale words; equals Sol-8's chunk formula).
  if((n/32)%NG||k%(split*BK)||k%(split*64))return cudaErrorInvalidValue;
  const int tiles=((n/32)/NG+W-1)/W;
  // r3 (Astra R2-1): ya is a 7-bit field; a split launch is refused unless 0 <= ya <= tiles <= 127 (no silent clamp).
  if(MT2==MT)ya=tiles;
  else if(ya<0||ya>tiles||tiles>127)return cudaErrorInvalidValue;
  dim3 grid((m+MT2*8-1)/(MT2*8),tiles,split);
  if(grid.y>65535)return cudaErrorInvalidValue;
  pgemm<F,MT,NG,W,BK,MT2,MINB><<<grid,W*32,C_::SMEM,stream>>>(x,w,s,inv,g,rec,b,y,p,m,n,k,pk,split,ya);
  if(split>1)finish<<<((int64_t)m*n+255)/256,256,0,stream>>>(p,inv,b,y,m,n,split,g,rec,0);
  return cudaGetLastError();
}

// Config id: MT | NG<<4 | W<<8 | (BK/32)<<12 | MT2<<16 | MINB<<20 | ya<<24 (ya = first column tile using MT2, 0..127;
// ignored when MT2 == MT). SOL20_CONFIGS(X) lists instantiations as X(F,MT,NG,W,BK) (MT2 = MT, MINB = 1, r1-equivalent);
// SOL20_CONFIGS2(X) as X(F,MT,NG,W,BK,MT2,MINB).
#ifndef SOL20_CONFIGS
#define SOL20_CONFIGS(X) X(true,8,2,4,32) X(false,8,2,4,32)
#endif
#ifndef SOL20_CONFIGS2
#define SOL20_CONFIGS2(X)
#endif
constexpr int pgemm_cfg(int mt,int ng,int w,int bk,int mt2=0,int minb=1){return mt|(ng<<4)|(w<<8)|((bk/32)<<12)|((mt2?mt2:mt)<<16)|(minb<<20);}

// Same argument contract and pinned (shape, split) admission as atlas_prefill; `cfg` selects the tile (no effect on bits).
extern "C" int atlas_pgemm(const half*x,const uint4*w,const uint32_t*s,const float*inv,float g,int rec,
    const float*b,float*y,float*p,int m,int n,int k,int pk,int split,int fp4,int cfg,cudaStream_t stream) {
  if(m<1||m>512||n<1||n%32||k<64||k%64||pk<k||pk%64||!x||!w||!inv||!y||
     (uintptr_t)x%16||(uintptr_t)w%16||split<1||split>4||(split>1&&!p)||
     (int64_t)n*pk>2147483647LL||(int64_t)m*n*split>2147483647LL) return cudaErrorInvalidValue;
  bool pinned=fp4 ? ((k==5120&&(n==17408||n==248320)&&split==1)||(k==17408&&n==5120&&split==2)) :
    ((k==6144&&n==5120&&split==1)||(k==5120&&(n==10240||n==6144||n==12288)&&split==1)||(k==5120&&n==1024&&split==4));
  if(!pinned||(fp4&&!s))return cudaErrorInvalidValue;
  if(cfg<0)return cudaErrorInvalidValue;                      // bit 31 is not part of any encoding
  const int base=cfg&0xffffff,ya=(cfg>>24)&127;
  #define SOL20_TRY(F_,MT_,NG_,W_,BK_) \
    if(bool(fp4)==F_&&(base==pgemm_cfg(MT_,NG_,W_,BK_)||cfg==(MT_|(NG_<<4)|(W_<<8)|((BK_/32)<<12)))) \
      return pgemm_launch<F_,MT_,NG_,W_,BK_>(x,w,s,inv,g,rec,b,y,p,m,n,k,pk,split,stream);
  SOL20_CONFIGS(SOL20_TRY)
  #undef SOL20_TRY
  #define SOL20_TRY2(F_,MT_,NG_,W_,BK_,MT2_,MINB_) \
    if(bool(fp4)==F_&&base==pgemm_cfg(MT_,NG_,W_,BK_,MT2_,MINB_)) \
      return pgemm_launch<F_,MT_,NG_,W_,BK_,MT2_,MINB_>(x,w,s,inv,g,rec,b,y,p,m,n,k,pk,split,stream,ya);
  SOL20_CONFIGS2(SOL20_TRY2)
  #undef SOL20_TRY2
  return cudaErrorInvalidValue;
}
extern "C" int atlas_pgemm_init(){return 0;}

// r3 (Astra R2-2): dispatch introspection for the gate. describe: configuration resolution and boundary validation
// only - the instantiation `cfg` resolves to for (fp4, n), out = {MT, NG, W, BK, MT2, MINB, effective ya, column tiles},
// returning 0 iff the id matches an instantiation (same match order as atlas_pgemm) and passes the same ya admission;
// it does NOT check atlas_pgemm's remaining launch preconditions (pinned shape, K/split, pointers, sizes, grid limits).
// inventory: every compiled instantiation as 7 ints {F, MT, NG, W, BK, MT2, MINB}; returns the count (writes <= cap).
extern "C" int atlas_pgemm_describe(int fp4,int n,int cfg,int*out) {
  if(cfg<0||!out||n<32||n%32)return cudaErrorInvalidValue;
  const int base=cfg&0xffffff,ya=(cfg>>24)&127;
  auto fill=[&](int mt,int ng,int w,int bk,int mt2,int minb)->int {
    if((n/32)%ng)return cudaErrorInvalidValue;
    int tiles=((n/32)/ng+w-1)/w,eff=tiles;
    if(mt2!=mt){if(ya>tiles||tiles>127)return cudaErrorInvalidValue;eff=ya;}
    int v[8]={mt,ng,w,bk,mt2,minb,eff,tiles};for(int i=0;i<8;i++)out[i]=v[i];return 0;
  };
  #define SOL20_DESC(F_,MT_,NG_,W_,BK_) \
    if(bool(fp4)==F_&&(base==pgemm_cfg(MT_,NG_,W_,BK_)||cfg==(MT_|(NG_<<4)|(W_<<8)|((BK_/32)<<12)))) return fill(MT_,NG_,W_,BK_,MT_,1);
  SOL20_CONFIGS(SOL20_DESC)
  #undef SOL20_DESC
  #define SOL20_DESC2(F_,MT_,NG_,W_,BK_,MT2_,MINB_) \
    if(bool(fp4)==F_&&base==pgemm_cfg(MT_,NG_,W_,BK_,MT2_,MINB_)) return fill(MT_,NG_,W_,BK_,MT2_,MINB_);
  SOL20_CONFIGS2(SOL20_DESC2)
  #undef SOL20_DESC2
  return cudaErrorInvalidValue;
}
extern "C" int atlas_pgemm_inventory(int*out,int cap) {
  int count=0;
  #define SOL20_INV(F_,MT_,NG_,W_,BK_) {if(out&&count<cap){int v[7]={F_,MT_,NG_,W_,BK_,MT_,1};for(int i=0;i<7;i++)out[7*count+i]=v[i];}count++;}
  SOL20_CONFIGS(SOL20_INV)
  #undef SOL20_INV
  #define SOL20_INV2(F_,MT_,NG_,W_,BK_,MT2_,MINB_) {if(out&&count<cap){int v[7]={F_,MT_,NG_,W_,BK_,MT2_,MINB_};for(int i=0;i<7;i++)out[7*count+i]=v[i];}count++;}
  SOL20_CONFIGS2(SOL20_INV2)
  #undef SOL20_INV2
  return count;
}

// ======================================================================================================================
// r4 (additive; r1-r3 code above unchanged): explicitly software-pipelined body for the captured prefill path.
//  - B prefetch ring of PF+1 packed 32-k steps (PF in {1,2}: loads issued PF steps before use).
//  - Decode pipelined one 8-k chunk ahead: while chunk j's HMMAs issue, chunk j+1 (or chunk 0 of the next 32-k step,
//    from ring slot 1) is decoded into a second register set. Per element the operand sequence is unchanged: the same
//    decode8/decode4_fp8/scale_fold calls on the same words, k-step 8j then 8j+4, j ascending, steps ascending.
//  - Device-side live row count (captured graphs run a fixed capacity M; the live token count is a runtime value,
//    e.g. metadata.live_counts[0]): L = live ? clamp(*live, 0, M) : M. Rows < L are computed exactly as r1-r3; every
//    output row in [L, M) is written +0.0 (split 1: in the epilogue, and CTAs whose tile starts at or after L store
//    zeros and exit before any barrier; split > 1: such CTAs exit, and `finish_live` writes +0.0 for rows >= L without
//    reading partials or inverse scales). Inverse scales are read only for rows < L.
// Template space: F; MT 1..8; NG 1..2; W 1..8; BK in {32,64,128}; PF in {1,2}; MINB 1..8.

template<bool F,int NG>
__device__ __forceinline__ void pgemm_scale(const BStep<F,NG>&b,uint32_t (&sc0)[NG],uint32_t (&sc1)[NG]) {
  if constexpr(F) {
    #pragma unroll
    for(int g=0;g<NG;g++){uint32_t ss=scale_fold(b.sc[g]);sc0[g]=(ss&65535)*0x10001u;sc1[g]=(ss>>16)*0x10001u;}
  }
}

// Decode 8-k chunk j of step b into d (identical calls/selection to pgemm_step lines above).
template<bool F,int NG>
__device__ __forceinline__ void pgemm_decode(const BStep<F,NG>&b,int j,const uint32_t (&sc0)[NG],const uint32_t (&sc1)[NG],
    uint32_t (&d)[NG][4]) {
  #pragma unroll
  for(int g=0;g<NG;g++) {
    if constexpr(F) {
      uint32_t word=j==0?b.r[g][0].x:j==1?b.r[g][0].y:j==2?b.r[g][0].z:b.r[g][0].w;
      decode8(word,j<2?sc0[g]:sc1[g],d[g]);
    } else {
      const uint4&q=b.r[g][j>>1];
      uint32_t lo=(j&1)?q.z:q.x,hi=(j&1)?q.w:q.y;
      decode4_fp8(lo,d[g][0],d[g][1]);decode4_fp8(hi,d[g][2],d[g][3]);
    }
  }
}

template<bool F,int MT,int NG,int W,int BK,int PF>
__device__ __forceinline__ void pgemm_body4(half*sa,const int mb,const int L,const half*__restrict__ x,
    const uint4*__restrict__ w,const uint32_t*__restrict__ s,const float*__restrict__ inv,float gain,int rec,
    const float*__restrict__ bias,float*__restrict__ y,float*__restrict__ partial,int M,int N,int K,int PK,int split) {
  using C_=PgemmCfg<F,MT,NG,W,BK>;
  constexpr int Q=C_::Q,LD=C_::LD,TILE=C_::TILE,AV=C_::AV,AC=C_::AC;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,quad=(lane>>2)&3,local=(lane&3)+(lane>>4)*4;
  const int nw0=(blockIdx.y*W+warp)*NG;
  const int chunk=K/split,begin=blockIdx.z*chunk,end=begin+chunk;   // host: K%(split*BK)==0, K%(split*64)==0
  const bool active=nw0<N/32;                                         // host: (N/32)%NG==0 (warp-uniform)
  const uint4*wb[NG];const uint32_t*sb[NG];
  #pragma unroll
  for(int g=0;g<NG;g++) {
    int nw=active?nw0+g:0;
    wb[g]=w+((int64_t)nw*(PK/Q)+begin/Q)*32+lane;
    sb[g]=F?s+((int64_t)nw*(PK/64)+begin/64)*32+lane:nullptr;
  }
  float c[MT][NG][8];
  #pragma unroll
  for(int r=0;r<MT;r++)
    #pragma unroll
    for(int g=0;g<NG;g++)
      #pragma unroll
      for(int e=0;e<8;e++)c[r][g][e]=0.f;
  uint4 ar[AC];
  auto load_a=[&](int k) {
    #pragma unroll
    for(int j=0;j<AC;j++) {
      int i=tid+j*W*32,row=i/(BK/8),kc=(i%(BK/8))*8;ar[j]=make_uint4(0,0,0,0);
      if(i<AV&&mb+row<L)ar[j]=__ldg(reinterpret_cast<const uint4*>(x+(int64_t)(mb+row)*K+k+kc));
    }
  };
  auto store_a=[&](half*dst) {
    #pragma unroll
    for(int j=0;j<AC;j++) {
      int i=tid+j*W*32,row=i/(BK/8),kc=(i%(BK/8))*8;
      if(i<AV)*reinterpret_cast<uint4*>(dst+row*LD+kc)=ar[j];
    }
  };
  BStep<F,NG> q[PF+1];                                 // q[i] = packed step begin+... (i steps ahead of the current one)
  uint32_t sc0[NG]={},sc1[NG]={},nc0[NG]={},nc1[NG]={},d[NG][4]={},dn[NG][4]={};
  load_a(begin);store_a(sa);
  if(active) {
    #pragma unroll
    for(int i=0;i<=PF;i++)pgemm_load_b<F,NG,Q>(q[i],wb,sb,begin+32*i,begin,end);
    pgemm_scale<F,NG>(q[0],sc0,sc1);
    pgemm_decode<F,NG>(q[0],0,sc0,sc1,d);             // chunk 0 of the first step
  }
  __syncthreads();
  int buffer=0;
  for(int base=begin;base<end;base+=BK) {
    const bool more=base+BK<end;
    if(more)load_a(base+BK);
    const half*cur=sa+buffer*TILE;
    #pragma unroll
    for(int st=0;st<BK;st+=32) {
      if(active) {
        #pragma unroll
        for(int j=0;j<4;j++) {
          uint4 af[MT];
          #pragma unroll
          for(int r=0;r<MT;r++)af[r]=*reinterpret_cast<const uint4*>(cur+(r*8+local)*LD+st+j*8);
          if(j<3)pgemm_decode<F,NG>(q[0],j+1,sc0,sc1,dn);         // next chunk of this step
          else {pgemm_scale<F,NG>(q[1],nc0,nc1);pgemm_decode<F,NG>(q[1],0,nc0,nc1,dn);}  // chunk 0 of the next step
          #pragma unroll
          for(int r=0;r<MT;r++) {
            #pragma unroll
            for(int g=0;g<NG;g++)mma(c[r][g],af[r].x,af[r].y,d[g][0],d[g][1]);   // k-step 8j
          }
          #pragma unroll
          for(int r=0;r<MT;r++) {
            #pragma unroll
            for(int g=0;g<NG;g++)mma(c[r][g],af[r].z,af[r].w,d[g][2],d[g][3]);   // k-step 8j+4
          }
          #pragma unroll
          for(int g=0;g<NG;g++)
            #pragma unroll
            for(int t=0;t<4;t++)d[g][t]=dn[g][t];
        }
        // advance the ring: slot 0 <- 1 <- ... <- PF, refill slot PF with the step PF+1 ahead (guarded at end)
        #pragma unroll
        for(int i=0;i<PF;i++)q[i]=q[i+1];
        pgemm_load_b<F,NG,Q>(q[PF],wb,sb,base+st+32*(PF+1),begin,end);
        #pragma unroll
        for(int g=0;g<NG;g++){sc0[g]=nc0[g];sc1[g]=nc1[g];}
      }
    }
    if(more)store_a(sa+(buffer^1)*TILE);
    __syncthreads();
    buffer^=1;
  }
  if(!active)return;
  #pragma unroll
  for(int r=0;r<MT;r++) {
    #pragma unroll
    for(int g=0;g<NG;g++) {
      #pragma unroll
      for(int e=0;e<8;e++) {
        int row=mb+r*8+(lane&1)+(e&2)+(lane>>4)*4,n=(nw0+g)*32+quad*8+(e&4)+(lane&2)+(e&1);
        if(row<M) {
          int64_t idx=(int64_t)row*N+n;
          if(split>1) {
            if(row<L){float v=c[r][g][e];if constexpr(F)v*=128.f;partial[(int64_t)blockIdx.z*M*N+idx]=v;}
          } else if(row<L) {
            float v=c[r][g][e];if constexpr(F)v*=128.f;
            store_output(y,idx,epilogue(v,inv[row],gain,rec,bias?bias[n]:0,0,bias!=nullptr),0);
          } else store_output(y,idx,0.f,0);
        }
      }
    }
  }
}

template<bool F,int MT,int NG,int W,int BK,int PF,int MINB>
__global__ __launch_bounds__(W*32,MINB) void pgemm4(const half*__restrict__ x,const uint4*__restrict__ w,
    const uint32_t*__restrict__ s,const float*__restrict__ inv,float gain,int rec,const float*__restrict__ bias,
    float*__restrict__ y,float*__restrict__ partial,int M,int N,int K,int PK,int split,const int*__restrict__ live) {
  extern __shared__ __align__(16) half sa[];
  const int L=live?min(max(__ldg(live),0),M):M;
  const int mb=blockIdx.x*MT*8;
  if(mb>=M)return;
  if(mb>=L) {                                          // CTA-uniform: no live row in this tile, no barrier reached
    if(split==1) {
      const int warp=threadIdx.x>>5,lane=threadIdx.x&31,nw0=(blockIdx.y*W+warp)*NG;
      if(nw0<N/32) {
        const int rows=min(MT*8,M-mb);
        for(int i=lane;i<rows*NG*32;i+=32) {
          int row=mb+i/(NG*32),n=nw0*32+i%(NG*32);
          store_output(y,(int64_t)row*N+n,0.f,0);
        }
      }
    }
    return;
  }
  pgemm_body4<F,MT,NG,W,BK,PF>(sa,mb,L,x,w,s,inv,gain,rec,bias,y,partial,M,N,K,PK,split);
}

__global__ void finish_live(const float*p,const float*inv,const float*bias,float*y,int M,int N,int split,float gain,
    int rec,int dtype,const int*live) {
  int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(int64_t)M*N)return;
  const int L=live?min(max(__ldg(live),0),M):M;
  if(i/N>=L){store_output(y,i,0.f,dtype);return;}
  float v=0;for(int s=0;s<split;s++)v+=p[(int64_t)s*M*N+i];   // Sol-8 `finish` arithmetic for live rows
  store_output(y,i,epilogue(v,inv[i/N],gain,rec,bias?bias[i%N]:0,dtype,bias!=nullptr),dtype);
}

template<bool F,int MT,int NG,int W,int BK,int PF,int MINB>
int pgemm4_launch(const half*x,const uint4*w,const uint32_t*s,const float*inv,float g,int rec,const float*b,float*y,
    float*p,int m,int n,int k,int pk,int split,const int*live,cudaStream_t stream) {
  using C_=PgemmCfg<F,MT,NG,W,BK>;
  static_assert((PF==1||PF==2)&&MINB>=1&&MINB<=8,"reviewed template space");
  if((n/32)%NG||k%(split*BK)||k%(split*64))return cudaErrorInvalidValue;
  dim3 grid((m+MT*8-1)/(MT*8),((n/32)/NG+W-1)/W,split);
  if(grid.y>65535)return cudaErrorInvalidValue;
  pgemm4<F,MT,NG,W,BK,PF,MINB><<<grid,W*32,C_::SMEM,stream>>>(x,w,s,inv,g,rec,b,y,p,m,n,k,pk,split,live);
  if(split>1)finish_live<<<((int64_t)m*n+255)/256,256,0,stream>>>(p,inv,b,y,m,n,split,g,rec,0,live);
  return cudaGetLastError();
}

// r4 id: MT | NG<<4 | W<<8 | (BK/32)<<12 | PF<<16 | MINB<<20 | 1<<28 (r4 marker). SOL20_CONFIGS4(X): X(F,MT,NG,W,BK,PF,MINB).
#ifndef SOL20_CONFIGS4
#define SOL20_CONFIGS4(X)
#endif
constexpr int pgemm4_cfg(int mt,int ng,int w,int bk,int pf,int minb){return mt|(ng<<4)|(w<<8)|((bk/32)<<12)|(pf<<16)|(minb<<20)|(1<<28);}

// Same contract as atlas_pgemm plus `live` (device int32, may be null = all M rows live).
extern "C" int atlas_pgemm4(const half*x,const uint4*w,const uint32_t*s,const float*inv,float g,int rec,
    const float*b,float*y,float*p,int m,int n,int k,int pk,int split,int fp4,int cfg,const int*live,cudaStream_t stream) {
  if(m<1||m>512||n<1||n%32||k<64||k%64||pk<k||pk%64||!x||!w||!inv||!y||
     (uintptr_t)x%16||(uintptr_t)w%16||(uintptr_t)live%4||split<1||split>4||(split>1&&!p)||
     (int64_t)n*pk>2147483647LL||(int64_t)m*n*split>2147483647LL) return cudaErrorInvalidValue;
  bool pinned=fp4 ? ((k==5120&&(n==17408||n==248320)&&split==1)||(k==17408&&n==5120&&split==2)) :
    ((k==6144&&n==5120&&split==1)||(k==5120&&(n==10240||n==6144||n==12288)&&split==1)||(k==5120&&n==1024&&split==4));
  if(!pinned||(fp4&&!s))return cudaErrorInvalidValue;
  #define SOL20_TRY4(F_,MT_,NG_,W_,BK_,PF_,MINB_) \
    if(bool(fp4)==F_&&cfg==pgemm4_cfg(MT_,NG_,W_,BK_,PF_,MINB_)) \
      return pgemm4_launch<F_,MT_,NG_,W_,BK_,PF_,MINB_>(x,w,s,inv,g,rec,b,y,p,m,n,k,pk,split,live,stream);
  SOL20_CONFIGS4(SOL20_TRY4)
  #undef SOL20_TRY4
  return cudaErrorInvalidValue;
}
// Configuration resolution for the gate (not full launch admission): out = {MT,NG,W,BK,PF,MINB}; 0 iff cfg matches.
extern "C" int atlas_pgemm4_describe(int fp4,int cfg,int*out) {
  if(!out)return cudaErrorInvalidValue;
  #define SOL20_DESC4(F_,MT_,NG_,W_,BK_,PF_,MINB_) \
    if(bool(fp4)==F_&&cfg==pgemm4_cfg(MT_,NG_,W_,BK_,PF_,MINB_)){int v[6]={MT_,NG_,W_,BK_,PF_,MINB_};for(int i=0;i<6;i++)out[i]=v[i];return 0;}
  SOL20_CONFIGS4(SOL20_DESC4)
  #undef SOL20_DESC4
  return cudaErrorInvalidValue;
}
extern "C" int atlas_pgemm4_inventory(int*out,int cap) {
  int count=0;
  #define SOL20_INV4(F_,MT_,NG_,W_,BK_,PF_,MINB_) {if(out&&count<cap){int v[7]={F_,MT_,NG_,W_,BK_,PF_,MINB_};for(int i=0;i<7;i++)out[7*count+i]=v[i];}count++;}
  SOL20_CONFIGS4(SOL20_INV4)
  #undef SOL20_INV4
  return count;
}
