// Sol-11o shipped build (v3 code, PINGPONG=0 SPANQ=128 UNR_PV=2) / Sol-11s: tensor-core prefill attention for long cached context on sm_70 (native mma.sync.m8n8k4).
// Qwen3.8-27B full attention: 24 query heads, 4 KV heads (6 per group), head_dim 256, paged BF16 KV
// [physical_page,128 tokens,4 kv heads,256], FP32 Q [M,24,256], FP32 out [M,24,256], one sequence.
//
// Exactness scheme (Argon, PROMPT8 Q1): Q per-row power-of-two scale, K/V per-(page,kv head)
// power-of-two scale kept as metadata (sol11s_meta), BF16->FP16 conversion in registers during the
// tile load (exact for BF16 data whose in-page dynamic range is <= 2^32; smaller elements round with
// absolute error <= 2^-40 of the page maximum). Scores, online softmax, normalisation and all
// accumulators are FP32. P is split into hi/lo FP16 halves of P*2^15 (P in [0,1]), both multiplied by V
// on the tensor cores into the same FP32 accumulator (22 significant bits, error <= 2^-39 absolute).
// The V page scale 2^sigma is folded into the FP32 accumulator rescale of the online softmax.
// Deterministic: fixed tile/step/split order, no atomics on data.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#define FULL 0xffffffffu
#ifndef TK
#define TK 32
#endif
#ifndef QG
#define QG 1
#endif
#ifndef QSPLIT
#define QSPLIT 1
#endif
#ifndef SPANQ
#define SPANQ 128  // split spans in tokens (128 = v1 split policy; 32 regresses context-0 chunks via empty splits)
#endif
static_assert(128%SPANQ==0&&SPANQ%TK==0,"split spans must be whole tiles inside 128-token pages");
#ifndef RSPERM
#define RSPERM 1   // 1: QK quad pairs with q0=1 hold Q rows ii^2, q1=1 read K rows n^4 -> select-free reduce-scatter
#endif
#ifndef PINGPONG
#define PINGPONG 0  // 1 (experimental, NOT shipped): warps 4..7 one barrier window behind; slower and fails synccheck
#endif
#ifndef UNR_PV
#define UNR_PV 2
#endif
#define STR2(x) #x
#define STR(x) STR2(x)
#define PRAGMA_UNR _Pragma(STR(unroll UNR_PV))
namespace {
constexpr int NW=12, NT=NW*32, QTILE=16;
constexpr int PS=(TK*2+16);                 // bytes per P row (conflict-free LDS.128 over rows)
constexpr int PW=8*PS*2;                    // bytes per warp for hi+lo
constexpr int SKB=TK*512;                   // bytes of one K/V tile in smem
// smem: K double buffer, V triple buffer (role B reads V(t-1) while role A reads V(t) and V(t+1) is written),
// warp-private P hi/lo, per-warp row scale table. 97,664 B for TK=32 (V100 limit 98,304).
constexpr int SMEM=5*SKB+NW*PW+NW*8*4;
static_assert(SMEM<=98304,"smem");
constexpr int NCH=TK*32;                    // 16-byte chunks per tile
constexpr int NLD=(NCH+NT-1)/NT;            // loads per thread per tile
constexpr int NG=TK/8;                      // key groups of 8
// meta V entry of an all-zero V page (Argon review cond. 2): any scale converts it exactly, so it never moves
// the accumulator units and is ignored by the admission scan.
constexpr int ZPAGE=1000;
// Admission (Argon review cond. 1, bound derived here): acc <= 2^15 (P) * 2^15.994 (FP16 V) * 2^15.022 (keys per
// split) * 2^S, S = ev(page) - min ev(earlier pages); FP32-safe iff S <= 81. Flag S > 80.
constexpr int VRANGE=80;
constexpr int IMAX=0x7fffffff;

__device__ __forceinline__ void mma_rc(float (&c)[8],uint32_t a0,uint32_t a1,uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};\n"
  :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]):"r"(a0),"r"(a1),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void mma_rr(float (&c)[8],uint32_t a0,uint32_t a1,uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};\n"
  :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]):"r"(a0),"r"(a1),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint4 lds128(uint32_t a){uint4 r;asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(r.x),"=r"(r.y),"=r"(r.z),"=r"(r.w):"r"(a));return r;}
__device__ __forceinline__ void sts128(uint32_t a,uint4 v){asm volatile("st.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(a),"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w));}
// CTA barrier 1 (all 384 threads). Executed once per tile by every warp at a role-dependent program point.
__device__ __forceinline__ void bar_cta(){asm volatile("bar.sync 1, 384;":::"memory");}
__device__ __forceinline__ void sts32(uint32_t a,uint32_t v){asm volatile("st.shared.u32 [%0],%1;"::"r"(a),"r"(v));}
__device__ __forceinline__ float pow2i(int e){return __int_as_float((127+e)<<23);}   // -126<=e<=127
__device__ __forceinline__ float pow2c(int e){return e<-126?0.f:(e>127?INFINITY:pow2i(e));}  // clamped
// 16-byte chunk c (0..31) of tile row `row` -> physical chunk; permutation inside each 128-byte window.
__device__ __forceinline__ int swz(int c,int row){return (c&~7)|(((c&7)+2*(row&3)+(c>>3))&7);}
__device__ __forceinline__ uint32_t cvt2(uint32_t w,float sc){
 half2 h=__floats2half2_rn(__uint_as_float(w<<16)*sc,__uint_as_float(w&0xffff0000u)*sc);return *(uint32_t*)&h;}
__device__ __forceinline__ uint32_t pack2(half a,half b){half2 h=__halves2half2(a,b);return *(uint32_t*)&h;}

// meta layout: int[(phys*4+kh)*2+{0:K,1:V}] = clamp(15 - exponent(max|x|), -120, 120)
template<bool DEV> __global__ void meta_kernel(const uint16_t*__restrict__ k,const uint16_t*__restrict__ v,const int*__restrict__ pages,
 int*__restrict__ meta,int*__restrict__ status,int length,int pb,const int*__restrict__ plan){
 if constexpr(DEV){if(plan[6]||(int)blockIdx.x>=plan[4])return;length=plan[5];}   // Flare: npages/length from the device plan
 int lp=pb+blockIdx.x,kh=blockIdx.y,phys=pages[lp],tid=threadIdx.x;
 int valid=min(128,length-lp*128);if(valid<=0)return;
 uint32_t mk=0,mv=0,bad=0;
 for(int c=tid;c<valid*32;c+=blockDim.x){
  int tok=c>>5,ch=c&31;size_t off=(((size_t)phys*128+tok)*4+kh)*256+ch*8;
  uint4 a=*(const uint4*)(k+off),b=*(const uint4*)(v+off);
  uint32_t w[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
  for(int j=0;j<8;j++){
   uint32_t lo=w[j]&0x7fff,hi=(w[j]>>16)&0x7fff;
   if(j<4){mk=max(mk,max(lo,hi));}else{mv=max(mv,max(lo,hi));}
   bad|=((lo&0x7f80)==0x7f80)|((hi&0x7f80)==0x7f80);
  }
 }
 for(int s=16;s;s>>=1){mk=max(mk,__shfl_xor_sync(FULL,mk,s));mv=max(mv,__shfl_xor_sync(FULL,mv,s));bad|=__shfl_xor_sync(FULL,bad,s);}
 __shared__ uint32_t sk[8],sv[8],sb[8];
 if((tid&31)==0){sk[tid>>5]=mk;sv[tid>>5]=mv;sb[tid>>5]=bad;}
 __syncthreads();
 if(tid==0){
  for(int w=1;w<blockDim.x/32;w++){mk=max(mk,sk[w]);mv=max(mv,sv[w]);bad|=sb[w];}
  auto sc=[](uint32_t m){if(m==0)return 0;int e=(int)(m>>7)-127;if(e<-126)e=-126;int s=15-e;return s<-120?-120:(s>120?120:s);};
  meta[(phys*4+kh)*2]=sc(mk);meta[(phys*4+kh)*2+1]=mv?sc(mv):ZPAGE;
  if(bad)atomicOr(status,1);
 }
}
// Cumulative V-scale admission over logical pages 0..npages-1, per kv head: ev(lp) - min_{p<lp} ev(p) <= VRANGE,
// skipping all-zero V pages. One block of 128 threads: warp = kv head, lane = contiguous chunk of pages,
// exclusive prefix-min of chunk minima by shuffles. Splits restart their accumulator, so the global prefix is
// conservative for every split.
template<bool DEV> __global__ void status_kernel(const int*__restrict__ pages,const int*__restrict__ meta,int*__restrict__ status,int npages,const int*__restrict__ plan){
 if constexpr(DEV){if(plan[6]){if(threadIdx.x==0)atomicOr(status,4);return;}npages=plan[4];}   // Argon: status != 0 for an inadmissible plan
 const int kh=threadIdx.x>>5,lane=threadIdx.x&31,per=(npages+31)>>5,b=min(npages,lane*per),e=min(npages,b+per);
 auto ev=[&](int lp){int x=meta[(pages[lp]*4+kh)*2+1];return x==ZPAGE?IMAX:x;};
 int mn=IMAX;
 for(int lp=b;lp<e;lp++)mn=min(mn,ev(lp));
 for(int s=1;s<32;s<<=1){int t=__shfl_up_sync(FULL,mn,s);if(lane>=s)mn=min(mn,t);}
 int run=__shfl_up_sync(FULL,mn,1);if(lane==0)run=IMAX;
 int bad=0;
 for(int lp=b;lp<e;lp++){int x=ev(lp);if(x!=IMAX){if(run!=IMAX&&x-run>VRANGE)bad=2;run=min(run,x);}}
 if(__any_sync(FULL,bad)&&lane==0)atomicOr(status,2);
}
__global__ void reset_kernel(int*status){*status=0;}

// Tile copy. Chunk n of thread (warp,lane) is cc = tid + n*NT = (warp+12n)*32 + lane: tile row warp+12n, 16-byte
// chunk `lane` (NT = 12*32). The swizzle depends on row&3 = warp&3 only, so every chunk address is one per-tile base
// plus a compile-time offset (same data, same smem locations as the tid-based form).
__device__ __forceinline__ void load_tile(uint4 (&r)[NLD],const uint16_t*__restrict__ src,const int*__restrict__ pages,
 int start,int kh,int len,int warp,int lane){
 const int phys=pages[start>>7],base=start&127;
 const uint16_t*p=src+((((size_t)phys*128+base+warp)*4+kh)*256+lane*8);
 #pragma unroll
 for(int n=0;n<NLD;n++){
  const int row=warp+n*NW;uint4 z={0,0,0,0};
  if(row<TK&&start+row<len)z=*(const uint4*)(p+(size_t)n*NW*1024);
  r[n]=z;}
}
__device__ __forceinline__ void store_tile(const uint4 (&r)[NLD],uint32_t sbase,float sc,int warp,int lane){
 const uint32_t d0=sbase+warp*512+swz(lane,warp)*16;
 #pragma unroll
 for(int n=0;n<NLD;n++){
  if(warp+n*NW<TK){
   uint4 o;o.x=cvt2(r[n].x,sc);o.y=cvt2(r[n].y,sc);o.z=cvt2(r[n].z,sc);o.w=cvt2(r[n].w,sc);
   sts128(d0+n*NW*512,o);}
 }
}

template<bool DEV> __global__ __launch_bounds__(NT,1) void tc_attention(const float*__restrict__ q,const uint16_t*__restrict__ kc,
 const uint16_t*__restrict__ vc,const int*__restrict__ pages,const int*__restrict__ meta,
 float*__restrict__ partial,float*__restrict__ stats,int M,int context,int span,const int*__restrict__ plan){
 if constexpr(DEV){   // Flare: the graph is captured at token capacity; the real (M, context, span, splits) come from plan_kernel
  if(plan[6])return;
  M=plan[0];context=plan[1];span=plan[2];
  if((int)blockIdx.x*QTILE>=M||(int)blockIdx.z>=plan[3])return;   // whole CTA, before any barrier
 }
 extern __shared__ __align__(128) unsigned char smem_raw[];
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,hd=warp>>1,mh=warp&1,kh=blockIdx.y,split=blockIdx.z;
 const int first=blockIdx.x*QTILE,len=context+M;
 const int qd=(lane>>2)&3,hh=lane>>4,ii=lane&3,q0=(lane>>2)&1,q1=(lane>>3)&1,jj=lane&3;
 const uint32_t sK0=(uint32_t)__cvta_generic_to_shared(smem_raw),sV0=sK0+2*SKB;
 const uint32_t pw=sV0+3*SKB+warp*PW,sqt=sV0+3*SKB+NW*PW+warp*32;
 const bool roleB=PINGPONG&&((warp>>2)&1);          // 2 role-A + 1 role-B warp on every SM sub-partition
 // ---- Q registers: lane holds row rho=ii+4*hh, d-quarter qd, 16 k-steps x 2 regs ----
 uint32_t Qa[32];
 {
  const int rq=(RSPERM&&q0)?(ii^2):ii;              // A-fragment row of this lane (true row rq+4*hh)
  int pos=first+8*mh+rq+4*hh;bool valid=pos<M;
  const float4*qp=(const float4*)(q+((size_t)(pos*24+kh*6+hd))*256+64*qd);
  float4 x[16];float mx=0.f;
  #pragma unroll
  for(int s=0;s<16;s++){x[s]=valid?qp[s]:make_float4(0,0,0,0);mx=fmaxf(mx,fmaxf(fmaxf(fabsf(x[s].x),fabsf(x[s].y)),fmaxf(fabsf(x[s].z),fabsf(x[s].w))));}
  mx=fmaxf(mx,__shfl_xor_sync(FULL,mx,RSPERM?6:4));mx=fmaxf(mx,__shfl_xor_sync(FULL,mx,8));   // the 4 d-quarters of one row
  int bits=__float_as_int(mx),sq=0;
  if(bits!=0){int e=(bits>>23)-127;if(e<-126)e=-126;sq=14-e;sq=sq<-120?-120:(sq>120?120:sq);}   // 14: FP32 (non-BF16) Q must not round up past FP16 max
  float sc=pow2i(sq);
  #pragma unroll
  for(int s=0;s<16;s++){
   half2 a=__floats2half2_rn(x[s].x*sc,x[s].y*sc),b=__floats2half2_rn(x[s].z*sc,x[s].w*sc);
   Qa[2*s]=*(uint32_t*)&a;Qa[2*s+1]=*(uint32_t*)&b;}
  if(qd==0)sts32(sqt+(rq+4*hh)*4,(uint32_t)sq);
 }
 __syncwarp();
 const int rowS=(jj&1)+2*q0+4*hh;                 // softmax row owned by this lane (0..7)
 int sqS;{uint32_t t;asm volatile("ld.shared.u32 %0,[%1];":"=r"(t):"r"(sqt+rowS*4));sqS=(int)t;}
 const float c1=pow2i(-(sqS+2));
 const int posS=first+8*mh+rowS;
 const int limit=min(context+first+QTILE,len);
 const int kb=split*span,ke=min(limit,kb+span);
 float acc[8][8];
 #pragma unroll
 for(int g=0;g<8;g++)
  #pragma unroll
  for(int e=0;e<8;e++)acc[g][e]=0.f;
 // V-scale state per split: sig = scale of the latest non-zero V page (accumulator units 2^(15+sig)), emin = smallest
 // scale (largest |V|) seen, for the overflow-safe combine. All-zero V pages (ZPAGE) never set them (review changes 1-2).
 float m=-INFINITY,l=0.f;int sig=ZPAGE,emin=ZPAGE;
 uint4 stg[NLD];
 int ek=0,ev=0;                                   // scales of the tile being processed (ev raw: may be ZPAGE)
 if(kb<ke){                                       // prologue: tile kb -> K buffer 0, V buffer 0
  const int ph=pages[kb>>7];ek=meta[(ph*4+kh)*2];ev=meta[(ph*4+kh)*2+1];
  load_tile(stg,kc,pages,kb,kh,len,warp,lane);store_tile(stg,sK0,pow2i(ek),warp,lane);
  load_tile(stg,vc,pages,kb,kh,len,warp,lane);store_tile(stg,sV0,pow2i(ev==ZPAGE?0:ev),warp,lane);
  bar_cta();
 }
 int kbi=0,vbi=0;
 // One barrier per tile. Barrier window t (between the t-th and (t+1)-th bar_cta of every warp):
 //   role A: QK(t) softmax(t) PV(t)        role B: PV(t-1) QK(t) softmax(t)
 // both roles store K(t+1) -> K[(t+1)&1] (last read by QK(t-1), window t-1) and V(t+1) -> V[(t+1)%3] (last read by
 // PV(t-2): role B in window t-1). Per row the (m, l, acc) update sequence is unchanged, so results equal v2 bit for bit.
 for(int start=kb;start<ke;start+=TK){
  const bool nxt=start+TK<ke;
  const uint32_t sKc=sK0+kbi*SKB,sKn=sK0+(kbi^1)*SKB,sVc=sV0+vbi*SKB,sVn=sV0+(vbi==2?0:vbi+1)*SKB;
  int nek=0,nevr=0;
  if(nxt){
   load_tile(stg,kc,pages,start+TK,kh,len,warp,lane);
   const int ph=pages[(start+TK)>>7];nek=meta[(ph*4+kh)*2];nevr=meta[(ph*4+kh)*2+1];
  }
  float y[NG][2];
  #pragma unroll
  for(int g=0;g<NG;g+=QG){
   float c[QG][8],cb[QG][8];
   #pragma unroll
   for(int u=0;u<QG;u++)
    #pragma unroll
    for(int e=0;e<8;e++){c[u][e]=0.f;cb[u][e]=0.f;}
   #pragma unroll
   for(int s2=0;s2<8;s2++){
    #pragma unroll
    for(int u=0;u<QG;u++){
     int row=(g+u)*8+ii+4*(RSPERM?(hh^q1):hh);
     uint4 b=lds128(sKc+row*512+swz(8*qd+s2,row)*16);
     mma_rc(c[u],Qa[4*s2],Qa[4*s2+1],b.x,b.y);
     if(QSPLIT==2)mma_rc(cb[u],Qa[4*s2+2],Qa[4*s2+3],b.z,b.w);
     else mma_rc(c[u],Qa[4*s2+2],Qa[4*s2+3],b.z,b.w);
    }
   }
   if(QSPLIT==2){
    #pragma unroll
    for(int u=0;u<QG;u++)
     #pragma unroll
     for(int e=0;e<8;e++)c[u][e]+=cb[u][e];
   }
   #pragma unroll
   for(int u=0;u<QG;u++){
    float s4[4];
    #pragma unroll
    for(int w=0;w<4;w++){
     #if RSPERM
     s4[w]=c[u][w]+__shfl_xor_sync(FULL,c[u][4+w],8);    // same operands and order as the select form
     #else
     float send=q1?c[u][w]:c[u][4+w],keep=q1?c[u][4+w]:c[u][w];s4[w]=keep+__shfl_xor_sync(FULL,send,8);
     #endif
    }
    #pragma unroll
    for(int w=0;w<2;w++){
     #if RSPERM
     y[g+u][w]=s4[w]+__shfl_xor_sync(FULL,s4[2+w],4);
     #else
     float send=q0?s4[w]:s4[2+w],keep=q0?s4[2+w]:s4[w];y[g+u][w]=keep+__shfl_xor_sync(FULL,send,4);
     #endif
    }
   }
  }
  if(nxt){store_tile(stg,sKn,pow2i(nek),warp,lane);load_tile(stg,vc,pages,start+TK,kh,len,warp,lane);}
  // ---- online softmax (row = rowS; this lane holds keys 8g+2*(jj>>1)+4*q1+{0,1}) ----
  const float c2=pow2i(-(ek+2));
  const bool full=(start+TK<=ke)&&(start+TK-1<=context+first);
  float tmax=-INFINITY;
  #pragma unroll
  for(int g=0;g<NG;g++)
   #pragma unroll
   for(int w=0;w<2;w++){
    float s=(y[g][w]*c1)*c2;
    if(!full){int kidx=start+8*g+2*(jj>>1)+4*q1+w;if(!(kidx<ke&&kidx<=context+posS&&posS<M))s=-INFINITY;}
    y[g][w]=s;tmax=fmaxf(tmax,s);}
  tmax=fmaxf(tmax,__shfl_xor_sync(FULL,tmax,2));tmax=fmaxf(tmax,__shfl_xor_sync(FULL,tmax,8));
  float mn=fmaxf(m,tmax);
  float alpha=(m==-INFINITY)?1.f:expf(m-mn);
  if(mn==-INFINITY)alpha=1.f;
  const float mns=(mn==-INFINITY)?0.f:mn;          // all-masked row: every y is -inf, expf(-inf)=+0 (as the old guard)
  float tsum=0.f;
  #pragma unroll
  for(int g=0;g<NG;g++){
   float p0=expf(y[g][0]-mns),p1=expf(y[g][1]-mns);
   tsum+=p0+p1;
   float v0=p0*32768.f,v1=p1*32768.f;half h0=__float2half_rn(v0),h1=__float2half_rn(v1);
   half l0=__float2half_rn(v0-__half2float(h0)),l1=__float2half_rn(v1-__half2float(h1));
   uint32_t off=rowS*PS+(8*g+2*(jj>>1)+4*q1)*2;
   sts32(pw+off,pack2(h0,h1));sts32(pw+8*PS+off,pack2(l0,l1));
  }
  l=l*alpha+tsum;m=mn;
  // V scale fold: acc is kept in units 2^(15+sig)
  float f=1.f;
  if(ev!=ZPAGE){
   if(sig==ZPAGE){sig=ev;emin=ev;}               // first non-zero V page of the split: acc is still +0, no rescale
   else{f=(ev==sig)?1.f:pow2c(ev-sig);sig=ev;emin=min(emin,ev);}
  }
  float aoth=__shfl_xor_sync(FULL,alpha,4);
  float al0=(q0?aoth:alpha)*f,al1=(q0?alpha:aoth)*f;
  if(nxt)store_tile(stg,sVn,pow2i(nevr!=ZPAGE?nevr:(sig==ZPAGE?0:sig)),warp,lane);   // same conversion rule as v2
  if(__any_sync(FULL,(al0!=1.f)|(al1!=1.f))){
   #pragma unroll
   for(int g=0;g<8;g++){
    #pragma unroll
    for(int e=0;e<8;e++)acc[g][e]*=((e>>1)&1)?al1:al0;}
  }
  if(roleB)bar_cta();
  __syncwarp();
  // ---- O += (Phi+Plo) V ----
  PRAGMA_UNR
  for(int t8=0;t8<NG;t8++){
   int rho=ii+4*hh;
   uint4 ah=lds128(pw+rho*PS+16*t8),al=lds128(pw+8*PS+rho*PS+16*t8);
   #pragma unroll
   for(int sub=0;sub<2;sub++){
    int st=2*t8+sub,row=4*st+ii;
    uint32_t h0=sub?ah.z:ah.x,h1=sub?ah.w:ah.y,l0=sub?al.z:al.x,l1=sub?al.w:al.y;
    uint4 b[4];
    #pragma unroll
    for(int G=0;G<4;G++)b[G]=lds128(sVc+row*512+swz(8*G+4*hh+qd,row)*16);
    #pragma unroll
    for(int G=0;G<4;G++){mma_rr(acc[2*G],h0,h1,b[G].x,b[G].y);mma_rr(acc[2*G+1],h0,h1,b[G].z,b[G].w);}
    #pragma unroll
    for(int G=0;G<4;G++){mma_rr(acc[2*G],l0,l1,b[G].x,b[G].y);mma_rr(acc[2*G+1],l0,l1,b[G].z,b[G].w);}
   }
  }
  if(!roleB)bar_cta();
  __syncwarp();                                    // role B rewrites its P buffer in softmax(t+1) before any barrier
  ek=nek;ev=nevr;kbi^=1;vbi=(vbi==2?0:vbi+1);
 }
 // ---- epilogue: unnormalised partial (true scale), row max and denominator ----
 float lsum=l;lsum+=__shfl_xor_sync(FULL,lsum,2);lsum+=__shfl_xor_sync(FULL,lsum,8);
 // partial = acc (units 2^(15+sig)); stats = {m, l, e=-15-sig, ebig=-15-emin}; |acc*2^e| <= 2^(46.02+ebig)
 if(sig==ZPAGE){sig=0;emin=0;}
 const float escale=(float)(-15-sig),ebig=(float)(-15-emin);
 #pragma unroll
 for(int b=0;b<2;b++){
  int row=(jj&1)+2*b+4*hh,pos=first+8*mh+row;
  if(pos>=M)continue;
  int qh=kh*6+hd;
  #pragma unroll
  for(int G=0;G<4;G++)
   #pragma unroll
   for(int u=0;u<2;u++)
    #pragma unroll
    for(int h2=0;h2<2;h2++){
     // C element e = 4*h2 + 2*b + w : d = 64G + 32*h2 + 8*qd + 4u + w + 2*(jj>>1)
     float2 o;o.x=acc[2*G+u][4*h2+2*b];o.y=acc[2*G+u][4*h2+2*b+1];
     int d=64*G+32*h2+8*qd+4*u+2*(jj>>1);
     *(float2*)(partial+(((size_t)(split*M+pos))*24+qh)*256+d)=o;}
 }
 if(q1==0&&(jj>>1)==0){
  int pos=posS;
  if(pos<M){int qh=kh*6+hd;size_t o=((size_t)(split*M+pos)*24+qh)*4;
   stats[o]=m;stats[o+1]=lsum;stats[o+2]=escale;stats[o+3]=ebig;}
 }
}
// x*2^e for e in [-240,240] with a single rounding (two exact power-of-two steps when 2^e is not a normal float)
__device__ __forceinline__ float ldexp2(float x,int e){
 if(e>=-126&&e<=127)return x*pow2i(e);
 return e<0?(x*pow2i(e+64))*pow2i(-64):(x*pow2i(e-64))*pow2i(64);}
// Fixed split order (deterministic). Split s holds acc_s (units 2^-e_s), m_s, l_s, ebig_s = -15 - min ev over its
// pages, with |acc_s*2^e_s| <= 2^(46.02+ebig_s). With D = max(0, max_s ebig_s - 76) (review change 1):
// out = (sum_s w_s * acc_s*2^(e_s-D) / sum_s w_s l_s) * 2^D; every term <= 2^122.02, the 8-split sum <= 2^125.02.
// D=0 (every attended V page has max|V| < 2^107) is bit-identical to scaling each partial by 2^e_s first.
template<bool DEV> __global__ void combine_kernel(const float*__restrict__ partial,const float*__restrict__ stats,float*__restrict__ out,
 const int*__restrict__ status,int M,int splits,const int*__restrict__ plan){
 int pos=blockIdx.x,qh=blockIdx.y,d=threadIdx.x;
 if constexpr(DEV){
  if(plan[6]){out[((size_t)pos*24+qh)*256+d]=__int_as_float(0x7fc00000);return;}   // inadmissible (M, context): loud NaN
  M=plan[0];splits=plan[3];
  if(pos>=M){out[((size_t)pos*24+qh)*256+d]=0.f;return;}                            // padded capacity rows are defined
 }
 if(*status){out[((size_t)pos*24+qh)*256+d]=__int_as_float(0x7fc00000);return;}
 float mx=-INFINITY;int E=-100000;
 for(int s=0;s<splits;s++){size_t o=((size_t)(s*M+pos)*24+qh)*4;float ms=stats[o];
  mx=fmaxf(mx,ms);if(ms!=-INFINITY)E=max(E,(int)stats[o+3]);}
 const int D=max(0,E-76);
 float den=0.f,res=0.f;
 for(int s=0;s<splits;s++){
  size_t o=((size_t)(s*M+pos)*24+qh);float ms=stats[o*4];
  float w=(ms==-INFINITY)?0.f:expf(ms-mx);
  den+=w*stats[o*4+1];res+=w*ldexp2(partial[o*256+d],(int)stats[o*4+2]-D);}
 out[((size_t)pos*24+qh)*256+d]=ldexp2(res/den,D);
}
}  // namespace

constexpr size_t PART_FLOATS_MAX=(size_t)8*512*24*256;
// Admission domain (astra-7): the real-world package serves 34,816 tokens per request (page table 272 pages); the split
// plan stays exact in double (nq <= 276 spans, waves <= 13, waves*span <= 3,588 << 2^53), the workspace does not depend
// on the context, and page metadata/status kernels index pages by the page table. 276 = ceil((34816 + 512) / 128).
constexpr int MAX_CONTEXT_TOKENS=34816, MAX_PAGE_TABLE=276;
// Split plan shared by the host entry (sol11s_attention) and the device plan_kernel (Argon review, required change 1):
// the split count with the least wave-quantised time, then the span in tokens. Exact integer values in double.
__host__ __device__ __forceinline__ void sol11s_split_plan(int M,int context,int splits_req,int&splits,int&span){
 const int len=context+M,nq=(len+SPANQ-1)/SPANQ,qt=(M+QTILE-1)/QTILE,base=qt*4;
 splits=splits_req;
 if(splits<=0){
  double best=1e30;splits=1;
  for(int s=1;s<=8;s++){int sp=((nq+s-1)/s);int waves=(base*s+79)/80;double t=waves*(double)sp;if(t<best-1e-9){best=t;splits=s;}}
 }
 if(splits>8)splits=8;
 span=((nq+splits-1)/splits)*SPANQ;
}
extern "C" size_t sol11s_attention_workspace(){return 16+PART_FLOATS_MAX*4+(size_t)8*512*24*4*4;}
extern "C" int sol11s_attention_init(void*ws,cudaStream_t s){
 cudaError_t e=cudaFuncSetAttribute(tc_attention<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
 if(e==cudaSuccess)e=cudaFuncSetAttribute(tc_attention<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);
 if(e==cudaSuccess)e=cudaMemsetAsync(ws,0,16,s);   // status word zeroed (Argon review cond. 3)
 return int(e);
}
// Update per-(physical page, kv head) scales for logical pages [page_begin,page_end) and re-validate.
extern "C" int sol11s_meta(const uint16_t*k,const uint16_t*v,const int*pages,int*meta,void*ws,int length,
 int page_begin,int page_end,cudaStream_t st){
 int*status=(int*)ws;
 if(page_begin==0)reset_kernel<<<1,1,0,st>>>(status);
 if(page_end>page_begin)meta_kernel<false><<<dim3(page_end-page_begin,4),256,0,st>>>(k,v,pages,meta,status,length,page_begin,nullptr);
 status_kernel<false><<<1,128,0,st>>>(pages,meta,status,(length+127)/128,nullptr);
 return int(cudaGetLastError());
}
extern "C" int sol11s_attention(const float*q,const uint16_t*k,const uint16_t*v,const int*pages,const int*meta,
 float*out,void*ws,int M,int context,int page_tokens,int splits_req,cudaStream_t st){
 if(M<1||M>512||context<0||context>MAX_CONTEXT_TOKENS||page_tokens!=128)return int(cudaErrorInvalidValue);
 const int qt=(M+QTILE-1)/QTILE;
 int splits,span;sol11s_split_plan(M,context,splits_req,splits,span);
 int*status=(int*)ws;float*partial=(float*)((char*)ws+16);float*stats=partial+PART_FLOATS_MAX;
 tc_attention<false><<<dim3(qt,4,splits),NT,SMEM,st>>>(q,k,v,pages,meta,partial,stats,M,context,span,nullptr);
 combine_kernel<false><<<dim3(M,24),256,0,st>>>(partial,stats,out,status,M,splits,nullptr);
 return int(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------
// Flare integration entry points (astra-6, Claude Fable 5.1). Flare replays captured CUDA graphs, so M, context
// and the split plan must come from device memory and every grid is sized for the graph's token capacity.
// plan_kernel reproduces the host split selection above bit for bit (same integer/double arithmetic) from the
// request metadata the prefill graph already owns: lengths[0] = context + M, cu[1] - cu[0] = M (one request per
// prefill graph). plan = int32[8]: {M, context, span, splits, npages, len, invalid, 0}.
// ---------------------------------------------------------------------------------------------------------
namespace {
__global__ void plan_kernel(const int*__restrict__ lengths,const int*__restrict__ cu,int*__restrict__ plan,int capacity,int max_pages){
 const int M=cu[1]-cu[0],len=lengths[0],context=len-M;
 const bool ok=M>=1&&M<=capacity&&M<=512&&context>=0&&context<=MAX_CONTEXT_TOKENS&&(len+127)/128<=max_pages;
 if(!ok){plan[0]=0;plan[1]=0;plan[2]=0;plan[3]=0;plan[4]=0;plan[5]=0;plan[6]=1;plan[7]=0;return;}
 int splits,span;sol11s_split_plan(M,context,0,splits,span);
 plan[0]=M;plan[1]=context;plan[2]=span;plan[3]=splits;plan[4]=(len+127)/128;plan[5]=len;plan[6]=0;plan[7]=0;
}
}  // namespace
// Derive the per-chunk plan from device request metadata. capacity = graph token capacity (<= 512).
extern "C" int sol11s_plan(const int*lengths,const int*cu,int*plan,int capacity,int max_pages,cudaStream_t st){
 if(capacity<1||capacity>512||max_pages<1||max_pages>MAX_PAGE_TABLE)return int(cudaErrorInvalidValue);
 plan_kernel<<<1,1,0,st>>>(lengths,cu,plan,capacity,max_pages);
 return int(cudaGetLastError());
}
// Full metadata recompute for every logical page of the request (page_begin = 0: resets the sticky status), with
// the page count taken from the plan; blocks beyond it exit. Metadata therefore never depends on history
// (restore, snapshot copies, eviction and page reuse change page contents without an append).
extern "C" int sol11s_meta_dev(const uint16_t*k,const uint16_t*v,const int*pages,int*meta,void*ws,const int*plan,int max_pages,cudaStream_t st){
 if(max_pages<1||max_pages>MAX_PAGE_TABLE)return int(cudaErrorInvalidValue);
 int*status=(int*)ws;
 reset_kernel<<<1,1,0,st>>>(status);
 meta_kernel<true><<<dim3(max_pages,4),256,0,st>>>(k,v,pages,meta,status,0,0,plan);
 status_kernel<true><<<1,128,0,st>>>(pages,meta,status,0,plan);
 return int(cudaGetLastError());
}
// Capacity-launched attention: grid (ceil(capacity/16), 4, 8); CTAs outside the plan exit; combine writes zeros
// for rows in [M, capacity), NaN for an inadmissible plan or a non-zero status word.
extern "C" int sol11s_attention_dev(const float*q,const uint16_t*k,const uint16_t*v,const int*pages,const int*meta,
 float*out,void*ws,const int*plan,int capacity,int page_tokens,cudaStream_t st){
 if(capacity<1||capacity>512||page_tokens!=128)return int(cudaErrorInvalidValue);
 int*status=(int*)ws;float*partial=(float*)((char*)ws+16);float*stats=partial+PART_FLOATS_MAX;
 tc_attention<true><<<dim3((capacity+QTILE-1)/QTILE,4,8),NT,SMEM,st>>>(q,k,v,pages,meta,partial,stats,0,0,0,plan);
 combine_kernel<true><<<dim3(capacity,24),256,0,st>>>(partial,stats,out,status,0,0,plan);
 return int(cudaGetLastError());
}
