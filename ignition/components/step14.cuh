// One packed-step pair (two uint4 weight words per lane) shared by the sol-14
// kernels. The decode and HMMA sequence is exactly sol-10's loop body
// (local_kernel.cuh lines 49-85); only the surrounding register pipeline moved.
template<int MT,bool F,int BK>
__device__ __forceinline__ void packed_pair(float (&c)[MT][8],uint4 raw0,uint4 raw1,uint32_t sb,
    half (*cur)[BK+SOL14_PAD],const int (&frow)[MT],int tile) {
  constexpr int Q=F?32:16;
#if SOL14_LDS128
  uint4 af[MT];
#endif
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
#if SOL14_LDS128
            // One 16-byte fragment load per (j,r) covers t=0 and t=1; per-row HMMA order (j, then t) is unchanged.
            if(t==0)af[r]=*reinterpret_cast<const uint4*>(&cur[frow[r]][tile+z*Q+j*8]);
            mma(c[r],t==0?af[r].x:af[r].z,t==0?af[r].y:af[r].w,b0,b1);
#else
            const uint32_t*a=reinterpret_cast<const uint32_t*>(&cur[frow[r]][tile+z*Q+j*8+t*4]);
            mma(c[r],a[0],a[1],b0,b1);
#endif
          }
        }
      }
    }else{
      #pragma unroll
      for(int t=0;t<4;t++) {
        uint32_t word=t==0?raw.x:t==1?raw.y:t==2?raw.z:raw.w,b0,b1;decode4_fp8(word,b0,b1);
        #pragma unroll
        for(int r=0;r<MT;r++) {
#if SOL14_LDS128
          if((t&1)==0)af[r]=*reinterpret_cast<const uint4*>(&cur[frow[r]][tile+z*Q+t*4]);
          mma(c[r],(t&1)==0?af[r].x:af[r].z,(t&1)==0?af[r].y:af[r].w,b0,b1);
#else
          const uint32_t*a=reinterpret_cast<const uint32_t*>(&cur[frow[r]][tile+z*Q+t*4]);
          mma(c[r],a[0],a[1],b0,b1);
#endif
        }
      }
    }
  }
}

// Four-deep register prefetch over one BK block. PIPE2 (two explicit register
// slots, loop body = two step pairs) removes the 9 register moves per pair and
// lets each load age two pairs before use; it needs BK >= 4Q. Otherwise the
// sol-10 move-based ring is used. Load addresses, load order and the consume
// order are identical in both forms, so the HMMA sequence per row is unchanged.
template<int MT,bool F,int BK,bool PIPE2>
__device__ __forceinline__ void block_steps(float (&c)[MT][8],half (*cur)[BK+SOL14_PAD],const int (&frow)[MT],
    const uint4*&wp,const uint32_t*&sp,int base,int end,
    uint4&next0,uint4&next1,uint4&next2,uint4&next3,uint32_t&nextscale,uint32_t&scale1) {
  constexpr int Q=F?32:16;
  if constexpr(PIPE2 && (BK%(4*Q)==0)) {
    #pragma unroll 1
    for(int tile=0;tile<BK;tile+=4*Q) {
      uint4 raw0=next0,raw1=next1;uint32_t sb=nextscale;wp+=64;
      if constexpr(F)sp+=32;
      if(base+tile+4*Q<end) {
        next0=__ldg(wp+64);next1=__ldg(wp+96);
        if constexpr(F)nextscale=__ldg(sp+32);
      }
      packed_pair<MT,F,BK>(c,raw0,raw1,sb,cur,frow,tile);
      raw0=next2;raw1=next3;sb=scale1;wp+=64;
      if constexpr(F)sp+=32;
      if(base+tile+6*Q<end) {
        next2=__ldg(wp+64);next3=__ldg(wp+96);
        if constexpr(F)scale1=__ldg(sp+32);
      }
      packed_pair<MT,F,BK>(c,raw0,raw1,sb,cur,frow,tile+2*Q);
    }
  } else {
    #pragma unroll 1
    for(int tile=0;tile<BK;tile+=2*Q) {
      uint4 raw0=next0,raw1=next1;uint32_t sb=nextscale;
      next0=next2;next1=next3;nextscale=scale1;wp+=64;
      if constexpr(F)sp+=32;
      if(base+tile+4*Q<end) {
        next2=__ldg(wp+64);next3=__ldg(wp+96);
        if constexpr(F)scale1=__ldg(sp+32);
      }
      packed_pair<MT,F,BK>(c,raw0,raw1,sb,cur,frow,tile);
    }
  }
}
