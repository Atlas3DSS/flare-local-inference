// Sol-21 r2: sol-13c with one change: load_block issues predicated loads (zero for positions >= len set by register moves
// before the load) instead of loading and zero-filling with a select. The select made every warp wait for HBM at the
// load site (ncu: 11 % of sol-13c's stall samples at those SELs), so neither the V load (meant to overlap QK) nor the
// next tile's K load (meant to overlap softmax and PV) overlapped anything. Same words, same zeros: bit-identical.
// Sol-13c: Sol-13b S1L with the Flare row contract: verify row j of a multi-row launch produces the SAME BITS as a
// single-row launch (R = 1) with length ctx0 + j + 1 on the same Q row and K/V. Changes vs sol-13b (NUMERICS.md,
// "Row semantics"): (1) row passes on boundary tiles: a token that the shared conversion flags (outside the window of
// the positions < ctx0 + 1) no longer takes an FP32 special path; the block is re-converted (reconvert) with the voter
// set of the first row that sees that token, and those rows' scores / PV are recomputed on the tensor (or raw) path,
// exactly as their single-row runs compute them; (2) per-row rung: row j uses ladder_P(ctx0 + j); an agent whose rows
// straddle a cutoff gets a second partition set. Row 0 and every R = 1 launch are bit-identical to sol-13b.
// Sol-13b: long-context verification attention on sm_70 tensor cores (mma.m8n8k4). PROMOTED build S1L = v2b + S1 + ladder:
// + append-time exact K/V encoding (variant +256): committed 32x64 blocks are stored as exact scaled FP16 by
// sol13b_encode_committed; for those tiles the hot loop skips the max pass and conversion (D from metadata).
// Tail / not-yet-encoded / inadmissible blocks take the unchanged in-register path. Bit-identical to v2b.
// Design follows kevin/PROMPT8.answer.md (Argon) with the PROMPT9 review applied:
//  * one CTA = (fixed 512-token partition, agent, KV head); 4 warps, warp w owns head-dim slice [64w,64w+64)
//  * K/V stay BF16 in HBM. Each warp loads its own 32x64 K (then V) block into registers, finds the block
//    max exponent with a warp reduction, converts BF16 -> power-of-two-scaled FP16 with integer ops (exact
//    for every admitted value), and stores the block to warp-private swizzled shared memory. No codec pass.
//  * Q is split per (row, 64-dim slice) into hi+lo FP16 with its own power-of-two scale; the A fragments
//    live in registers for the whole partition. S partials of the 4 slices are unscaled per (row, slice)
//    and summed in a fixed order through shared memory; FP32 online softmax; P split into hi+lo FP16.
//  * O[rows,64] accumulates in FP32 registers per warp; partitions are merged in a fixed per-row order.
//  * Blocks that fail admission (span > 2^30 within a 32x64 block, non-finite) and query slices whose
//    two-half representation is off by more than 2^-20 componentwise take an FP32 SIMT path for that
//    block / row only. No row ever selects another row's arithmetic.
#include "sm70_decode_attention.h"   // = sol-21/attention.h = sol-13c/attention.h verbatim (astra-19: the only edit to sol-21/attention.cu)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <math.h>
#include <algorithm>

namespace {
constexpr int TILE = 32;
constexpr int MAXROWS = 20;           // 4 agents x 5 rows
// Partition ladder (variant +128), chosen per agent from its own committed context ctx0 = length - R only (row-invariant).
// Cutoffs subtract MAX_R = 5 so length = ctx0 + R never exceeds 10*P: <= 10 splits x 16 groups = one wave of 160 CTAs.
constexpr int MAX_R = 5, LADDER_SPLITS = 10;
__host__ __device__ __forceinline__ int ladder_P(int ctx0, int plong) {
  return ctx0 <= 10 * 128 - MAX_R ? 128 : (ctx0 <= 10 * 256 - MAX_R ? 256 : plong);  // ctx0 <= 1275 / <= 2555
}
// sol-13c per-row rung: row j of an agent (committed context ctx0, R rows) uses ladder_P(ctx0 + j), the rung its own
// single-row decode uses. Rows [0, jb) share rung PA, rows [jb, R) rung PB (jb = R: one rung; R <= 5 crosses at most one
// cutoff). Straddling agent: CTAs x < nA compute rung PA (nA covers the longest PA row, visible ctx0 + jb) for rows
// [0, jb) into slots x; CTAs x in [nA, nA + ceil(len/PB)) compute rung PB for rows [jb, R) into slots x - nA. The two
// rungs share slot numbers (disjoint rho = 6j + h rows, Argon PROMPT26), so the merge reads slots 0..chunks-1 for every
// row and the workspace is unchanged; only the grid grows (15 instead of 10 x-blocks at a straddle).
struct Rung { int PA, PB, jb, nA; };
__host__ __device__ __forceinline__ Rung rung_plan(int ctx0, int R, int plong) {
  Rung g;
  g.PA = g.PB = ladder_P(ctx0, plong);
  g.jb = R;
  for (int j = 1; j < R; ++j)
    if (g.jb == R && ladder_P(ctx0 + j, plong) != g.PA) { g.jb = j; g.PB = ladder_P(ctx0 + j, plong); }
  g.nA = g.jb < R ? (ctx0 + g.jb + g.PA - 1) / g.PA : 0;
  return g;
}
// S1 metadata: one byte per (page, 32-token tile, KV head, 64-dim slice, K|V): 0..224 = D (block holds exact
// bf16*2^(112-D) FP16), META_UNENC = raw BF16 not yet encoded, META_RAW = raw BF16 kept (inadmissible block).
constexpr uint8_t META_UNENC = 0xFE, META_RAW = 0xFF;
// sol-13c row passes on boundary tiles. 0 (default, "lazy"): a row gets its own conversion only when the conversion
// that would otherwise serve it flagged a token the row sees; by the NUMERICS.md lemma every other row's conversion
// already equals its single-row run's. 1 ("eager", Argon PROMPT26 Q2 literally): every row j whose visible set in the
// tile differs from the previous row's gets its own pass, loading only positions < ctx0 + j + 1 (others zero) and
// converting with all of them voting = load_block(len = vis_j) + convert_block<false>. Both must give identical bits
// (qualification compares the two builds); lazy does no extra work unless a token is flagged.
#ifndef ROWPASS_EAGER
#define ROWPASS_EAGER 0
#endif
// Rows rho < return value are final with the conversion whose flags are `spec` (first row to re-pass, times 6; 32 = all).
__device__ __forceinline__ int pass_end(unsigned spec, int jlo, int start, int ctx0, int R) {
#if ROWPASS_EAGER
  int jn = spec ? start + __ffs(spec) - 1 - ctx0 : R;
  const int je = max(jlo + 1, start - ctx0);  // next row that sees one more token of this tile
  if (je < R && ctx0 + je < start + 32) jn = min(jn, je);
  return jn < R ? 6 * jn : 32;
#else
  return spec ? 6 * (start + __ffs(spec) - 1 - ctx0) : 32;
#endif
}
__host__ __device__ __forceinline__ size_t meta_index(int page, int tile, int head, int slice, int isv) {
  return ((((size_t(page) * 4 + tile) * 4 + head) * 4 + slice) << 1) | isv;
}
constexpr int SM_KV = 0;              // 4 warps x 4096 B  K or V block (warp private)
constexpr int SM_SP = 16384;          // 4 warps x 32 rows x 32 tokens FP32 score partials
constexpr int SM_P = 32768;           // 32 rows x 144 B : per 4-token group {hi[4], lo[4]} FP16
constexpr int SM_ST = SM_P + 32 * 144;// 32 rows x {alpha, 1/den}
constexpr int SM_VIS = SM_ST + 32 * 8;     // 32 rows: visible length (0 = padded row)
constexpr int SM_EQ = SM_VIS + 32 * 4;      // 4 warps x 32 rows: query slice exponents
constexpr int SM_PG = SM_EQ + 4 * 32 * 4;     // partition page table (<= 32 pages)
constexpr int SM_BYTES = SM_PG + 32 * 4;

// Branchless x * 2^e (Argon review): exact whenever the result is a normal number; e in [-252, 254].
__device__ __forceinline__ float scale2(float x, int e) {
  int e1 = max(min(e, 127), -126);
  x = __fmul_rn(x, __uint_as_float(unsigned(e1 + 127) << 23));
  int e2 = max(min(e - e1, 127), -126);
  return __fmul_rn(x, __uint_as_float(unsigned(e2 + 127) << 23));
}
__device__ __forceinline__ void mma_rc(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
               "{%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_rr(float (&c)[8], uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
               "{%0,%1,%2,%3,%4,%5,%6,%7};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t sel) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %1, %2;" : "=r"(d) : "r"(a), "r"(sel));
  return d;
}
__device__ __forceinline__ uint4 ld_stream(const uint16_t* p) {
  uint4 v;
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
// sol-21: predicated streaming load: returns the 16 bytes at p if in, else zero; no select on the loaded value, so the
// warp waits for the data at its first use, not here.
__device__ __forceinline__ uint4 ld_stream_if(const uint16_t* p, bool in) {
  uint4 v;
  asm volatile("{\n\t.reg .pred q;\n\tsetp.ne.b32 q, %5, 0;\n\tmov.b32 %0, 0;\n\tmov.b32 %1, 0;\n\tmov.b32 %2, 0;\n\tmov.b32 %3, 0;\n\t"
               "@q ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];\n\t}"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "r"(int(in)));
  return v;
}
__device__ __forceinline__ float bf2f(uint32_t bits16) { return __uint_as_float(bits16 << 16); }
__device__ __forceinline__ uint32_t pack2(__half a, __half b) {
  return uint32_t(__half_as_ushort(a)) | (uint32_t(__half_as_ushort(b)) << 16);
}
// Score-partial swizzle (pairs of FP32): conflict-free C-fragment STS.64 and softmax LDS.128.
__device__ __forceinline__ int sp_h(int rho) { return (((rho >> 3) & 3) << 1) | ((rho & 1) << 3); }

struct Params {
  const float* q; const uint16_t* k; const uint16_t* v;
  const int32_t* pages; const int32_t* lens; const int32_t* cu;
  float* partO; float* partS; uint16_t* dbg_h; int32_t* dbg_e; const uint8_t* meta;
  int pagecols, P, debug, shortp, enc;
};

// Loads this warp's 32-token x 64-dim BF16 block (8 x 16 B per lane, 4 full 128-B lines per instruction).
// Positions >= len are zero filled (never read from memory).
// sol-21: branch-free predicated loads (ld_stream_if): slots >= len are not loaded and read as zero (sol-13c loaded them
// and zeroed them with a select, which stalled the warp on HBM right here). Same words, same zeros.
// P1: the page id comes from the CTA's shared page table (no dependent global load per block).
__device__ __forceinline__ void load_block(uint4 (&b)[8], const uint16_t* base, int page, int start, int len,
                                           int head, int warp, int lane) {
  const int slot0 = start & 127;
  const uint16_t* p0 = base + ((size_t(page) * 128 + slot0 + (lane >> 3)) * 4 + head) * 256 + 64 * warp + 8 * (lane & 7);
#pragma unroll
  for (int i = 0; i < 8; ++i) b[i] = ld_stream_if(p0 + size_t(4 * i) * 4 * 256, start + 4 * i + (lane >> 3) < len);
}
__device__ __forceinline__ void load_block_page(uint4 (&b)[8], const uint16_t* base, int page, int start, int len,
                                                int head, int warp, int lane) {
  const int slot0 = start & 127;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int t = 4 * i + (lane >> 3);
    b[i] = make_uint4(0, 0, 0, 0);
    if (start + t < len)
      b[i] = ld_stream(base + ((size_t(page) * 128 + slot0 + t) * 4 + head) * 256 + 64 * warp + 8 * (lane & 7));
  }
}
__device__ __forceinline__ void store_k_block(unsigned char* kv, const uint4 (&b)[8], int lane) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int t = 4 * i + (lane >> 3);
    *reinterpret_cast<uint4*>(kv + t * 128 + (((lane & 7) ^ (t & 7)) << 4)) = b[i];
  }
}
__device__ __forceinline__ void store_v_block(unsigned char* kv, const uint4 (&b)[8], int lane) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int t = 4 * i + (lane >> 3), g = (t & 1) | (((t >> 1) & 1) << 3), u = 2 * (lane & 7);
    *reinterpret_cast<uint2*>(kv + t * 128 + ((u ^ g) << 3)) = make_uint2(b[i].x, b[i].y);
    *reinterpret_cast<uint2*>(kv + t * 128 + (((u + 1) ^ g) << 3)) = make_uint2(b[i].z, b[i].w);
  }
}

// Block admission + exact in-register BF16 -> FP16 conversion (span <= 2^30 within the block, finite).
// Result FP16 = BF16 * 2^(112-D). Returns D, sets raw=true when the block must take the FP32 path
// (then b is left in an unspecified state and must be reloaded).
// BND (boundary tile, start+TILE > ctx0+1): only positions < vlimit = ctx0+1 (seen by EVERY verify row) vote for D
// and for raw. A later verify row's token (position >= vlimit) that does not fit the window [D+1, D+30] is zeroed
// in the tile and reported in `spec` (bit = token in tile). sol-13c: the rows that see a flagged token are recomputed
// by a row pass (reconvert with their own voter set); no FP32 special path. So a later verify row can never change an
// earlier row's arithmetic, and every row's arithmetic equals its single-row run's.
template <bool BND>
__device__ __forceinline__ int convert_block(uint4 (&b)[8], bool& raw, unsigned& spec, int vlimit, int start, int lane) {
  uint32_t mh = 0, ml = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const bool vote = !BND || start + 4 * i + (lane >> 3) < vlimit;
    const uint32_t w[4] = {b[i].x, b[i].y, b[i].z, b[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t a = vote ? (w[j] & 0x7FFF7FFFu) : 0u;
      mh = max(mh, a);
      ml = max(ml, a << 16);
    }
  }
  uint32_t m = max(mh, ml);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
  const int F = int(m >> 23);
  spec = 0;
  if (F == 255) { raw = true; return 0; }
  const int D = max(F - 30, 0);
  const uint32_t Kb = (0x8000u - (uint32_t(D) << 7)) * 0x10001u;
  const uint32_t Kv = (0x8000u - (uint32_t(D + 1) << 7)) * 0x10001u;
  const uint32_t Kh = (0x8000u - (uint32_t(D + 31) << 7)) * 0x10001u;  // bit 15 per half: above the window
  uint32_t bad = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const bool vote = !BND || start + 4 * i + (lane >> 3) < vlimit;
    uint32_t w[4] = {b[i].x, b[i].y, b[i].z, b[i].w};
    uint32_t oor = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t x = w[j], a = x & 0x7FFF7FFFu;
      const uint32_t bb = a + Kb;            // per half: 0x8000 + a - (D<<7), no carry between halves
      const uint32_t vv = a + Kv;            // bit 15 per half: FP16 result is a normal number
      const uint32_t nz = a + 0x7FFF7FFFu;   // bit 15 per half: value is nonzero
      if (BND && !vote) oor |= (nz & ~vv) | (a + Kh);
      else bad |= nz & ~vv;
      const uint32_t msk = prmt(vv, 0xBB99u);// 0xFFFF per normal half (sign-replicate bytes 1 and 3)
      w[j] = ((bb << 3) & msk & 0x7FF87FF8u) | (x & 0x80008000u);
    }
    if (BND) {
      const bool sp = !vote && (oor & 0x80008000u) != 0;
      const unsigned bl = __ballot_sync(0xffffffffu, sp);
#pragma unroll
      for (int k = 0; k < 4; ++k)
        if ((bl >> (8 * k)) & 0xFFu) spec |= 1u << (4 * i + k);
      if ((bl >> (lane & 24)) & 0xFFu) w[0] = w[1] = w[2] = w[3] = 0u;  // whole token zeroed in this slice
    }
    b[i] = make_uint4(w[0], w[1], w[2], w[3]);
  }
  raw = __any_sync(0xffffffffu, (bad & 0x80008000u) != 0);
  if (raw) spec = 0;
  return D;
}

// Rare FP32 paths, kept out of line so they do not raise register pressure of the tensor-core path.
// Score fallback: lane = token. Rows with raw K (whole block) or a flagged query slice use FP32 dots.
// sol-13c: rr = rlo | rhi << 8, rows rho in [rlo, rhi) (a row pass recomputes only the rows it finalizes and later ones).
__device__ __noinline__ void qk_fallback(float* spw, const unsigned char* kv, const float* q0, bool kraw, unsigned qmask,
                                         int Dk, int rr, int lane) {
  const int t = lane;
  for (int rho = rr & 0xFF; rho < (rr >> 8); ++rho) {
    const bool need = kraw || ((qmask >> ((rho & 3) + 16 * ((rho >> 2) & 1) + 4 * (rho >> 3))) & 1);
    if (!need) continue;
    const int j = rho / 6, hh = rho - 6 * j;
    const float* qrow = q0 + (size_t(j) * 24 + hh) * 256;
    float dot = 0.f;
    for (int d = 0; d < 64; ++d) {
      const uint16_t bits = *reinterpret_cast<const uint16_t*>(kv + t * 128 + (((d >> 3) ^ (t & 7)) << 4) + ((d & 7) << 1));
      const float kval = kraw ? bf2f(bits) : scale2(__half2float(__ushort_as_half(bits)), Dk - 112);
      dot = __fmaf_rn(__ldg(qrow + d), kval, dot);
    }
    spw[rho * 32 + 2 * ((t >> 1) ^ sp_h(rho)) + (t & 1)] = __fmul_rn(dot, 0.0625f);
  }
}
// Raw-V fallback: lane = dim of the 32-dim half b; out[rho*32+lane] = sum_t p(t)/den * v(t,dim), ascending t.
// Each term is normalized (p * 2^-15 / den) before multiplying raw V, so V near the BF16 maximum cannot overflow.
__device__ __noinline__ void pv_fallback(float* out, const unsigned char* kv, const unsigned char* ps, const float2* st,
                                         int b, int rows, int lane) {
  const int dim = 32 * b + lane;
  for (int rho = 0; rho < rows; ++rho) {
    float sum = 0.f;
    const float w = __fmul_rn(st[rho].y, 0x1p-15f);
    for (int t = 0; t < TILE; ++t) {
      const __half* prow = reinterpret_cast<const __half*>(ps + rho * 144 + (t >> 2) * 16);
      const float pv = __fmul_rn(__fadd_rn(__half2float(prow[t & 3]), __half2float(prow[4 + (t & 3)])), w);
      if (pv == 0.f) continue;  // masked / zero-weight token: never form 0 * (Inf|NaN) (Argon review)
      const int g = (t & 1) | (((t >> 1) & 1) << 3);
      const uint16_t bits = *reinterpret_cast<const uint16_t*>(kv + t * 128 + (((dim >> 2) ^ g) << 3) + ((dim & 3) << 1));
      sum = __fmaf_rn(pv, bf2f(bits), sum);
    }
    out[rho * 32 + lane] = sum;
  }
}

// sol-13c row pass (rare path, out of line): re-convert this warp's 32x64 block of a boundary tile with voters =
// positions < vlimit, where vlimit = ctx0 + j + 1 is the visible length of the first row j that sees a token flagged by
// the previous conversion. Per element and per decision this is exactly convert_block<true>(vlimit): same D, same raw
// decision, same flagged-and-zeroed tokens, same FP16 bits (identical expressions), so a row finalized by this pass gets
// the block its own single-row run converts (NUMERICS.md lemma). Streamed global -> shared in 8 token chunks (two passes
// over L2-resident data) so it needs few registers. Positions >= len are zero (never loaded), as in load_block.
// Stores the converted block, or the raw BF16 block if it is not admitted, to kv with the K (isv = 0) or V swizzle.
// pk = start | isv << 16. Returns spec | (D | raw << 8) << 32.
__device__ __forceinline__ void store_chunk(unsigned char* kv, int t, int lane, const uint32_t (&w)[4], int isv) {
  if (!isv) {
    *reinterpret_cast<uint4*>(kv + t * 128 + (((lane & 7) ^ (t & 7)) << 4)) = make_uint4(w[0], w[1], w[2], w[3]);
  } else {
    const int g = (t & 1) | (((t >> 1) & 1) << 3), u = 2 * (lane & 7);
    *reinterpret_cast<uint2*>(kv + t * 128 + ((u ^ g) << 3)) = make_uint2(w[0], w[1]);
    *reinterpret_cast<uint2*>(kv + t * 128 + (((u + 1) ^ g) << 3)) = make_uint2(w[2], w[3]);
  }
}
__device__ __noinline__ unsigned long long reconvert(unsigned char* kv, const uint16_t* tile, int pk, int vlimit, int len) {
  const int lane = threadIdx.x & 31, start = pk & 0xFFFF, isv = pk >> 16;
  const uint16_t* src = tile + size_t(lane >> 3) * 4 * 256 + 8 * (lane & 7);
  uint32_t mh = 0, ml = 0;
#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    const int pos = start + 4 * i + (lane >> 3);
    if (pos < vlimit && pos < len) {  // voters only (positions >= len are zero: they never raise the max)
      const uint4 x = __ldg(reinterpret_cast<const uint4*>(src + size_t(4 * i) * 4 * 256));
      const uint32_t w[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t a = w[j] & 0x7FFF7FFFu;
        mh = max(mh, a);
        ml = max(ml, a << 16);
      }
    }
  }
  uint32_t m = max(mh, ml);
#pragma unroll
  for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
  const int F = int(m >> 23);
  unsigned spec = 0;
  int D = 0;
  bool raw = F == 255;
  if (!raw) {
    D = max(F - 30, 0);
    const uint32_t Kb = (0x8000u - (uint32_t(D) << 7)) * 0x10001u;
    const uint32_t Kv = (0x8000u - (uint32_t(D + 1) << 7)) * 0x10001u;
    const uint32_t Kh = (0x8000u - (uint32_t(D + 31) << 7)) * 0x10001u;
    uint32_t bad = 0;
  #pragma unroll 1
  for (int i = 0; i < 8; ++i) {
      const int t = 4 * i + (lane >> 3), pos = start + t;
      const bool vote = pos < vlimit;
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if (pos < len) x = __ldg(reinterpret_cast<const uint4*>(src + size_t(4 * i) * 4 * 256));
      uint32_t w[4] = {x.x, x.y, x.z, x.w};
      uint32_t oor = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {  // convert_block<true>, element for element
        const uint32_t xx = w[j], a = xx & 0x7FFF7FFFu;
        const uint32_t bb = a + Kb, vv = a + Kv, nz = a + 0x7FFF7FFFu;
        if (!vote) oor |= (nz & ~vv) | (a + Kh);
        else bad |= nz & ~vv;
        const uint32_t msk = prmt(vv, 0xBB99u);
        w[j] = ((bb << 3) & msk & 0x7FF87FF8u) | (xx & 0x80008000u);
      }
      const bool sp = !vote && (oor & 0x80008000u) != 0;
      const unsigned bl = __ballot_sync(0xffffffffu, sp);
#pragma unroll
      for (int k = 0; k < 4; ++k)
        if ((bl >> (8 * k)) & 0xFFu) spec |= 1u << (4 * i + k);
      if ((bl >> (lane & 24)) & 0xFFu) w[0] = w[1] = w[2] = w[3] = 0u;  // whole token zeroed in this slice
      store_chunk(kv, t, lane, w, isv);
    }
    raw = __any_sync(0xffffffffu, (bad & 0x80008000u) != 0);
  }
  if (raw) {  // not admitted: this warp's block as original BF16 (FP32 SIMT path), positions >= len zero
    spec = 0;
  #pragma unroll 1
  for (int i = 0; i < 8; ++i) {
      const int t = 4 * i + (lane >> 3);
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if (start + t < len) x = __ldg(reinterpret_cast<const uint4*>(src + size_t(4 * i) * 4 * 256));
      const uint32_t w[4] = {x.x, x.y, x.z, x.w};
      store_chunk(kv, t, lane, w, isv);
    }
  }
  return (unsigned long long)spec | ((unsigned long long)(unsigned(D) | (unsigned(raw) << 8)) << 32);
}

template <int MT>
__global__ void __launch_bounds__(128, 2) attn_kernel(Params p) {
  extern __shared__ __align__(16) unsigned char sm[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int group = blockIdx.y, req = group >> 2, head = group & 3;
  const int begin = p.cu[req], R = p.cu[req + 1] - begin, len = p.lens[req];
  constexpr int CAP = MT == 3 ? 4 : 5;
  if (R <= 0 || R > CAP) return;
  const int ctx0 = len - R;  // local row j sees positions < ctx0 + j + 1
  // sol-13c per-row rung (rung_plan): this CTA (slot blockIdx.x) serves rows [rlo, rhi) at partition P. Without a
  // straddle (every agent except at ctx0 in [cutoff-3, cutoff] with R >= 2) that is all rows at ladder_P(ctx0).
  // (rlo/rhi are recomputed for the partial stores at the end so they are not live in the tile loop.)
  int P = p.P, split = blockIdx.x, vend = len;  // vend: visible length of the last served row (= len unless straddle)
  if (p.shortp) {
    const Rung g = rung_plan(ctx0, R, p.P);
    P = g.PA;
    if (g.jb < R) {
      if (split < g.nA) vend = ctx0 + g.jb;
      else { P = g.PB; split -= g.nA; }
    }
  }
  const int t_begin = split * P;
  if (t_begin >= vend) return;
  const int t_end = min(vend, t_begin + P);
  const int nrows = 6 * R;
  const int qp = (lane >> 2) & 3, loc = (lane & 3) | ((lane >> 4) << 2);
  unsigned char* kv = sm + SM_KV + warp * 4096;
  float* sp = reinterpret_cast<float*>(sm + SM_SP);
  unsigned char* ps = sm + SM_P;
  float2* st = reinterpret_cast<float2*>(sm + SM_ST);
  int* vist = reinterpret_cast<int*>(sm + SM_VIS);
  if (tid < 32) vist[tid] = tid < nrows ? ctx0 + tid / 6 + 1 : 0;  // published by barrier B1 of the first tile
  const int* pg = p.pages + req * p.pagecols;

  int* pgs = reinterpret_cast<int*>(sm + SM_PG);
  const int npg = (t_end - t_begin + 127) >> 7;
  if (tid < npg) pgs[tid] = pg[(t_begin >> 7) + tid];
  uint4 buf[8];
  load_block(buf, p.k, pg[t_begin >> 7], t_begin, len, head, warp, lane);
  // S1 metadata of the current tile (this warp's slice): K byte | V byte << 8 (adjacent bytes, one 16-bit load; sol-13c
  // packs them in one register to keep the 5-row kernel free of spills in the tile loop).
  int mkv = META_RAW | (META_RAW << 8);
  if (p.enc) {
    const int pg0 = pg[t_begin >> 7], tl0 = (t_begin & 127) >> 5;
    mkv = __ldg(reinterpret_cast<const unsigned short*>(p.meta + meta_index(pg0, tl0, head, warp, 0)));
  }

  // ---- query slice: this lane owns A row rho_a = 8*qp + loc of slice [64w, 64w+64)
  const int rho_a = 8 * qp + loc, ja = rho_a / 6, hha = rho_a - 6 * ja;
  const bool arow = qp < MT && rho_a < nrows;
  uint32_t qh[32], ql[32];
  int eq = 0;
  bool qbad = false;
  {
    const float* qrow = p.q + (size_t(begin + ja) * 24 + head * 6 + hha) * 256 + 64 * warp;
    float x[64];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
      if (arow) f = __ldg(reinterpret_cast<const float4*>(qrow) + i);
      x[4 * i] = f.x; x[4 * i + 1] = f.y; x[4 * i + 2] = f.z; x[4 * i + 3] = f.w;
    }
    float mx = 0.f;
#pragma unroll
    for (int d = 0; d < 64; ++d) mx = fmaxf(mx, fabsf(x[d]));
    eq = mx == 0.f ? 0 : 14 - ilogbf(mx);
#pragma unroll
    for (int d = 0; d < 64; d += 2) {
      __half h[2], l[2];
#pragma unroll
      for (int u = 0; u < 2; ++u) {
        const float xs = scale2(x[d + u], eq);
        h[u] = __float2half_rn(xs);
        l[u] = __float2half_rn(__fsub_rn(xs, __half2float(h[u])));
        const float back = scale2(__fadd_rn(__half2float(h[u]), __half2float(l[u])), -eq);
        qbad |= fabsf(__fsub_rn(back, x[d + u])) > __fmul_rn(fabsf(x[d + u]), 0x1p-20f);
      }
      qh[d >> 1] = pack2(h[0], h[1]);
      ql[d >> 1] = pack2(l[0], l[1]);
    }
    qbad = qbad && arow;
    if (p.debug && split == 0 && arow) {
      const size_t qi = size_t(begin + ja) * 24 + head * 6 + hha;
      uint32_t* dh = reinterpret_cast<uint32_t*>(p.dbg_h + qi * 512 + 64 * warp);
      uint32_t* dl = reinterpret_cast<uint32_t*>(p.dbg_h + qi * 512 + 256 + 64 * warp);
#pragma unroll
      for (int i = 0; i < 32; ++i) { dh[i] = qh[i]; dl[i] = ql[i]; }
      p.dbg_e[qi * 8 + warp * 2] = eq;
      p.dbg_e[qi * 8 + warp * 2 + 1] = qbad;
    }
  }
  const unsigned qmask = __ballot_sync(0xffffffffu, qbad);
  // Per-row slice exponents for the C-fragment rows, read per tile from a warp-private table (saves registers).
  int* eqs = reinterpret_cast<int*>(sm + SM_EQ) + warp * 32;
  eqs[rho_a] = eq;
  __syncwarp();

  float acc[MT][2][8];
#pragma unroll
  for (int a = 0; a < MT; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int c = 0; c < 8; ++c) acc[a][b][c] = 0.f;
  float m_run[2] = {-CUDART_INF_F, -CUDART_INF_F}, d_run[2] = {0.f, 0.f};
  const int sr0 = tid >> 3, sc = tid & 7;  // softmax role: rows sr0 and sr0+16, tokens 4sc..4sc+3

  __syncthreads();  // publish the page table (and query exponent tables) before the first tile
  for (int start = t_begin; start < t_end; start += TILE) {
    // ---------------- K block: admit, convert, store (warp private) ----------------
    bool kraw = false;
    unsigned kspec = 0, vspec = 0;
    // Boundary tile: holds later verify rows' tokens. The shared conversion uses positions <= ctx0 as voters (row 0's
    // prefix); a token it flags starts a row pass for the rows that see it (sol-13c, below).
    const bool bnd = start + TILE > ctx0 + 1;
    const int mk = mkv & 0xFF, mv = mkv >> 8;
    const bool kenc = p.enc && !bnd && mk <= 224, venc = p.enc && !bnd && mv <= 224;
    int Dk;
    if (kenc) Dk = mk;  // S1: block stored as exact scaled FP16 by the writer (same D convert_block<false> picks)
    else {
      Dk = bnd ? convert_block<true>(buf, kraw, kspec, ctx0 + 1, start, lane)
               : convert_block<false>(buf, kraw, kspec, 0, start, lane);
      if (kraw) load_block(buf, p.k, pgs[(start - t_begin) >> 7], start, len, head, warp, lane);
    }
    if (p.debug && kraw && lane == 0) atomicAdd(p.dbg_e + MAXROWS * 24 * 8, 1);  // harness only
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = 4 * i + (lane >> 3);
      *reinterpret_cast<uint4*>(kv + t * 128 + (((lane & 7) ^ (t & 7)) << 4)) = buf[i];
    }
    load_block(buf, p.v, pgs[(start - t_begin) >> 7], start, len, head, warp, lane);  // V of this tile, consumed after QK
    __syncwarp();
    // ---------------- S partial = Q_w K_w^T on tensor cores ----------------
    // P4: n-outer; each 8x8 score tile accumulates in 2 independent chains of 64 products (hi and lo halves of Q),
    // combined as (hi + lo): shorter FP32 chains, 2-way ILP, 16 accumulator registers instead of 32.
    // P3: per-row power-of-two factors f1*f2 (identical to scale2's two multiplies), computed once per tile.
    float* spw = sp + warp * 1024;
    // sol-13c row passes. Pass 0 uses the shared conversion. If it flagged a token (boundary tile only; first flagged
    // position p, seen by rows j >= p - ctx0), rows rho < khi = 6(p - ctx0) are final, and the block is re-converted with
    // the voter set of row khi/6 (reconvert); that pass rewrites rows rho >= klo = khi, and so on. Each pass strictly
    // advances (its flagged tokens lie at or beyond its own voter limit), at most R - 1 re-passes per block. A row's
    // final pass never flags a token the row sees, so its D, raw decision and FP16 bits equal its single-row run's.
    // Interior tiles: kspec = 0, one pass, identical to sol-13b.
    for (int klo = 0;;) {
    const int khi = pass_end(kspec, klo / 6, start, ctx0, R);
    float f1[2], f2[2];
#pragma unroll
    for (int vb = 0; vb < 2; ++vb) {
      const int E = -(eqs[8 * qp + (lane & 1) + 4 * (lane >> 4) + 2 * vb] + 112 - Dk) - 4;
      const int e1 = max(min(E, 127), -126), e2 = max(min(E - e1, 127), -126);
      f1[vb] = __uint_as_float(unsigned(e1 + 127) << 23);
      f2[vb] = __uint_as_float(unsigned(e2 + 127) << 23);
    }
    // v2b: two n-tiles per iteration -> 4 independent HMMA chains (hi/lo x 2 tiles). Each element's summation
    // order is exactly v2's (k-chunks ascending, hi and lo chains separate, then hi + lo): bit-identical scores.
#pragma unroll
    for (int n0 = 0; n0 < 4; n0 += 2) {
      float sh[2][8], sl[2][8];
#pragma unroll
      for (int u = 0; u < 2; ++u)
#pragma unroll
        for (int c = 0; c < 8; ++c) sh[u][c] = sl[u][c] = 0.f;
      if (!kraw) {
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const uint4 ka = *reinterpret_cast<const uint4*>(kv + (8 * n0 + loc) * 128 + ((kk ^ loc) << 4));
          const uint4 kb = *reinterpret_cast<const uint4*>(kv + (8 * n0 + 8 + loc) * 128 + ((kk ^ loc) << 4));
          mma_rc(sh[0], qh[4 * kk], qh[4 * kk + 1], ka.x, ka.y);
          mma_rc(sh[1], qh[4 * kk], qh[4 * kk + 1], kb.x, kb.y);
          mma_rc(sl[0], ql[4 * kk], ql[4 * kk + 1], ka.x, ka.y);
          mma_rc(sl[1], ql[4 * kk], ql[4 * kk + 1], kb.x, kb.y);
          mma_rc(sh[0], qh[4 * kk + 2], qh[4 * kk + 3], ka.z, ka.w);
          mma_rc(sh[1], qh[4 * kk + 2], qh[4 * kk + 3], kb.z, kb.w);
          mma_rc(sl[0], ql[4 * kk + 2], ql[4 * kk + 3], ka.z, ka.w);
          mma_rc(sl[1], ql[4 * kk + 2], ql[4 * kk + 3], kb.z, kb.w);
        }
      }
      if (qp < MT) {
#pragma unroll
        for (int u = 0; u < 2; ++u)
#pragma unroll
          for (int vb = 0; vb < 2; ++vb) {
            const int rho = 8 * qp + (lane & 1) + 4 * (lane >> 4) + 2 * vb, h = sp_h(rho);
            if (rho < klo) continue;  // finalized by an earlier pass
#pragma unroll
            for (int vh = 0; vh < 2; ++vh) {
              const int c = 4 * vh + 2 * vb, pair = 4 * (n0 + u) + ((lane >> 1) & 1) + 2 * vh;
              const float s0 = __fadd_rn(sh[u][c], sl[u][c]);
              const float s1 = __fadd_rn(sh[u][c + 1], sl[u][c + 1]);
              *reinterpret_cast<float2*>(spw + rho * 32 + 2 * (pair ^ h)) =
                  make_float2(__fmul_rn(__fmul_rn(s0, f1[vb]), f2[vb]), __fmul_rn(__fmul_rn(s1, f1[vb]), f2[vb]));
            }
          }
      }
    }
    if (kraw || qmask) {  // FP32 fallback: whole block (raw K) or individual query rows (bad split), rows >= klo
      __syncwarp();
      qk_fallback(spw, kv, p.q + (size_t(begin) * 24 + head * 6) * 256 + 64 * warp, kraw, qmask, Dk, klo | (nrows << 8), lane);
      __syncwarp();
    }
    if (khi >= nrows) break;  // every row is final (always, unless a flagged token is seen by some row)
    klo = khi;
    __syncwarp();  // all lanes finished reading kv
    {
      const unsigned long long rc = reconvert(kv, p.k + ((size_t(pgs[(start - t_begin) >> 7]) * 128 + (start & 127)) * 4 + head) * 256 + 64 * warp,
                                              start, ctx0 + 1 + klo / 6, ROWPASS_EAGER ? ctx0 + 1 + klo / 6 : len);
      kspec = unsigned(rc);
      Dk = int(rc >> 32) & 0xFF;
      kraw = ((rc >> 40) & 1) != 0;
    }
    if (p.debug && lane == 0) atomicAdd(p.dbg_e + MAXROWS * 24 * 8 + 2, 1);  // harness only: K row passes
    __syncwarp();
    }  // row passes (K)
    // ---------------- V block: admit, convert, store (after all lanes finished reading K) ----------------
    bool vraw = false;
    int Dv;
    if (venc) Dv = mv;
    else {
      Dv = bnd ? convert_block<true>(buf, vraw, vspec, ctx0 + 1, start, lane)
               : convert_block<false>(buf, vraw, vspec, 0, start, lane);
      if (vraw) load_block(buf, p.v, pgs[(start - t_begin) >> 7], start, len, head, warp, lane);
    }
    if (p.debug && vraw && lane == 0) atomicAdd(p.dbg_e + MAXROWS * 24 * 8 + 1, 1);  // harness only
    __syncwarp();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = 4 * i + (lane >> 3), g = (t & 1) | (((t >> 1) & 1) << 3), u = 2 * (lane & 7);
      *reinterpret_cast<uint2*>(kv + t * 128 + ((u ^ g) << 3)) = make_uint2(buf[i].x, buf[i].y);
      *reinterpret_cast<uint2*>(kv + t * 128 + (((u + 1) ^ g) << 3)) = make_uint2(buf[i].z, buf[i].w);
    }
    if (start + TILE < t_end) {
      const int ns = start + TILE, npg = pgs[(ns - t_begin) >> 7];
      load_block(buf, p.k, npg, ns, len, head, warp, lane);
      if (p.enc) {  // metadata of the next tile, issued with its K payload
        mkv = __ldg(reinterpret_cast<const unsigned short*>(p.meta + meta_index(npg, (ns & 127) >> 5, head, warp, 0)));
      }
    }
    __syncthreads();  // B1: all score partials written; previous PV finished reading P/stats
    // ---------------- softmax (FP32), fixed reduction order ----------------
#pragma unroll
    for (int which = 0; which < 2; ++which) {
      const int rho = sr0 + 16 * which;
      if (rho < MT * 8) {
        const int vis = vist[rho];
        const float* base = sp + rho * 32 + 2 * ((2 * sc) ^ sp_h(rho));
        const float4 a0 = *reinterpret_cast<const float4*>(base);
        const float4 a1 = *reinterpret_cast<const float4*>(base + 1024);
        const float4 a2 = *reinterpret_cast<const float4*>(base + 2048);
        const float4 a3 = *reinterpret_cast<const float4*>(base + 3072);
        float sv[4] = {__fadd_rn(__fadd_rn(a0.x, a1.x), __fadd_rn(a2.x, a3.x)),
                       __fadd_rn(__fadd_rn(a0.y, a1.y), __fadd_rn(a2.y, a3.y)),
                       __fadd_rn(__fadd_rn(a0.z, a1.z), __fadd_rn(a2.z, a3.z)),
                       __fadd_rn(__fadd_rn(a0.w, a1.w), __fadd_rn(a2.w, a3.w))};
        float mx = -CUDART_INF_F;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          if (start + 4 * sc + k >= vis) sv[k] = -CUDART_INF_F;
          mx = fmaxf(mx, sv[k]);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        const float mn = fmaxf(m_run[which], mx);
        const float a = m_run[which] == -CUDART_INF_F ? 0.f : expf(__fsub_rn(m_run[which], mn));
        float pr[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) pr[k] = sv[k] == -CUDART_INF_F ? 0.f : expf(__fsub_rn(sv[k], mn));
        float l = __fadd_rn(__fadd_rn(pr[0], pr[1]), __fadd_rn(pr[2], pr[3]));
        l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 1));
        l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 2));
        l = __fadd_rn(l, __shfl_xor_sync(0xffffffffu, l, 4));
        const float old = __fmul_rn(d_run[which], a), dn = __fadd_rn(old, l);
        if (sc == 0) st[rho] = make_float2(dn > 0.f ? __fdiv_rn(old, dn) : 0.f, dn > 0.f ? __fdiv_rn(1.f, dn) : 0.f);
        // sol-13c (Argon PROMPT26 Q3.1): a row with no visible token in this tile keeps its state, as its single-row run
        // never enters the tile. For finite state this is what the update computes anyway (mn = m, dn = d exactly); it
        // differs only for a NaN running max (fmaxf(NaN, -inf) = -inf), i.e. non-finite visible K.
        if (start < vis) {
          m_run[which] = mn;
          d_run[which] = dn;
        }
        __half hi[4], lo[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float x = __fmul_rn(pr[k], 32768.f);
          hi[k] = __float2half_rn(x);
          lo[k] = __float2half_rn(__fsub_rn(x, __half2float(hi[k])));
        }
        *reinterpret_cast<uint4*>(ps + rho * 144 + sc * 16) =
            make_uint4(pack2(hi[0], hi[1]), pack2(hi[2], hi[3]), pack2(lo[0], lo[1]), pack2(lo[2], lo[3]));
      }
    }
    __syncthreads();  // B2: P and stats ready
    // ---------------- O_w += P V_w on tensor cores, per 32-dim half ----------------
    // sol-13c row passes for V (same rule as K): each pass updates rows rho in [vlo, vhi) exactly once per tile. Pass 0
    // (shared conversion) is the hot path; the re-passes run only if the shared conversion flagged a token that some row
    // sees (boundary tile). The PV body is one inlined lambda used by both (kept out of a loop on the hot path:
    // a loop around it costs registers/spills in the 5-row kernel).
    auto pv_rows = [&](const int vlo, const int vhi, const int Dv, const bool vraw) {
      const int vshift = -15 - (112 - Dv);
      // invd in [2^-9, 1]: invd*2^vshift is a normal number iff Dv >= 10; tiny-V blocks scale the tile sum instead.
      const bool vfast = vraw || Dv >= 16;
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        float tl[MT][8];
#pragma unroll
        for (int rt = 0; rt < MT; ++rt)
#pragma unroll
          for (int c = 0; c < 8; ++c) tl[rt][c] = 0.f;
        if (!vraw) {
          const int c = 8 * b + 2 * qp + (lane >> 4), g = (lane & 1) | (((lane >> 1) & 1) << 3);
#pragma unroll
          for (int s4 = 0; s4 < 8; ++s4) {
            const int t = 4 * s4 + (lane & 3);
            const uint2 vb2 = *reinterpret_cast<const uint2*>(kv + t * 128 + ((c ^ g) << 3));
#pragma unroll
            for (int rt = 0; rt < MT; ++rt) {
              const uint4 pa = *reinterpret_cast<const uint4*>(ps + (8 * rt + loc) * 144 + s4 * 16);
              mma_rr(tl[rt], pa.x, pa.y, vb2.x, vb2.y);
              mma_rr(tl[rt], pa.z, pa.w, vb2.x, vb2.y);
            }
          }
        }
        else {  // FP32 fallback for a V block that failed admission; result staged in this warp's score area
          __syncwarp();
          pv_fallback(spw, kv, ps, st, b, min(MT * 8, nrows), lane);
          __syncwarp();
#pragma unroll
          for (int rt = 0; rt < MT; ++rt)
#pragma unroll
            for (int c = 0; c < 8; ++c)
              tl[rt][c] = spw[(8 * rt + (lane & 1) + 4 * (lane >> 4) + 2 * ((c >> 1) & 1)) * 32 + 8 * qp +
                              2 * ((lane >> 1) & 1) + (c & 1) + 4 * (c >> 2)];
          __syncwarp();
        }
#pragma unroll
        for (int rt = 0; rt < MT; ++rt)
#pragma unroll
          for (int vb = 0; vb < 2; ++vb) {
            const int rho = 8 * rt + (lane & 1) + 4 * (lane >> 4) + 2 * vb;
            if (start < vist[rho] && rho >= vlo && rho < vhi) {  // skip tiles with no visible token; rows of this pass
              const float2 x = st[rho];
              if (vfast) {
                const float be = vraw ? 1.f : scale2(x.y, vshift);  // raw path is already normalized
#pragma unroll
                for (int c = 2 * vb; c < 8; c += (c & 1) ? 3 : 1)  // c in {2vb, 2vb+1, 2vb+4, 2vb+5}
                  acc[rt][b][c] = __fmaf_rn(tl[rt][c], be, __fmul_rn(acc[rt][b][c], x.x));
              } else {
#pragma unroll
                for (int c = 2 * vb; c < 8; c += (c & 1) ? 3 : 1)
                  acc[rt][b][c] = __fmaf_rn(scale2(tl[rt][c], vshift), x.y, __fmul_rn(acc[rt][b][c], x.x));
              }
            }
          }
      }
    };
    const int vhi0 = pass_end(vspec, 0, start, ctx0, R);
    pv_rows(0, vhi0, Dv, vraw);
    if (vhi0 < nrows) {
      const uint16_t* vtile = p.v + ((size_t(pgs[(start - t_begin) >> 7]) * 128 + (start & 127)) * 4 + head) * 256 + 64 * warp;
      for (int vlo = vhi0;;) {
        __syncwarp();  // all lanes finished reading kv (and the pv_fallback staging)
        const unsigned long long rc = reconvert(kv, vtile, start | (1 << 16), ctx0 + 1 + vlo / 6, ROWPASS_EAGER ? ctx0 + 1 + vlo / 6 : len);
        if (p.debug && lane == 0) atomicAdd(p.dbg_e + MAXROWS * 24 * 8 + 3, 1);  // harness only: V row passes
        __syncwarp();
        const int vhi = pass_end(unsigned(rc), vlo / 6, start, ctx0, R);
        pv_rows(vlo, vhi, int(rc >> 32) & 0xFF, ((rc >> 40) & 1) != 0);
        if (vhi >= nrows) break;
        vlo = vhi;
      }
    }
    __syncwarp();  // all lanes finished reading V before the next K store
  }
  // ---------------- partition outputs: normalized partial, max, mass ----------------
  // Partition slot = split (rung PB CTAs of a straddling agent reuse rung PA's slot numbers for their own rows).
  int rlo = 0, rhi = R, slot = blockIdx.x;  // rows this CTA serves (all rows unless the agent straddles a cutoff)
  if (p.shortp) {
    const Rung g = rung_plan(ctx0, R, p.P);
    if (g.jb < R) { if (slot < g.nA) rhi = g.jb; else { rlo = g.jb; slot -= g.nA; } }
  }
  const size_t cta = size_t(slot) * 16 + group;
#pragma unroll
  for (int rt = 0; rt < MT; ++rt)
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int c = 0; c < 8; c += 2) {
        const int rho = 8 * rt + (lane & 1) + 4 * (lane >> 4) + 2 * ((c >> 1) & 1);
        const int dim = 64 * warp + 32 * b + 8 * qp + 2 * ((lane >> 1) & 1) + 4 * (c >> 2);
        if (rho >= 6 * rlo && rho < 6 * rhi)  // served rows (all rows unless straddle)
          *reinterpret_cast<float2*>(p.partO + (cta * 32 + rho) * 256 + dim) = make_float2(acc[rt][b][c], acc[rt][b][c + 1]);
      }
#pragma unroll
  for (int which = 0; which < 2; ++which) {
    const int rho = sr0 + 16 * which;
    if (sc == 0 && rho >= 6 * rlo && rho < 6 * rhi)
      *reinterpret_cast<float2*>(p.partS + (cta * 32 + rho) * 2) = make_float2(m_run[which], d_run[which]);
  }
}

// S1 writer: encode every fully committed, not-yet-encoded 32x64 block of K and V in place.
// "Committed" = every position of the block is <= ctx0 = length - R (seen by every verify row of this step; never
// rewritten). The D chosen here equals what convert_block<false> computes in the attention kernel for the same block,
// so encoded and raw modes are bit-identical. Grid: (32-token blocks of the longest agent, 16 agent-heads); 8 warps:
// warp & 3 = 64-dim slice, warp >> 2 = K or V. In place: each lane reads its whole block slice before writing it.
__global__ void __launch_bounds__(256) encode_kernel(uint16_t* K, uint16_t* V, uint8_t* meta, const int32_t* pages,
                                                     const int32_t* lens, const int32_t* cu, int pagecols, int recent) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, slice = w & 3, isv = w >> 2;
  const int group = blockIdx.y, req = group >> 2, head = group & 3;
  const int R = cu[req + 1] - cu[req], len = lens[req];
  const int limit = R > 0 ? len - R + 1 : len;  // positions < limit are final
  // recent > 0 (decode step): only the last `recent` fully committed blocks; recent == 0 (prefill): every block.
  const int blk = recent ? limit / TILE - recent + int(blockIdx.x) : int(blockIdx.x);
  if (blk < 0) return;
  const int start = blk * TILE;
  if (start + TILE > limit) return;
  const int page = pages[req * pagecols + (start >> 7)], tile = (start & 127) >> 5;
  uint8_t* m = meta + meta_index(page, tile, head, slice, isv);
  if (*m != META_UNENC) return;  // already encoded, or kept raw
  uint16_t* base = isv ? V : K;
  uint4 b[8];
  load_block_page(b, base, page, start, len, head, slice, lane);
  bool raw = false;
  unsigned spec = 0;
  const int D = convert_block<false>(b, raw, spec, 0, start, lane);
  if (!raw) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = 4 * i + (lane >> 3);
      *reinterpret_cast<uint4*>(base + ((size_t(page) * 128 + (start & 127) + t) * 4 + head) * 256 + 64 * slice + 8 * (lane & 7)) = b[i];
    }
  }
  __syncwarp();
  if (lane == 0) *m = raw ? META_RAW : uint8_t(D);
}

// Fixed-order merge of a row's visible partitions. One block per (row, query head), 64 threads x float4.
__global__ void __launch_bounds__(64) merge_kernel(const float* __restrict__ partO, const float* __restrict__ partS,
                                                   const int32_t* cu, const int32_t* lens, float* out, int Plong, int cap,
                                                   int shortp) {
  __shared__ float fac[64];
  const int row = blockIdx.x / 24, qh = blockIdx.x % 24, tid = threadIdx.x;
  if (row >= cu[4]) return;
  int req = 0;
#pragma unroll
  for (int r = 1; r < 4; ++r) req += row >= cu[r];
  const int R = cu[req + 1] - cu[req], j = row - cu[req];
  const int group = req * 4 + qh / 6, rho = j * 6 + qh % 6;
  float4* dst = reinterpret_cast<float4*>(out + (size_t(row) * 24 + qh) * 256) + tid;
  if (R > cap) { *dst = make_float4(CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F, CUDART_NAN_F); return; }
  // sol-13c: the row's own rung ladder_P(ctx0 + j) and its slot range (rung_plan, identical to the attention CTAs).
  const int ctx0 = lens[req] - R, vis = ctx0 + j + 1;
  int P = Plong;
  if (shortp) {
    const Rung g = rung_plan(ctx0, R, Plong);
    P = j < g.jb ? g.PA : g.PB;
  }
  const int chunks = (vis + P - 1) / P;
  if (tid < 32) {
    float m[2], d[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int s = tid + 32 * u;
      m[u] = -CUDART_INF_F; d[u] = 0.f;
      if (s < chunks) {
        const float2 x = *reinterpret_cast<const float2*>(partS + ((size_t(s) * 16 + group) * 32 + rho) * 2);
        m[u] = x.x; d[u] = x.y;
      }
    }
    float M = fmaxf(m[0], m[1]);
#pragma unroll
    for (int o = 16; o; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    float w[2];
#pragma unroll
    for (int u = 0; u < 2; ++u) w[u] = m[u] == -CUDART_INF_F ? 0.f : __fmul_rn(d[u], expf(__fsub_rn(m[u], M)));
    float D = __fadd_rn(w[0], w[1]);
#pragma unroll
    for (int o = 16; o; o >>= 1) D = __fadd_rn(D, __shfl_xor_sync(0xffffffffu, D, o));
#pragma unroll
    for (int u = 0; u < 2; ++u) fac[tid + 32 * u] = D > 0.f ? __fdiv_rn(w[u], D) : 0.f;
  }
  __syncthreads();
  float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int s = 0; s < chunks; ++s) {
    const float4 x = *(reinterpret_cast<const float4*>(partO + ((size_t(s) * 16 + group) * 32 + rho) * 256) + tid);
    const float f = fac[s];
    o.x = __fmaf_rn(x.x, f, o.x); o.y = __fmaf_rn(x.y, f, o.y);
    o.z = __fmaf_rn(x.z, f, o.z); o.w = __fmaf_rn(x.w, f, o.w);
  }
  *dst = o;
}

size_t splits(int n, int p) { return size_t((n + p - 1) / p); }
// Workspace: maximum slot count over all rungs (short rungs never exceed LADDER_SPLITS). Unchanged from sol-13b: a
// straddling agent's two rungs share slots (rung_plan).
size_t wsplits(int n, int p) { return std::max(splits(n, p), std::min<size_t>(LADDER_SPLITS, splits(n, 128))); }
// Grid x (sol-13c): an agent whose rows straddle cutoff c (ctx0 <= c < ctx0 + R - 1, so length >= c + 2) needs
// ceil((c+1)/PA) + ceil(length/PB) CTAs per head, length <= min(n, c + MAX_R): 10 + 5 = 15 at both cutoffs for P = 512.
size_t gsplits(int n, int p) {
  size_t g = wsplits(n, p);
  const int cut[2] = {10 * 128 - MAX_R, 10 * 256 - MAX_R};
  for (int c : cut) {
    const int pa = ladder_P(c, p), pb = ladder_P(c + 1, p);
    if (pa != pb && n >= c + 2) g = std::max(g, splits(c + 1, pa) + splits(std::min(n, c + MAX_R), pb));
  }
  return g;
}
size_t partO_bytes(int n, int p) { return wsplits(n, p) * 16 * 32 * 256 * 4; }
size_t partS_bytes(int n, int p) { return wsplits(n, p) * 16 * 32 * 2 * 4; }
bool valid_cfg(int n, int p) { return n >= 1 && n <= 32768 && p >= 32 && p <= 4096 && p % 32 == 0 && splits(n, p) <= 64; }
}  // namespace

extern "C" size_t sol13b_debug_offset(int n, int p) { return partO_bytes(n, p) + partS_bytes(n, p); }
extern "C" size_t sol13b_workspace_bytes(int n, int p) {
  if (!valid_cfg(n, p)) return 0;
  return sol13b_debug_offset(n, p) + size_t(MAXROWS) * 24 * 512 * 2 + size_t(MAXROWS) * 24 * 8 * 4 + 16;
}
extern "C" int sol13b_init(void) {
  int e;
  if ((e = cudaFuncSetAttribute(attn_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_BYTES))) return e;
  if ((e = cudaFuncSetAttribute(attn_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_BYTES))) return e;
  if ((e = cudaFuncSetAttribute(attn_kernel<3>, cudaFuncAttributePreferredSharedMemoryCarveout, 100))) return e;
  if ((e = cudaFuncSetAttribute(attn_kernel<4>, cudaFuncAttributePreferredSharedMemoryCarveout, 100))) return e;
  return 0;
}
extern "C" int sol13b_attention(const float* q, const uint16_t* k, const uint16_t* v, const uint16_t*, const uint16_t*,
                                const int32_t* meta, const int32_t* pages, const int32_t* lens, const int32_t* cu, float* out,
                                void* work, int pagecols, int maxlen, int partition, int variant, void* stream) {
  const int rows = variant & 63, debug = (variant & 64) != 0, shortp = (variant & 128) != 0, enc = (variant & 256) != 0;
  if (!q || !k || !v || !pages || !lens || !cu || !out || !work || !valid_cfg(maxlen, partition) ||
      pagecols < (maxlen + 127) / 128 || (rows != 4 && rows != 5) || (variant & ~511) || (enc && !meta))
    return cudaErrorInvalidValue;
  Params p;
  p.q = q; p.k = k; p.v = v; p.pages = pages; p.lens = lens; p.cu = cu;
  char* w = static_cast<char*>(work);
  p.partO = reinterpret_cast<float*>(w);
  p.partS = reinterpret_cast<float*>(w + partO_bytes(maxlen, partition));
  p.dbg_h = reinterpret_cast<uint16_t*>(w + sol13b_debug_offset(maxlen, partition));
  p.dbg_e = reinterpret_cast<int32_t*>(w + sol13b_debug_offset(maxlen, partition) + size_t(MAXROWS) * 24 * 512 * 2);
  p.pagecols = pagecols; p.P = partition; p.debug = debug; p.shortp = shortp; p.enc = enc;
  p.meta = reinterpret_cast<const uint8_t*>(meta);
  auto s = static_cast<cudaStream_t>(stream);
  const size_t gx = shortp ? gsplits(maxlen, partition) : splits(maxlen, partition);
  const dim3 grid(unsigned(gx), 16);
  if (rows == 4) attn_kernel<3><<<grid, 128, SM_BYTES, s>>>(p);
  else attn_kernel<4><<<grid, 128, SM_BYTES, s>>>(p);
  int e = cudaGetLastError();
  if (e) return e;
  merge_kernel<<<MAXROWS * 24, 64, 0, s>>>(p.partO, p.partS, cu, lens, out, partition, rows, shortp);
  return cudaGetLastError();
}

// S1 writer-side ABI. meta: per layer, sol13b_meta_bytes(physical_pages) bytes, initialised to 0xFE (unencoded) when a
// page is allocated. Call after the step's K/V tokens are written (BF16) and before attention reads the layer, on the
// same stream; it encodes every block that is fully committed (all positions <= length - R) and still unencoded.
// Committed blocks are never rewritten; if positions are ever rewritten, write BF16 and reset their bytes to 0xFE.
extern "C" size_t sol13b_meta_bytes(int physical_pages) { return size_t(physical_pages) * 4 * 4 * 4 * 2; }
// recent_blocks: 0 = scan every block (after prefill); k > 0 = only each agent's last k fully committed blocks (a
// decode step commits <= 5 tokens, so k = 2 covers it): grid (k, 16) of tiny CTAs that exit after one metadata byte.
extern "C" int sol13b_encode_committed(uint16_t* k, uint16_t* v, uint8_t* meta, const int32_t* pages, const int32_t* lens,
                                       const int32_t* cu, int pagecols, int max_length, int recent_blocks, void* stream) {
  if (!k || !v || !meta || !pages || !lens || !cu || max_length < 1 || max_length > 32768 || pagecols < (max_length + 127) / 128 ||
      recent_blocks < 0 || recent_blocks > 1024)
    return cudaErrorInvalidValue;
  const unsigned gx = recent_blocks ? unsigned(recent_blocks) : unsigned((max_length + TILE - 1) / TILE);
  encode_kernel<<<dim3(gx, 16), 256, 0, static_cast<cudaStream_t>(stream)>>>(k, v, meta, pages, lens, cu, pagecols, recent_blocks);
  return cudaGetLastError();
}
