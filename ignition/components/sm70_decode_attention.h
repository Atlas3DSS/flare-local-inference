#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Sol-13b: long-context verification attention for sm_70 tensor cores (build S1L = v2b + S1 + partition ladder).
 * Same call shape as sol13_attention so an integrator can swap kernels.
 *
 * Q/Out: packed [M,24,256] FP32, RoPE applied. cu[5]: prefix sum of 0..5 rows per agent (empty agents allowed).
 * lengths[4] include the new rows; local row j of an agent with R rows sees positions < length-R+j+1.
 * K/V: device [physical_page,128,4,256] 16-bit words, one layer per call. kh/vh are ignored (may be NULL).
 *
 * variant = R_cap | flags:
 *   R_cap 4 -> up to 4 rows per agent, 5 -> up to 5 (both plans give identical row bits).
 *   +128  partition ladder: per agent, from its committed context ctx0 = length - R only (row-invariant):
 *         ctx0 <= 1275 -> 128-token partitions, <= 2555 -> 256, else `partition` (512). Recommended.
 *   +256  encoded K/V (S1): `meta` (uint8_t*, cast to const int32_t* to keep the sol13 argument list) describes each
 *         32-token x 64-dim block: 0..224 = block stored as exact bf16*2^(112-D) FP16 (written by
 *         sol13b_encode_committed); 0xFE/0xFF = block stored as raw BF16. Without +256 all of K/V must be raw BF16.
 *         Encoded and raw mode give identical bits for the same logical K/V.
 *   +64   harness only: dump the query hi/lo representation and debug counters into the workspace.
 * partition: 512 recommended (pinned; multiple of 32 in [32,4096]). A row's bits depend only on its Q row, its
 * visible K/V, its committed context ctx0 and the plan (partition, ladder flag). Not on R, packing, other agents,
 * max_length or R_cap.
 * max_length <= 32768 must cover every active length; pagecols >= ceil(max_length/128). Visible K/V finite.
 * Graph capturable; caller stream; caller workspace; no allocation, no host sync, no atomics. Returns cudaError_t. */
int sol13b_init(void); /* call once before CUDA graph capture */
size_t sol13b_workspace_bytes(int max_length, int partition);
int sol13b_attention(const float* q, const uint16_t* k, const uint16_t* v,
                     const uint16_t* kh, const uint16_t* vh, const int32_t* meta,
                     const int32_t* pages, const int32_t* lengths, const int32_t* cu,
                     float* out, void* work, int pagecols, int max_length,
                     int partition, int variant, void* stream);
size_t sol13b_debug_offset(int max_length, int partition); /* harness only */

/* S1 writer side (append-time exact encoding; replaces any per-step codec pass).
 * meta: per layer, sol13b_meta_bytes(physical_pages) bytes; set a page's bytes to 0xFE when the page is allocated.
 * After a layer's new K/V tokens are written (BF16) and before attention reads that layer, on the same stream:
 *   sol13b_encode_committed(k, v, meta, pages, lengths, cu, pagecols, max_length, recent_blocks, stream)
 * encodes, in place, every 32x64 block whose positions are all <= ctx0 = length - R ("committed": seen by every verify
 * row, never rewritten) and whose byte is still 0xFE. Blocks spanning > 2^30 stay BF16 (byte 0xFF). Tail blocks that
 * hold later verify rows' tokens are never encoded (attention handles them exactly in registers).
 * recent_blocks: 0 after prefill (scan all blocks); 2 for a decode step (ctx0 advances by <= 5 tokens per step):
 * 32 tiny CTAs, ~2 us per layer. Integrators may instead fuse the same per-block routine into their KV writer.
 * If committed positions are ever rewritten (cache truncation/reuse), write BF16 and reset those bytes to 0xFE. */
size_t sol13b_meta_bytes(int physical_pages);
int sol13b_encode_committed(uint16_t* k, uint16_t* v, uint8_t* meta, const int32_t* pages, const int32_t* lengths,
                            const int32_t* cu, int pagecols, int max_length, int recent_blocks, void* stream);
#ifdef __cplusplus
}
#endif
