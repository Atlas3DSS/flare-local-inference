// Sol-15: bit-identical, token-parallel replacement for sm70_gdn._convolution with PREFILL=True on Volta.
//
// Reference: Triton 3.8 `_convolution` (CHANNELS=10240, HISTORY=7, PREFILL=True, BLOCK=128, num_warps=4,
// enable_fp_fusion=False), triton-cache M5HBNTBK..., .text sha256 e546a05c... (the astra-7 prefill128 graph runs it
// 48x per chunk, grid (requests, 80)). The reference walks tokens sequentially per channel only to shift its three
// history registers; every token's output depends on inputs alone, so tokens are independent:
//   h(p) = p >= start ? bf16(X[p]) : (fresh ? 0 : History[slot, ch, p - start + 3])        (p in start-3 .. end-1)
//   out[t] = bf16(div.full(c, add.rn(ex2.approx(mul.rn(c, -log2e)), 1))),
//   c = bf16(add.rn(add.rn(add.rn(mul.rn(h(t-3), w0), mul.rn(h(t-2), w1)), mul.rn(h(t-1), w2)), mul.rn(bf16(X[t]), w3)))
// exactly the reference PTX order (r36 = h0*w0, r37 = h1*w1, r38 = r36 + r37, r39 = r38 + h2*w2, r40 = r39 + x*w3).
// History after the chunk = h(end-3), h(end-2), h(end-1) in columns 0..2 and zeros in columns 3..HISTORY-1; a slot < 0
// writes zero outputs and leaves history untouched; request >= NumSeqs or end <= start does nothing.
// Two launches keep it race-free: conv_main reads the old history, conv_history (stream-ordered after) writes the new.
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int CHANNELS = 10240;
constexpr int TOKENS_PER_CTA = 16;

__device__ __forceinline__ float ex2_approx(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float div_full(float a, float b) { float y; asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float bf16_round(float v) {
    uint32_t bits = __float_as_uint(v);
    uint32_t rounded = (bits + ((bits >> 16) & 1u) + 32767u) & 0xFFFF0000u;
    uint32_t nan = (bits & 0xFFBF0000u) | 0x00400000u;
    return __uint_as_float((bits & 0x7FFFFFFFu) > 0x7F800000u ? nan : rounded);
}

__device__ __forceinline__ float tap(const float* __restrict__ x, int x_stride, const float* __restrict__ history,
                                     bool fresh, int start, int p, int ch) {
    if (p >= start) return bf16_round(__ldg(x + (int64_t)p * x_stride + ch));
    return fresh ? 0.0f : __ldg(history + (p - start + 3));
}

// grid: (ceil(capacity / TOKENS_PER_CTA), CHANNELS / 128, requests); block 128 (one channel per thread).
__global__ void __launch_bounds__(128) conv_main(
        const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ history,
        const int* __restrict__ starts, const int64_t* __restrict__ slots, const uint8_t* __restrict__ fresh_flags,
        const int* __restrict__ num_seqs, float* __restrict__ output,
        int x_stride, int weight_stride, int history_stride, int channel_stride) {
    const int request = blockIdx.z;
    if (request >= *num_seqs) return;
    const int start = starts[request], end = starts[request + 1];
    const int t0 = start + blockIdx.x * TOKENS_PER_CTA;
    if (t0 >= end) return;
    const int ch = blockIdx.y * 128 + threadIdx.x;
    const int t1 = min(end, t0 + TOKENS_PER_CTA);
    const int64_t slot = slots[request];
    if (slot < 0) {
        for (int t = t0; t < t1; ++t) output[(int64_t)t * CHANNELS + ch] = 0.0f;
        return;
    }
    const bool fresh = fresh_flags[request] != 0;
    const float* hist = history + slot * (int64_t)history_stride + (int64_t)ch * channel_stride;
    const float w0 = weight[(int64_t)ch * weight_stride], w1 = weight[(int64_t)ch * weight_stride + 1];
    const float w2 = weight[(int64_t)ch * weight_stride + 2], w3 = weight[(int64_t)ch * weight_stride + 3];
    float h0 = tap(x, x_stride, hist, fresh, start, t0 - 3, ch);
    float h1 = tap(x, x_stride, hist, fresh, start, t0 - 2, ch);
    float h2 = tap(x, x_stride, hist, fresh, start, t0 - 1, ch);
#pragma unroll 4
    for (int t = t0; t < t1; ++t) {
        const float value = bf16_round(__ldg(x + (int64_t)t * x_stride + ch));
        const float s = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(h0, w0), __fmul_rn(h1, w1)), __fmul_rn(h2, w2)),
                                  __fmul_rn(value, w3));
        const float c = bf16_round(s);
        const float activated = bf16_round(div_full(c, __fadd_rn(ex2_approx(__fmul_rn(c, -1.44269502f)), 1.0f)));
        output[(int64_t)t * CHANNELS + ch] = activated;
        h0 = h1; h1 = h2; h2 = value;
    }
}

// grid: (CHANNELS / 128, requests); block 128. Runs after conv_main in stream order.
template <int HISTORY>
__global__ void __launch_bounds__(128) conv_history(
        const float* __restrict__ x, float* __restrict__ history, const int* __restrict__ starts,
        const int64_t* __restrict__ slots, const uint8_t* __restrict__ fresh_flags, const int* __restrict__ num_seqs,
        int x_stride, int history_stride, int channel_stride) {
    const int request = blockIdx.y;
    if (request >= *num_seqs) return;
    const int start = starts[request], end = starts[request + 1];
    if (end <= start) return;
    const int64_t slot = slots[request];
    if (slot < 0) return;
    const int ch = blockIdx.x * 128 + threadIdx.x;
    const bool fresh = fresh_flags[request] != 0;
    float* hist = history + slot * (int64_t)history_stride + (int64_t)ch * channel_stride;
    // read all three before writing (overlapping columns when end - start < 3)
    const float a = tap(x, x_stride, hist, fresh, start, end - 3, ch);
    const float b = tap(x, x_stride, hist, fresh, start, end - 2, ch);
    const float c = tap(x, x_stride, hist, fresh, start, end - 1, ch);
    hist[0] = a; hist[1] = b; hist[2] = c;
#pragma unroll
    for (int column = 3; column < HISTORY; ++column) hist[column] = 0.0f;
}
}  // namespace

extern "C" int sol15_conv_prefill(const float* x, const float* weight, float* history, const int* starts,
                                  const int64_t* slots, const uint8_t* fresh, const int* num_seqs, float* output,
                                  int x_stride, int weight_stride, int history_stride, int channel_stride,
                                  int history_columns, int capacity, int requests, cudaStream_t stream) {
    if (capacity <= 0 || requests <= 0 || history_columns != 7) return (int)cudaErrorInvalidValue;
    // Argon required check (REVIEW-ARGON-CONV.md): conv_main reads boundary taps of x across CTAs while writing output.
    if (static_cast<const void*>(x) == static_cast<const void*>(output)) return (int)cudaErrorInvalidValue;
    dim3 main_grid((capacity + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA, CHANNELS / 128, requests);
    conv_main<<<main_grid, 128, 0, stream>>>(x, weight, history, starts, slots, fresh, num_seqs, output, x_stride,
                                             weight_stride, history_stride, channel_stride);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    conv_history<7><<<dim3(CHANNELS / 128, requests), 128, 0, stream>>>(x, history, starts, slots, fresh, num_seqs,
                                                                         x_stride, history_stride, channel_stride);
    return (int)cudaGetLastError();
}
