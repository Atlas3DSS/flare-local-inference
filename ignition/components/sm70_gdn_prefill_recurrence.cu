// Sol-15: bit-identical, faster replacement for sm70_gdn._recurrence with PREFILL=True on Volta (sm_70).
//
// The reference is the Triton 3.8 kernel in src-mtp/ignition/components/sm70_gdn.py, compiled with num_warps=4,
// num_stages=1, enable_fp_fusion=False (triton-cache XNAG7YQR..., code sha256 eff220e8..., the function the astra-7
// prefill128 graph runs 48x per chunk). Every floating-point operation below reproduces one PTX instruction of that
// kernel, in the same order, with the same rounding mode and FTZ flavour, so outputs and recurrent state are
// bit-identical by construction (the GPU gate proves it):
//   * q/k L2 norm: Triton holds element i of the 128-vector in thread i of 4 warps (#blocked1). Each warp reduces its
//     32 elements with shfl.bfly 16,8,4,2,1 (own + other), lane 0 of warp w stores W_w, every thread reloads
//     W[tid & 3] and reduces with shfl.bfly 2,1 -> total = (W0 + W2) + (W1 + W3). Then add.rn 1e-6, sqrt.rn,
//     div.full.f32 1/x, mul.rn (inv * x), and for q a second mul.rn by 0.0883883461.
//   * state tile (#blocked: sizePerThread [1,4], threadsPerWarp [1,32]): lane l holds keys 4l..4l+3 of a row; a row
//     dot product is ((p0 + p1) + (p2 + p3)) in-thread, then shfl.bfly 16,8,4,2,1 (own + other).
//   * decay = ex2.approx(mul.rn(mul.rn(softplus, -ex2.approx(mul.rn(A_log, log2e))), log2e)); softplus selects the
//     inlined libdevice __nv_log1pf sequence (copied verbatim, incl. its .rz.ftz/.rn.ftz flavours) when a <= 20.
//   * beta = bf16(div.full(1, add.rn(ex2.approx(mul.rn(bf16(b), -log2e)), 1))).
//   * S = S*decay (mul.rn); r = row dot(S, k); delta = (v - r)*beta; S = S + k*delta (mul.rn then add.rn, no FMA);
//     out = bf16(row dot(S, q)).
// Speed comes only from restructuring: (1) the per-token q/k norms, decay and beta are computed ONCE per token in a
// fully parallel pre-pass instead of 48x redundantly inside the sequential loop with three CTA barriers per token;
// (2) the sequential scan keeps whole rows inside one warp, so the token loop has no barrier and no shared memory;
// (3) next-token operands are prefetched while the current token computes.
//
// Build: nvcc -O3 -arch=compute_70 -code=sm_70 --fmad=false -Xcompiler -fPIC -shared (CUDA 12.9).
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int KEY_HEADS = 16, VALUE_HEADS = 48, D = 128;
constexpr int GROUP = VALUE_HEADS / KEY_HEADS;      // 3 value heads share one key head
constexpr int V_OFFSET = 2 * KEY_HEADS * D;          // 4096: value columns inside a mixed row
constexpr float LOG2E = 1.44269502f;                 // 0f3FB8AA3B
constexpr float NEG_LOG2E = -1.44269502f;            // 0fBFB8AA3B
constexpr float Q_SCALE = 0.0883883461f;             // 0f3DB504F3
constexpr float NORM_EPS = 9.99999997e-07f;          // 0f358637BD

__device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sub_rn(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float ex2_approx(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float div_full(float a, float b) { float y; asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float sqrt_rn(float x) { float y; asm("sqrt.rn.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

// _bf16 exactly as the PTX: bits + ((bits >> 16) & 1) + 0x7FFF, & 0xFFFF0000; NaN -> (bits & 0xFFBF0000) | 0x400000.
__device__ __forceinline__ float bf16_round(float v) {
    uint32_t bits = __float_as_uint(v);
    uint32_t rounded = (bits + ((bits >> 16) & 1u) + 32767u) & 0xFFFF0000u;
    uint32_t nan = (bits & 0xFFBF0000u) | 0x00400000u;
    return __uint_as_float((bits & 0x7FFFFFFFu) > 0x7F800000u ? nan : rounded);
}

// libdevice __nv_log1pf as inlined into the reference PTX (lines "add.rz.ftz.f32 %r117" .. "$L__BB0_10"), branches
// replaced by predication with identical results.
__device__ __forceinline__ float log1p_reference(float e) {
    float r;
    asm("{\n\t"
        ".reg .b32 e, r, one, t117, t118, t119, t120, t121, t122, t123, t124, t125, t126, t127, t128, t129, t130, t131, t132;\n\t"
        ".reg .b32 t133, t134, t135, t136, t137, t138, t139, t140, t141, t142, t143, t144, t145, t146, t147, t148;\n\t"
        ".reg .pred p8, p9, p10, p11, n8, p12;\n\t"
        "mov.b32 e, %1;\n\t"
        "mov.b32 one, 0f3F800000;\n\t"
        "add.rz.ftz.f32 t117, e, one;\n\t"
        "add.s32 t118, t117, -1061158912;\n\t"
        "and.b32 t119, t118, -8388608;\n\t"
        "sub.s32 t120, e, t119;\n\t"
        "sub.s32 t121, 1082130432, t119;\n\t"
        "mov.b32 t122, 0fBF800000;\n\t"
        "mov.b32 t123, 0f3E800000;\n\t"
        "fma.rn.ftz.f32 t124, t123, t121, t122;\n\t"
        "add.rn.f32 t125, t124, t120;\n\t"
        "cvt.rn.f32.s32 t126, t119;\n\t"
        "mul.rn.f32 t127, t126, 0f34000000;\n\t"
        "mov.b32 t128, 0f3DD80012;\n\t"
        "mov.b32 t129, 0fBD39BF78;\n\t"
        "fma.rn.ftz.f32 t130, t129, t125, t128;\n\t"
        "mov.b32 t131, 0fBE0778E0;\n\t"
        "fma.rn.ftz.f32 t132, t130, t125, t131;\n\t"
        "mov.b32 t133, 0f3E146475;\n\t"
        "fma.rn.ftz.f32 t134, t132, t125, t133;\n\t"
        "mov.b32 t135, 0fBE2A68DD;\n\t"
        "fma.rn.ftz.f32 t136, t134, t125, t135;\n\t"
        "mov.b32 t137, 0f3E4CAF9E;\n\t"
        "fma.rn.ftz.f32 t138, t136, t125, t137;\n\t"
        "mov.b32 t139, 0fBE800042;\n\t"
        "fma.rn.ftz.f32 t140, t138, t125, t139;\n\t"
        "mov.b32 t141, 0f3EAAAAE6;\n\t"
        "fma.rn.ftz.f32 t142, t140, t125, t141;\n\t"
        "mov.b32 t143, 0fBF000000;\n\t"
        "fma.rn.ftz.f32 t144, t142, t125, t143;\n\t"
        "mul.rn.f32 t145, t125, t144;\n\t"
        "fma.rn.ftz.f32 t146, t145, t125, t125;\n\t"
        "mov.b32 t147, 0f3F317218;\n\t"
        "fma.rn.ftz.f32 r, t127, t147, t146;\n\t"
        "setp.lt.u32 p8, e, 2139095040;\n\t"          // reference: @p8 bra $L__BB0_10 (done)
        "setp.lt.s32 p9, e, -1082130431;\n\t"         // reference: @p9 bra $L__BB0_9 (skip the inf fma)
        "or.pred p11, p8, p9;\n\t"
        "mov.b32 t148, 0f7F800000;\n\t"
        "@!p11 fma.rn.ftz.f32 r, e, t148, t148;\n\t"
        "setp.eq.f32 p10, e, 0f00000000;\n\t"        // $L__BB0_9: e == 0 -> -0.0 (only reached when !p8)
        "not.pred n8, p8;\n\t"
        "and.pred p12, p10, n8;\n\t"
        "@p12 mov.b32 r, 0f80000000;\n\t"
        "mov.b32 %0, r;\n\t"
        "}"
        : "=f"(r) : "f"(e));
    return r;
}

// One warp reduces x[32j + lane] for j = 0..3 exactly as warps 0..3 of the reference do, then combines the four
// warp sums as the reference's cross-warp step does: (W0 + W2) + (W1 + W3).
__device__ __forceinline__ float norm_sum_reference(float x0, float x1, float x2, float x3) {
    float w0 = mul_rn(x0, x0), w1 = mul_rn(x1, x1), w2 = mul_rn(x2, x2), w3 = mul_rn(x3, x3);
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        w0 = add_rn(w0, __shfl_xor_sync(0xffffffffu, w0, offset));
        w1 = add_rn(w1, __shfl_xor_sync(0xffffffffu, w1, offset));
        w2 = add_rn(w2, __shfl_xor_sync(0xffffffffu, w2, offset));
        w3 = add_rn(w3, __shfl_xor_sync(0xffffffffu, w3, offset));
    }
    return add_rn(add_rn(w0, w2), add_rn(w1, w3));
}

// Pre-pass: one CTA per token row of the capacity; rows outside [starts[0], starts[num_seqs]) exit at once. Warp w
// (0..15) normalizes q and k of key head w; threads
// 0..47 compute decay and beta of value head tid. Outputs: qn/kn [capacity][16][128], decay/beta [capacity][48].
__global__ void __launch_bounds__(512) prepare_kernel(
        const float* __restrict__ mixed, const float* __restrict__ a, const float* __restrict__ b,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ decay, float* __restrict__ beta,
        const int* __restrict__ starts, const int* __restrict__ num_seqs, int mixed_stride, int a_stride, int b_stride) {
    const int64_t token = blockIdx.x;
    // Argon required change 1: only rows of live requests are read (never uninitialized capacity tail rows).
    const int sequences = *num_seqs;
    if (sequences <= 0 || token < starts[0] || token >= starts[sequences]) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float* row = mixed + token * (int64_t)mixed_stride;
    {
        const float* q = row + warp * D;
        const float* k = row + KEY_HEADS * D + warp * D;
        float q0 = q[lane], q1 = q[32 + lane], q2 = q[64 + lane], q3 = q[96 + lane];
        float k0 = k[lane], k1 = k[32 + lane], k2 = k[64 + lane], k3 = k[96 + lane];
        const float q_inv = div_full(1.0f, sqrt_rn(add_rn(norm_sum_reference(q0, q1, q2, q3), NORM_EPS)));
        const float k_inv = div_full(1.0f, sqrt_rn(add_rn(norm_sum_reference(k0, k1, k2, k3), NORM_EPS)));
        float* qo = qn + (token * KEY_HEADS + warp) * D;
        float* ko = kn + (token * KEY_HEADS + warp) * D;
        qo[lane] = mul_rn(mul_rn(q_inv, q0), Q_SCALE);
        qo[32 + lane] = mul_rn(mul_rn(q_inv, q1), Q_SCALE);
        qo[64 + lane] = mul_rn(mul_rn(q_inv, q2), Q_SCALE);
        qo[96 + lane] = mul_rn(mul_rn(q_inv, q3), Q_SCALE);
        ko[lane] = mul_rn(k_inv, k0);
        ko[32 + lane] = mul_rn(k_inv, k1);
        ko[64 + lane] = mul_rn(k_inv, k2);
        ko[96 + lane] = mul_rn(k_inv, k3);
    }
    if (threadIdx.x < VALUE_HEADS) {
        const int head = threadIdx.x;
        const float neg_scale = -ex2_approx(mul_rn(a_log[head], LOG2E));   // neg.f32 of ex2.approx: exact negation
        const float av = add_rn(dt_bias[head], bf16_round(a[token * (int64_t)a_stride + head]));
        const float e = ex2_approx(mul_rn(av, LOG2E));
        const float softplus = (av <= 20.0f) ? log1p_reference(e) : av;    // setp.le.f32: NaN selects av
        decay[token * VALUE_HEADS + head] = ex2_approx(mul_rn(mul_rn(softplus, neg_scale), LOG2E));
        const float bb = bf16_round(b[token * (int64_t)b_stride + head]);
        const float sig = div_full(1.0f, add_rn(ex2_approx(mul_rn(bb, NEG_LOG2E)), 1.0f));
        beta[token * VALUE_HEADS + head] = bf16_round(sig);
    }
}

template <int R>
struct Operands {
    float4 k, q;
    float decay, beta, v[R];
};

template <int R>
__device__ __forceinline__ void load_operands(Operands<R>& o, const float* __restrict__ qn, const float* __restrict__ kn,
        const float* __restrict__ decay, const float* __restrict__ beta, const float* __restrict__ mixed,
        int64_t token, int key_head, int head, int lane, int row0, int mixed_stride) {
    o.k = __ldg(reinterpret_cast<const float4*>(kn + (token * KEY_HEADS + key_head) * D) + lane);
    o.q = __ldg(reinterpret_cast<const float4*>(qn + (token * KEY_HEADS + key_head) * D) + lane);
    o.decay = __ldg(decay + token * VALUE_HEADS + head);
    o.beta = __ldg(beta + token * VALUE_HEADS + head);
    const float* v = mixed + token * (int64_t)mixed_stride + V_OFFSET + head * D + row0;
#pragma unroll
    for (int i = 0; i < R; ++i) o.v[i] = __ldg(v + i);
}

__device__ __forceinline__ float row_sum_reference(float p0, float p1, float p2, float p3) {
    float s = add_rn(add_rn(p0, p1), add_rn(p2, p3));
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) s = add_rn(s, __shfl_xor_sync(0xffffffffu, s, offset));
    return s;
}

// Sequential scan. Each warp owns R consecutive rows of one value head; CTA = WARPS warps; grid.x = 128/(WARPS*R),
// grid.y = requests * 48 (request-major, as the reference's program_id(1)).
// MIN_BLOCKS keeps every CTA of a 48-head grid resident in ONE wave (a second wave would double the scan time).
template <int WARPS, int R, int MIN_BLOCKS>
__global__ void __launch_bounds__(WARPS * 32, MIN_BLOCKS) scan_kernel(
        const float* __restrict__ mixed, const float* __restrict__ qn, const float* __restrict__ kn,
        const float* __restrict__ decay, const float* __restrict__ beta, float* __restrict__ state,
        const int* __restrict__ starts, const int64_t* __restrict__ initial_slots, const int64_t* __restrict__ final_slots,
        const int64_t* __restrict__ checkpoint_slots, const int* __restrict__ checkpoint_offsets,
        const uint8_t* __restrict__ fresh_flags, const int* __restrict__ num_seqs, float* __restrict__ output,
        int mixed_stride, int state_stride) {
    const int request = blockIdx.y / VALUE_HEADS, head = blockIdx.y % VALUE_HEADS, key_head = head / GROUP;
    if (request >= *num_seqs) return;
    const int start = starts[request], end = starts[request + 1];
    if (end <= start) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * WARPS + warp) * R;
    const int64_t final_slot = final_slots[request];
    if (final_slot == 0) {
        if (lane < R)
            for (int position = start; position < end; ++position)
                output[(int64_t)position * (VALUE_HEADS * D) + head * D + row0 + lane] = 0.0f;
        return;
    }
    const int64_t initial_slot = initial_slots[request];
    const int64_t checkpoint_slot = checkpoint_slots[request];
    const int boundary = checkpoint_offsets[request];
    const bool load = (initial_slot != 0) && (fresh_flags[request] == 0);
    float s[R][4];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (load)
            x = *reinterpret_cast<const float4*>(state + initial_slot * state_stride + (int64_t)head * D * D
                                                 + (int64_t)(row0 + i) * D + 4 * lane);
        s[i][0] = x.x; s[i][1] = x.y; s[i][2] = x.z; s[i][3] = x.w;
    }
    Operands<R> cur, nxt;
    load_operands<R>(cur, qn, kn, decay, beta, mixed, start, key_head, head, lane, row0, mixed_stride);
    for (int position = start; position < end; ++position) {
        if (position + 1 < end)
            load_operands<R>(nxt, qn, kn, decay, beta, mixed, position + 1, key_head, head, lane, row0, mixed_stride);
        const float k[4] = {cur.k.x, cur.k.y, cur.k.z, cur.k.w};
        const float q[4] = {cur.q.x, cur.q.y, cur.q.z, cur.q.w};
        float out[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
            float sd[4];
#pragma unroll
            for (int c = 0; c < 4; ++c) sd[c] = mul_rn(s[i][c], cur.decay);
            const float r = row_sum_reference(mul_rn(sd[0], k[0]), mul_rn(sd[1], k[1]),
                                              mul_rn(sd[2], k[2]), mul_rn(sd[3], k[3]));
            const float delta = mul_rn(sub_rn(cur.v[i], r), cur.beta);
#pragma unroll
            for (int c = 0; c < 4; ++c) s[i][c] = add_rn(sd[c], mul_rn(k[c], delta));
            out[i] = bf16_round(row_sum_reference(mul_rn(s[i][0], q[0]), mul_rn(s[i][1], q[1]),
                                                  mul_rn(s[i][2], q[2]), mul_rn(s[i][3], q[3])));
        }
        float* o = output + (int64_t)position * (VALUE_HEADS * D) + head * D + row0;
#pragma unroll
        for (int i = 0; i < R; ++i)
            if (lane == i) o[i] = out[i];
        if (checkpoint_slot > 0 && position - start + 1 == boundary) {
#pragma unroll
            for (int i = 0; i < R; ++i)
                *reinterpret_cast<float4*>(state + checkpoint_slot * state_stride + (int64_t)head * D * D
                                           + (int64_t)(row0 + i) * D + 4 * lane) = make_float4(s[i][0], s[i][1], s[i][2], s[i][3]);
        }
        cur = nxt;
    }
#pragma unroll
    for (int i = 0; i < R; ++i)
        *reinterpret_cast<float4*>(state + final_slot * state_stride + (int64_t)head * D * D
                                   + (int64_t)(row0 + i) * D + 4 * lane) = make_float4(s[i][0], s[i][1], s[i][2], s[i][3]);
}

// ---- Variant 3/4 (Sol-15 v2, after G3 = 10.5 ms): reduce-scatter row sums, 4 rows per warp. ----
// The reference reduces each row with a full butterfly in which lanes l and l^off compute the same node
// (own + other, commutative). With four rows in a warp, every butterfly node is still computed from exactly the same two
// operands, but each node only once: offset 16 splits the rows (lanes < 16 finish rows 0,1; lanes >= 16 rows 2,3),
// offset 8 splits again (lane group g = lane >> 3 finishes row g), offsets 4,2,1 are the reference's own steps inside
// the group. Every add is the reference's add.rn on the reference's operand pair, so the row totals are bit-identical;
// SHFL per row-reduction drops from 5 to 1.5 and FADD from 5 to 1.5 (in-thread (p0+p1)+(p2+p3) unchanged).
__device__ __forceinline__ float scatter_sum4(float p0, float p1, float p2, float p3, bool hi16, bool b8) {
    const float send0 = hi16 ? p0 : p2, send1 = hi16 ? p1 : p3;
    const float keep0 = hi16 ? p2 : p0, keep1 = hi16 ? p3 : p1;
    const float a0 = add_rn(keep0, __shfl_xor_sync(0xffffffffu, send0, 16));   // row (hi16 ? 2 : 0), node (l, l^16)
    const float a1 = add_rn(keep1, __shfl_xor_sync(0xffffffffu, send1, 16));   // row (hi16 ? 3 : 1)
    const float send = b8 ? a0 : a1, keep = b8 ? a1 : a0;
    float b = add_rn(keep, __shfl_xor_sync(0xffffffffu, send, 8));             // row lane >> 3, node (l, l^8)
    b = add_rn(b, __shfl_xor_sync(0xffffffffu, b, 4));
    b = add_rn(b, __shfl_xor_sync(0xffffffffu, b, 2));
    b = add_rn(b, __shfl_xor_sync(0xffffffffu, b, 1));
    return b;
}

struct OperandsRS {
    float4 k, q;
    float decay, beta, v;   // v of row row0 + (lane >> 3)
};

__device__ __forceinline__ void load_operands_rs(OperandsRS& o, const float* __restrict__ qn, const float* __restrict__ kn,
        const float* __restrict__ decay, const float* __restrict__ beta, const float* __restrict__ mixed,
        int64_t token, int key_head, int head, int lane, int row0, int mixed_stride) {
    o.k = __ldg(reinterpret_cast<const float4*>(kn + (token * KEY_HEADS + key_head) * D) + lane);
    o.q = __ldg(reinterpret_cast<const float4*>(qn + (token * KEY_HEADS + key_head) * D) + lane);
    o.decay = __ldg(decay + token * VALUE_HEADS + head);
    o.beta = __ldg(beta + token * VALUE_HEADS + head);
    o.v = __ldg(mixed + token * (int64_t)mixed_stride + V_OFFSET + head * D + row0 + (lane >> 3));
}

template <int WARPS, int MIN_BLOCKS>
__global__ void __launch_bounds__(WARPS * 32, MIN_BLOCKS) scan_rs_kernel(
        const float* __restrict__ mixed, const float* __restrict__ qn, const float* __restrict__ kn,
        const float* __restrict__ decay, const float* __restrict__ beta, float* __restrict__ state,
        const int* __restrict__ starts, const int64_t* __restrict__ initial_slots, const int64_t* __restrict__ final_slots,
        const int64_t* __restrict__ checkpoint_slots, const int* __restrict__ checkpoint_offsets,
        const uint8_t* __restrict__ fresh_flags, const int* __restrict__ num_seqs, float* __restrict__ output,
        int mixed_stride, int state_stride) {
    constexpr int R = 4;
    const int request = blockIdx.y / VALUE_HEADS, head = blockIdx.y % VALUE_HEADS, key_head = head / GROUP;
    if (request >= *num_seqs) return;
    const int start = starts[request], end = starts[request + 1];
    if (end <= start) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * WARPS + warp) * R;
    const bool hi16 = lane >= 16, b8 = (lane & 8) != 0, leader = (lane & 7) == 0;
    const int group = lane >> 3;
    const int64_t final_slot = final_slots[request];
    if (final_slot == 0) {
        if (lane < R)
            for (int position = start; position < end; ++position)
                output[(int64_t)position * (VALUE_HEADS * D) + head * D + row0 + lane] = 0.0f;
        return;
    }
    const int64_t initial_slot = initial_slots[request];
    const int64_t checkpoint_slot = checkpoint_slots[request];
    const int boundary = checkpoint_offsets[request];
    const bool load = (initial_slot != 0) && (fresh_flags[request] == 0);
    float s[R][4];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        float4 x = make_float4(0.f, 0.f, 0.f, 0.f);
        if (load)
            x = *reinterpret_cast<const float4*>(state + initial_slot * state_stride + (int64_t)head * D * D
                                                 + (int64_t)(row0 + i) * D + 4 * lane);
        s[i][0] = x.x; s[i][1] = x.y; s[i][2] = x.z; s[i][3] = x.w;
    }
    OperandsRS cur, nxt;
    load_operands_rs(cur, qn, kn, decay, beta, mixed, start, key_head, head, lane, row0, mixed_stride);
    for (int position = start; position < end; ++position) {
        if (position + 1 < end)
            load_operands_rs(nxt, qn, kn, decay, beta, mixed, position + 1, key_head, head, lane, row0, mixed_stride);
        const float k[4] = {cur.k.x, cur.k.y, cur.k.z, cur.k.w};
        const float q[4] = {cur.q.x, cur.q.y, cur.q.z, cur.q.w};
        float sd[R][4], part[R];
#pragma unroll
        for (int i = 0; i < R; ++i) {
#pragma unroll
            for (int c = 0; c < 4; ++c) sd[i][c] = mul_rn(s[i][c], cur.decay);
            part[i] = add_rn(add_rn(mul_rn(sd[i][0], k[0]), mul_rn(sd[i][1], k[1])),
                             add_rn(mul_rn(sd[i][2], k[2]), mul_rn(sd[i][3], k[3])));
        }
        const float r = scatter_sum4(part[0], part[1], part[2], part[3], hi16, b8);   // row `group` total
        const float d = mul_rn(sub_rn(cur.v, r), cur.beta);                            // delta of row `group`
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const float delta = __shfl_sync(0xffffffffu, d, 8 * i);
#pragma unroll
            for (int c = 0; c < 4; ++c) s[i][c] = add_rn(sd[i][c], mul_rn(k[c], delta));
            part[i] = add_rn(add_rn(mul_rn(s[i][0], q[0]), mul_rn(s[i][1], q[1])),
                             add_rn(mul_rn(s[i][2], q[2]), mul_rn(s[i][3], q[3])));
        }
        const float out = bf16_round(scatter_sum4(part[0], part[1], part[2], part[3], hi16, b8));
        if (leader) output[(int64_t)position * (VALUE_HEADS * D) + head * D + row0 + group] = out;
        if (checkpoint_slot > 0 && position - start + 1 == boundary) {
#pragma unroll
            for (int i = 0; i < R; ++i)
                *reinterpret_cast<float4*>(state + checkpoint_slot * state_stride + (int64_t)head * D * D
                                           + (int64_t)(row0 + i) * D + 4 * lane) = make_float4(s[i][0], s[i][1], s[i][2], s[i][3]);
        }
        cur = nxt;
    }
#pragma unroll
    for (int i = 0; i < R; ++i)
        *reinterpret_cast<float4*>(state + final_slot * state_stride + (int64_t)head * D * D
                                   + (int64_t)(row0 + i) * D + 4 * lane) = make_float4(s[i][0], s[i][1], s[i][2], s[i][3]);
}

template <int WARPS, int MIN_BLOCKS>
cudaError_t launch_scan_rs(const float* mixed, const float* qn, const float* kn, const float* dec, const float* bet,
                           float* state, const int* starts, const int64_t* initial, const int64_t* final_,
                           const int64_t* checkpoints, const int* offsets, const uint8_t* fresh, const int* num_seqs,
                           float* output, int mixed_stride, int state_stride, int requests, cudaStream_t stream) {
    dim3 grid(D / (WARPS * 4), requests * VALUE_HEADS);
    scan_rs_kernel<WARPS, MIN_BLOCKS><<<grid, WARPS * 32, 0, stream>>>(mixed, qn, kn, dec, bet, state, starts, initial,
        final_, checkpoints, offsets, fresh, num_seqs, output, mixed_stride, state_stride);
    return cudaGetLastError();
}

template <int WARPS, int R, int MIN_BLOCKS>
cudaError_t launch_scan(const float* mixed, const float* qn, const float* kn, const float* dec, const float* bet,
                        float* state, const int* starts, const int64_t* initial, const int64_t* final_,
                        const int64_t* checkpoints, const int* offsets, const uint8_t* fresh, const int* num_seqs,
                        float* output, int mixed_stride, int state_stride, int requests, cudaStream_t stream) {
    static_assert(D % (WARPS * R) == 0, "rows per CTA must divide 128");
    dim3 grid(D / (WARPS * R), requests * VALUE_HEADS);
    scan_kernel<WARPS, R, MIN_BLOCKS><<<grid, WARPS * 32, 0, stream>>>(mixed, qn, kn, dec, bet, state, starts, initial, final_,
        checkpoints, offsets, fresh, num_seqs, output, mixed_stride, state_stride);
    return cudaGetLastError();
}

}  // namespace

extern "C" {

// Workspace bytes for a capacity of `capacity` token rows: qn + kn [capacity][16][128] and decay + beta [capacity][48].
size_t sol15_gdn_prefill_workspace(int capacity) {
    return (size_t)capacity * (2 * KEY_HEADS * D + 2 * VALUE_HEADS) * sizeof(float);
}

// Drop-in for _recurrence(PREFILL=True): same tensors and strides (elements), same grid semantics over `requests`.
// `capacity` = rows of mixed/a/b/output. `variant` selects the scan tiling (0: 4 warps x 2 rows, 1: 4 warps x 4 rows,
// 2: 2 warps x 4 rows; 3: 4 warps x 4 rows reduce-scatter; 4: 2 warps x 4 rows reduce-scatter); all variants are
// bit-identical (row arithmetic does not depend on the tiling or on which lane computes a butterfly node).
int sol15_gdn_prefill(const float* mixed, const float* a, const float* b, const float* a_log, const float* dt_bias,
                      float* state, const int* starts, const int64_t* initial, const int64_t* final_,
                      const int64_t* checkpoints, const int* checkpoint_offsets, const uint8_t* fresh,
                      const int* num_seqs, float* output, int mixed_stride, int a_stride, int b_stride,
                      int state_stride, int capacity, int requests, void* workspace, int variant, cudaStream_t stream) {
    if (capacity <= 0 || requests <= 0 || !workspace) return (int)cudaErrorInvalidValue;
    if (state_stride & 3) return (int)cudaErrorInvalidValue;   // float4 state rows
    float* qn = static_cast<float*>(workspace);
    float* kn = qn + (size_t)capacity * KEY_HEADS * D;
    float* dec = kn + (size_t)capacity * KEY_HEADS * D;
    float* bet = dec + (size_t)capacity * VALUE_HEADS;
    prepare_kernel<<<capacity, 512, 0, stream>>>(mixed, a, b, a_log, dt_bias, qn, kn, dec, bet, starts, num_seqs,
                                                 mixed_stride, a_stride, b_stride);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    switch (variant) {
    case 0: e = launch_scan<4, 2, 10>(mixed, qn, kn, dec, bet, state, starts, initial, final_, checkpoints,
                                  checkpoint_offsets, fresh, num_seqs, output, mixed_stride, state_stride, requests, stream); break;
    case 1: e = launch_scan<4, 4, 6>(mixed, qn, kn, dec, bet, state, starts, initial, final_, checkpoints,
                                  checkpoint_offsets, fresh, num_seqs, output, mixed_stride, state_stride, requests, stream); break;
    case 2: e = launch_scan<2, 4, 12>(mixed, qn, kn, dec, bet, state, starts, initial, final_, checkpoints,
                                  checkpoint_offsets, fresh, num_seqs, output, mixed_stride, state_stride, requests, stream); break;
    case 3: e = launch_scan_rs<4, 6>(mixed, qn, kn, dec, bet, state, starts, initial, final_, checkpoints,
                                     checkpoint_offsets, fresh, num_seqs, output, mixed_stride, state_stride, requests, stream); break;
    case 4: e = launch_scan_rs<2, 12>(mixed, qn, kn, dec, bet, state, starts, initial, final_, checkpoints,
                                      checkpoint_offsets, fresh, num_seqs, output, mixed_stride, state_stride, requests, stream); break;
    default: return (int)cudaErrorInvalidValue;
    }
    return (int)e;
}

}  // extern "C"
