"""V100 bit identity of astra-21's prefill launch changes against the launches they replace.

* Sol-20 r4 projections write a fused member's column slice of the parent output in place (output row stride `ldy`):
  every bit of the slice equals the member computed into its own contiguous [m, n] buffer (the former scratch that
  `destination.copy_` moved), padded rows [live, m) are +0.0, and no element outside the slice is written.
* Single-pass BF16 row projections: one multi-row launch over all rows equals the decode instantiation's 20-row calls
  bit for bit, for every compiled multi-row instantiation, contiguous and strided operands, and row counts that are not
  multiples of R.
"""

import ctypes as C
import os

import pytest
import torch


from ignition.components.sm70_dense import ROWS


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned V100 device")
POISON = 0x7FC0DEAD
# Every admitted (K, N) of each tactic: GDN in_proj_ba; MTP fc and draft down; MTP k/v.
SHAPES = {22: ((5120, 96),), 42: ((10240, 5120), (17408, 5120)), 54: ((5120, 1024),)}


def _require_v100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")


def _bits(tensor):
    return tensor.contiguous().view(torch.int32)


def _packed(n, pk, fp4, generator):
    """Random packed weights (bytes as the kernels read them) and finite E4M3 scale words for NVFP4."""
    values = torch.randint(0, 256, (n * pk // (2 if fp4 else 1),), dtype=torch.uint8, generator=generator)
    if not fp4:
        values[(values & 127) == 127] = 0x38          # finite E4M3 only (load_weight rejects NaN codes)
        return values.cuda(), None
    scales = torch.randint(0x20, 0x48, (n * pk // 16,), dtype=torch.uint8, generator=generator)
    return values.cuda(), scales.cuda()


@pytest.mark.parametrize("fp4,k,widths", [
    (True, 5120, (17408, 17408)),           # NVFP4 gate/up members (split 1)
    (True, 17408, (5120,)),                 # NVFP4 down (split 2 + finish_live), contiguous parent
    (False, 5120, (10240, 6144)),           # FP8 GDN qkv/z members
    (False, 5120, (12288, 1024, 1024)),     # FP8 attention q+gate/k/v members (k/v split 4)
])
@pytest.mark.parametrize("m,live", [(128, 128), (128, 113), (128, 20), (256, 256), (256, 200), (512, 512), (512, 300)])  # prefill-lab EXP-7: capacities 256, 512
@torch.no_grad()
def test_pgemm4_member_slices_are_written_in_place_with_contiguous_bits(fp4, k, widths, m, live):
    from ignition.components import sm70_prefill
    from ignition.components.sm70_projection_tc import ptr

    _require_v100()
    if not os.environ.get("FLARE_CUTLASS_INCLUDE"):
        pytest.skip("the SM70 prefill library requires FLARE_CUTLASS_INCLUDE (CUTLASS include directory)")
    generator = torch.Generator().manual_seed(2101 + k + live + m)
    lib = sm70_prefill.library(None, os.environ["FLARE_CUTLASS_INCLUDE"])
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    total, kind = sum(widths), "nvfp4" if fp4 else "tensor_fp8"
    x = (torch.randn((m, k), generator=generator) * 0.25).half().cuda()
    inverse = (torch.rand((m,), generator=generator) + 0.5).cuda()
    count = torch.tensor([live], dtype=torch.int32, device="cuda")
    # Poisoned guard columns on both sides of the parent (Argon R1 note, Astra R1 required change 1): before each member
    # launch the whole storage is snapshotted, and afterwards every element outside that member's slice must be unchanged.
    guard = 32
    storage = torch.empty((m, guard + total + guard), device="cuda")
    storage.view(torch.int32).fill_(POISON)
    parent = storage[:, guard:guard + total]
    references, column = [], 0
    for n in widths:
        split = sm70_prefill.PINNED[kind, k, n][1]
        cfg = sm70_prefill.tile(kind, k, n, m)
        values, scales = _packed(n, k, fp4, generator)
        partial = torch.empty((max(split, 1) * m * n,), device="cuda")
        gain = 0.03125 if fp4 else 0.0078125
        reference = torch.empty((m, n), device="cuda")
        reference.view(torch.int32).fill_(POISON)
        assert lib.atlas_pgemm4(ptr(x), ptr(values), ptr(scales), ptr(inverse), gain, 0, None, ptr(reference),
                                ptr(partial), m, n, k, k, split, fp4, cfg, ptr(count), n, stream) == 0
        before = storage.clone()
        assert lib.atlas_pgemm4(ptr(x), ptr(values), ptr(scales), ptr(inverse), gain, 0, None,
                                ptr(parent[:, column:column + n]), ptr(partial), m, n, k, k, split, fp4, cfg, ptr(count),
                                guard + total + guard, stream) == 0
        torch.cuda.synchronize()
        outside = torch.ones((m, guard + total + guard), dtype=torch.bool, device="cuda")
        outside[:, guard + column:guard + column + n] = False
        assert torch.equal(_bits(storage)[outside], _bits(before)[outside])
        assert torch.equal(_bits(parent[:, column:column + n]), _bits(reference))
        assert not bool(_bits(reference[live:]).any())                      # padded rows +0.0
        assert not bool((_bits(reference[:live]) == POISON).any())          # every live element written
        references.append(reference)
        column += n
    assert column == total
    # After every member: earlier slices were not clobbered by later launches, and the guard columns are untouched.
    assert torch.equal(_bits(parent), _bits(torch.cat(references, dim=1)))
    assert bool((_bits(storage[:, :guard]) == POISON).all()) and bool((_bits(storage[:, guard + total:]) == POISON).all())
    # ldy < n is refused before any launch.
    assert lib.atlas_pgemm4(ptr(x), ptr(values), ptr(scales), ptr(inverse), 1.0, 0, None, ptr(reference),
                            ptr(partial), m, widths[-1], k, k, split, fp4, cfg, ptr(count), widths[-1] - 32,
                            stream) != 0


@pytest.mark.parametrize("tactic,k,n", [(t, k, n) for t in sorted(ROWS) for k, n in SHAPES[t]])
@pytest.mark.parametrize("rows", [21, 113, 128, 256, 512])
@pytest.mark.parametrize("strided", [False, True])
@torch.no_grad()
def test_single_pass_rows_equal_twenty_row_decode_launches(tactic, k, n, rows, strided):
    from ignition.components import sm70_dense

    _require_v100()
    generator = torch.Generator().manual_seed(2102 + tactic + rows)
    lib = sm70_dense.library(None)
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    pad, guard = (64 if strided else 0), 32
    source = torch.empty((rows, k + pad), device="cuda")[:, :k]
    values = torch.randn((rows, k), generator=generator)
    values[: rows // 2] = values[: rows // 2].bfloat16().float()        # BF16-valued and arbitrary FP32 rows
    source.copy_(values.cuda())
    weight = torch.randn((n, k), generator=generator).bfloat16().cuda().contiguous()
    expected = torch.empty((rows, n + pad), device="cuda")[:, :n]
    # The single-pass output lives inside poisoned guard rows and columns (Astra R3): nothing outside [rows, n] may change.
    storage = torch.empty((rows + guard, n + pad + guard), device="cuda")
    actual = storage[:rows, :n]
    for out in (expected, storage):
        out.view(torch.int32).fill_(POISON)
    xs, ys = source.stride(0), expected.stride(0)
    for begin in range(0, rows, sm70_dense.MAX_ROWS):
        live = min(sm70_dense.MAX_ROWS, rows - begin)
        assert lib.sol9_dense(C.c_void_p(source.data_ptr() + begin * xs * 4), C.c_void_p(weight.data_ptr()), None,
                              C.c_void_p(expected.data_ptr() + begin * ys * 4), None, live, n, k, xs, ys, 0, 0,
                              tactic, stream) == 0
    r, w, c = sm70_dense.ROWS[tactic]
    config = r | w << 8 | c << 16
    assert lib.sol9_dense_rows(C.c_void_p(source.data_ptr()), C.c_void_p(weight.data_ptr()),
                               C.c_void_p(actual.data_ptr()), rows, n, k, xs, actual.stride(0), tactic, config,
                               stream) == 0
    torch.cuda.synchronize()
    assert torch.equal(_bits(actual), _bits(expected))
    assert not bool((_bits(actual) == POISON).any())
    assert bool((_bits(storage[rows:]) == POISON).all()) and bool((_bits(storage[:, n:]) == POISON).all())
    # Uncompiled instantiations and oversized row counts are refused.
    assert lib.sol9_dense_rows(C.c_void_p(source.data_ptr()), C.c_void_p(weight.data_ptr()),
                               C.c_void_p(actual.data_ptr()), rows, n, k, xs, ys, tactic, 3 | 3 << 8, stream) != 0
    assert lib.sol9_dense_rows(C.c_void_p(source.data_ptr()), C.c_void_p(weight.data_ptr()),
                               C.c_void_p(actual.data_ptr()), 513, n, k, xs, ys, tactic, config, stream) != 0


@pytest.mark.parametrize("m,live", [(128, 128), (128, 113), (256, 256), (256, 200), (512, 512), (512, 300)])  # prefill-lab EXP-7: capacities 256, 512
@torch.no_grad()
def test_fused_swiglu_producer_equals_compiled_swiglu_then_produce(m, live):
    """astra-21 item 2: atlas_produce_swiglu on the gate|up parent equals the production pair - the Inductor-compiled
    canonical BF16 SwiGLU (compile_forward, as SM70DecoderLayer) followed by kernels14 `produce` - bit for bit in the
    FP16 rows and inverse scales of every live row, including NaN/inf/zero/subnormal/tie/huge operands and all-zero rows."""
    from ignition.components import sm70_prefill
    from ignition.components.sm70_precision import round_bf16
    from ignition.components.sm70_projection_tc import library as projection_library, ptr
    from ignition.preparation.compilation import compile_forward

    _require_v100()
    if not os.environ.get("FLARE_CUTLASS_INCLUDE"):
        pytest.skip("the SM70 prefill library requires FLARE_CUTLASS_INCLUDE (CUTLASS include directory)")
    k = 17408
    generator = torch.Generator().manual_seed(2103 + live + m)
    values = torch.randn((m, 2 * k), generator=generator) * torch.logspace(-3, 3, m)[:, None]
    special = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1e-40, -3e-39, 1e30, -1e30, 88.7,
                            -88.7, 104.0, -104.0, 1.00390625, 1.01171875, -2.0078125])
    values[3, :special.numel()] = special                               # gate specials
    values[4, k:k + special.numel()] = special                          # up specials
    values[5].zero_()                                                   # amax 0 -> shift 0, inverse 1
    values[6, ::97] = float("nan")                                      # NaN never sets the row maximum
    # Row maxima exactly at and just below powers of two (ilogbf boundary; Astra R2 note): silu(g) == g in BF16 for
    # g >= 32, so with up = 1 the row maximum is round_bf16(g).
    for row, peak in ((7, 1024.0), (8, 1020.0), (9, 2.0 ** 100), (10, 2.0 ** 100 * (1 - 2.0 ** -8)), (11, 2.0 ** -100)):
        values[row, :k] = torch.rand(k, generator=generator) * min(peak, 1.0) * 0.5
        values[row, k:] = 1.0
        values[row, 1234] = peak
    # BF16 nearest-even ties and their neighbours in gate and up (low 16 bits 0x8000, 0x7fff, 0x8001).
    ties = (torch.randint(0x3c00, 0x4400, (2, 3 * 512), generator=generator, dtype=torch.int32) << 16)
    ties |= torch.tensor([0x8000, 0x7fff, 0x8001], dtype=torch.int32).repeat(512)
    values[12, :3 * 512] = ties[0].view(torch.float32)
    values[12, k:k + 3 * 512] = ties[1].view(torch.float32)
    gate_up = values.cuda()
    product = torch.empty((m, k), device="cuda")

    def swiglu():
        gate, up = gate_up.chunk(2, dim=-1)
        gate, up = round_bf16(gate), round_bf16(up)
        product.copy_(round_bf16(round_bf16(torch.nn.functional.silu(gate)) * up))

    compile_forward(swiglu)()
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    half_ref, inverse_ref = torch.empty((m, k), dtype=torch.float16, device="cuda"), torch.empty(m, device="cuda")
    assert projection_library(None).atlas_produce(ptr(product), ptr(half_ref), ptr(inverse_ref), m, k, 0, 1.0,
                                                  stream) == 0
    lib = sm70_prefill.library(None, os.environ["FLARE_CUTLASS_INCLUDE"])
    half = torch.full((m, k), float("nan"), dtype=torch.float16, device="cuda")
    inverse = torch.full((m,), float("nan"), device="cuda")
    half_before, inverse_before = half.clone(), inverse.clone()
    count = torch.tensor([live], dtype=torch.int32, device="cuda")
    assert lib.atlas_produce_swiglu(ptr(gate_up), 2 * k, ptr(half), ptr(inverse), m, k, ptr(count), stream) == 0
    torch.cuda.synchronize()
    assert torch.equal(half[:live].view(torch.int16), half_ref[:live].view(torch.int16))
    assert torch.equal(inverse[:live].view(torch.int32), inverse_ref[:live].view(torch.int32))
    # Padded rows keep their exact prior bits (Astra R2 note: compare bits, not NaN-ness).
    assert torch.equal(half[live:].view(torch.int16), half_before[live:].view(torch.int16))
    assert torch.equal(inverse[live:].view(torch.int32), inverse_before[live:].view(torch.int32))
    # Admission: the parent row stride must hold gate and up.
    assert lib.atlas_produce_swiglu(ptr(gate_up), k, ptr(half), ptr(inverse), m, k, ptr(count), stream) != 0
