"""Plan sizing, GPU selection, capacity errors, and the plan report from declared costs and injected free memory."""

import io
import json
import re
from types import MappingProxyType

import pytest

from flare import planning
from flare.cli import display
from flare.errors import CapacityError, UsageError
from flare.recipes.declaration import Checkpoint, RankCosts, Recipe

GIB, MIB = 1 << 30, 1 << 20


def recipe(name, device_gib, ranks, *, speculative_rows, mapped_gib=0):
    checkpoint = Checkpoint(name, f"lab/{name}", "ab12cd34" + "0" * 32, "DemoForCausalLM",
                            stored_bytes=(device_gib + mapped_gib) * GIB, device_bytes=device_gib * GIB,
                            mapped_host_bytes=mapped_gib * GIB)
    return Recipe(name=name, checkpoints=(checkpoint,), context=262144, page_tokens=64, concurrency=4,
                  max_concurrency=64, speculative_rows=speculative_rows, prefill_chunk=4096, prepare=None, usage=None,
                  mapped_host_name="lookup table",
                  ranks=MappingProxyType({size: RankCosts(100 * MIB // size, 2 * MIB // size, 4 * GIB)
                                          for size in ranks}))


# One rank holds 60 GiB of weights, 4 GiB of graphs, 100 MiB per state row, and 2 MiB per page.
# At concurrency 4 it keeps 45 live rows, and its minimum of 512 pages costs 1 GiB.
SOLO = recipe("solo", 60, (1, 2, 4), speculative_rows=7, mapped_gib=4)
PAIR = recipe("pair", 160, (2, 4), speculative_rows=0)


@pytest.fixture(autouse=True)
def headroom(monkeypatch):
    monkeypatch.setattr(planning, "HEADROOM_BYTES", 4 * GIB)


def gpu(index, free_gib, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, int(free_gib * GIB),
                        capability)


def derive(model=SOLO, gpus=(gpu(0, 95),), *, selected=None, concurrency=4, reserve=0, prefill_chunk=None,
           measured=None):
    return planning.plan(model, model.checkpoints[0], tuple(gpus), selected=selected, concurrency=concurrency,
                         reserve_bytes=int(reserve * GIB), prefill_chunk=prefill_chunk, measured=measured)


def report(plan, **options):
    stream = io.StringIO()
    display.show(display.plan(plan, **options), stream=stream)
    return stream.getvalue()


def test_pages_take_all_remaining_supply():
    limited = derive()
    assert limited.fits and limited.cache_states == limited.cache_target == 32
    assert limited.supply_bytes == 91 * GIB and limited.live_bytes == 45 * 100 * MIB
    # 91 GiB of supply less weights, graphs, 45 live rows, and 32 cached rows leaves 9,974 pages.
    assert limited.pages == (91 * GIB - 64 * GIB - 77 * 100 * MIB) // (2 * MIB) == 9974
    assert limited.demand_pages == 16384 and limited.minimum_pages == 512
    assert limited.context == 262144 and limited.guaranteed_tokens == 9974 // 4 * 64
    assert limited.snapshots == 32 + 12

    assert derive(gpus=(gpu(0, 95), gpu(1, 95))).world_size == 1

    # One request needs 4,096 pages at full context; the rest of the supply holds cached prefixes.
    whole = derive(concurrency=1)
    assert whole.demand_pages == 4096 and whole.pages == 12824 and whole.guaranteed_tokens == 262144
    assert whole.supply_bytes - whole.page_bytes - whole.fixed_bytes - whole.cache_bytes < 2 * MIB


def test_live_state_counts_null_live_speculative_and_admission_rows():
    assert SOLO.live_rows(4) == 1 + 4 * 8 + 12
    assert SOLO.live_rows(5) == 1 + 8 * 8 + 15
    assert PAIR.live_rows(4) == 1 + 4 + 12
    assert SOLO.snapshot_rows(32, 4) == 44


def test_prefix_cache_follows_concurrency_and_shrinks_under_pressure():
    assert derive(concurrency=1).cache_states == 8
    assert derive(concurrency=2).cache_states == 16
    # 10.6 GiB remain for cache and pages; a quarter of that holds 27 states, fewer than 32.
    capped = derive(reserve=12)
    assert capped.cache_target == capped.cache_states == 27 and capped.pages > capped.minimum_pages
    # With 32 requests on two GPUs, the minimum pages leave room for 19 of the 25 targeted states.
    two = (gpu(0, 95), gpu(1, 95))
    tight = derive(gpus=two, selected=(0, 1), concurrency=32, reserve=34.8)
    assert tight.fits and (tight.cache_target, tight.cache_states) == (25, 19)
    assert tight.minimum_pages <= tight.pages < tight.minimum_pages + 50
    assert "19 states for prompt reuse; 25 without memory pressure" in report(tight)
    floor = derive(reserve=20)
    assert floor.fits and floor.cache_states == planning.CACHE_FLOOR


def test_prefill_chunk_defaults_to_the_recipe_and_accepts_an_override():
    assert derive().prefill_chunk == 4096
    assert derive(prefill_chunk=1024).prefill_chunk == 1024
    assert derive(prefill_chunk=262144).prefill_chunk == 262144
    for chunk in (0, 100, 262208):
        with pytest.raises(UsageError, match="multiple of 64 from 64 through 262,144"):
            derive(prefill_chunk=chunk)
    # Twenty GiB of reserve leaves pages for 59,776 tokens, the longest sequence this plan allows.
    with pytest.raises(UsageError, match="65,536 exceeds the 59,776-token context"):
        derive(reserve=20, prefill_chunk=65536)


def test_measured_bytes_replace_the_estimates():
    seen = []

    def measured(world_size, concurrency, chunk, name):
        seen.append((world_size, concurrency, chunk, name))
        return planning.Measurement(62 * GIB, 6 * GIB)

    plan = derive(measured=measured)
    assert plan.measured and (plan.weights_bytes, plan.graph_bytes) == (62 * GIB, 6 * GIB)
    assert "measured on this host" in report(plan)
    assert (1, 4, 4096, "NVIDIA RTX PRO 6000 Blackwell") in seen
    estimate = derive()
    assert not estimate.measured and estimate.pages - plan.pages == 4 * GIB // (2 * MIB)
    assert (estimate.weights_bytes, estimate.graph_bytes) == (60 * GIB, 4 * GIB)


def test_first_fitting_gpus_and_smallest_supported_tensor_parallel_size():
    assert [g.index for g in derive(gpus=(gpu(0, 10), gpu(1, 95), gpu(2, 95))).gpus] == [1]
    assert [g.index for g in derive(gpus=(gpu(0, 95, (9, 0)), gpu(1, 95))).gpus] == [1]
    assert [g.index for g in derive(PAIR, (gpu(0, 95), gpu(1, 20), gpu(2, 95))).gpus] == [0, 2]
    split = derive(gpus=(gpu(0, 50), gpu(1, 50)))
    assert split.world_size == 2 and split.weights_bytes == 30 * GIB and "memory per GPU" in report(split)
    assert derive(PAIR, tuple(gpu(index, 60) for index in range(4))).world_size == 4


def test_explicit_gpus_set_the_tensor_parallel_size():
    gpus = tuple(gpu(index, 95) for index in range(4))
    assert derive(gpus=gpus, selected=(2, 3)).world_size == 2
    with pytest.raises(CapacityError, match=r"pair needs 2 or 4 GPUs; 1 selected; use --gpus 0-1"):
        derive(PAIR, gpus, selected=(0,))
    with pytest.raises(CapacityError, match=r"solo needs 1 or 2 or 4 GPUs; 3 selected; use --gpus 0-3"):
        derive(gpus=gpus, selected=(0, 1, 2))
    with pytest.raises(UsageError, match="names a GPU this host does not have"):
        derive(gpus=gpus, selected=(4,))
    with pytest.raises(CapacityError, match="not an SM120 or SM121 device"):
        derive(gpus=(gpu(0, 95, (9, 0)),), selected=(0,))
    with pytest.raises(CapacityError, match="pair needs 2 or 4 SM120 or SM121 GPUs; this host has 1"):
        derive(PAIR, (gpu(0, 95),))


@pytest.mark.parametrize("limit", [0, 65])
def test_concurrency_outside_the_recipe_range_is_a_usage_error(limit):
    with pytest.raises(UsageError, match="must be 1 through 64"):
        derive(concurrency=limit)


def option_arguments(option, arguments):
    """Apply one printed option to the arguments of `derive`."""
    arguments = dict(arguments)
    if match := re.fullmatch(r'\[memory\] reserve = "(\d+)G?"', option):
        arguments["reserve"] = int(match[1])
    elif match := re.fullmatch(r"--max-concurrency (\d+)", option):
        arguments["concurrency"] = int(match[1])
    elif match := re.fullmatch(r"--gpus (\S+)", option):
        arguments["selected"] = planning.parse_gpus(match[1])
    else:
        raise AssertionError(option)
    return arguments


@pytest.mark.parametrize("arguments, expected", [
    (dict(gpus=(gpu(0, 95),), selected=(0,), reserve=24), ['[memory] reserve = "20G"', "--max-concurrency 1"]),
    (dict(gpus=(gpu(0, 95), gpu(1, 95)), selected=(0,), reserve=24),
     ['[memory] reserve = "20G"', "--max-concurrency 1", "--gpus 0-1"]),
    (dict(gpus=(gpu(0, 73.9),)), ["--max-concurrency 3"]),
    (dict(gpus=(gpu(0, 77),), concurrency=8), ["--max-concurrency 4"]),
    (dict(gpus=(gpu(0, 80), gpu(1, 80)), reserve=40), ['[memory] reserve = "38G"', "--max-concurrency 2"]),
])
def test_every_capacity_error_option_fits_on_its_own(arguments, expected):
    with pytest.raises(CapacityError) as error:
        derive(**arguments)
    message = str(error.value)
    assert message.startswith("needs ") and "minimum pages" in message and "\nhas " in message
    options = re.search(r"Options: (.*)\.$", message)[1].split(", or ")
    assert options == expected
    for option in options:
        assert derive(**option_arguments(option, arguments)).fits, option


def test_capacity_error_states_requirement_and_supply():
    with pytest.raises(CapacityError) as error:
        derive(gpus=(gpu(0, 95),), selected=(0,), reserve=24)
    assert str(error.value).splitlines()[:3] == [
        "needs 60.0 GiB weights + 4.0 GiB graphs + 4.4 GiB live state + 0.8 GiB prefix cache",
        "+ 1.0 GiB minimum pages (4 × 8,192 tokens) = 70.2 GiB;",
        "has 95.0 GiB free − 4.0 GiB headroom − 24.0 GiB reserve = 67.0 GiB.",
    ]


def test_capacity_error_without_a_working_setting_says_so():
    with pytest.raises(CapacityError, match="No setting makes solo fit on this host"):
        derive(gpus=(gpu(0, 40),))
    with pytest.raises(CapacityError, match="No setting makes solo fit on GPU 0"):
        derive(gpus=(gpu(0, 40),), selected=(0,))


def test_automatic_selection_explains_the_closest_tensor_parallel_size():
    with pytest.raises(CapacityError) as error:
        derive(gpus=(gpu(0, 80), gpu(1, 80)), reserve=40)
    assert "= 37.1 GiB per GPU;" in str(error.value)


def test_plan_report_and_fit_summary():
    plan = derive(reserve=2)
    rule = "  " + "─" * 91
    assert report(plan, restriction="1").splitlines() == [
        "solo",
        "  checkpoint      lab/solo@ab12cd34",
        "  architecture    DemoForCausalLM",
        "  GPUs            0 · RTX PRO 6000 Blackwell · tensor-parallel 1",
        "                  indices follow CUDA_VISIBLE_DEVICES=1",
        "  concurrency     4 requests",
        "  context         262,144 tokens",
        "  prefill chunk   4,096 tokens",
        "  host memory     4.0 GiB · lookup table mapped from the checkpoint",
        "",
        "  memory                GiB",
        rule,
        "  weights              60.0   checkpoint tensors",
        "  graphs and scratch    4.0   estimate; never prepared on this host",
        "  live state            4.4   4 active requests",
        "  prefix cache          3.1   32 states for prompt reuse",
        "  KV pages             17.5   572,800 tokens; full context for 4 requests would take 32.0 GiB",
        "  headroom              4.0   kept free through warm-up and graph capture",
        "  reserve               2.0   [memory] reserve",
        rule,
        "  free                 95.0   GPU 0",
        "",
        "  guarantees 4 × 143,168 tokens at full concurrency; a single request up to 262,144",
    ]
    assert display.fit(plan) == "1 GPU · 4 requests of 143,168 tokens, or 2 of 262,144"
    assert display.fit(derive(concurrency=1)) == "1 GPU · 1 request of 262,144 tokens"
    pair = derive(PAIR, tuple(gpu(index, 95) for index in range(2)))
    assert display.fit(pair).startswith("2 GPUs · 4 requests of ")
    assert "least free of GPUs 0-1" in report(pair) and "host memory" not in report(pair)
    single = report(derive(concurrency=1))
    assert "1 active request\n" in single and "full context" not in single
    record = json.loads(json.dumps(plan.to_dict()))
    assert record["pages"] == plan.pages == 8950 and record["snapshots"] == 44
    assert record["reserve_bytes"] == 2 * GIB and record["measured"] is False


def test_gpu_lists_and_restrictions():
    assert planning.parse_gpus("0") == (0,)
    assert planning.parse_gpus("0,2") == (0, 2)
    assert planning.parse_gpus("0-3") == (0, 1, 2, 3)
    assert planning.format_gpus((0, 1, 2)) == "0-2" and planning.format_gpus((0, 2)) == "0,2"
    for text in ("", "a", "0-", "1,1", "3-1", "0,,1"):
        with pytest.raises(UsageError):
            planning.parse_gpus(text)
    gpus = tuple(gpu(index, 95) for index in range(3))
    restricted = planning.restrict_gpus(gpus, "2,GPU-0000")
    assert [(g.index, g.uuid) for g in restricted] == [(0, "GPU-0002"), (1, "GPU-0000")]
    assert planning.restrict_gpus(gpus, "") == ()
    for text in ("7", "GPU-", "1,1", "GPU-9"):
        with pytest.raises(UsageError):
            planning.restrict_gpus(gpus, text)


def test_measurement_records_require_the_same_gpu_model(tmp_path):
    path = planning.measurement_path(tmp_path, SOLO.checkpoints[0], 1, 4, 4096)
    assert path.name == "solo-ab12cd34-tp1-c4-p4096.json"
    assert planning.read_measurement(path, "A") is None
    planning.write_measurement(path, "A", planning.Measurement(456, 123))
    assert planning.read_measurement(path, "A") == planning.Measurement(456, 123)
    assert planning.read_measurement(path, "B") is None
    path.write_text(json.dumps({"gpu": "A", "graph_bytes": 123}))
    assert planning.read_measurement(path, "A") is None
