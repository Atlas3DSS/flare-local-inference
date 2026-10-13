"""Plan sizing, GPU selection, and capacity errors from declared costs and injected free memory."""

import json
from dataclasses import replace
import re
from types import MappingProxyType

import pytest

from ignition import planning
from ignition.cli import display
from ignition.errors import CapacityError, UsageError
from ignition.recipes.declaration import Checkpoint, RankCosts, Recipe, Speculation

GIB, MIB = 1 << 30, 1 << 20


DRAFTING = Speculation(draft_tokens=3)


def recipe(name, device_gib, ranks, *, speculation=None, mapped_gib=0, draft_gib=0):
    checkpoint = Checkpoint(name, f"lab/{name}", "ab12cd34" + "0" * 32, "DemoForCausalLM",
                            stored_bytes=(device_gib + mapped_gib) * GIB, device_bytes=device_gib * GIB,
                            mapped_host_bytes=mapped_gib * GIB, draft_bytes=draft_gib * GIB)
    speculating = speculation is not None
    return Recipe(name=name, checkpoints=(checkpoint,), context=262144, page_tokens=64, concurrency=4,
                  max_concurrency=64, max_draft_tokens=7 if speculating else 0, prefill_chunk=4096, prepare=None,
                  usage=None, mapped_host_name="lookup table", speculation=speculation, history_copy=speculating,
                  ranks=MappingProxyType({size: RankCosts(100 * MIB // size, 2 * MIB // size, 4 * GIB,
                                                          draft_page_bytes=MIB // (2 * size) if speculating else 0)
                                          for size in ranks}),
                  plain_ranks=MappingProxyType({size: RankCosts(80 * MIB // size, 3 * MIB // (2 * size), 3 * GIB)
                                                for size in ranks} if speculating else {}))


# While drafting, one rank holds 60 GiB of weights, 2 GiB of them the draft model, 4 GiB of graphs,
# 100 MiB per state row, and 2 MiB per page, a quarter of it the draft model's. At concurrency 4 it
# keeps 29 live rows, and its minimum of 512 pages costs 1 GiB. Without drafting it holds 3 GiB of
# graphs, 80 MiB per state row, and 1.5 MiB per page.
SOLO = recipe("solo", 60, (1, 2, 4), speculation=DRAFTING, mapped_gib=4, draft_gib=2)
PAIR = recipe("pair", 160, (2, 4))


@pytest.fixture(autouse=True)
def headroom(monkeypatch):
    monkeypatch.setattr(planning, "HEADROOM_BYTES", 4 * GIB)


def gpu(index, free_gib, capability=(12, 0)):
    return planning.Gpu(index, "NVIDIA RTX PRO 6000 Blackwell", f"GPU-{index:04d}", 96 * GIB, int(free_gib * GIB),
                        capability)


def spark(index, free_gib):
    return planning.Gpu(index, "NVIDIA GB10", f"GPU-{index:04d}", 121 * GIB, int(free_gib * GIB), (12, 1),
                        unified=True)


DEFAULT = object()


def derive(model=SOLO, gpus=(gpu(0, 95),), *, selected=None, concurrency=4, reserve=0, prefill_chunk=None,
           speculation=DEFAULT):
    return planning.plan(model, model.checkpoints[0], tuple(gpus), selected=selected, concurrency=concurrency,
                         reserve_bytes=int(reserve * GIB), prefill_chunk=prefill_chunk,
                         speculation=model.speculation if speculation is DEFAULT else speculation)


def test_pages_take_all_remaining_supply():
    limited = derive()
    assert limited.fits and limited.cache_states == limited.cache_target == 32
    assert limited.supply_bytes == 91 * GIB and limited.live_bytes == 29 * 100 * MIB
    # 91 GiB of supply less weights, graphs, 29 live rows, and 32 cached rows leaves 10,774 pages.
    assert limited.pages == (91 * GIB - 64 * GIB - 61 * 100 * MIB) // (2 * MIB) == 10774
    assert limited.demand_pages == 16384 and limited.minimum_pages == 512
    assert limited.context == 262144 and limited.guaranteed_tokens == 10774 // 4 * 64
    assert limited.snapshots == 32 + 12

    assert derive(gpus=(gpu(0, 95), gpu(1, 95))).world_size == 1

    # One request needs 4,096 pages at full context; the rest of the supply holds cached prefixes.
    whole = derive(concurrency=1)
    assert whole.demand_pages == 4096 and whole.pages == 13024 and whole.guaranteed_tokens == 262144
    assert whole.supply_bytes - whole.page_bytes - whole.fixed_bytes - whole.cache_bytes < 2 * MIB


def test_live_state_counts_null_live_speculative_and_admission_rows():
    assert SOLO.live_rows(4, DRAFTING) == 1 + 4 * 4 + 12
    assert SOLO.live_rows(5, DRAFTING) == 1 + 8 * 4 + 15
    assert SOLO.live_rows(4, None) == PAIR.live_rows(4, None) == 1 + 4 + 12
    assert SOLO.snapshot_rows(32, 4) == 44


def test_prefix_cache_follows_concurrency_and_shrinks_under_pressure():
    assert derive(concurrency=1).cache_states == 8
    assert derive(concurrency=2).cache_states == 16
    # 12.2 GiB remain for cache and pages; a quarter of that holds 31 states, fewer than 32.
    capped = derive(reserve=12)
    assert capped.cache_target == capped.cache_states == 31 and capped.pages > capped.minimum_pages
    # With 32 requests on two GPUs, the minimum pages leave room for 19 of the 25 targeted states.
    two = (gpu(0, 95), gpu(1, 95))
    tight = derive(gpus=two, selected=(0, 1), concurrency=32, reserve=41.05)
    assert tight.fits and (tight.cache_target, tight.cache_states) == (25, 19)
    assert tight.minimum_pages <= tight.pages < tight.minimum_pages + 50
    floor = derive(reserve=21)
    assert floor.fits and floor.cache_states == planning.CACHE_FLOOR


def test_speculation_settings_select_weights_rows_and_pages():
    drafting = derive()
    assert drafting.speculation == DRAFTING and drafting.live_rows == 29
    assert (drafting.weights_bytes, drafting.graph_bytes, drafting.costs.page_bytes) == (60 * GIB, 4 * GIB, 2 * MIB)
    # Without drafting the draft model is not loaded, rows shrink, and each request keeps no verification rows.
    plain = derive(speculation=None)
    assert (plain.weights_bytes, plain.graph_bytes, plain.live_rows) == (58 * GIB, 3 * GIB, 17)
    assert plain.live_bytes == 17 * 80 * MIB and plain.cache_states == 32
    assert plain.pages == (91 * GIB - 61 * GIB - 49 * 80 * MIB) * 2 // (3 * MIB) == 17866
    assert plain.to_dict()["speculation"] is None
    # Drafts copied from history keep the verification rows and load no draft model or draft pages.
    copied = derive(speculation=Speculation(3, proposal_source="copy"))
    assert (copied.weights_bytes, copied.live_rows, copied.costs.page_bytes) == (58 * GIB, 29, 3 * MIB // 2)
    assert copied.costs.state_row_bytes == drafting.costs.state_row_bytes and copied.pages == 15730
    split = derive(gpus=(gpu(0, 50), gpu(1, 50)), speculation=None)
    assert split.world_size == 2 and split.weights_bytes == 29 * GIB and split.costs.state_row_bytes == 40 * MIB


def test_speculation_settings_are_validated_and_described():
    assert DRAFTING.widths == (1, 2, 3, 4) and DRAFTING.fixed_width is None and DRAFTING.learned
    fixed = Speculation(2, adaptive=False)
    assert fixed.widths == (1, 2, 3) and fixed.fixed_width == 3
    assert Speculation(0).widths == (1,) and Speculation(0, adaptive=False).fixed_width == 1
    record = derive(speculation=Speculation(5, history_drafts=True)).to_dict()["speculation"]
    assert record == {"draft_tokens": 5, "adaptive": True, "proposal_source": "mtp", "history_drafts": True}
    assert display.speculation(None) == "off"
    assert display.speculation(DRAFTING) == "up to 3 draft tokens per round · MTP module"
    assert display.speculation(fixed) == "exactly 2 draft tokens per round · MTP module"
    assert display.speculation(Speculation(4, history_drafts=True)) == (
        "up to 4 draft tokens per round · MTP module and request history")
    assert display.speculation(Speculation(1, proposal_source="copy")) == (
        "up to 1 draft token per round · request history")
    assert display.speculation(None, brief=True) == "speculation off"
    assert display.speculation(DRAFTING, brief=True) == "up to 3 draft tokens"
    assert display.speculation(fixed, brief=True) == "exactly 2 draft tokens"
    assert display.speculation(Speculation(1, proposal_source="copy"), brief=True) == (
        "up to 1 draft token from history")


def test_headroom_grows_with_prefill_chunk_tokens_past_4096():
    # The fixture sets four GiB of headroom; each 4096 chunk tokens past 4096 add half a GiB.
    assert derive().headroom_bytes == derive(prefill_chunk=1024).headroom_bytes == 4 * GIB
    assert derive(prefill_chunk=8192).headroom_bytes == 4 * GIB + GIB // 2
    assert derive(prefill_chunk=8192).supply_bytes == derive().supply_bytes - GIB // 2
    assert derive(prefill_chunk=8192).to_dict()["headroom_bytes"] == 4 * GIB + GIB // 2


def test_prefill_chunk_defaults_to_the_recipe_and_accepts_an_override():
    assert derive().prefill_chunk == 4096
    assert derive(prefill_chunk=1024).prefill_chunk == 1024
    # A whole-context chunk's activations need about 32 GiB more headroom.
    assert derive(gpus=(gpu(0, 140),), prefill_chunk=262144).prefill_chunk == 262144
    for chunk in (0, 100, 262208):
        with pytest.raises(UsageError, match="multiple of 64 from 64 through 262,144"):
            derive(prefill_chunk=chunk)
    # 14 GiB of reserve and the chunk's 11.5 GiB headroom leave pages for 61,824 tokens, the longest
    # sequence this plan allows.
    with pytest.raises(UsageError, match="65,536 exceeds the 61,824-token context"):
        derive(reserve=14, prefill_chunk=65536)


def test_context_parallel_pages_hold_every_rank_share_unless_disabled():
    sharded = replace(PAIR, ranks=MappingProxyType({size: replace(costs, page_tokens=64 * size)
                                                    for size, costs in PAIR.ranks.items()}))
    pair = (gpu(0, 95), gpu(1, 95))
    shared = planning.plan(sharded, sharded.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                           speculation=None)
    whole = planning.plan(sharded, sharded.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                          speculation=None, context_parallel=False)
    assert (shared.context_parallel, shared.page_tokens) == (True, 128)
    assert (whole.context_parallel, whole.page_tokens) == (False, 64)
    assert shared.pages == whole.pages and shared.tokens == 2 * whole.tokens
    assert derive().context_parallel is False
    with pytest.raises(UsageError, match="does not shard attention context"):
        planning.plan(PAIR, PAIR.checkpoints[0], pair, selected=None, concurrency=4, reserve_bytes=0,
                      speculation=None, context_parallel=False)


def test_first_fitting_gpus_and_smallest_supported_tensor_parallel_size():
    assert [g.index for g in derive(gpus=(gpu(0, 10), gpu(1, 95), gpu(2, 95))).gpus] == [1]
    assert [g.index for g in derive(gpus=(gpu(0, 95, (9, 0)), gpu(1, 95))).gpus] == [1]
    assert [g.index for g in derive(PAIR, (gpu(0, 95), gpu(1, 20), gpu(2, 95))).gpus] == [0, 2]
    split = derive(gpus=(gpu(0, 50), gpu(1, 50)))
    assert split.world_size == 2 and split.weights_bytes == 30 * GIB
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
    with pytest.raises(CapacityError, match="not an SM103, SM120, or SM121 device"):
        derive(gpus=(gpu(0, 95, (9, 0)),), selected=(0,))
    with pytest.raises(CapacityError, match="pair needs 2 or 4 SM103, SM120, or SM121 GPUs; this host has 1"):
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
    elif option == "--no-mtp":
        arguments["speculation"] = None
    else:
        raise AssertionError(option)
    return arguments


@pytest.mark.parametrize("arguments, expected", [
    (dict(gpus=(gpu(0, 95),), selected=(0,), reserve=24),
     ['[memory] reserve = "22G"', "--max-concurrency 2", "--no-mtp"]),
    (dict(gpus=(gpu(0, 95), gpu(1, 95)), selected=(0,), reserve=24),
     ['[memory] reserve = "22G"', "--max-concurrency 2", "--no-mtp", "--gpus 0-1"]),
    (dict(gpus=(gpu(0, 72.5),)), ["--max-concurrency 3", "--no-mtp"]),
    (dict(gpus=(gpu(0, 74),), concurrency=8), ["--max-concurrency 4", "--no-mtp"]),
    (dict(gpus=(gpu(0, 80), gpu(1, 80)), reserve=40),
     ['[memory] reserve = "39G"', "--max-concurrency 2", "--no-mtp"]),
    # Without speculation the plan has no drafting to give up.
    (dict(gpus=(gpu(0, 95),), selected=(0,), reserve=30, speculation=None), ['[memory] reserve = "27G"']),
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
        "needs 60.0 GiB weights + 4.0 GiB graphs + 2.8 GiB live state + 0.8 GiB prefix cache",
        "+ 1.0 GiB minimum pages (4 × 8,192 tokens) = 68.6 GiB;",
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
    assert "= 36.3 GiB per GPU;" in str(error.value)


def test_estimate_record():
    plan = derive(reserve=2)
    record = json.loads(json.dumps(plan.to_dict()))
    assert record["pages"] == plan.pages == 9750 and record["snapshots"] == 44
    assert record["reserve_bytes"] == 2 * GIB and "measured" not in record


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
def test_unified_memory_gpus_plan_from_their_reported_free_memory():
    # The PLE table reaches the GPU through compact row delivery, so no mapped table is subtracted.
    discrete, unified = derive(gpus=(gpu(0, 95),)), derive(gpus=(spark(0, 95),))
    assert discrete.free_bytes == unified.free_bytes == 95 * GIB


