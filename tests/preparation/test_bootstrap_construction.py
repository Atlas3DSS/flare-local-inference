"""Exercise production constructor wiring without loading weights or CUDA programs."""

import importlib
import inspect
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes
from torch.utils._pytree import tree_leaves, tree_map

from flare.preparation import inspect_qwen38_checkpoint
from flare.preparation.qwen38 import MODEL_ID, REVISION
from flare.recipes.qwen38 import bootstrap


class MetadataValues(TorchDispatchMode):
    """Track small integer metadata on CPU while all large/model storage stays meta."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._local_scalar_dense.default:
            value = getattr(args[0], "_construction_value", None)
            if value is None:
                raise AssertionError("scalar read has no recorded metadata value")
            with _disable_current_modes():
                return value.item()
        result = func(*args, **kwargs)
        tensors = [x for x in tree_leaves(result) if isinstance(x, FakeTensor)]
        kinds = (
            torch.bool,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint32,
            torch.uint64,
        )
        if not tensors or any(
            x.dtype not in kinds or x.numel() > 65536 for x in tensors
        ):
            return result
        operands = [x for x in tree_leaves((args, kwargs)) if isinstance(x, FakeTensor)]
        if any(getattr(x, "_construction_value", None) is None for x in operands):
            return result

        def cpu(x):
            if isinstance(x, FakeTensor):
                return x._construction_value
            if isinstance(x, torch.device) and x.type == "cuda":
                return torch.device("cpu")
            return x

        with _disable_current_modes():
            values = func(*tree_map(cpu, args), **tree_map(cpu, kwargs))
        for fake, value in zip(tree_leaves(result), tree_leaves(values), strict=True):
            if isinstance(fake, FakeTensor):
                fake._construction_value = value
        return result


class ConstructionComplete(Exception):
    pass


class RecordingSession:
    def __init__(self, **kwargs):
        self.requests = {}
        self.stages = {}
        self.finalizers = []
        self.arena = None
        self.closed = False
        self.resources = {}
        self.memory = {}

    def __enter__(self):
        return self

    def declare(self, requests, *, stage="weights", coordinator=None):
        for request in requests:
            assert request.name not in self.requests
            self.memory[request.name] = request.plan.memory_requirements()
            self.requests[request.name] = request
            self.stages[request.name] = stage

    def register_finalizer(self, callback):
        self.finalizers.append(callback)

    def register_benchmark_resource(self, key, resource, *, restore):
        self.resources[key] = (resource, restore)

    def register_scratch(self, arena):
        self.arena = arena

    def tune(self, **kwargs):
        raise ConstructionComplete

    def close(self):
        self.closed = True


def construction_contract(session, signatures):
    """Export declarations without process-local plan or allocation identities."""
    from b12x.preparation import device

    owners = {}

    def encode(value):
        if value is None or type(value) in (str, int, float, bool):
            return value
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        if is_dataclass(value):
            return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
        raise TypeError(f"unsupported construction metadata: {type(value).__name__}")

    def allocation_key(value):
        if isinstance(value, tuple):
            return [allocation_key(item) for item in value]
        if type(value) is object or hasattr(value, "component_id"):
            return {"owner": owners.setdefault(value, len(owners))}
        return encode(value)

    def memory(requirements):
        return {
            "scratch": encode(requirements.scratch),
            "scratch_nbytes": requirements.scratch_nbytes,
            "persistent": [
                {"key": allocation_key(item.key),
                 "required_nbytes": item.required_nbytes,
                 "resident_nbytes": item.resident_nbytes}
                for item in requirements.persistent
            ],
            "pending_persistent_nbytes": requirements.pending_persistent_nbytes,
        }

    def plan_contract(plan):
        result = {
            "component_id": plan.component_id,
            "dependencies": encode(plan.dependencies),
            "shared": plan.shared,
            "memory": memory(plan.memory_requirements()),
        }
        if hasattr(plan, "variants"):
            result.update(
                composite_semantic_version=plan.composite_semantic_version,
                capacity_metadata=encode(plan.capacity_metadata),
                variants={str(count): plan_contract(child)
                          for count, child in plan.variants.items()},
            )
            return result
        contract = plan.contract
        detected = device.detect_device(
            plan._device if plan._device is not None else getattr(plan.query, "device", None)
        )
        configuration = contract.configure(
            plan.query, device=detected.identity, override=plan.override, search=False
        )
        result.update(
            versions={name: getattr(contract, name) for name in (
                "query_schema_version", "config_schema_version", "semantic_version",
                "candidate_contract_version")},
            query=encode(contract.encode_query(plan.query)),
            invocation=encode(plan.invocation),
            default_config=encode(contract.config_payload(configuration.default)),
            override=None if plan.override is None else encode(contract.config_payload(plan.override)),
            requested_device=encode(plan._device),
            detected_device=encode(detected),
        )
        return result

    def factories(value):
        if value is None:
            return None
        if isinstance(value, Mapping):
            assert all(callable(callback) for callback in value.values())
            return list(value)
        assert callable(value)
        return "scalar"

    return {
        "requests": [{
            "name": request.name,
            "plan": plan_contract(request.plan),
            "dependencies": encode(request.dependencies),
            "collective": encode(request.collective),
            "retain_benchmark_call": request.retain_benchmark_call,
            "prepare_call": factories(request.prepare_call),
            "benchmark_call": factories(request.benchmark_call),
        } for request in session.requests.values()],
        "graph_signatures": encode(signatures),
        "scratch_arena_nbytes": session.arena.nbytes,
    }


class CompiledKernel:
    def __getitem__(self, grid):
        return lambda *args, **kwargs: None


@pytest.fixture(scope="module")
def checkpoint():
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub"
        / ("models--" + MODEL_ID.replace("/", "--"))
        / "snapshots"
        / REVISION
    )
    if not snapshot.exists():
        pytest.skip("exact Qwen checkpoint metadata is not available")
    manifest = inspect_qwen38_checkpoint(snapshot)
    from flare.recipes.qwen38.ple import table_layout

    config, layout = table_layout(manifest, "cuda:0")
    return manifest, config, layout


def _install_boundaries(monkeypatch, checkpoint):
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
    from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
    from b12x.preparation import device
    from b12x.preparation.types import DeviceIdentity
    from triton.runtime.jit import JITFunction

    from flare.preparation import tuning
    from flare.recipes.qwen38 import (
        attention,
        gdn,
        hyperconnection,
        moe,
        mtp,
        mtp_weights,
        ple,
        vision,
    )

    manifest, table_config, table_layout = checkpoint
    detected = device.DetectedDevice(
        ordinal=0,
        identity=DeviceIdentity(
            vendor="nvidia",
            compute_capability=(12, 0),
            sm_count=170,
            product_name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        ),
    )
    monkeypatch.setattr(device, "detect_device", lambda *args: detected)

    class DeclaredStream:
        def __init__(self, device):
            self.device = torch.device(device)
            self.cuda_stream = 0

    class DeclaredEvent:
        def record(self, stream):
            self.device = stream.device

    for name, replacement in {
        "set_device": lambda *a: None,
        "synchronize": lambda *a: None,
        "current_device": lambda: 0,
        "get_device_capability": lambda *a: (12, 0),
        "get_device_properties": lambda *a: SimpleNamespace(
            major=12,
            minor=0,
            multi_processor_count=170,
            name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
            uuid="CPU-construction",
            shared_memory_per_block_optin=101376,
            shared_memory_per_multiprocessor=101376,
        ),
        "mem_get_info": lambda *a: (96 << 30, 96 << 30),
        "memory_allocated": lambda *a: 0,
        "device": lambda *a: nullcontext(),
        "Stream": DeclaredStream,
        "Event": DeclaredEvent,
    }.items():
        monkeypatch.setattr(torch.cuda, name, replacement)

    def forbidden(*args, **kwargs):
        raise AssertionError("constructor smoke must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    from flashinfer import sampling

    old_storage_pointer = torch.UntypedStorage.data_ptr
    monkeypatch.setattr(
        torch.UntypedStorage,
        "data_ptr",
        lambda storage: (
            storage._cdata * 256
            if storage.device.type == "meta"
            else old_storage_pointer(storage)
        ),
    )
    old_pointer = torch.Tensor.data_ptr
    monkeypatch.setattr(
        torch.Tensor,
        "data_ptr",
        lambda t: (
            t.untyped_storage()._cdata * 256 + t.storage_offset() * t.element_size()
            if isinstance(t, FakeTensor)
            else old_pointer(t)
        ),
    )

    def warmup(kernel, *args, **kwargs):
        launch_options = {"grid", "num_warps", "num_stages", "enable_fp_fusion"}
        inspect.signature(kernel.fn).bind(
            *args,
            **{
                key: value for key, value in kwargs.items() if key not in launch_options
            },
        )
        return CompiledKernel()

    monkeypatch.setattr(JITFunction, "warmup", warmup)
    for name in ("top_k_mask_logits", "softmax", "top_p_renorm_probs"):
        monkeypatch.setattr(
            sampling, name, lambda tensor, *a, **kw: torch.empty_like(tensor)
        )

    def sampled(tensor, *args, **kwargs):
        ids = torch.zeros(tensor.shape[0], dtype=torch.int32, device=tensor.device)
        return (
            (ids, torch.ones_like(ids, dtype=torch.bool))
            if kwargs.get("return_valid")
            else ids
        )

    monkeypatch.setattr(sampling, "sampling_from_probs", sampled)
    monkeypatch.setattr(
        sampling,
        "chain_speculative_sampling",
        lambda draft, ids, target, *a, **kw: (
            torch.zeros(
                (draft.shape[0], draft.shape[1] + 1),
                dtype=torch.int32,
                device=draft.device,
            ),
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        sampling,
        "_get_cache_buf",
        lambda name, nbytes, device, **kw: torch.empty(
            nbytes, dtype=torch.uint8, device=device
        ),
    )
    for op in (
        attention.qsa,
        attention.paged_kv_write,
        hyperconnection.hc,
        gdn.short_conv,
        gdn.gdn_prefill,
        gdn.gdn_decode,
        moe.fused_moe,
        ple.ple_embedding,
        vision.varlen,
        mtp.mtp_feedback,
    ):
        if hasattr(op, "bind"):
            original = op.bind

            def bind(plan, *args, _original=original, **kwargs):
                inspect.signature(_original).bind(plan, *args, **kwargs)
                return SimpleNamespace(plan=plan, **kwargs)

            monkeypatch.setattr(op, "bind", bind)
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "I64": torch.int64,
        "I32": torch.int32,
        "U8": torch.uint8,
        "U32": torch.uint32,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E8M0": torch.float8_e8m0fnu,
    }

    def load_tensor(loader, name):
        record = manifest.tensors[name]
        return torch.empty(
            record.shape, dtype=dtypes[record.dtype], device=loader.device
        )

    def pack(values, scales, *, recipe="mxfp8", **kwargs):
        n, k = values.shape[:2]
        return MXFP8LinearWeight(
            MXFP8Rows(
                values.reshape(n, k, 1),
                scales.reshape(1, n, -1),
                torch.empty(
                    (32, 4, (n + 127) // 128, 4, (k + 127) // 128, 1),
                    dtype=torch.uint8,
                    device=values.device,
                ),
            ),
            k,
            k,
            n,
        )

    def load_linear(loader, prefix):
        record = manifest.tensors[prefix + ".weight"]
        n, k = record.shape
        if record.dtype == "F8_E4M3":
            return pack(
                load_tensor(loader, prefix + ".weight"),
                load_tensor(loader, prefix + ".weight_scale"),
            )
        assert record.dtype == "U8"
        logical = k * 2
        stored = (logical + 127) // 128 * 128
        return NVFP4LinearWeight(
            torch.empty((n, stored // 2), dtype=torch.uint8, device=loader.device),
            torch.empty(
                (32, 4, (n + 127) // 128, 4, (stored // 16 + 3) // 4, 1),
                dtype=torch.float8_e4m3fn,
                device=loader.device,
            ),
            load_tensor(loader, prefix + ".weight_scale_2"),
            "multiplier",
            logical,
            n,
        )

    class Sources(dict):
        def __init__(self, loader):
            self.loader = loader

        def __missing__(self, key):
            return load_tensor(self.loader, key)

    def enter(loader):
        loader._sources = Sources(loader)
        loader._session = SimpleNamespace(flush=lambda: None)
        return loader

    loader = bootstrap.ModelWeights.load.__func__.__globals__["QwenWeightLoader"]
    monkeypatch.setattr(loader, "__enter__", enter)
    monkeypatch.setattr(loader, "__exit__", lambda *a: None)
    monkeypatch.setattr(loader, "load_tensor", load_tensor)
    monkeypatch.setattr(loader, "load_linear", load_linear)
    monkeypatch.setattr(
        loader, "_copy", lambda self, destination, source: destination.copy_(source)
    )
    monkeypatch.setattr(moe.blockscaled, "pack_weight", pack)

    def experts(loader, *args, **kwargs):
        from dataclasses import replace

        from b12x._lib.quant.nvfp4_csf import Nvfp4CsfBatch, Nvfp4CsfDecoder
        from b12x.moe._shared.kernels.w4a16.prepare import W4A16PackedWeights
        from b12x.moe.fused_moe._impl import (
            B12XFP4ExpertWeights,
            _PreparedWeightRepresentation,
        )

        mode = "a16" if isinstance(loader, mtp_weights.MTPWeightLoader) else "a4"
        plan = moe.fused_moe.plan_weights(
            source=moe.fused_moe.PackedSource(
                format="modelopt_nvfp4", w13_layout="w13"
            ),
            activation=moe.fused_moe.ActivationSpec(
                mode=mode, nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=moe.fused_moe.MoEGeometry(
                num_experts=512, hidden_size=2560, intermediate_size=640
            ),
        )
        data = {}
        for name, shape, dtype in (
            ("w1_fp4", (512, 1280, 1280), torch.uint8),
            ("w2_fp4", (512, 2560, 320), torch.uint8),
            ("w1_blockscale", (512, 1280, 160), torch.float8_e4m3fn),
            ("w2_blockscale", (512, 2560, 40), torch.float8_e4m3fn),
            ("a1_gscale", (512,), torch.float32),
            ("a2_gscale", (512,), torch.float32),
            ("w1_alphas", (512,), torch.float32),
            ("w2_alphas", (512,), torch.float32),
        ):
            data[name] = torch.empty(shape, dtype=dtype, device=loader.device)
        if mode == "a4":
            plan = replace(plan, _impl=replace(plan._impl, nvfp4_inline_scales=True))
            planes = []
            for rows, columns, projections in (
                (1280, 160, ("up_proj", "gate_proj")),
                (2560, 40, ("down_proj",)),
            ):
                count = sum(
                    manifest.tensors[
                        f"model.language_model.layers.{args[0]}.mlp.experts.{e}.{proj}.weight_scale.nvfp4_csf_exceptions"
                    ].nbytes
                    for e in range(512)
                    for proj in projections
                )
                plane = Nvfp4CsfBatch(
                    torch.empty(
                        (512, rows // 128, 128 * (1 + columns // 2)),
                        dtype=torch.uint8,
                        device=loader.device,
                    ),
                    torch.empty(count, dtype=torch.uint8, device=loader.device),
                    torch.empty(
                        (512, rows // 128 + 1), dtype=torch.int64, device=loader.device
                    ),
                    rows,
                    columns,
                )
                plane.validate()
                planes.append(plane)
            data["nvfp4_csf"] = Nvfp4CsfDecoder(
                *planes,
                (),
                (),
                torch.empty(512, dtype=torch.int32, device=loader.device),
                inline_scales=(),
                indexed_programs=(),
            )
        else:
            packed = W4A16PackedWeights(
                data["w1_fp4"],
                data["w1_blockscale"],
                data["w1_alphas"],
                data["w2_fp4"],
                data["w2_blockscale"],
                data["w2_alphas"],
                torch.empty(0, dtype=torch.uint8, device=loader.device),
                2560,
                640,
                512,
                True,
                torch.bfloat16,
            )
            data["representation"] = _PreparedWeightRepresentation(
                quant_mode="w4a16",
                layout=plan._impl.required_weight_layout("w4a16"),
                value=packed,
            )
        value = B12XFP4ExpertWeights(plan=plan._impl, **data)
        return moe.fused_moe.PreparedExperts(plan=plan, _impl=value)

    monkeypatch.setattr(loader, "load_experts", experts)
    monkeypatch.setattr(mtp_weights.MTPWeightLoader, "load_experts", experts)

    def load_table(cls, manifest, device, *, progress=None):
        geometry = ple.ple_embedding.allocate_geometry(
            table_layout.geometry, device=device
        )
        storage = SimpleNamespace(
            weight=torch.empty(
                table_layout.weight_shape, dtype=torch.uint8, device=device
            ),
            weight_scale=torch.empty(
                table_layout.weight_scale_shape,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            weight_scale_2=torch.empty(1, dtype=torch.float32, device=device),
            close=lambda: None,
        )
        return ple.MappedPLETable(table_config, table_layout, geometry, storage)

    monkeypatch.setattr(ple.MappedPLETable, "load", classmethod(load_table))
    session = RecordingSession()
    monkeypatch.setattr(
        bootstrap, "preparation_session", lambda **kwargs: session
    )
    monkeypatch.setattr(tuning, "RecordedPreparationSession", RecordingSession)
    monkeypatch.setattr(bootstrap, "inspect_qwen38_checkpoint", lambda path: manifest)
    signatures = []
    original = bootstrap.GraphSignature

    def signature(*args, **kwargs):
        result = original(*args, **kwargs)
        signatures.append(result)
        return result

    monkeypatch.setattr(bootstrap, "GraphSignature", signature)
    constructors = defaultdict(list)
    modules = (
        "attention",
        "decoder",
        "decoder_state",
        "gdn",
        "hyperconnection",
        "maintenance",
        "mtp",
        "ple",
        "sampling",
        "speculation",
        "speculative_model",
        "speculative_state",
        "draft_state",
        "state",
        "vision",
        "vision_state",
    )
    for name in ("flare.components.linear", *("flare.recipes.qwen38." + name for name in modules)):
        module = importlib.import_module(name)
        for cls in vars(module).values():
            if not inspect.isclass(cls) or cls.__module__ != module.__name__:
                continue
            if not (
                cls.__name__.startswith("Prepared")
                or cls.__name__
                in ("SpeculativePreparation", "UnquantizedProjection", "QuantizedProjection")
            ):
                continue
            original_init = cls.__init__

            @wraps(original_init)
            def initialize(instance, *args, _init=original_init, _cls=cls, **kwargs):
                bound = inspect.signature(_init).bind(instance, *args, **kwargs)
                _init(instance, *args, **kwargs)
                constructors[_cls.__name__].append((instance, bound.arguments))

            monkeypatch.setattr(cls, "__init__", initialize)
    return session, signatures, constructors


@pytest.mark.parametrize("proposal", ["ordinary", "copy", "mtp"])
def test_prepare_model_constructs_real_catalog_on_meta(
    monkeypatch, checkpoint, proposal
):
    assert not torch.cuda.is_initialized()
    session, signatures, constructors = _install_boundaries(monkeypatch, checkpoint)
    config = bootstrap.PreparationConfig(
        mtp=proposal != "ordinary",
        speculative_proposal_source="copy" if proposal == "copy" else "mtp",
        history_drafts=proposal == "copy",
    )
    with (
        FakeTensorMode(allow_non_fake_inputs=True),
        MetadataValues(),
        pytest.raises(ConstructionComplete),
    ):
        bootstrap.prepare_model(checkpoint[0].snapshot, config)
    assert session.closed
    expected = {
        "vision": 4,
        "prefill": 6,
        "resume": 3,
        "init": 3,
        "restore": 3,
        "snapshot": 3,
    }
    expected.update({"decode": 3} if proposal == "ordinary" else {"speculative": 14})
    assert Counter(signature.kind for signature in signatures) == expected
    assert {
        signature.token_capacity
        for signature in signatures
        if signature.kind == "prefill"
    } == set(config.prefill_capacities)
    assert {
        signature.token_capacity
        for signature in signatures
        if signature.kind == "vision"
    } == {256, 1024, 4096, 16384}
    assert {
        (s.batch_capacity, s.token_capacity)
        for s in signatures
        if s.kind == "speculative"
    } == (set(config.speculative_shapes) if proposal != "ordinary" else set())
    assert all(signature.sampling_mode == "general" for signature in signatures)
    assert all(
        ("draft_page_tables" in signature.inputs) == (proposal == "mtp")
        for signature in signatures
        if signature.kind in ("prefill", "resume", "speculative")
    )
    decoders = constructors["PreparedDecoder"]
    assert len(decoders) == (9 if proposal == "ordinary" else 20)
    assert len(constructors["PreparedVision"]) == 4
    assert len(constructors["PreparedMaintenance"]) == 9
    assert len(constructors["PreparedResume"]) == 3
    assert len(constructors["PreparedGDN"]) == 36 * len(decoders)
    assert len(constructors["PreparedPLE"]) == len(decoders)
    for decoder, args in decoders:
        assert len(decoder.layers) == 48
        assert args["workspaces"] is session.arena
        assert args["completion_records"].shape == (
            2,
            4,
            8 if proposal == "ordinary" else 24,
        )
        assert args["completion_records"].dtype == torch.int64
        pool = args["state"].pool
        assert pool.backing.dtype == torch.uint8
        assert pool.backing.shape[0] == 5 + pool.layout.snapshot_capacity(config.snapshot_budget_bytes) + (0 if proposal == "ordinary" else 28)
        assert decoder.scratch.residual.dtype == torch.bfloat16
        assert decoder.scratch.logits.dtype == torch.bfloat16
        assert pool.layout.speculative_tokens == (0 if proposal == "ordinary" else 7)
        assert decoder.scratch.residual.shape == (args["state"].tokens, 10240)
        assert decoder.scratch.logits.shape == (
            (args["state"].tokens if decoder.verification else args["state"].rows),
            248320,
        )
        assert len(decoder.qsa_states) == (13 if proposal == "mtp" else 12)
        for index, layer in enumerate(decoder.layers):
            assert type(layer[1]).__name__ == (
                "PreparedAttention" if index % 4 == 3 else "PreparedGDN"
            )
    assert len(constructors["PreparedMTP"]) == (31 if proposal == "mtp" else 0)
    assert len(constructors["PreparedCheckpointCaches"]) == 9
    for owner, _ in constructors["PreparedMaintenance"]:
        assert owner.transfers.exact.pool is owner.pool
        assert owner.transfers.exact.tables is owner.forward_inputs.page_tables
        assert (owner.transfers.exact.draft is not None) == (proposal == "mtp")
    assert len(constructors["PreparedStochasticVerification"]) == (
        0 if proposal == "ordinary" else 14
    )
    assert len(constructors["PreparedCopyProposals"]) == (
        14 if proposal == "copy" else 0
    )
    for _, args in constructors["PreparedMTP"]:
        assert args["workspace"] is session.arena
        assert args["target_hidden"].shape[1:] == (10240,)
        assert args["scratch"].logits.shape[1] == 248320
    assert (
        session.arena.nbytes == 0
        and session.arena.tensor.dtype == torch.uint8
        and session.arena._finalizers
        and not session.arena.frozen
    )
    assert set(session.stages.values()) == {"weights", "state"}
    for name, request in session.requests.items():
        if request.plan.component_id in ("attention.qsa", "sequence.gdn_decode", "sequence.gdn_prefill"):
            assert session.stages[name] == "state"
    assert len(session.resources) == 1
    borrowed, _ = next(iter(session.resources.values()))
    assert (
        borrowed.borrowed_nbytes == borrowed.nbytes and borrowed.allocated_nbytes == 0
    )
    families = {request.plan.component_id for request in session.requests.values()}
    assert {"moe.decode", "sequence.gdn_prefill", "sequence.ple_embedding"}.issubset(
        families
    )
    assert all(memory.scratch_nbytes >= 0 for memory in session.memory.values())
    contract = construction_contract(session, signatures)
    assert [request["name"] for request in contract["requests"]] == list(session.requests)
    assert len(contract["graph_signatures"]) == len(signatures)
    assert not torch.cuda.is_initialized()


def test_recording_metadata_preserves_values_and_rejects_unknown_scalars():
    with FakeTensorMode(), MetadataValues():
        metadata = torch.zeros(4, dtype=torch.int64)
        metadata[2] = 7
        assert metadata[2].item() == 7
        assert metadata.any().item() is True
        assert metadata[0].item() == 0
        with pytest.raises(AssertionError, match="no recorded metadata value"):
            torch.empty(1, dtype=torch.float32).item()


def test_real_moe_constructor_rejects_incompatible_loaded_geometry(
    monkeypatch, checkpoint
):
    from flare.recipes.qwen38.weights import QwenWeightLoader

    session, _signatures, constructors = _install_boundaries(monkeypatch, checkpoint)
    load = QwenWeightLoader.load_tensor

    def malformed(loader, name):
        value = load(loader, name)
        if name == "model.language_model.layers.0.mlp.gate.weight":
            return value[:-1]
        return value

    monkeypatch.setattr(QwenWeightLoader, "load_tensor", malformed)
    with (
        FakeTensorMode(allow_non_fake_inputs=True),
        MetadataValues(),
        pytest.raises(ValueError, match="router weight must have geometry"),
    ):
        bootstrap.prepare_model(checkpoint[0].snapshot)
    assert len(constructors["PreparedVision"]) == 4
    assert not constructors["PreparedDecoder"]
    assert session.closed
    assert not torch.cuda.is_initialized()
