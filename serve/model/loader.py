"""HuggingFace model loading with transport-aware tensor loading.

Instantiates the HF model on the meta device (zero memory), then loads
safetensor shards directly to GPU. Recipes describe whether a tensor is
replicated or sharded; the loader realizes that intent either by loading
locally on every rank or by reading once per node and sending slices to
followers on the fast interconnect.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import pathlib
import time
from typing import Optional

import safetensors.torch as sf
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from serve.logging import get_logger, start_load_session
from serve.model.ops import precompute_rope_freqs
from serve.tp.group import TPGroup


# -- recipe registry -------------------------------------------------------

_RECIPES = {}


def register_recipe(model_type: str):
    """Decorator to register a surgery recipe for a model_type."""

    def wrapper(fn):
        _RECIPES[model_type] = fn
        return fn

    return wrapper


# -- metadata --------------------------------------------------------------


@dataclass(frozen=True)
class TensorMeta:
    shape: tuple[int, ...]
    dtype: torch.dtype
    shard_file: str


@dataclass
class LoadMetrics:
    storage_bytes_read: int = 0
    transport_bytes_sent: int = 0
    transport_bytes_received: int = 0
    tensors_loaded: int = 0


_SAFE_DTYPE_MAP = dict(sf._TYPES)
LOGGER = get_logger(__name__)


def _normalize_backend(backend: str, world_size: int) -> str:
    if backend == "auto":
        return "distributed" if world_size > 1 else "local"
    if backend not in {"local", "distributed"}:
        raise ValueError(f"unsupported load backend {backend!r}")
    if backend == "distributed" and world_size == 1:
        return "local"
    return backend


def _round_up(size: int, multiple: int) -> int:
    return ((size + multiple - 1) // multiple) * multiple


def _resolve_local_topology(rank: int, world_size: int) -> tuple[int, int, int]:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, local_world_size)))
    if local_world_size <= 0:
        local_world_size = world_size
    node_start = rank - local_rank
    if node_start < 0 or node_start + local_world_size > world_size:
        node_start = 0
        local_world_size = world_size
        local_rank = rank
    return local_rank, local_world_size, node_start


def _chunked(seq: list[int], size: int) -> list[list[int]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [seq[i:i + size] for i in range(0, len(seq), size)]


class TensorIndex:
    """Metadata-only view over the safetensor checkpoint."""

    def __init__(self, model_path: str | pathlib.Path):
        self.model_path = pathlib.Path(model_path)
        index_path = self.model_path / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            self.weight_map: dict[str, str] = dict(index["weight_map"])
        else:
            single = self.model_path / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors found in {self.model_path}")
            with sf.safe_open(str(single), framework="pt", device="cpu") as handle:
                self.weight_map = {k: single.name for k in handle.keys()}
        self._meta_files: dict[str, object] = {}
        self._meta_cache: dict[str, TensorMeta] = {}
        self._all_keys = set(self.weight_map.keys())
        self._total_bytes: int | None = None

    def keys(self) -> set[str]:
        return self._all_keys

    def exists(self, key: str) -> bool:
        return key in self._all_keys

    def meta(self, key: str) -> TensorMeta:
        meta = self._meta_cache.get(key)
        if meta is not None:
            return meta
        shard_file = self.weight_map[key]
        handle = self._meta_files.get(shard_file)
        if handle is None:
            path = str(self.model_path / shard_file)
            handle = sf.safe_open(path, framework="pt", device="cpu")
            self._meta_files[shard_file] = handle
        tensor_slice = handle.get_slice(key)
        meta = TensorMeta(
            shape=tuple(int(x) for x in tensor_slice.get_shape()),
            dtype=_SAFE_DTYPE_MAP[tensor_slice.get_dtype()],
            shard_file=shard_file,
        )
        self._meta_cache[key] = meta
        return meta

    def close(self) -> None:
        self._meta_files.clear()

    def total_bytes(self) -> int:
        if self._total_bytes is None:
            total = 0
            for key in self._all_keys:
                meta = self.meta(key)
                tensor_bytes = int(torch.empty((), dtype=meta.dtype).element_size())
                for dim in meta.shape:
                    tensor_bytes *= dim
                total += tensor_bytes
            self._total_bytes = total
        return self._total_bytes


# -- safetensor loading ----------------------------------------------------


class ShardedLoader:
    """Intent-aware tensor loader for model recipes."""

    def __init__(
        self,
        model_path: str | pathlib.Path,
        device: str = "cuda",
        *,
        tp_group: Optional[TPGroup] = None,
        backend: str = "auto",
    ):
        self.model_path = pathlib.Path(model_path)
        self.device = torch.device(device)
        self.tp_group = tp_group
        self.rank = tp_group.rank if tp_group is not None else 0
        self.world_size = tp_group.world_size if tp_group is not None else 1
        self.backend = _normalize_backend(backend, self.world_size)
        self.index = TensorIndex(model_path)
        self.metrics = LoadMetrics()
        self._load_start_time = time.time()

        self.local_rank, self.local_world_size, self.node_start = _resolve_local_topology(
            self.rank, self.world_size
        )
        self.node_ranks = list(range(self.node_start, self.node_start + self.local_world_size))
        self.node_leader_rank = self.node_start
        self.is_node_leader = self.rank == self.node_leader_rank
        self._p2p_window = max(1, min(self.local_world_size, 4))

        self._device_files: dict[str, object] = {}
        self._allow_data_reads = self.backend == "local" or self.is_node_leader
        self._load_session = None
        if self.rank == 0:
            self._load_session = start_load_session(
                model_name=self.model_path.name,
                backend=self.backend,
                total_bytes=self.index.total_bytes(),
                tp_world_size=self.world_size,
            )

    def keys(self) -> set[str]:
        return self.index.keys()

    def exists(self, key: str) -> bool:
        return self.index.exists(key)

    def shape(self, key: str) -> tuple[int, ...]:
        return self.index.meta(key).shape

    def dtype(self, key: str) -> torch.dtype:
        return self.index.meta(key).dtype

    def get(self, key: str) -> torch.Tensor:
        return self.tensor(key)

    def optional(self, key: str) -> torch.Tensor | None:
        if not self.exists(key):
            return None
        return self.tensor(key)

    def scalar(self, key: str, default=None):
        if not self.exists(key):
            return default
        tensor = self.tensor(key)
        if tensor.numel() != 1:
            raise ValueError(f"scalar({key!r}) expected one element, got {tuple(tensor.shape)}")
        return tensor.item()

    def tensor(self, key: str) -> torch.Tensor:
        meta = self.index.meta(key)
        out = torch.empty(meta.shape, dtype=meta.dtype, device=self.device)
        self._load_full_into(out, key)
        return out

    def dim0_shard(
        self,
        key: str,
        *,
        unit: int = 1,
        start: int = 0,
        length: int | None = None,
        shard_world_size: int | None = None,
        replica_group_size: int = 1,
        pad: bool = True,
    ) -> torch.Tensor:
        return self._dim_shard(
            key,
            dim=0,
            unit=unit,
            start=start,
            length=length,
            shard_world_size=shard_world_size,
            replica_group_size=replica_group_size,
            pad=pad,
            out=None,
        )

    def dim0_shard_shape(
        self,
        key: str,
        *,
        unit: int = 1,
        start: int = 0,
        length: int | None = None,
        shard_world_size: int | None = None,
        replica_group_size: int = 1,
        pad: bool = True,
    ) -> tuple[int, ...]:
        return self._shard_shape(
            key,
            dim=0,
            unit=unit,
            start=start,
            length=length,
            shard_world_size=shard_world_size,
            replica_group_size=replica_group_size,
            pad=pad,
        )

    def dim1_shard(
        self,
        key: str,
        *,
        unit: int = 1,
        start: int = 0,
        length: int | None = None,
        shard_world_size: int | None = None,
        replica_group_size: int = 1,
        pad: bool = True,
    ) -> torch.Tensor:
        return self._dim_shard(
            key,
            dim=1,
            unit=unit,
            start=start,
            length=length,
            shard_world_size=shard_world_size,
            replica_group_size=replica_group_size,
            pad=pad,
            out=None,
        )

    def dim1_shard_shape(
        self,
        key: str,
        *,
        unit: int = 1,
        start: int = 0,
        length: int | None = None,
        shard_world_size: int | None = None,
        replica_group_size: int = 1,
        pad: bool = True,
    ) -> tuple[int, ...]:
        return self._shard_shape(
            key,
            dim=1,
            unit=unit,
            start=start,
            length=length,
            shard_world_size=shard_world_size,
            replica_group_size=replica_group_size,
            pad=pad,
        )

    def load_into_dim0_shard(
        self,
        out: torch.Tensor,
        key: str,
        *,
        unit: int = 1,
        start: int = 0,
        length: int | None = None,
        shard_world_size: int | None = None,
        replica_group_size: int = 1,
        pad: bool = True,
    ) -> None:
        self._dim_shard(
            key,
            dim=0,
            unit=unit,
            start=start,
            length=length,
            shard_world_size=shard_world_size,
            replica_group_size=replica_group_size,
            pad=pad,
            out=out,
        )

    def load_into_dim1_shard(
        self,
        out: torch.Tensor,
        key: str,
        *,
        unit: int = 1,
        start: int = 0,
        length: int | None = None,
        shard_world_size: int | None = None,
        replica_group_size: int = 1,
        pad: bool = True,
    ) -> None:
        self._dim_shard(
            key,
            dim=1,
            unit=unit,
            start=start,
            length=length,
            shard_world_size=shard_world_size,
            replica_group_size=replica_group_size,
            pad=pad,
            out=out,
        )

    def evict_all(self) -> None:
        self._device_files.clear()
        self.index.close()

    def log_metrics(self) -> None:
        if self._load_session is not None:
            self._load_session.finish()
        elapsed_ms = int((time.time() - self._load_start_time) * 1000.0)
        vals = torch.tensor(
            [
                self.metrics.storage_bytes_read,
                self.metrics.transport_bytes_sent,
                self.metrics.transport_bytes_received,
                self.metrics.tensors_loaded,
                elapsed_ms,
            ],
            dtype=torch.long,
            device=self.device,
        )
        if self.tp_group is not None:
            dist.all_reduce(vals, op=dist.ReduceOp.SUM, group=self.tp_group.process_group)
        if self.rank == 0:
            storage_gib = vals[0].item() / (1024**3)
            sent_gib = vals[1].item() / (1024**3)
            recv_gib = vals[2].item() / (1024**3)
            load_s = vals[4].item() / 1000.0 / max(1, self.world_size)
            LOGGER.info(
                f"Load backend={self.backend} storage={storage_gib:.2f} GiB "
                f"sent={sent_gib:.2f} GiB recv={recv_gib:.2f} GiB "
                f"tensors={vals[3].item()} load_time={load_s:.1f}s"
            )

    def start_layer_progress(self, description: str, *, total: int) -> None:
        if self._load_session is not None:
            self._load_session.start_layers(description, total=total)

    def advance_layer_progress(self, *, advance: int = 1, description: str | None = None) -> None:
        if self._load_session is not None:
            self._load_session.advance_layers(advance=advance, description=description)

    def _open_device_file(self, shard_file: str):
        if not self._allow_data_reads:
            raise RuntimeError("this rank is not allowed to read checkpoint data directly")
        handle = self._device_files.get(shard_file)
        if handle is None:
            path = str(self.model_path / shard_file)
            handle = sf.safe_open(path, framework="pt", device=str(self.device))
            self._device_files[shard_file] = handle
        return handle

    def _load_full_local(self, key: str) -> torch.Tensor:
        meta = self.index.meta(key)
        tensor = self._open_device_file(meta.shard_file).get_tensor(key)
        num_bytes = tensor.numel() * tensor.element_size()
        self.metrics.storage_bytes_read += num_bytes
        if self._load_session is not None:
            self._load_session.advance_storage(num_bytes)
        self.metrics.tensors_loaded += 1
        return tensor

    def _load_slice_local(self, key: str, *, dim: int, start: int, length: int) -> torch.Tensor:
        meta = self.index.meta(key)
        valid_start = max(0, min(start, meta.shape[dim]))
        valid_end = max(valid_start, min(start + length, meta.shape[dim]))
        slices = [slice(None)] * len(meta.shape)
        slices[dim] = slice(valid_start, valid_end)
        tensor = self._open_device_file(meta.shard_file).get_slice(key)[tuple(slices)]
        num_bytes = tensor.numel() * tensor.element_size()
        self.metrics.storage_bytes_read += num_bytes
        if self._load_session is not None:
            self._load_session.advance_storage(num_bytes)
        self.metrics.tensors_loaded += 1
        return tensor

    def _load_full_into(self, out: torch.Tensor, key: str) -> None:
        if self.backend == "local":
            out.copy_(self._load_full_local(key))
            return
        if self.is_node_leader:
            out.copy_(self._load_full_local(key))
            followers = [dst for dst in self.node_ranks if dst != self.rank]
            self._batch_send([(out, dst) for dst in followers])
        else:
            self._batch_recv([out])

    def _dim_shard(
        self,
        key: str,
        *,
        dim: int,
        unit: int,
        start: int,
        length: int | None,
        shard_world_size: int | None,
        replica_group_size: int,
        pad: bool,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        meta = self.index.meta(key)
        logical_world = shard_world_size or self.world_size
        if replica_group_size <= 0:
            raise ValueError("replica_group_size must be positive")
        if logical_world <= 0:
            raise ValueError("shard_world_size must be positive")
        if logical_world * replica_group_size != self.world_size:
            raise ValueError(
                f"logical_world={logical_world} replica_group_size={replica_group_size} "
                f"must multiply to tp_world_size={self.world_size}"
            )
        total = meta.shape[dim] - start if length is None else length
        local_shape = self._local_shard_shape(
            meta.shape,
            dim=dim,
            total=total,
            logical_rank=self.rank // replica_group_size,
            logical_world=logical_world,
            unit=unit,
            pad=pad,
        )
        if out is None:
            out = torch.empty(local_shape, dtype=meta.dtype, device=self.device)
        elif tuple(out.shape) != tuple(local_shape):
            raise ValueError(
                f"output shape mismatch for {key}: expected {tuple(local_shape)}, got {tuple(out.shape)}"
            )

        if self.backend == "local":
            piece = self._load_local_shard_piece(
                key,
                dim=dim,
                start=start,
                total=total,
                logical_rank=self.rank // replica_group_size,
                logical_world=logical_world,
                unit=unit,
                pad=pad,
            )
            out.copy_(piece)
            return out

        if self.is_node_leader:
            full_tensor = self._load_full_local(key)
            logical_ranks = self._node_logical_ranks(replica_group_size)
            inflight_reqs: list[dist.Work] = []
            inflight_bytes = 0
            inflight_keepalive: list[torch.Tensor] = []
            for logical_rank_chunk in _chunked(logical_ranks, self._p2p_window):
                send_plan: list[tuple[torch.Tensor, int]] = []
                keepalive: list[torch.Tensor] = []
                for logical_rank in logical_rank_chunk:
                    piece = self._slice_shard_piece(
                        full_tensor,
                        dim=dim,
                        start=start,
                        total=total,
                        logical_rank=logical_rank,
                        logical_world=logical_world,
                        unit=unit,
                        pad=pad,
                    )
                    keepalive.append(piece)
                    for dst in self._ranks_for_logical_rank(logical_rank, replica_group_size):
                        if dst not in self.node_ranks:
                            continue
                        if dst == self.rank:
                            out.copy_(piece)
                        else:
                            send_plan.append((piece, dst))
                if inflight_reqs:
                    self._wait_batch_send(inflight_reqs, inflight_bytes)
                    inflight_reqs = []
                    inflight_keepalive.clear()
                    inflight_bytes = 0
                inflight_reqs, inflight_bytes = self._launch_batch_send(send_plan)
                inflight_keepalive = keepalive
            if inflight_reqs:
                self._wait_batch_send(inflight_reqs, inflight_bytes)
        else:
            self._batch_recv([out])

        return out

    def _launch_batch_send(self, send_plan: list[tuple[torch.Tensor, int]]) -> tuple[list[dist.Work], int]:
        if not send_plan:
            return [], 0
        assert self.tp_group is not None
        ops = [
            dist.P2POp(dist.isend, tensor, dst, group=self.tp_group.process_group)
            for tensor, dst in send_plan
        ]
        reqs = dist.batch_isend_irecv(ops)
        num_bytes = sum(tensor.numel() * tensor.element_size() for tensor, _dst in send_plan)
        return reqs, num_bytes

    def _wait_batch_send(self, reqs: list[dist.Work], num_bytes: int) -> None:
        for req in reqs:
            req.wait()
        self.metrics.transport_bytes_sent += num_bytes
        if self._load_session is not None:
            self._load_session.advance_fanout(num_bytes)

    def _batch_send(self, send_plan: list[tuple[torch.Tensor, int]]) -> None:
        reqs, num_bytes = self._launch_batch_send(send_plan)
        self._wait_batch_send(reqs, num_bytes)

    def _batch_recv(self, recv_tensors: list[torch.Tensor]) -> None:
        if not recv_tensors:
            return
        assert self.tp_group is not None
        ops = [
            dist.P2POp(dist.irecv, tensor, self.node_leader_rank, group=self.tp_group.process_group)
            for tensor in recv_tensors
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        self.metrics.transport_bytes_received += sum(
            tensor.numel() * tensor.element_size() for tensor in recv_tensors
        )

    def _shard_shape(
        self,
        key: str,
        *,
        dim: int,
        unit: int,
        start: int,
        length: int | None,
        shard_world_size: int | None,
        replica_group_size: int,
        pad: bool,
    ) -> tuple[int, ...]:
        meta = self.index.meta(key)
        logical_world = shard_world_size or self.world_size
        total = meta.shape[dim] - start if length is None else length
        return self._local_shard_shape(
            meta.shape,
            dim=dim,
            total=total,
            logical_rank=self.rank // replica_group_size,
            logical_world=logical_world,
            unit=unit,
            pad=pad,
        )

    def _node_logical_ranks(self, replica_group_size: int) -> list[int]:
        logical_ranks = []
        seen = set()
        for rank in self.node_ranks:
            logical_rank = rank // replica_group_size
            if logical_rank not in seen:
                seen.add(logical_rank)
                logical_ranks.append(logical_rank)
        return logical_ranks

    def _ranks_for_logical_rank(self, logical_rank: int, replica_group_size: int) -> list[int]:
        start = logical_rank * replica_group_size
        return list(range(start, start + replica_group_size))

    def _local_shard_shape(
        self,
        shape: tuple[int, ...],
        *,
        dim: int,
        total: int,
        logical_rank: int,
        logical_world: int,
        unit: int,
        pad: bool,
    ) -> tuple[int, ...]:
        shard_size, _valid_len, _local_start = self._shard_layout(
            total=total,
            logical_rank=logical_rank,
            logical_world=logical_world,
            unit=unit,
            pad=pad,
        )
        out_shape = list(shape)
        out_shape[dim] = shard_size
        return tuple(out_shape)

    def _shard_layout(
        self,
        *,
        total: int,
        logical_rank: int,
        logical_world: int,
        unit: int,
        pad: bool,
    ) -> tuple[int, int, int]:
        if pad:
            padded = _round_up(total, logical_world * unit)
            shard_size = padded // logical_world
        else:
            if total % logical_world != 0:
                raise ValueError(f"cannot shard total={total} across logical_world={logical_world}")
            shard_size = total // logical_world
        local_start = logical_rank * shard_size
        valid_len = max(0, min(shard_size, total - local_start))
        return shard_size, valid_len, local_start

    def _load_local_shard_piece(
        self,
        key: str,
        *,
        dim: int,
        start: int,
        total: int,
        logical_rank: int,
        logical_world: int,
        unit: int,
        pad: bool,
    ) -> torch.Tensor:
        meta = self.index.meta(key)
        shard_size, valid_len, local_start = self._shard_layout(
            total=total,
            logical_rank=logical_rank,
            logical_world=logical_world,
            unit=unit,
            pad=pad,
        )
        full_shape = list(meta.shape)
        full_shape[dim] = shard_size
        out = torch.zeros(tuple(full_shape), dtype=meta.dtype, device=self.device)
        if valid_len == 0:
            self.metrics.tensors_loaded += 1
            return out
        piece = self._load_slice_local(
            key,
            dim=dim,
            start=start + local_start,
            length=valid_len,
        )
        slices = [slice(None)] * out.ndim
        slices[dim] = slice(0, valid_len)
        out[tuple(slices)].copy_(piece)
        return out

    def _slice_shard_piece(
        self,
        tensor: torch.Tensor,
        *,
        dim: int,
        start: int,
        total: int,
        logical_rank: int,
        logical_world: int,
        unit: int,
        pad: bool,
    ) -> torch.Tensor:
        shard_size, valid_len, local_start = self._shard_layout(
            total=total,
            logical_rank=logical_rank,
            logical_world=logical_world,
            unit=unit,
            pad=pad,
        )
        full_shape = list(tensor.shape)
        full_shape[dim] = shard_size
        out = torch.zeros(tuple(full_shape), dtype=tensor.dtype, device=tensor.device)
        if valid_len == 0:
            self.metrics.tensors_loaded += 1
            return out
        slices = [slice(None)] * tensor.ndim
        slices[dim] = slice(start + local_start, start + local_start + valid_len)
        piece = tensor[tuple(slices)]
        out_slices = [slice(None)] * out.ndim
        out_slices[dim] = slice(0, valid_len)
        out[tuple(out_slices)].copy_(piece)
        self.metrics.tensors_loaded += 1
        return out


# -- public API ------------------------------------------------------------


class LoadedModel:
    """The result of loading and surgery: layers + embeddings + head."""

    def __init__(
        self,
        layers: nn.ModuleList,
        embed_tokens: nn.Embedding,
        final_norm_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        config: object,
    ):
        self.layers = layers
        self.embed_tokens = embed_tokens
        self.final_norm_weight = final_norm_weight
        self.lm_head_weight = lm_head_weight
        self.cos = cos
        self.sin = sin
        self.config = config


def load_model(
    model_path: str,
    device: torch.device | str = "cuda",
    tp_group: Optional[TPGroup] = None,
    dtype: torch.dtype = torch.bfloat16,
    load_backend: str = "auto",
) -> LoadedModel:
    """Load a HF model and apply b12x layer surgery."""

    del dtype  # Weight dtype is model-specific today.

    device_str = str(torch.device(device))
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    model_type = hf_config.model_type

    if model_type not in _RECIPES:
        import serve.model.recipe_qwen3_5  # noqa: F401

    if model_type not in _RECIPES:
        raise ValueError(
            f"no surgery recipe registered for model_type={model_type!r}. "
            f"available: {list(_RECIPES.keys())}"
        )

    instantiate_config = hf_config
    if hasattr(hf_config, "text_config"):
        instantiate_config = hf_config.text_config
        instantiate_config.model_type = instantiate_config.model_type or model_type
    LOGGER.info(f"Instantiating {model_type} on meta device")
    with torch.device("meta"):
        hf_model = AutoModelForCausalLM.from_config(instantiate_config)

    loader = ShardedLoader(model_path, device=device_str, tp_group=tp_group, backend=load_backend)

    recipe_fn = _RECIPES[model_type]
    result = recipe_fn(hf_model, hf_config, loader, device_str, tp_group)

    del hf_model
    loader.log_metrics()
    loader.evict_all()
    torch.cuda.empty_cache()

    return result


# -- register known recipes ------------------------------------------------


@register_recipe("minimax_m2")
def _apply_minimax_m2(hf_model, hf_config, loader, device, tp_group):
    from serve.model.recipe_minimax_m2 import build_config, extract_layer

    world_size = tp_group.world_size if tp_group is not None else 1
    cfg = build_config(hf_config, tp_world_size=world_size)

    max_seq_len = getattr(hf_config, "max_position_embeddings", 32768)
    cos, sin = precompute_rope_freqs(
        cfg.head_dim, cfg.rotary_dim, max_seq_len, base=cfg.rope_base, device=device
    )

    layers = nn.ModuleList()
    loader.start_layer_progress("MiniMax M2 layers", total=cfg.num_layers)
    for i in range(cfg.num_layers):
        layer = extract_layer(hf_model.model.layers[i], i, cfg, tp_group, device, loader)
        layers.append(layer)
        loader.advance_layer_progress(description=f"MiniMax M2 layers [{i + 1}/{cfg.num_layers}]")

    embed_weight = loader.tensor("model.embed_tokens.weight").to(torch.bfloat16)
    embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, _weight=embed_weight)
    final_norm_weight = loader.tensor("model.norm.weight")
    lm_head_weight = loader.tensor("lm_head.weight").to(torch.bfloat16)

    return LoadedModel(
        layers=layers,
        embed_tokens=embed_tokens,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
        cos=cos,
        sin=sin,
        config=cfg,
    )
