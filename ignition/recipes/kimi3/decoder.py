"""Kimi K3 attention residuals and latent MoE composed into complete forwards."""

from dataclasses import dataclass

import torch
import triton

from ignition.components.collectives import PreparedAllGather, PreparedAllReduce
from ignition.components.embedding import PreparedShardedEmbedding
from ignition.components.linear import UnquantizedProjection
from ignition.components.moe import PreparedRouter, PreparedRoutedExperts, RouterScratch
from ignition.components.recurrent import DeltaMetadata, _commit_convolution
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.components.kda_speculation import PreparedKDAReplay

from .attention import PreparedMLA
from .kda import PreparedKDA
from .model import attention_residual, rms_norm, situ


@dataclass(frozen=True)
class DecoderStorage:
    scratch: dict
    hidden_columns: torch.Tensor
    latent_columns: torch.Tensor

    @classmethod
    def allocate(cls, config, shard, device, *, scratch=None):
        return cls({} if scratch is None else scratch,
            torch.arange(*shard.partition(config.hidden), device=device),
            torch.arange(*shard.partition(config.text["routed_expert_hidden_size"]), device=device))

    def scratch_tensors(self):
        return tuple(self.scratch.values())


@dataclass(frozen=True)
class Scratch:
    activations: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    @classmethod
    def allocate(cls, tokens, rows, config, world_size, device, *, activations=None):
        def zeros(shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        return cls(zeros((tokens, config.hidden)) if activations is None else activations, zeros((rows, config.hidden)),
                   zeros((rows, config.vocabulary // world_size)), zeros((rows, config.vocabulary)))

    def rows(self, tokens, rows):
        return type(self)(self.activations[:tokens], self.hidden[:rows], self.local_logits[:rows], self.logits[:rows])


class PreparedMLP:
    def __init__(self, weights, source, *, session, name, buffers):
        rows, device = source.shape[0], source.device
        def buffer(field, width):
            if field not in buffers or buffers[field].shape[0] < rows:
                buffers[field] = torch.zeros((rows, width), dtype=source.dtype, device=device)
            return buffers[field][:rows]
        self.input = buffer("input", weights["gate_up"].shape[0] // 2)
        self.gate_up = UnquantizedProjection(source, weights["gate_up"], buffer("gate_up", 2 * self.input.shape[1]),
                                            session=session, name=name + ".gate_up")
        self.down = UnquantizedProjection(self.input, weights["down"], buffer("output", source.shape[1]),
                                         session=session, name=name + ".down")

    def __call__(self, source):
        self.input.copy_(situ(self.gate_up(source)))
        return self.down(self.input)


class PreparedLatentMoE:
    def __init__(self, weights, source, config, shard, *, runtime, router_runtime, coordinator, session, name, workspace, buffers, storage):
        tokens, hidden = source.shape
        c, device = config.text, source.device
        latent = c["routed_expert_hidden_size"]

        def buffer(field, width, dtype=torch.bfloat16):
            if field not in buffers or buffers[field].shape[0] < tokens:
                buffers[field] = torch.zeros((tokens, width), dtype=dtype, device=device)
            return buffers[field][:tokens]

        self.source = buffer("source", hidden // shard.world_size)
        self.hidden_columns, self.latent_columns = storage.hidden_columns, storage.latent_columns
        self.latent = buffer("latent", latent)
        self.expert_output = buffer("expert_output", latent)
        self.latent_sum = buffer("latent_sum", latent)
        self.up_input = buffer("up_input", latent // shard.world_size)
        self.output = buffer("output", hidden)
        self.down = UnquantizedProjection(self.source, weights["latent_down"], buffer("down", latent),
                                         session=session, name=name + ".latent_down")
        self.up = UnquantizedProjection(self.up_input, weights["latent_up"], self.output,
                                       session=session, name=name + ".latent_up")
        self.norm = rms_norm(weights["latent_norm"], c["rms_norm_eps"])
        self.down_reduce = PreparedAllReduce(self.down.output, self.latent, runtime=runtime, coordinator=coordinator,
                                            session=session, name=name + ".down_reduce")
        self.expert_reduce = PreparedAllReduce(self.expert_output, self.latent_sum, runtime=runtime,
                                              coordinator=coordinator, session=session, name=name + ".expert_reduce")
        logits = buffer("router", c["num_experts"], torch.float32)
        local_logits = buffer("local_router", weights["router"].shape[0], torch.float32)
        self.router_logits = logits
        local_experts = local_logits.shape[1]
        router_width = (local_experts + 31) // 32 * 32
        self.router_input = buffer("router_input", router_width, torch.float32)
        self.gathered_logits = buffer("gathered_router", router_width * shard.world_size, torch.float32)
        self.router_columns = self.gathered_logits.view(tokens, shard.world_size, router_width)[:, :, :local_experts]
        self.router_output = logits.view(tokens, shard.world_size, local_experts)
        self.router_projection = UnquantizedProjection(source, weights["router"], local_logits, session=session, name=name + ".router")
        # Gathering BF16 views transports the FP32 logits without conversion or arithmetic.
        self.router_gather = PreparedAllGather(self.router_input.view(torch.bfloat16), self.gathered_logits.view(torch.bfloat16),
            runtime=router_runtime, coordinator=coordinator, session=session, name=name + ".router_gather")
        if "routes" not in buffers:
            buffers["routes"] = RouterScratch.allocate(tokens, device, experts=c["num_experts"], top_k=c["num_experts_per_token"])
        routes = buffers["routes"].rows(tokens)
        self.router = PreparedRouter(logits, routes, score_func="sigmoid", correction_bias=weights["correction"],
            renormalize=True, routed_scaling_factor=c["routed_scaling_factor"], session=session, name=name + ".route")
        self.experts = PreparedRoutedExperts(self.latent, weights["experts"], routes.ids, routes.weights,
            self.expert_output, workspace=workspace, session=session, name=name + ".experts")
        self.shared = PreparedMLP(weights["shared"], source, session=session, name=name + ".shared",
                                  buffers=buffers.setdefault("shared", {}))

    def __call__(self, source):
        logits = self.router_projection(source)
        self.router_input[:, :logits.shape[1]].copy_(logits)
        self.router_gather(self.router_input.view(torch.bfloat16))
        self.router_output.copy_(self.router_columns)
        self.router(self.router_logits)
        self.source.copy_(source.index_select(1, self.hidden_columns))
        self.down_reduce(self.down(self.source))
        self.expert_reduce(self.experts())
        self.up_input.copy_(self.norm(self.latent_sum).index_select(1, self.latent_columns))
        self.up(self.up_input)
        self.output.add_(self.shared(source))
        return self.output


class PreparedDecoder:
    def __init__(self, weights, scratch, metadata, pool, *, mode, runtime, vocabulary_runtime, embedding_runtime, router_runtime, vision_state,
                 coordinator, session, name, workspace, buffers, kda_records=None):
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.vision_state = vision_state if mode == "prefill" else None
        config, source = weights.config, scratch.activations
        self.local_embedding = torch.zeros((source.shape[0], weights.embedding.shape[1]), dtype=source.dtype, device=source.device)
        self.embedding = PreparedShardedEmbedding(weights.embedding, metadata.token_ids, self.local_embedding, source,
            runtime=embedding_runtime, coordinator=coordinator,
            num_rows=metadata.live_counts[:1], session=session, name=name + ".embedding")
        self.delta = DeltaMetadata.allocate(metadata.starts.numel(), source.device,
            verification_columns=8 if mode == "verify" else 1)
        self.buffers, self.layers = buffers.scratch, []
        self.target_layers = () if weights.draft is None else config.draft.target_layers
        self.auxiliary = None
        if self.target_layers:
            if "auxiliary" not in self.buffers:
                self.buffers["auxiliary"] = torch.zeros((source.shape[0], len(self.target_layers) * config.hidden),
                    dtype=source.dtype, device=source.device)
            self.auxiliary = self.buffers["auxiliary"][:source.shape[0]]
        for index, w in enumerate(weights.layers):
            prefix = f"{name}.layer{index}"
            kind = config.layers[index]
            kwargs = dict(mode=mode, session=session, name=prefix + ".attention", workspace=workspace,
                          buffers=self.buffers.setdefault(kind, {}))
            attention = (PreparedKDA(w["attention"], source, pool.recurrent(index), self.delta,
                eps=config.text["rms_norm_eps"], records=kda_records.layer(index) if mode == "verify" else None,
                **kwargs) if kind == "kda" else
                PreparedMLA(w["attention"], source, metadata, pool.attention(index), **kwargs))
            if "mlp" in w:
                mlp = PreparedMLP(w["mlp"], source, session=session, name=prefix + ".mlp",
                                  buffers=self.buffers.setdefault("dense", {}))
            else:
                mlp = PreparedLatentMoE(w, source, config, weights.shard, runtime=runtime,
                    router_runtime=router_runtime,
                    coordinator=coordinator, session=session, name=prefix + ".moe", workspace=workspace,
                    buffers=self.buffers.setdefault("moe", {}), storage=buffers)
            # The sum buffer is consumed before the next attention or MLP reduction.
            if "total" not in self.buffers:
                self.buffers["total"] = torch.zeros_like(source)
            total = self.buffers["total"][:source.shape[0]]
            reduce = PreparedAllReduce(source, total, runtime=runtime, coordinator=coordinator,
                                       session=session, name=prefix + ".reduce")
            self.layers.append((w, attention, mlp, reduce,
                rms_norm(w["input_layernorm"], config.text["rms_norm_eps"]),
                rms_norm(w["post_attention_layernorm"], config.text["rms_norm_eps"])))
        self.norm = rms_norm(weights.norm, config.text["rms_norm_eps"])
        self.head = PreparedVocabularyProjection(scratch.hidden, weights.head, scratch.local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head")
        self.replay = None
        if mode == "verify":
            indices = kda_records.layers
            states = [pool.recurrent(i).recurrent[:kda_records.corrections.shape[1]] for i in indices]
            self.replay = PreparedKDAReplay(pool.backing, states, kda_records, self.layers[indices[0]][1].plan,
                self.delta.state_indices[:, 0], session=session, workspace=workspace)
            regions = [pool.layout.region(f"layer{i}.convolution") for i in indices]
            self.convolution_offsets = torch.tensor([r.offset // 2 for r in regions], dtype=torch.int64, device=source.device)
            self.convolution_shape = regions[0].shape

    def commit(self, counts):
        self.replay(counts)
        channels, columns = self.convolution_shape
        torch.library.wrap_triton(_commit_convolution)[
            (self.convolution_offsets.numel(), counts.numel(), triton.cdiv(channels, 128))](
            self.pool.backing.view(torch.bfloat16), self.convolution_offsets, self.metadata.state_rows, counts,
            self.pool.layout.row_bytes // 2, channels, columns, 128)

    def __call__(self):
        m, d, c = self.metadata, self.delta, self.weights.config.text
        active = m.lengths > 0
        d.query_start_loc.copy_(m.query_start_loc)
        d.convolution_slots.copy_(torch.where(active, m.state_rows, -1))
        d.fresh.copy_(m.starts == 0)
        d.state_indices[:, 0].copy_(m.state_rows)
        d.initial_state_indices.copy_(torch.where(m.starts > 0, m.state_rows, 0))
        d.final_state_indices.copy_(m.state_rows)
        d.num_tokens.copy_(m.live_counts[:1])
        d.num_seqs.copy_(m.live_counts[1:])
        local = self.embedding.lookup()
        if self.vision_state is not None:
            self.vision_state.inject(m, local)
        prefix, blocks = self.embedding.gather(local), ()
        for index, (w, attention, mlp, reduce, input_norm, post_norm) in enumerate(self.layers):
            hidden = (attention_residual(prefix, blocks, w["self_attention_res_norm"], w["self_attention_res_proj"],
                                         c["rms_norm_eps"]) if blocks else prefix)
            boundary = index % c["attn_res_block_size"] == 0
            if boundary:
                blocks = (*blocks, prefix)
            attended = reduce(attention(input_norm(hidden)))
            prefix = attended.clone() if boundary else prefix + attended
            hidden = attention_residual(prefix, blocks, w["mlp_res_norm"], w["mlp_res_proj"], c["rms_norm_eps"])
            prefix = prefix + reduce(mlp(post_norm(hidden)))
            if index in self.target_layers:
                column = self.target_layers.index(index) * c["hidden_size"]
                self.auxiliary[:, column:column + c["hidden_size"]].copy_(prefix)
        hidden = attention_residual(prefix, blocks, self.weights.output_res_norm,
                                    self.weights.output_res_proj, c["rms_norm_eps"])
        self.scratch.hidden.copy_(self.norm(hidden[m.sample_indices]))
        self.head(self.scratch.hidden)
