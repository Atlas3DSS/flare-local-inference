"""Exact-checkpoint decoder weight ownership and canonical module parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from flare.preparation.checkpoint import CheckpointManifest
from flare.preparation.qwen38 import TEXT_PREFIX
from flare.preparation.scratch import ScratchArena

from .model import create_model_tree
from .hyperconnection import packed_down_projection
from .moe import QwenMoeWeights
from .vision import VisionWeights
from .weights import QwenWeightLoader, allocate_expert_scale_scratch


@dataclass(frozen=True)
class DecoderLayerWeights:
    attention: dict[str, object]
    moe: QwenMoeWeights


@dataclass(frozen=True)
class ModelWeights:
    """Own the canonical tree and every packed GPU weight used by its recipes.

    Canonical quantized Linear and expert parameters remain on meta storage;
    their prepared replacements own the checkpoint's packed representations.
    PLE table ownership is separate because those tables reside in mapped host
    memory. All expert layers share the same scale-expansion workspace and
    therefore execute serially.
    """

    tree: torch.nn.Module
    layers: tuple[DecoderLayerWeights, ...]
    vision: VisionWeights
    expert_scale_scratch: tuple[torch.Tensor, torch.Tensor]
    allocated_bytes: int

    @classmethod
    def load(cls, manifest: CheckpointManifest, device: torch.device,
             *, progress: Callable[[str], None] | None = None) -> ModelWeights:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("model weights require a CUDA device")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
            raise ValueError("the Qwen recipe requires SM120 or SM121")
        free_bytes, _ = torch.cuda.mem_get_info(device)
        stored = sum(tensor.nbytes for tensor in manifest.tensors.values()
                     if tensor.component not in ("mtp", "ple_tables"))
        # Additional layouts, state, graph memory, and cache pages are planned separately.
        if free_bytes < stored + (8 << 30):
            raise MemoryError(f"model weight loading requires at least {stored + (8 << 30)} free CUDA bytes; found {free_bytes}")
        tree = create_model_tree(manifest)
        before = torch.cuda.memory_allocated(device)
        scale_scratch = allocate_expert_scale_scratch(device)
        layers = []
        with QwenWeightLoader(manifest, device) as loader:
            def parameter(path: str):
                value = loader.load_tensor(path)
                parent, _, attribute = path.rpartition(".")
                setattr(tree.get_submodule(parent), attribute, torch.nn.Parameter(value, requires_grad=False))

            def hyperconnection(prefix: str):
                names = ("hc_norm", "input_mix_weight_down", "input_mix_weight_up")
                if tree.get_submodule(prefix).block_inject_weight is not None:
                    names += ("block_inject_weight",)
                for name in names:
                    parameter(f"{prefix}.{name}.weight")
                packed_down_projection(tree.get_submodule(prefix))

            parameter(f"{TEXT_PREFIX}.embed_tokens.weight")
            parameter("lm_head.weight")
            hyperconnection(f"{TEXT_PREFIX}.hyper_connection_mixer")
            for layer in range(48):
                prefix = f"{TEXT_PREFIX}.layers.{layer}"
                hyperconnection(f"{prefix}.attn_hyper_connection")
                hyperconnection(f"{prefix}.mlp_hyper_connection")
                if layer % 4 == 3:
                    attention_prefix = f"{prefix}.self_attn"
                    load_projections = loader.load_qsa_projections
                    parameters = ("q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight",
                                  "indexer.k_layernorm.weight")
                else:
                    attention_prefix = f"{prefix}.linear_attn"
                    load_projections = loader.load_gdn_projections
                    parameters = ("A_log", "dt_bias", "conv1d.weight", "norm.weight")
                for name in parameters:
                    parameter(f"{attention_prefix}.{name}")
                attention = load_projections(attention_prefix)
                moe = QwenMoeWeights.load(loader, layer, scale_scratch=scale_scratch)
                module = tree.get_submodule(f"{prefix}.mlp")
                module.gate.weight = torch.nn.Parameter(moe.router, requires_grad=False)
                module.shared_expert_gate.weight = torch.nn.Parameter(moe.shared.gate, requires_grad=False)
                layers.append(DecoderLayerWeights(attention, moe))
                if progress is not None:
                    progress(f"loaded decoder layer {layer + 1}/48")

            prefix = f"{TEXT_PREFIX}.layers.1.ple"
            for name in ("key_proj.weight", "value_proj.weight", "conv1d.weight", "norm_key.weight",
                         "norm_query.weight", "norm_conv.weight"):
                parameter(f"{prefix}.{name}")
            vision = VisionWeights.load(tree.model.visual, loader)
        torch.cuda.synchronize(device)
        return cls(tree, tuple(layers), vision, scale_scratch,
                   torch.cuda.memory_allocated(device) - before)


@dataclass(frozen=True)
class DecoderScratch:
    """Shared intermediates for serialized complete-forward graph variants."""

    embedding: torch.Tensor
    residual: torch.Tensor
    hidden: torch.Tensor
    sample_hidden: torch.Tensor
    logits: torch.Tensor
    hyperconnection: object
    gdn: object
    qsa: object
    ple: object
    moe: object

    @classmethod
    def allocate(cls, tokens: int, device, *, sample_rows: int = 4):
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("decoder token capacity must be a positive integer")
        if type(sample_rows) is not int or sample_rows <= 0:
            raise ValueError("decoder sampling capacity must be positive")
        from .attention import QSAScratch
        from .gdn import GDNScratch
        from .hyperconnection import HyperconnectionScratch
        from .moe import QwenMoeScratch
        from .ple import PLEScratch

        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)

        return cls(zeros(tokens, 2560), zeros(tokens, 10240), zeros(tokens, 2560),
            zeros(sample_rows, 2560), zeros(sample_rows, 248320), HyperconnectionScratch.allocate(tokens, device),
            GDNScratch.allocate(tokens, device), QSAScratch.allocate(tokens, device),
            PLEScratch.allocate(tokens, device), QwenMoeScratch.allocate(tokens, device))

    def rows(self, tokens: int, sequences: int):
        if (type(tokens) is not int or type(sequences) is not int
                or not 0 < tokens <= self.embedding.shape[0]
                or not 1 <= sequences <= self.sample_hidden.shape[0]):
            raise ValueError("decoder view exceeds persistent scratch capacity")

        def rows(value):
            return type(value)(*(getattr(value, field)[:tokens] for field in value.__dataclass_fields__))

        return type(self)(self.embedding[:tokens], self.residual[:tokens], self.hidden[:tokens],
            self.sample_hidden[:sequences], self.logits[:sequences], self.hyperconnection.rows(tokens),
            self.gdn.rows(tokens), rows(self.qsa), rows(self.ple), self.moe.rows(tokens))


class PreparedDecoder:
    """Capture the canonical 48-layer residual composition with prepared operators.

    Weights, hybrid state, paged attention caches and all escaping tensors have
    explicit owners. Scratch and workspaces are shared across serialized layer
    calls and graph variants. Finalize scratch bindings before capture.
    Python executes this object
    only during preparation and capture.
    """

    def __init__(self, weights: ModelWeights, scratch: DecoderScratch, state,
                 qsa_states, rotary, ple_table, workspaces: ScratchArena,
                 completion_records, frame_index, submission_sequence,
                 *, session, name: str, progress=None,
                 layer_outputs: torch.Tensor | None = None, embedding_injection=None,
                 stage_outputs: dict[str, torch.Tensor] | None = None,
                 max_cache_pages: int | None = None,
                 residual_output: torch.Tensor | None = None, verification: bool = False):
        from b12x.preparation import PreparedCall
        from b12x.sequence import embedding

        from .attention import PreparedAttention
        from .gdn import PreparedGDN
        from .hyperconnection import PreparedHyperconnection
        from .linear import BF16Projection
        from .moe import PreparedQwenMoe
        from .ple import PreparedPLE

        self.weights, self.scratch, self.state = weights, scratch, state
        self.verification = verification
        if residual_output is not None and (
                residual_output.shape != scratch.residual.shape
                or residual_output.dtype != torch.bfloat16
                or residual_output.device != scratch.residual.device):
            raise ValueError("target feedback requires persistent BF16 [tokens,10240] storage")
        self.residual_output = residual_output
        if layer_outputs is not None and (layer_outputs.shape != (48, state.tokens, 10240)
                or layer_outputs.dtype != torch.bfloat16 or layer_outputs.device != scratch.hidden.device):
            raise ValueError("layer trace requires BF16 [48,token_capacity,10240] on the model device")
        self.layer_outputs = layer_outputs
        self.embedding_injection = embedding_injection
        self.stage_outputs = {} if stage_outputs is None else stage_outputs
        self.workspaces, self.ple_table, self.qsa_states = workspaces, ple_table, qsa_states
        source = scratch.hyperconnection.block_input
        language = weights.tree.model.language_model
        self.embedding_weight = language.embed_tokens.weight
        self.embedding_args = dict(out=scratch.embedding, num_rows=state.gdn.num_tokens)
        self.embedding_plan = embedding.plan(embedding.query_from_call(self.embedding_weight,
            state.ple.token_ids, **self.embedding_args), device=source.device)
        session.prepare((self.embedding_plan.request(name=f"{name}.embedding", prepare_call=lambda materialized:
            PreparedCall(run=lambda: materialized.run(self.embedding_weight, state.ple.token_ids,
                                                       **self.embedding_args))),))
        self.layers = []
        for index, (module, layer_weights) in enumerate(zip(language.layers, weights.layers, strict=True)):
            prefix = f"{name}.layer{index}"
            attn_hc = PreparedHyperconnection(module.attn_hyper_connection, scratch.hyperconnection,
                priming_state=scratch.residual, session=session, name=f"{prefix}.attn_hc")
            mlp_hc = PreparedHyperconnection(module.mlp_hyper_connection, scratch.hyperconnection,
                priming_state=scratch.residual, session=session, name=f"{prefix}.mlp_hc")
            if index % 4 == 3:
                attention = PreparedAttention(module.self_attn, layer_weights.attention, source,
                    scratch.qsa, qsa_states[index], state.qsa, rotary, session=session, name=f"{prefix}.qsa",
                    workspace=workspaces, projection_workspace=workspaces,
                    max_cache_pages=max_cache_pages, draft_tokens=state.pool.layout.speculative_tokens)
            else:
                attention = PreparedGDN(module.linear_attn, layer_weights.attention, source,
                    scratch.gdn, state.pool.gdn[index], state.gdn, mode=state.mode,
                    session=session, name=f"{prefix}.gdn", checkpoint_export=False,
                    workspace=workspaces, projection_workspace=workspaces)
            moe = PreparedQwenMoe(source, layer_weights.moe, session=session, name=f"{prefix}.moe",
                scratch=scratch.moe, workspace=workspaces,
                projection_workspace=workspaces)
            self.layers.append((attn_hc, attention, mlp_hc, moe))
            if progress is not None:
                progress(f"prepared {name} layer {index + 1}/48")
        module = language.layers[1].ple
        self.ple = PreparedPLE(module,
            {key: getattr(module, key).weight for key in ("key_proj", "value_proj")},
            scratch.residual, scratch.ple, state.pool.ple_convolution, state.ple, ple_table,
            mode=state.mode, session=session, name=f"{name}.ple", workspace=workspaces)
        self.final_mixer = PreparedHyperconnection(language.hyper_connection_mixer,
            scratch.hyperconnection, priming_state=scratch.residual,
            session=session, name=f"{name}.final_mixer")
        for index, (attn_hc, _, mlp_hc, _) in enumerate(self.layers):
            attn_hc.prepare_combine_norm(mlp_hc, priming_state=scratch.residual,
                session=session, name=f"{name}.layer{index}.attn_combine_norm")
            if index != 0:
                following = self.layers[index + 1][0] if index + 1 < len(self.layers) else self.final_mixer
                mlp_hc.prepare_combine_norm(following, priming_state=scratch.residual,
                    session=session, name=f"{name}.layer{index}.mlp_combine_norm")
        state.prepare_hidden(scratch.hidden, scratch.sample_hidden)
        state.prepare_residual(scratch.residual)
        self.lm_head = BF16Projection(scratch.sample_hidden, weights.tree.lm_head.weight,
            scratch.logits, session=session, name=f"{name}.lm_head")
        self.sampler = None
        if not verification:
            from .sampling import PreparedSampling

            state, inputs = self.state, self.state.inputs
            self.sampler = PreparedSampling(self.scratch.logits, state.requests,
                inputs.row_slot_ids[:state.rows], inputs.row_generations[:state.rows],
                inputs.live_rows, completion_records, frame_index, submission_sequence,
                sample_enabled=inputs.sample_enabled[:state.rows])
            self.sampler.prepare()

    def __call__(self):
        from b12x.sequence import embedding

        sampler = self.sampler
        s = self.scratch
        if self.embedding_injection is not None:
            self.embedding_injection.validate()
        self.state.begin()
        s.embedding.zero_()
        embedding.run(self.embedding_weight, self.state.ple.token_ids, plan=self.embedding_plan,
                      **self.embedding_args)
        if self.embedding_injection is not None:
            self.embedding_injection.inject()
        if self.stage_outputs:
            self.stage_outputs["embedding"].copy_(s.embedding)
        s.residual.view(-1, 4, 2560).copy_(s.embedding[:, None, :])
        if self.stage_outputs:
            self.stage_outputs["residual_in"].copy_(s.residual)
        for index, (attn_hc, attention, mlp_hc, moe) in enumerate(self.layers):
            if index == 1:
                s.residual.add_(self.ple(s.residual))
            mixed = attn_hc.mix(s.residual, pre_normalized=index not in (0, 1))
            if index == 0 and self.stage_outputs:
                self.stage_outputs["attn_mix"].copy_(mixed)
                for field in ("normalized", "projected_down", "bottleneck", "gate_logits", "injection_logits"):
                    self.stage_outputs[f"attn_{field}"].copy_(getattr(s.hyperconnection, field))
            block = attention(mixed)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["attn_output"].copy_(block)
                for field in ("projected_qkv", "convolved_qkv", "recurrence", "gated"):
                    self.stage_outputs[f"gdn_{field}"].copy_(getattr(s.gdn, field))
            attn_hc.combine_norm(s.residual, block)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["attn_residual"].copy_(s.residual)
            mixed = mlp_hc.mix(s.residual, pre_normalized=True)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["mlp_mix"].copy_(mixed)
                for field in ("normalized", "projected_down", "bottleneck", "gate_logits", "injection_logits"):
                    self.stage_outputs[f"mlp_{field}"].copy_(getattr(s.hyperconnection, field))
            block = moe(mixed)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["mlp_output"].copy_(block)
                self.stage_outputs["moe_router_logits"].copy_(moe.router_logits)
                self.stage_outputs["moe_ids"].copy_(moe.router.ids)
                self.stage_outputs["moe_weights"].copy_(moe.router.weights)
                self.stage_outputs["moe_shared"].copy_(moe.shared.output)
                self.stage_outputs["moe_routed"].copy_(moe.routed_output)
            if index == 0:
                s.residual.copy_(mlp_hc.combine(s.residual, block))
            else:
                mlp_hc.combine_norm(s.residual, block)
            if index == 0 and self.stage_outputs:
                self.stage_outputs["mlp_residual"].copy_(s.residual)
            if self.layer_outputs is not None:
                self.layer_outputs[index].copy_(s.residual)
        if self.residual_output is not None:
            self.residual_output.copy_(s.residual)
        s.hidden.copy_(self.final_mixer.mix(s.residual, pre_normalized=True))
        self.state.finish()
        self.lm_head(s.sample_hidden)
        if sampler is not None:
            sampler()
