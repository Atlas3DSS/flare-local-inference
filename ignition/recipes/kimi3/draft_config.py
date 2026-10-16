"""MLA DFlash2 geometry and its external checkpoint declaration."""

from dataclasses import dataclass
import json
from pathlib import Path

REPOSITORY = "lightseekorg/kimi-k3-dflash2"
REVISION = "e77935fb4804e17eb55085bffd045eae1d779769"
CHECKPOINT = Path("/data/models/Kimi-K3-DFlash2")


@dataclass(frozen=True)
class DraftConfig:
    text: dict

    @classmethod
    def load(cls, snapshot, target):
        c = json.loads((Path(snapshot) / "config.json").read_text())
        expected = dict(architectures=["DFlash2DraftModel"], hidden_size=target.hidden,
            vocab_size=target.vocabulary, hidden_act="silu", attention_bias=False,
            is_causal=False, q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128,
            qk_rope_head_dim=64, v_head_dim=128, mla_use_output_gate=False, fc_norm=False,
            num_attention_heads=64, num_key_value_heads=64, num_hidden_layers=5,
            intermediate_size=14336, tie_word_embeddings=False)
        for field, value in expected.items():
            if c.get(field) != value:
                raise ValueError(f"unsupported Kimi DFlash2 {field}: {c.get(field)!r}")
        d = c["dflash_config"]
        for field, value in dict(block_size=8, attention_mode="mla", conv_kernel_size=2,
                conv_group_size=16, selector_rank=256, selector_top_k=16).items():
            if d.get(field) != value:
                raise ValueError(f"unsupported Kimi DFlash2 {field}: {d.get(field)!r}")
        taps = d["target_layer_ids"]
        if not taps or sorted(set(taps)) != taps or not all(0 <= i < len(target.layers) for i in taps):
            raise ValueError("DFlash2 feature taps must be ordered target decoder indices")
        if c["layer_types"] != ["sliding_attention"] * 4 + ["full_attention"] or c["sliding_window"] != 4096:
            raise ValueError("Kimi DFlash2 requires four 4096-token windows and one full attention layer")
        if not 0 <= d["mask_token_id"] < target.vocabulary:
            raise ValueError("DFlash2 mask token must belong to the target vocabulary")
        rope = c["rope_parameters"]
        if rope.get("rope_type") != "yarn" or rope.get("mscale") != rope.get("mscale_all_dim"):
            raise ValueError("Kimi DFlash2 requires YaRN with equal rotary and full-head scaling")
        return cls(c)

    @property
    def block_size(self):
        return self.text["dflash_config"]["block_size"]

    @property
    def noise_token(self):
        return self.text["dflash_config"]["mask_token_id"]

    @property
    def target_layers(self):
        return tuple(self.text["dflash_config"]["target_layer_ids"])

    @property
    def layers(self):
        return self.text["num_hidden_layers"]
