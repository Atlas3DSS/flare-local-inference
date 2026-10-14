"""Structural extraction and model-card defaults at the runtime package boundary."""

from pathlib import Path
import json

from google.protobuf.json_format import MessageToDict, ParseError
import pytest

from ignition.preparation.frontend import image_settings, prepare_frontend, tokenizer_settings
from ignition.preparation.package_file import combine, publish, read_index
from ignition.preparation.package_proto import serving
from ignition.preparation.runtime_package_pb2 import BundleIndex, ModelIndex
from ignition.recipes import recipes
from ignition.recipes.declaration import SamplingDefaults


@pytest.mark.parametrize("name,top_k", [("qwen38", 20), ("qwen38_dense", 20), ("glm53", -1), ("deepseek41", -1),
                                      ("deepseek4", -1), ("mimo_v2", -1)])
def test_default_mode_uses_manually_declared_model_card_sampling(name, top_k):
    defaults = next(recipe for recipe in recipes() if recipe.name == name).sampling_defaults
    assert defaults.as_dict() == dict(seed=0, temperature=1., top_p=.95, top_k=top_k,
        min_p=0., repetition_penalty=1., presence_penalty=0., frequency_penalty=0.)


@pytest.mark.parametrize("temperature", [0., -0., -1., float("nan"), float("inf")])
def test_recipe_defaults_require_positive_finite_temperature(temperature):
    with pytest.raises(ValueError, match="invalid recipe sampling defaults"):
        SamplingDefaults(temperature=temperature, top_p=.95, top_k=20)


def test_tokenizer_metadata_preserves_token_contents_and_omits_unconsumed_fields():
    source = dict(bos_token={"content": "<bos>", "special": True}, eos_token="<eos>",
        pad_token=None, additional_special_tokens=[{"content": "<extra>"}, "<other>"],
        model_max_length=123456, tokenizer_class="RemoteTokenizer", chat_template="unused")
    assert tokenizer_settings(source) == dict(bos_token="<bos>", eos_token="<eos>",
        additional_special_tokens={"values": ["<extra>", "<other>"]})
    assert tokenizer_settings({}) == {}
    assert tokenizer_settings({"additional_special_tokens": []}) == {"additional_special_tokens": {"values": []}}
    with pytest.raises(ValueError, match="clean_up_tokenization_spaces"):
        tokenizer_settings({"clean_up_tokenization_spaces": True})
    with pytest.raises(ValueError, match="special token"):
        tokenizer_settings({"bos_token": {"unused": True}})


def test_sampling_config_does_not_enter_prepared_frontend(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text('{"eos_token":"<eos>"}')
    (tmp_path / "generation_config.json").write_text('{"do_sample":false,"temperature":0,"top_k":1}')
    assert prepare_frontend(tmp_path, eos_token_ids=17) == {
        "tokenizer": {"eos_token": "<eos>"}, "eos_token_ids": [17]}


@pytest.mark.parametrize("ids", [[], [1, 1], [-1], [2**32], [True], list(range(9))])
def test_eos_metadata_rejects_invalid_ids(tmp_path, ids):
    with pytest.raises(ValueError, match="EOS token IDs"):
        prepare_frontend(tmp_path, eos_token_ids=ids)


def qwen_source():
    processor = dict(processor_class="Qwen3VLProcessor", image_processor_type="Qwen2VLImageProcessorFast",
        patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[.5] * 3, image_std=[.5] * 3,
        size=dict(shortest_edge=65536, longest_edge=16777216), unused="discard")
    model = dict(model_type="qwen4_exp", image_token_id=248056, video_token_id=248057,
        vision_config=dict(patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
            in_channels=3, num_position_embeddings=2304), quantization_config={"unused": True})
    return processor, model


def test_image_metadata_is_a_tight_consumed_contract():
    processor, model = qwen_source()
    assert image_settings(processor, model, "qwen") == {"qwen": dict(min_pixels=65536,
        max_pixels=16777216, position_side=48, image_token_id=248056, video_token_id=248057)}
    for key, value in (("do_resize", False), ("patch_size", 14), ("rescale_factor", 1.)):
        changed = {**processor, key: value}
        with pytest.raises(ValueError, match="unsupported"):
            image_settings(changed, model, "qwen")


def server():
    return dict(model_id="fixture", served_names=["fixture-alias"], reserve_bytes=1 << 30,
        queue_capacity=16, frontend_workers=4, max_body_bytes=32 << 20,
        context_hint="prepared context", capacity_hint="prepared concurrency",
        sampling_defaults=SamplingDefaults(temperature=1., top_p=.95, top_k=-1).as_dict(),
        frontend=dict(tokenizer={"eos_token": "<eos>"}, eos_token_ids=[17],
            image={"mimo": dict(min_pixels=3136, max_pixels=12845056,
                image_mean=[0.48145466, 0.4578275, 0.40821073], image_std=[0.26862954, 0.26130258, 0.27577711])}),
        assets={"tokenizer": {}, "chat_template": {}})


def test_typed_settings_survive_publication_and_tensor_parallel_assembly(tmp_path):
    paths = []
    for rank in range(2):
        index = ModelIndex(version=4, ranks=[BundleIndex()])
        serving(index, server())
        assert index.contract.model_id == "fixture"
        assert list(index.contract.served_names) == ["fixture-alias"]
        assert index.contract.reserve_bytes == 1 << 30
        assert index.contract.frontend_workers == 4
        assert index.HasField("sampling_defaults") and index.sampling_defaults.HasField("seed")
        assert index.frontend.image.WhichOneof("profile") == "mimo"
        path = tmp_path / f"rank-{rank}.safetensors"
        publish(path, index, [])
        paths.append(path)
    result = tmp_path / "model.safetensors"
    combine(result, paths, [])
    index, _ = read_index(result)
    assert MessageToDict(index.sampling_defaults, preserving_proto_field_name=True) == {
        **server()["sampling_defaults"], "seed": "0"}
    assert list(index.frontend.eos_token_ids) == [17]
    assert index.frontend.tokenizer.eos_token == "<eos>"
    assert len(index.frontend.image.mimo.image_mean) == 3
    assert not index.assets


def test_runtime_config_blobs_and_partial_sampling_defaults_are_rejected():
    for name in ("generation_config", "tokenizer_config", "model_config", "preprocessor_config"):
        source = server()
        source["assets"][name] = {}
        with pytest.raises(ValueError, match="runtime assets"):
            serving(ModelIndex(), source)
    source = server()
    del source["sampling_defaults"]["top_k"]
    with pytest.raises(ValueError, match="every parameter"):
        serving(ModelIndex(), source)
    source = server()
    source["frontend"]["tokenizer"]["unused"] = True
    with pytest.raises(ParseError, match="has no field"):
        serving(ModelIndex(), source)


@pytest.mark.parametrize("recipe", recipes(), ids=lambda recipe: recipe.name)
def test_cached_checkpoint_structural_metadata_extracts_without_model_weights(recipe):
    checkpoint = recipe.checkpoints[0]
    snapshot = (Path.home() / ".cache/huggingface/hub" / ("models--" + checkpoint.repository.replace("/", "--"))
                / "snapshots" / checkpoint.revision)
    if not (snapshot / "tokenizer_config.json").exists():
        pytest.skip("checkpoint structural metadata is not cached")
    profiles = {"qwen38": "qwen", "mimo_v2": "mimo", "deepseek4": "deepseek4"}
    preprocessor = Path("ignition/recipes/deepseek4/assets/preprocessor_config.json") if recipe.name == "deepseek4" else None
    token_source = snapshot / "generation_config.json"
    if not token_source.exists():
        token_source = snapshot / "config.json"
    eos = json.loads(token_source.read_text())["eos_token_id"]
    result = prepare_frontend(snapshot, eos_token_ids=eos, image_profile=profiles.get(recipe.name), preprocessor=preprocessor)
    index = ModelIndex()
    source = server()
    source["frontend"] = result
    source["sampling_defaults"] = recipe.sampling_defaults.as_dict()
    serving(index, source)
    assert index.frontend.tokenizer == ModelIndex.FromString(index.SerializeToString()).frontend.tokenizer
    if recipe.name in profiles:
        assert index.frontend.image.WhichOneof("profile") == profiles[recipe.name]
    else:
        assert not index.frontend.HasField("image")
