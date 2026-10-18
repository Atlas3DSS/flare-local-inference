"""Extract and validate the structural frontend fields consumed by Helios."""

import json
from pathlib import Path


TOKEN_FIELDS = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")


def _token(value):
    if isinstance(value, dict):
        value = value.get("content")
    if not isinstance(value, str):
        raise ValueError("special token must be a string or an object with string content")
    return value


def tokenizer_settings(config):
    if not isinstance(config, dict):
        raise ValueError("tokenizer configuration must be an object")
    if config.get("clean_up_tokenization_spaces") is True:
        raise ValueError("frontend requires tokenizer clean_up_tokenization_spaces=false")
    result = {name: _token(config[name]) for name in TOKEN_FIELDS if config.get(name) is not None}
    additional = config.get("additional_special_tokens")
    if additional is not None:
        if not isinstance(additional, list):
            raise ValueError("additional_special_tokens must be a list")
        result["additional_special_tokens"] = {"values": [_token(value) for value in additional]}
    return result


def _expected(config, values, label):
    for key, expected in values.items():
        value = config.get(key)
        if value != expected or (type(expected) in (bool, int) and type(value) is not type(expected)):
            raise ValueError(f"unsupported {label} {key}")


def image_settings(config, model, profile):
    if profile == "radio":
        _expected(config, {"image_processor_type": "NemotronH_Omni_Reasoning_V3ImageProcessor",
            "patch_size": 16, "downsample_ratio": .5, "max_model_len": 16384,
            "min_num_patches": 1024, "max_num_patches": 13312,
            "norm_mean": [0.48145466, 0.4578275, 0.40821073],
            "norm_std": [0.26862954, 0.26130258, 0.27577711]}, "RADIO image processor")
        _expected(model, {"patch_size": 16, "downsample_ratio": .5, "ps_version": "v2",
            "img_context_token_id": 18}, "RADIO image configuration")
        _expected(model["vision_config"], {"max_img_size": 2048, "patch_size": 16}, "RADIO vision configuration")
        return {"radio": {}}
    if profile == "moonvit":
        _expected(config["media_proc_cfg"], {"in_patch_limit": 65536, "patch_size": 14,
            "image_mean": [0.5] * 3, "image_std": [0.5] * 3, "merge_kernel_size": 2,
            "fixed_output_tokens": None, "patch_limit_on_one_side": 512,
            "transparent_bg_fill_stage": "after_resize", "transparent_bg_config": {
                "pattern": "chessboard", "chessboard_square_size": 8,
                "chessboard_square_on_top_left": True, "chessboard_white_value": 255,
                "chessboard_gray_value": 180}}, "MoonViT image processor")
        _expected(model["vision_config"], {"patch_size": 14, "init_pos_emb_height": 64,
            "init_pos_emb_width": 64, "pos_emb_interpolation_mode": "bilinear",
            "merge_kernel_size": [2, 2]}, "MoonViT vision configuration")
        return {"moonvit": {}}
    if profile == "deepseek_row_major":
        vision = model["vision_config"]
        _expected(vision, {"patch_size": 14, "downsample_ratio": 3, "max_image_tokens": 1024,
            "min_pixels": 295936, "max_wh_ratio": None}, "DeepSeek row-major vision configuration")
        return {profile: dict(min_pixels=vision["min_pixels"], max_tokens=vision["max_image_tokens"])}
    if profile == "deepseek4":
        _expected(config, {"image_processor_type": "DeepSeekV4ImageProcessor", "patch_size": 14,
            "merge_size": 3, "max_image_tokens": 384, "min_pixels": 147456,
            "max_width_height_ratio": 8}, "DeepSeek image processor")
        _expected(model, {"vision_patch_size": 14, "vision_downsample_ratio": 3,
            "vision_max_n_token": 384, "vision_min_pixels": 147456,
            "vision_max_wh_ratio": 8}, "DeepSeek vision configuration")
        return {"deepseek4": {}}
    if profile not in ("qwen", "mimo"):
        raise ValueError("unsupported image preprocessing profile")
    qwen = profile == "qwen"
    mean = [0.5] * 3 if qwen else [0.48145466, 0.4578275, 0.40821073]
    std = [0.5] * 3 if qwen else [0.26862954, 0.26130258, 0.27577711]
    _expected(config, {"processor_class": "Qwen3VLProcessor" if qwen else "Qwen2_5_VLProcessor",
        "image_processor_type": "Qwen2VLImageProcessorFast" if qwen else "Qwen2VLImageProcessor",
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": mean, "image_std": std}, "image processor")
    optional = {"resample": 3, "rescale_factor": 1 / 255,
        "do_resize": True, "do_rescale": True, "do_normalize": True, "do_convert_rgb": True}
    _expected(config, {key: value for key, value in optional.items() if key in config}, "image processor")
    _expected(model, {"model_type": "qwen4_exp" if qwen else "mimo_v2"}, "model configuration")
    vision = model["vision_config"]
    _expected(vision, {"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2}, "vision configuration")
    if qwen:
        _expected(vision, {"in_channels": 3, "num_position_embeddings": 2304}, "vision configuration")
        _expected(config["size"], {"shortest_edge": 65536, "longest_edge": 16777216}, "resize area")
        result = dict(min_pixels=65536, max_pixels=16777216, position_side=48)
        for name in ("image_token_id", "video_token_id"):
            value = model[name]
            if type(value) is not int or not 0 <= value < 2**32:
                raise ValueError(f"invalid model {name}")
            result[name] = value
    else:
        _expected(config, {"min_pixels": 3136, "max_pixels": 12845056}, "resize area")
        _expected(model, {"image_token_id": 151655}, "model configuration")
        result = dict(min_pixels=3136, max_pixels=12845056, image_mean=mean, image_std=std)
    return {profile: result}


def prepare_frontend(snapshot, *, eos_token_ids, image_profile=None, preprocessor=None):
    """Resolve checkpoint structure during preparation; sampling is recipe-owned."""
    snapshot = Path(snapshot)
    if type(eos_token_ids) is int:
        eos_token_ids = [eos_token_ids]
    eos_token_ids = list(eos_token_ids)
    if (not eos_token_ids or len(eos_token_ids) > 8 or len(set(eos_token_ids)) != len(eos_token_ids)
            or any(type(value) is not int or not 0 <= value < 2**32 for value in eos_token_ids)):
        raise ValueError("EOS token IDs must be one through eight distinct uint32 values")
    result = {"tokenizer": tokenizer_settings(json.loads((snapshot / "tokenizer_config.json").read_text())),
              "eos_token_ids": eos_token_ids}
    if image_profile is not None:
        path = snapshot / "preprocessor_config.json" if preprocessor is None else Path(preprocessor)
        result["image"] = image_settings({} if image_profile == "deepseek_row_major" else json.loads(path.read_text()),
            json.loads((snapshot / "config.json").read_text()), image_profile)
    return result
