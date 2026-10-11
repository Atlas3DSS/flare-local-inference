"""Preparation policy for the Qwen 3.8 operator catalog."""

from flare.preparation.tuning import RecordedPreparationSession


def preparation_session(*, device, checkpoint, namespace=None, coordinator=None):
    return RecordedPreparationSession(
        device=device, checkpoint=checkpoint, namespace=namespace, coordinator=coordinator,
        workload_schema="qwen38-active-operators-v1",
        numerical_policy="qsa16-gdn-sequential-k1-dense-mxfp8-quantized-vision-nvfp4-a16-moe-a4-route-fp32",
        search_limits={
            "bf16_projection": "all eligible b12x backends and rows_per_tile configurations",
            "mxfp8_projection": "fixed quantized wrapper with default inner dense lowering",
            "qsa": "fixed attention_splits=16 and singleton kernel configuration",
            "gdn_prefill": "sequential k_split=1; v_split, stages, and window_tiles remain tunable",
            "operator_scratch": "one arena sized to the complete candidate envelope",
        },
    )
