"""Preparation policy for the Qwen 3.8 operator catalog."""

from ignition.preparation.tuning import RecordedPreparationSession


def preparation_session(*, device, checkpoint, namespace=None, coordinator=None, headroom_bytes=0):
    return RecordedPreparationSession(
        device=device, checkpoint=checkpoint, namespace=namespace, coordinator=coordinator,
        headroom_bytes=headroom_bytes,
        workload_schema="qwen38-active-operators-v1",
        numerical_policy="qsa16-gdn-sequential-k1-dense-mxfp8-auto-vision-nvfp4-a16-moe-a4-route-fp32",
        search_limits={
            "bf16_projection": "all eligible b12x backends and rows_per_tile configurations",
            "mxfp8_projection": "BF16 or quantized activations with eligible b12x projection configurations",
            "qsa": "fixed attention_splits=16 and singleton kernel configuration",
            "gdn_prefill": "sequential k_split=1; v_split, stages, and window_tiles remain tunable",
            "operator_scratch": "one arena sized to the complete candidate envelope",
        },
    )
