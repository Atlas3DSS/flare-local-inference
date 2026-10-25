#!/usr/bin/env python3
"""Prepare, strictly qualify and optionally export the SM70 MTP catalog (replay-commit)."""
# Release replay-commit validation and numerical helpers are colocated in tools/.
# Paths, input fixtures and signing credentials are explicit; serving stays native.
import argparse
import hashlib
from contextlib import nullcontext
import json
from pathlib import Path
import os
import sys
import time
import traceback


# --skip-reference-gates skips only the gate that reads the FP32 reference*.safetensors files: the
# full-vocabulary accuracy (KL/top-1) gate, tools/accuracy.py. Every other gate runs and must pass.
REFERENCE_GATE_INPUTS = ('measure_accuracy', 'short_reference', 'long_reference')
SKIPPED_REFERENCE_GATES = {'reference_gates': 'skipped', 'reason': '--skip-reference-gates'}


def skips_reference_gates(args):
    # The option is absent from the parsed arguments unless given, so default reports keep their keys.
    return getattr(args, 'skip_reference_gates', False) is True


def qualification(args):
    """The skipped-gate record for the report and the signed package, or None when every gate runs."""
    return dict(SKIPPED_REFERENCE_GATES) if skips_reference_gates(args) else None


def export_qualification(args):
    """Publisher fields that stamp the signed package; none unless reference gates are skipped."""
    if not skips_reference_gates(args):
        return {}
    from ignition.preparation.protection import Qualification
    return {'qualification': Qualification(**SKIPPED_REFERENCE_GATES)}


def require_export_gates(args):
    if not args.output:
        return
    if skips_reference_gates(args):
        if not all((args.validate, args.check_invariance, args.check_commit_growth, args.check_consumers,
                    args.measure_components, args.measure, args.signing_dir)):
            raise ValueError('export with --skip-reference-gates still requires every strict/invariance/commit/consumer/component/tau gate and signing inputs')
    elif not all((args.validate, args.check_invariance, args.check_commit_growth,
                  args.check_consumers, args.measure_components, args.measure_accuracy,
                  args.measure, args.signing_dir, args.short_reference, args.long_reference)):
        raise ValueError('export requires every strict/invariance/commit/consumer/component/accuracy/tau gate and signing inputs')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True, help='existing directory for flat validation outputs')
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--kernel-cache', type=Path, required=True)
    parser.add_argument('--signing-dir', type=Path, help='external development-signing key directory; required for export')
    parser.add_argument('--short-reference', type=Path)
    parser.add_argument('--long-reference', type=Path)
    parser.add_argument('--cutlass-include', type=Path,
        help='CUTLASS include directory required by reviewed SM70 prefill')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--legacy-dense', action='store_true', help='diagnostic cuBLAS A/B; cannot export')
    parser.add_argument('--depth', type=int, choices=(2, 3, 4), default=4)
    parser.add_argument('--lease-gib', type=int, default=30)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--chunk', type=int, default=128)
    parser.add_argument('--capture-activations', action='store_true',
        help='retain per-layer prefill observations for diagnostics; cannot export or measure')
    parser.add_argument('--steering-max-sites', type=int, default=0,
        help='prepare additive mlp_out/residual inputs with this per-request site limit; zero disables')
    parser.add_argument('--compile-workers', type=int, required=True)
    parser.add_argument('--compiler-context-mib', type=int, default=400,
        help='measured SM70 metadata-only forward compiler context budget')
    parser.add_argument('--pages', type=int, default=86)
    parser.add_argument('--measure', type=Path)
    parser.add_argument('--measure-batches', type=int, nargs='+', choices=(1, 4), default=[1])
    parser.add_argument('--measure-prompt-ids', nargs='+')
    parser.add_argument('--measure-components', action='store_true')
    parser.add_argument('--measure-accuracy', action='store_true',
        help='full-vocabulary KL/top1 on the published short and long references')
    parser.add_argument('--skip-reference-gates', action='store_true', default=argparse.SUPPRESS,
        help='skip only the accuracy gate, which needs the FP32 reference files; every other gate still '
             'runs, and the report and signed package record the skip')
    parser.add_argument('--check-consumers', action='store_true')
    parser.add_argument('--check-invariance', action='store_true',
        help='512-token ordinary/speculative and solo/co-scheduled greedy checks')
    parser.add_argument('--check-commit-growth', action='store_true',
        help='full-target recurrent commit after logical backing-capacity growth')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--diagnose-rows', action='store_true',
        help='diagnose cross-row arithmetic with the cached production graphs; cannot export')
    parser.add_argument('--diagnose-long', action='store_true',
        help='trace long coding-prompt layer/norm arithmetic at B1/B4; cannot export or measure')
    parser.add_argument('--diagnose-long-widths', type=int, nargs='+', choices=(4, 5),
        help='target verification widths for --diagnose-long; default: available widths 4 and 5')
    parser.add_argument('--diagnostic-uncompiled', action='store_true',
        help='capture prepared eager operators for numerical diagnosis; never export this run')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if skips_reference_gates(args) and any(getattr(args, name) for name in REFERENCE_GATE_INPUTS):
        parser.error('--skip-reference-gates skips the accuracy gate; omit --measure-accuracy, '
                     '--short-reference and --long-reference')
    try: require_export_gates(args)
    except ValueError as error: parser.error(str(error))
    if args.legacy_dense and args.output:
        parser.error('legacy dense A/B is diagnostic and cannot export')
    if (args.diagnostic_uncompiled or args.diagnose_rows or args.diagnose_long or args.capture_activations) and (
            args.output or args.measure or args.measure_components):
        parser.error('arithmetic diagnostic modes cannot export or produce performance measurements')
    if args.diagnose_long_widths and not args.diagnose_long:
        parser.error('--diagnose-long-widths requires --diagnose-long')
    if not args.root.is_dir():
        parser.error('--root must be an existing output directory')
    if args.output and not (args.signing_dir and args.measure):
        parser.error('export requires --signing-dir and --measure coding-token/tau qualification')
    if args.measure_accuracy and not (args.short_reference and args.long_reference):
        parser.error('--measure-accuracy requires both --short-reference and --long-reference')
    if args.output and args.output.exists():
        parser.error('refusing to overwrite an existing package; choose a new output')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from cuda.pathfinder import load_nvidia_dynamic_lib
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.package_storage import PackageStorage
    from ignition.recipes.qwen38_dense.bootstrap import PreparationConfig, prepare
    from ignition.preparation.compilation import _OPTIONS
    from ignition.preparation import compile_pool
    if args.legacy_dense:
        from ignition.components import sm70_dense
        sm70_dense.tactic = lambda query: None

    def progress(message):
        print(time.strftime('%H:%M:%S'), message, flush=True)

    result = {'status': 'started', 'args': vars(args), 'torch': torch.__version__}
    if qualification(args) is not None:
        result['qualification'] = qualification(args)
    start = time.monotonic()
    try:
        torch.cuda.set_device(0)
        if torch.cuda.get_device_capability() != (7, 0):
            raise RuntimeError('qualification requires sm_70')
        # Spend the CPU budget within the number of compiler CUDA contexts
        # that actually fits the lease, rather than spawning 86 GPU contexts.
        _OPTIONS['compile_threads'] = args.compile_workers
        # All 23 full dense forwards used 368 MiB per worker. Retain 32 MiB
        # per-worker margin plus the separate 256 MiB lease headroom below.
        compile_pool._DISCRETE_WORKER_DEVICE_BYTES = args.compiler_context_mib << 20
        total_memory = torch.cuda.get_device_properties(0).total_memory
        lease_headroom = max(0, total_memory - (args.lease_gib << 30)) + (256 << 20)
        config = PreparationConfig(context=args.context, max_active=args.concurrency,
            prefill_chunk_size=args.chunk, compile_workers=args.compile_workers,
            snapshot_budget_bytes=172 << 20, headroom_bytes=lease_headroom,
            preparation_pages=args.pages, steering_max_sites=args.steering_max_sites,
            capture_activations=args.capture_activations,
            compile_decoder=not args.diagnostic_uncompiled,
            mtp=True, speculative_widths=tuple(range(1, args.depth + 2)),
            retain_graphs=True, kernel_cache=args.kernel_cache, cutlass_include=args.cutlass_include)
        recorder = ModuleImages(load_nvidia_dynamic_lib('cupti').abs_path) if args.output else None
        with recorder if recorder is not None else nullcontext():
            storage = PackageStorage(device=0)
            with storage:
                prepared = prepare(args.checkpoint, config, progress=progress, package_storage=storage)
        from ignition.components.sm70_projection_tc import ADMISSIONS
        result.update(status='prepared', memory=prepared.memory_report, projection_admissions=ADMISSIONS,
            prepared_identity=prepared.server_config['identity'],
            steering=prepared.server_config['model_contract'].get('steering'),
            preparation_free_bytes=torch.cuda.mem_get_info()[0], preparation_total_bytes=torch.cuda.mem_get_info()[1],
            default_draft='checkpoint-selected; no checkpoint overlay applied by this harness')
        with prepared:
            if args.diagnose_long:
                from validate_replay import diagnose_long_sequence
                result['long_diagnostics'] = diagnose_long_sequence(
                    prepared, args.root, checkpoint=args.checkpoint, prompts_path=args.prompts,
                    progress=progress, widths=args.diagnose_long_widths)
                if result['long_diagnostics']['status'] != 'complete':
                    raise RuntimeError('long arithmetic diagnosis failed; see report')
            if args.measure_accuracy:
                from accuracy import measure as measure_accuracy
                result['accuracy'] = measure_accuracy(prepared, args.root, short_reference=args.short_reference,
                    long_reference=args.long_reference, progress=progress)
            if args.diagnose_rows:
                from tokenizers import Tokenizer
                from validate_replay import _Harness, PROMPTS
                with torch.inference_mode():
                    harness = _Harness(prepared)
                    tokenizer = Tokenizer.from_file(prepared.server_config['assets']['tokenizer']['path'])
                    prompts = [tokenizer.encode(text, add_special_tokens=False).ids for text in PROMPTS]
                    try:
                        result['diagnostics'] = {str(batch): harness.trace_first_divergence(prompts[:batch])
                                                 for batch in (1, 4)}
                    finally:
                        result['cleanup'] = harness.cleanup()
            if args.validate:
                from validate_replay import validate
                result['validation'] = validate(prepared, args.root, progress=progress)
                if not result['validation']['passed']:
                    raise RuntimeError('strict MTP validation failed; package not exported')
            if args.check_invariance:
                from validate_replay import validate_long_sequence
                result['long_sequence'] = validate_long_sequence(prepared, args.root, checkpoint=args.checkpoint,
                    prompts_path=args.prompts, progress=progress)
                if not result['long_sequence']['passed']:
                    raise RuntimeError('long-sequence invariance failed; package not exported')
            if args.check_commit_growth:
                from validate_replay import validate_commit_capacity_growth
                result['commit_capacity_growth'] = validate_commit_capacity_growth(
                    prepared, args.root, progress=progress)
                if not result['commit_capacity_growth']['passed']:
                    raise RuntimeError('commit capacity growth failed; package not exported')
            if args.measure:
                from measure_mtp import measure
                previous_measurement = (json.loads(args.measure.read_text())
                                        if args.measure.exists() else None)
                result['measurement'] = measure(prepared, args.root, checkpoint=args.checkpoint, prompts_path=args.prompts, depths=tuple(range(3, args.depth + 1)),
                    batches=tuple(args.measure_batches), prompt_ids=args.measure_prompt_ids,
                    qualification=result.get('validation'),
                    output=None if previous_measurement is not None else args.measure, progress=progress)
                if result['measurement']['status'] != 'complete':
                    raise RuntimeError('MTP measurement failed; package not exported')
                if previous_measurement is not None:
                    current = result['measurement']
                    fields = ('depth', 'batch', 'repetition', 'prompt_ids', 'output_token_ids')
                    golden = lambda report: [[run[key] for key in fields] for run in report['runs']]
                    if (previous_measurement.get('status') != 'complete'
                            or previous_measurement['prompt_file_sha256'] != current['prompt_file_sha256']
                            or previous_measurement['summary'] != current['summary']
                            or golden(previous_measurement) != golden(current)):
                        raise RuntimeError('rebuilt package changed the fixed coding-token/tau goldens')
                    current['prior_measurement_identity'] = True
                    args.measure.write_text(json.dumps(current, indent=2, allow_nan=False) + '\n')
            if args.check_consumers:
                from check_consumers import check
                result['consumer_identity'] = check(prepared, args.root, progress=progress)
                if not result['consumer_identity']['passed']:
                    raise RuntimeError('consumer identity failed; package not exported')
            if args.measure_components:
                from measure_components import measure as measure_components
                result['components'] = measure_components(prepared, args.root, checkpoint=args.checkpoint, prompts_path=args.prompts,
                    qualification=result.get('validation'), depths=tuple(range(3, args.depth + 1)),
                    batches=tuple(args.measure_batches),
                    prompt_ids=args.measure_prompt_ids or ('code-00', 'code-01', 'code-02', 'code-03'),
                    progress=progress)
                if result['components']['status'] != 'complete':
                    raise RuntimeError('component measurement failed; package not exported')
            if args.output:
                if not args.validate:
                    raise ValueError('export requires strict validation')
                if not args.check_invariance:
                    raise ValueError('export requires the SPEC addendum long-sequence invariance checks')
                if not (args.check_consumers and args.check_commit_growth and args.measure_components):
                    raise ValueError('export requires the reviewed consumer, commit growth and component gates')
                if not skips_reference_gates(args) and (
                        not args.measure_accuracy or result.get('accuracy', {}).get('status') != 'complete'):
                    raise ValueError('export requires the SPEC addendum full-vocabulary accuracy measurements')
                from dataclasses import replace
                from types import SimpleNamespace
                from ignition.development import publisher
                from ignition.preparation.model_card import model_license
                from ignition.preparation.model_package import export_package
                signer = replace(publisher(SimpleNamespace(path=args.signing_dir / 'config.toml')),
                    license=model_license(args.checkpoint, 'nvidia/Qwen3.8-27B-NVFP4',
                                          '482ca0f3832238542f8f5295dde86b5f22711d80'),
                    **export_qualification(args))
                export_package(prepared, recorder, storage, args.output, options={}, publisher=signer,
                               progress=progress, staging=args.root)
                with args.output.open('rb') as primary:
                    primary_sha256 = hashlib.file_digest(primary, 'sha256').hexdigest()
                result.update(status='exported', package=str(args.output),
                    package_primary_sha256=primary_sha256, fingerprint=signer.fingerprint.hex())
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    except BaseException as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        raise
    finally:
        from ignition.components.sm70_projection_tc import ADMISSIONS
        result['projection_admissions'] = ADMISSIONS
        result['elapsed_seconds'] = time.monotonic() - start
        args.report.write_text(json.dumps(result, indent=2, default=str) + '\n')


if __name__ == '__main__':
    main()
