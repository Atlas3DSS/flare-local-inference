"""Full-vocabulary comparisons on the published short and long token contexts."""

import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from validate_replay import _LongSequenceHarness


class _AccuracyHarness(_LongSequenceHarness):
    def __init__(self, prepared, chunk):
        self.chunk = chunk
        super().__init__(prepared)

    def name(self, kind, batch, mtp, *, width=None):
        if kind != 'prefill':
            return super().name(kind, batch, mtp, width=width)
        names = [name for name, signature in self.prepared.signatures.items()
                 if signature.kind == kind and signature.batch_capacity == 1
                 and signature.speculation_enabled == mtp and signature.token_capacity == 128]
        if len(names) != 1:
            raise ValueError(f'accuracy needs a retained prefill{self.chunk} graph')
        return names[0]


def measure(prepared, root, progress=print, *, short_reference, long_reference):
    root = Path(root)
    from validate_candidate import MODEL_ID, REVISION, _reference, _sha256, distribution_metrics

    started = time.monotonic()
    report = {'status': 'started', 'native_runtime_tested': False, 'model_id': MODEL_ID,
              'revision': REVISION, 'references': {}, 'numerical_pass': None,
              'note': 'Measured full-vocabulary KL(reference || candidate), FP64 CPU reduction; no invented pass threshold.'}
    artifacts = {}
    try:
        for label, path, chunk in [('short', Path(short_reference), 1),
                                   ('long', Path(long_reference), 128)]:
            with safe_open(path, framework='pt', device='cpu') as source:
                metadata = source.metadata()
            if metadata.get('model_id') != MODEL_ID or metadata.get('revision') != REVISION:
                raise ValueError('reference checkpoint identity differs')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            provenance = json.loads(path.with_suffix('.json').read_text())
            if provenance.get('model_id') != MODEL_ID or provenance.get('revision') != REVISION:
                raise ValueError('reference JSON checkpoint identity differs')
            checkpoint = Path(prepared.server_config['assets']['tokenizer']['path']).parent
            for filename, field in [('config.json', 'config_sha256'),
                                    ('model.safetensors.index.json', 'index_sha256')]:
                if _sha256(checkpoint / filename) != provenance[field]:
                    raise ValueError('prepared checkpoint differs from reference provenance: ' + filename)
            if provenance.get('artifact_sha256', digest) != digest:
                raise ValueError('reference artifact differs from its provenance digest')
            if label == 'short':
                rows, _ = _reference(path)
                cases = [(prefix, torch.cat((ids, generated[:-1])),
                          torch.arange(ids.numel() + generated.numel() - 1),
                          torch.cat((prefill, decode)))
                         for prefix, ids, generated, prefill, decode in rows]
            else:
                tensors = load_file(path, device='cpu')
                prefixes = sorted({key.split('.')[0] for key in tensors})
                cases = [(prefix, tensors[prefix + '.input_ids'], tensors[prefix + '.positions'],
                          tensors[prefix + '.logits']) for prefix in prefixes]
                for (_, ids, positions, _), item in zip(cases, provenance['prompts'], strict=True):
                    if ids.tolist() != item['prompt_ids'] + item['generated_ids'] or positions.tolist() != item['saved_positions']:
                        raise ValueError('long reference JSON contexts/positions disagree with artifact')
            result = {'reference': str(path), 'reference_sha256': digest, 'metadata': metadata,
                      'execution': ('one live token per retained prefill128 replay at every position, matching Astra-1 sequential mode'
                                    if chunk == 1 else 'retained 128-token prefill to first measured position, then ordinary decode on published tokens'),
                      'prompts': []}
            report['references'][label] = result
            all_reference, all_candidate = [], []
            with torch.inference_mode():
                harness = _AccuracyHarness(prepared, chunk)
                harness.seeds = [17]
                try:
                    for prefix, ids, positions, reference in cases:
                        if (ids.dtype != torch.int64 or positions.dtype != torch.int64
                                or positions.ndim != 1 or len(positions) != len(reference)
                                or not bool((positions[1:] > positions[:-1]).all())
                                or int(positions[0]) < 0 or int(positions[-1]) >= len(ids)
                                or reference.shape[1] != harness.requests.vocab_size):
                            raise ValueError('reference positions/logits/context disagree')
                        start, end = int(positions[0]), int(positions[-1])
                        harness.initialize([ids[:start + 1].tolist()], mtp=False)
                        prefill_name = harness.name('prefill', 1, False)
                        active = harness.catalog[prefill_name].decoder.scratch.logits
                        wanted = set(positions.tolist())
                        collected = []
                        for position in range(start, end + 1):
                            if position > start:
                                if chunk == 1:
                                    harness.rows([0], lengths=[1])
                                    harness.inputs.token_ids[0] = ids[position]
                                    harness.replay(prefill_name, [0])
                                else:
                                    harness.decode_inputs([int(ids[position])], [0])
                                    active = harness.catalog[harness.name('decode', 1, False)].decoder.scratch.logits
                            if position in wanted:
                                collected.append(active[0].detach().cpu().float().clone())
                        candidate = torch.stack(collected)
                        metrics, per_position = distribution_metrics(reference, candidate)
                        result['prompts'].append({'id': prefix, **metrics,
                            'top1_disagreement_positions': positions[per_position['reference_top1'] != per_position['candidate_top1']].tolist()})
                        key = label + '.' + prefix
                        artifacts[key + '.input_ids'] = ids
                        artifacts[key + '.positions'] = positions
                        artifacts[key + '.candidate_logits'] = candidate
                        artifacts.update({key + '.' + name: value for name, value in per_position.items()})
                        all_reference.append(reference)
                        all_candidate.append(candidate)
                        progress(f'accuracy {label} {prefix}: KL={metrics["kld_mean"]:.8g}, top1={metrics["positions"] - metrics["top1_disagreements"]}/{metrics["positions"]}')
                    result['aggregate'], _ = distribution_metrics(torch.cat(all_reference), torch.cat(all_candidate))
                finally:
                    result['cleanup'] = harness.cleanup()
                if not result['cleanup']['passed']:
                    raise RuntimeError('accuracy measurement did not restore startup/pool bytes')
            result['status'] = 'complete'
        save_file(artifacts, root / 'accuracy.safetensors', metadata={'model_id': MODEL_ID, 'revision': REVISION})
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (root / 'accuracy.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report
