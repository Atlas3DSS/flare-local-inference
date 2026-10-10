"""Compare complete greedy token continuations against vLLM request variation."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from .provenance import file_record, write_json


def common_prefix(actual, reference):
    if len(actual) != 64 or len(reference) != 64:
        raise ValueError('agreement requires exactly 64 observed tokens per continuation')
    for tokens in (actual, reference):
        if any(type(token) is not int or not 0 <= token < 248320 for token in tokens):
            raise ValueError('continuation contains an invalid token ID')
    return next((index for index, (left, right) in enumerate(zip(actual, reference, strict=True))
                 if left != right), 64)


def compare_cases(prompts, actual, reference):
    expected = {prompt['id'] for prompt in prompts}
    if set(actual) != expected or set(reference) != expected or len(expected) != len(prompts):
        raise ValueError('observations must cover every distinct corpus prompt exactly once')
    cases = []
    for prompt in prompts:
        a, r = actual[prompt['id']], reference[prompt['id']]
        for observation in (a, r):
            if (observation['prompt_tokens'] != prompt['prompt_tokens']
                    or observation['category'] != prompt['category']):
                raise ValueError(f'{prompt["id"]}: observed context or category differs from corpus')
            if observation['finish_reason'] != 'length':
                raise ValueError(f'{prompt["id"]}: generation did not reach the fixed output length')
        prefix = common_prefix(a['generated_tokens'], r['generated_tokens'])
        cases.append({'id': prompt['id'], 'category': prompt['category'],
            'prompt_token_count': len(prompt['prompt_tokens']), 'agreement_length': prefix,
            'first_divergence_position': None if prefix == 64 else prefix,
            'actual_token_at_divergence': None if prefix == 64 else a['generated_tokens'][prefix],
            'reference_token_at_divergence': None if prefix == 64 else r['generated_tokens'][prefix]})

    def summary(values):
        lengths = [entry['agreement_length'] for entry in values]
        return {'prompts': len(values), 'median_agreement_length': statistics.median(lengths),
                'mean_agreement_length': statistics.mean(lengths),
                'full_64_token_agreement_count': lengths.count(64),
                'full_64_token_agreement_fraction': lengths.count(64) / len(lengths),
                'agreement_length_histogram': dict(sorted(Counter(lengths).items()))}

    return {'summary': summary(cases), 'categories': {
        category: summary([entry for entry in cases if entry['category'] == category])
        for category in sorted({entry['category'] for entry in cases})}, 'cases': cases}


def load_observations(path, *, repetitions):
    path = Path(path)
    receipt = json.loads((path.parent / 'capture.json').read_text())
    if receipt['observations']['sha256'] != file_record(path)['sha256']:
        raise ValueError('observation file differs from its completed capture receipt')
    result = {index: {} for index in range(repetitions)}
    count = 0
    for line in path.read_text().splitlines():
        observation = json.loads(line)
        repetition = observation['repetition']
        if type(repetition) is not int or repetition not in result:
            raise ValueError('observation repetition differs from the declared experiment')
        if observation['id'] in result[repetition]:
            raise ValueError('duplicate observation for a prompt and repetition')
        if observation.get('prefix_cache_reset') is not True:
            raise ValueError('observation does not confirm cold prompt execution')
        result[repetition][observation['id']] = observation
        count += 1
    if receipt['completed_observations'] != count or receipt['repetitions'] != repetitions:
        raise ValueError('capture observation count or repetition contract differs')
    if receipt['generation_contract'] != {
            'tokens': 64, 'temperature': 0, 'ignore_eos': True,
            'teacher_forcing': False, 'prefix_reuse': False}:
        raise ValueError('capture generation contract differs')
    return result, receipt


def main():
    from .generation_corpus import load_corpus

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--flare', type=Path, required=True)
    parser.add_argument('--vllm', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    corpus = load_corpus(args.corpus)
    actual, actual_receipt = load_observations(args.flare, repetitions=1)
    reference, reference_receipt = load_observations(args.vllm, repetitions=2)
    for path, receipt in ((args.flare, actual_receipt), (args.vllm, reference_receipt)):
        captured = load_corpus(path.parent / 'corpus.json', expected_sha256=receipt['corpus']['sha256'])
        if captured != corpus:
            raise ValueError('captured corpus differs from comparison corpus')
    pairs = {'vllm_repeat': compare_cases(corpus['prompts'], reference[1], reference[0]),
             'flare_vs_vllm_0': compare_cases(corpus['prompts'], actual[0], reference[0]),
             'flare_vs_vllm_1': compare_cases(corpus['prompts'], actual[0], reference[1])}
    floor = pairs['vllm_repeat']['summary']
    measurements = ('median_agreement_length', 'full_64_token_agreement_fraction')
    write_json(args.output, {'schema_version': 1, 'corpus': file_record(args.corpus),
        'sources': {name: {'observations': file_record(path), 'capture': file_record(path.parent / 'capture.json')}
                    for name, path in (('flare', args.flare), ('vllm', args.vllm))},
        'metric': 'Number of identical leading raw generated token IDs, from 0 through 64; larger is better.',
        'eos_policy': 'Exactly 64 generated tokens, including tokens produced after EOS; no teacher forcing.',
        'pairs': pairs,
        'observed_reference_floor': {name: floor[name] for name in measurements},
        'at_least_observed_reference_agreement': {
            pair: {name: pairs[pair]['summary'][name] >= floor[name] for name in measurements}
            for pair in ('flare_vs_vllm_0', 'flare_vs_vllm_1')},
        'scope': 'Fixed prompt continuation agreement. Reference floor comes from two cold requests per prompt; task success is not measured.'})


if __name__ == '__main__':
    main()
