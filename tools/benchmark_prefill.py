#!/usr/bin/env python3
"""Exact-length, uncached native prefill/TTFT measurements on an exclusive endpoint.

New deterministic public token prompts; not historical traces. This client does
not launch a server or establish GPU allocation. Pure GPU prefill duration is
unavailable in the current native metrics and is never inferred from TTFT.

Requests stay within the /v1/completions contract: token-ID prompts with
continuous usage statistics and no token observation extensions, which the
server accepts only on /v1/chat/completions. First-output time and output-token
counts come from the SSE stream itself.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

if __package__:
    from .benchmark_mtp import percentile
    from .native import metrics, prometheus_values
else:
    from benchmark_mtp import percentile
    from native import metrics, prometheus_values

LENGTHS = (2048, 8192, 22528, 30720)
CONCURRENCIES = (1, 4)
COUNTERS = ('prefill_tokens', 'generated_tokens', 'completed', 'failed', 'reused_tokens')
REQUEST_COUNTERS = ('flare_requests_admitted_total', 'flare_requests_cancelled_total')
PREFILL_BASIS = ('successful server prefill-token counter delta / batch first-token wall window; '
                'includes queueing, prefill, first-token generation and transport; not pure GPU prefill')
TTFT_BASIS = 'client POST start to first SSE chunk carrying output; includes queueing, transport and first sampling'
OBSERVATION_BASIS = ('first observed output = first SSE chunk whose choices[0].text is non-empty or whose '
                     'continuous usage shows completion_tokens >= 1; output-token counts from per-chunk usage')


class CompletionObservations:
    """Observe a /v1/completions stream without per-token IDs or timestamps."""

    def __init__(self, output_tokens):
        self.limit = output_tokens
        self.events, self.samples, self.text, self.finish_reasons = [], [], [], []
        self.usage = self.first = self.last = None
        self.usage_count = self.first_count = self.observed = 0

    def consume(self, event, elapsed):
        self.events.append({'seconds': elapsed, 'event': event})
        if not isinstance(event, dict):
            raise ValueError('SSE data must be a JSON object')
        if 'error' in event:
            raise RuntimeError(event['error'])
        performance = event.get('performance')
        reused = (event.get('reused_prompt_tokens', 0),
                  performance.get('reused_prompt_tokens', 0) if isinstance(performance, dict) else 0)
        if any(value != 0 for value in reused):
            raise ValueError('request received prefix-cache credit')
        text = ''
        for choice in event.get('choices') or []:
            if choice.get('index', 0) != 0:
                raise ValueError('benchmark expects one completion per request')
            text += choice.get('text') or ''
            if choice.get('finish_reason') is not None:
                self.finish_reasons.append(choice['finish_reason'])
        self.text.append(text)
        usage = event.get('usage')
        if usage is not None:
            count = usage.get('completion_tokens') if isinstance(usage, dict) else None
            if type(count) is not int or count < self.usage_count:
                raise ValueError('completion usage must be a monotonic native token count')
            self.usage, self.usage_count = usage, count
        elif text:
            raise ValueError('output chunk lacks continuous usage; token counts cannot be derived')
        if text and self.usage_count < 1:
            raise ValueError('output text arrived without a counted output token')
        if self.usage_count > self.limit:
            raise ValueError('native output-token count exceeds requested workload')
        if self.first is None and (text or self.usage_count >= 1):
            self.first, self.first_count = elapsed, self.usage_count
        if self.first is not None and (self.last is None or self.usage_count > self.observed):
            self.last, self.observed = elapsed, self.usage_count
            self.samples.append([elapsed, self.usage_count])

    def result(self):
        duration = None if self.first is None or self.last <= self.first else self.last - self.first
        details = self.usage.get('prompt_tokens_details') if isinstance(self.usage, dict) else None
        return {'output_tokens': self.observed, 'first_token_seconds': self.first,
            'first_observed_tokens': self.first_count, 'last_token_seconds': self.last,
            'decode_tokens_per_second': None if duration is None else (self.observed - self.first_count) / duration,
            'usage': self.usage, 'observations': self.samples, 'observation_basis': OBSERVATION_BASIS,
            'raw_events': self.events, 'output_text': ''.join(self.text), 'finish_reasons': self.finish_reasons,
            'cached_tokens': details.get('cached_tokens') if isinstance(details, dict) else None}


def exact_prompt(tokenizer, length, case_id):
    digest=hashlib.sha256(case_id.encode()).hexdigest()
    prefix=f'{digest}\nWrite and explain a Python function that merges sorted integer lists.\nExamples:\n'
    head=tokenizer.encode(prefix,add_special_tokens=False).ids
    body=tokenizer.encode('def example(values):\n    return sorted(set(values))\n',add_special_tokens=False).ids
    if not body or len(head)>=length:
        raise ValueError('tokenizer cannot build the requested exact-length public prompt')
    ids=(head+body*((length-len(head)+len(body)-1)//len(body)))[:length]
    if len(ids)!=length or any(type(i) is not int or i<0 for i in ids):
        raise ValueError('invalid generated token prompt')
    return ids


def stream_request(base, model, prompt, case_id, output_tokens, barrier, timeout):
    body={'model':model,'prompt':prompt,'max_tokens':output_tokens,'stream':True,
          'ignore_eos':True,'seed':7035,'temperature':.7,'top_p':.95,'top_k':20,
          'stream_options':{'include_usage':True,'continuous_usage_stats':True}}
    encoded=json.dumps(body).encode()
    observed=CompletionObservations(output_tokens)
    record={'case_id':case_id,'status':'failed','requested_prompt_tokens':len(prompt),
            'prompt_token_ids':prompt,'request_bytes':len(encoded),'sampling':{'seed':7035,'temperature':.7,'top_p':.95,'top_k':20}}
    barrier.wait(timeout=timeout)
    started=time.perf_counter();record['started_monotonic_seconds']=started
    try:
        request=urllib.request.Request(base+'/v1/completions',data=encoded,
                                       headers={'Content-Type':'application/json'})
        done=False
        with urllib.request.urlopen(request,timeout=timeout) as response:
            record['http_headers_seconds']=time.perf_counter()-started
            for line in response:
                if not line.startswith(b'data:'):continue
                payload=line[5:].strip()
                if payload==b'[DONE]':done=True;break
                event=json.loads(payload)
                observed.consume(event,time.perf_counter()-started)
        usage=observed.usage or {}
        if not done:raise ValueError('stream ended without [DONE]')
        if type(usage.get('prompt_tokens')) is not int or usage['prompt_tokens']!=len(prompt):
            raise ValueError('server-reported prompt tokens do not match exact target')
        details=usage.get('prompt_tokens_details')
        cached=details.get('cached_tokens') if isinstance(details,dict) else None
        if type(cached) is not int or cached!=0:
            raise ValueError('request received prefix-cache credit or reported no cache count')
        if usage.get('completion_tokens')!=output_tokens or observed.observed!=output_tokens:
            raise ValueError('incomplete or inconsistent native output-token budget')
        if observed.first is None:
            raise ValueError('no SSE chunk carried output')
        record.update(status='complete',cached_tokens=cached,cache_proof='request usage plus required batch counter proof')
    except Exception as error:
        record['error']={'type':type(error).__name__,'message':str(error)}
    finally:
        record.update(observed.result())
        record['transport_ttft_seconds']=observed.first
        record['elapsed_seconds']=time.perf_counter()-started
        record['ended_monotonic_seconds']=time.perf_counter()
    return record


def checked_snapshot(result):
    if result['errors']:raise ValueError('native metric snapshots unavailable: '+str(result['errors']))
    value=result['json']
    for name in ('active','pending',*COUNTERS):
        if type(value.get(name)) is not int or value[name]<0:
            raise ValueError('missing or invalid native metric: '+name)
    if value['active'] or value['pending']:raise ValueError('endpoint is not idle; other traffic excluded')
    counters=prometheus_values(result.get('prometheus',''))
    for name in REQUEST_COUNTERS:
        if type(counters.get(name)) is not int or counters[name]<0:
            raise ValueError('missing or invalid native metric: '+name)
    return result


def validate_batch(batch):
    before,after=batch['metrics_before']['json'],batch['metrics_after']['json']
    delta={name:after[name]-before[name] for name in COUNTERS}
    batch['metric_deltas']=delta
    expected={'completed':batch['concurrency'],'prefill_tokens':batch['prompt_length']*batch['concurrency'],
              'generated_tokens':batch['output_tokens']*batch['concurrency'],'failed':0,'reused_tokens':0}
    if delta!=expected:
        raise ValueError('metric counters show cache reuse, other traffic, reset, or incomplete work: '+str(delta))
    before_requests=prometheus_values(batch['metrics_before']['prometheus'])
    after_requests=prometheus_values(batch['metrics_after']['prometheus'])
    request_delta={name:after_requests[name]-before_requests[name] for name in REQUEST_COUNTERS}
    batch['request_metric_deltas']=request_delta
    if request_delta!={'flare_requests_admitted_total':batch['concurrency'], 'flare_requests_cancelled_total':0}:
        raise ValueError('admission/cancellation counters show other traffic or reset: '+str(request_delta))
    if any(r['status']!='complete' for r in batch['requests']):
        raise ValueError('one or more exact-length requests failed')
    first=min(r['started_monotonic_seconds'] for r in batch['requests'])
    last=max(r['started_monotonic_seconds']+r['transport_ttft_seconds'] for r in batch['requests'])
    if last<=first:raise ValueError('invalid batch first-token window')
    batch.update(first_token_wall_window_seconds=last-first,
                 effective_prefill_tokens_per_second=delta['prefill_tokens']/(last-first),
                 prefill_rate_basis=PREFILL_BASIS,pure_gpu_prefill_tokens_per_second=None,
                 pure_gpu_prefill_unavailable_reason='native release exposes prefill token counters but no isolated prefill GPU duration',
                 status='complete')


def run_batch(base, model, tokenizer, *, length, concurrency, repetition, phase, run_id, output_tokens, timeout):
    batch={'status':'running','phase':phase,'prompt_length':length,'concurrency':concurrency,
           'repetition':repetition,'output_tokens':output_tokens,'requests':[]}
    try:
        inputs=[]
        for slot in range(concurrency):
            case=f'{run_id}/{phase}/{length}/{concurrency}/{repetition}/{slot}'
            inputs.append((case,exact_prompt(tokenizer,length,case)))
        if len({tuple(ids[:128]) for _,ids in inputs})!=concurrency:
            raise ValueError('early prompt tokens collide; cache isolation is not established')
        batch['metrics_before']=metrics(base)
        checked_snapshot(batch['metrics_before'])
        barrier=threading.Barrier(concurrency)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures=[pool.submit(stream_request,base,model,ids,case,output_tokens,barrier,timeout)
                     for case,ids in inputs]
            batch['requests']=[future.result() for future in futures]
        batch['metrics_after']=metrics(base)
        checked_snapshot(batch['metrics_after'])
        validate_batch(batch)
    except Exception as error:
        batch.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
    return batch


def summarize(batches):
    result=[]
    for length in LENGTHS:
        for concurrency in CONCURRENCIES:
            rows=[b for b in batches if b['phase']=='measured' and b['prompt_length']==length and b['concurrency']==concurrency]
            if not rows:continue
            requests=[r for b in rows for r in b['requests']]
            good=[r for b in rows if b['status']=='complete' for r in b['requests'] if r['status']=='complete']
            ttft=[r['transport_ttft_seconds'] for r in good]
            rates=[b['effective_prefill_tokens_per_second'] for b in rows if b['status']=='complete']
            decode=[r['decode_tokens_per_second'] for r in good if r['decode_tokens_per_second'] is not None]
            result.append({'prompt_tokens':length,'concurrency':concurrency,'measured_batches':len(rows),
                'successful_requests':len(good),'failed_batches':sum(b['status']!='complete' for b in rows),
                'transport_ttft_seconds':{'p50':percentile(ttft,.5),'p95':percentile(ttft,.95)},
                'effective_prefill_tokens_per_second_median':percentile(rates,.5),'prefill_rate_basis':PREFILL_BASIS,
                'pure_gpu_prefill_tokens_per_second':None,
                'median_request_decode_tokens_per_second':percentile(decode,.5),
                'decode_rate_unavailable_requests':len(good)-len(decode)})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--tokenizer',type=Path,required=True,help='tokenizer.json from the exact served checkpoint')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',required=True,help='new public case namespace for this run; record it for reproduction')
    parser.add_argument('--exclusive-endpoint',action='store_true',help='confirm no other clients use this native server')
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--output-tokens',type=int,default=32)
    parser.add_argument('--timeout',type=float,default=1800)
    args=parser.parse_args()
    if not args.exclusive_endpoint or args.warmups<1 or args.repetitions<3 or args.output_tokens<2 or args.timeout<=0:
        parser.error('exclusive endpoint, >=1 separate warmup, >=3 measured repetitions, >=2 output tokens and positive timeout required')
    if args.output.exists():parser.error('output already exists; preserve previous measurements')
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    report={'status':'running','input_set':'public-exact-prefill-v1-new-baseline','options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'tokenizer_sha256':hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
            'transport_ttft_basis':TTFT_BASIS,'observation_basis':OBSERVATION_BASIS,
            'request_contract':'/v1/completions token-ID prompts; no token observation extensions',
            'batches':[],'summary':[],'pure_gpu_prefill_available':False}
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with args.output.open('x'):pass
    try:
        for length in LENGTHS:
            for concurrency in CONCURRENCIES:
                for phase,count in [('warmup',args.warmups),('measured',args.repetitions)]:
                    for repetition in range(count):
                        batch=run_batch(args.base_url.rstrip('/'),args.model,tokenizer,length=length,
                              concurrency=concurrency,repetition=repetition,phase=phase,run_id=args.run_id,
                              output_tokens=args.output_tokens,timeout=args.timeout)
                        report['batches'].append(batch);report['summary']=summarize(report['batches']);save()
                        if batch['status']!='complete':raise RuntimeError('prefill measurement failed; see saved raw batch')
        report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:save()


if __name__=='__main__':main()
