"""Exercise the real HTTP client against a CPU-only SSE/metrics fixture."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from types import SimpleNamespace
import unittest

from tools import benchmark_prefill as bench


class TokenizerFixture:
    def encode(self,text,add_special_tokens=False):
        return SimpleNamespace(ids=list(text.encode()))


class PrefillTests(unittest.TestCase):
    def setUp(self):
        self.lock=threading.Lock();self.mode='normal';self.received=[];self.max_active=0
        self.counters=dict(active=0,pending=0,prefill_tokens=0,generated_tokens=0,completed=0,failed=0,reused_tokens=0,
                           flare_requests_admitted_total=0,flare_requests_cancelled_total=0)
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                with owner.lock:data=dict(owner.counters)
                raw=json.dumps(data).encode() if 'format=json' in self.path else ''.join(
                    f'{name} {data[name]}\n' for name in bench.REQUEST_COUNTERS
                    if not (owner.mode=='missing_counter' and name=='flare_requests_cancelled_total')).encode()
                self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.assert_path=self.path
                if self.path!='/v1/completions':self.send_error(404);return
                count=len(body['prompt']);output=body['max_tokens']
                with owner.lock:
                    owner.received.append(body);owner.counters['active']+=1
                    owner.counters['flare_requests_admitted_total']+=1
                    if owner.mode=='foreign_cancelled':
                        owner.counters['flare_requests_admitted_total']+=1
                        owner.counters['flare_requests_cancelled_total']+=1
                    owner.max_active=max(owner.max_active,owner.counters['active'])
                    owner.counters['prefill_tokens']+=count
                    if owner.mode=='metric_cache':owner.counters['reused_tokens']+=1
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                try:
                    for i in range(output):
                        time.sleep(.025)
                        event={'choices':[{'index':0,'token_ids':[100+i],'text':'x'}],
                               'usage':{'prompt_tokens':count+(owner.mode=='wrong_length'),
                                        'completion_tokens':i+1,'prompt_tokens_details':{'cached_tokens':int(owner.mode=='cache')}}}
                        if owner.mode=='top_level_cache':event['reused_prompt_tokens']=1
                        self.wfile.write(b'data: '+json.dumps(event).encode()+b'\n\n');self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
                finally:
                    with owner.lock:
                        owner.counters['active']-=1;owner.counters['generated_tokens']+=output
                        owner.counters['completed']+=1+int(owner.mode=='foreign')
                try:self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                except (BrokenPipeError,ConnectionResetError):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
        self.addCleanup(self.server.server_close);self.addCleanup(self.server.shutdown)

    def run_batch(self,**kwargs):
        return bench.run_batch(self.base,'cpu-fixture',TokenizerFixture(),length=kwargs.get('length',2048),
                    concurrency=kwargs.get('concurrency',1),repetition=0,phase=kwargs.get('phase','measured'),
                    run_id='public-cpu-regression',output_tokens=2,timeout=5)

    def test_all_exact_lengths_at_both_concurrencies_with_native_usage_proof(self):
        batches=[]
        for length in bench.LENGTHS:
            for concurrency in bench.CONCURRENCIES:
                batch=self.run_batch(length=length,concurrency=concurrency);batches.append(batch)
                self.assertEqual(batch['status'],'complete',batch.get('error'))
                self.assertEqual(batch['metric_deltas']['prefill_tokens'],length*concurrency)
                self.assertTrue(all(r['usage']['prompt_tokens']==length for r in batch['requests']))
                self.assertTrue(all(r['transport_ttft_seconds']>0 for r in batch['requests']))
                self.assertTrue(all(r['decode_tokens_per_second']>0 for r in batch['requests']))
                self.assertIsNone(batch['pure_gpu_prefill_tokens_per_second'])
                self.assertIn('not pure GPU',batch['prefill_rate_basis'])
        self.assertGreaterEqual(self.max_active,2)
        self.assertEqual(len(bench.summarize(batches)),8)

    def test_wrong_server_prompt_length_fails_and_preserves_raw_events(self):
        self.mode='wrong_length';batch=self.run_batch()
        self.assertEqual(batch['status'],'failed')
        request=batch['requests'][0]
        self.assertIn('exact target',request['error']['message']);self.assertTrue(request['raw_events'])
        self.assertEqual(request['usage']['prompt_tokens'],2049)

    def test_cached_usage_or_metric_cache_credit_fails(self):
        for mode in ('cache','metric_cache','top_level_cache'):
            self.mode=mode
            batch=self.run_batch()
            self.assertEqual(batch['status'],'failed',mode)
            self.assertIn('metrics_before',batch);self.assertIn('metrics_after',batch)
            if mode=='top_level_cache':
                self.assertTrue(batch['requests'][0]['raw_events'])

    def test_other_traffic_counter_mismatch_fails(self):
        self.mode='foreign';batch=self.run_batch()
        self.assertEqual(batch['status'],'failed')
        self.assertIn('other traffic',batch['error']['message'])
        self.assertEqual(batch['metric_deltas']['completed'],2)
        summary=bench.summarize([batch])[0]
        self.assertEqual(summary['successful_requests'],0)
        self.assertIsNone(summary['transport_ttft_seconds']['p50'])

    def test_foreign_admission_cancelled_before_prefill_fails(self):
        self.mode='foreign_cancelled';batch=self.run_batch()
        self.assertEqual(batch['status'],'failed')
        self.assertEqual(batch['metric_deltas']['completed'],1)
        self.assertEqual(batch['request_metric_deltas']['flare_requests_cancelled_total'],1)
        self.assertIn('other traffic',batch['error']['message'])

    def test_missing_cancellation_counter_fails_before_requests(self):
        self.mode='missing_counter';batch=self.run_batch()
        self.assertEqual(batch['status'],'failed')
        self.assertIn('missing or invalid',batch['error']['message'])
        self.assertEqual(self.received,[])

    def test_warmup_is_separate_from_summary_and_early_prefixes_are_distinct(self):
        warm=self.run_batch(phase='warmup');measured=self.run_batch()
        self.assertEqual(warm['status'],'complete');self.assertEqual(measured['status'],'complete')
        summary=bench.summarize([warm,measured])[0]
        self.assertEqual(summary['measured_batches'],1);self.assertEqual(summary['successful_requests'],1)
        self.assertNotEqual(warm['requests'][0]['prompt_token_ids'][:128],measured['requests'][0]['prompt_token_ids'][:128])
        a=bench.exact_prompt(TokenizerFixture(),2048,'same-case')
        self.assertEqual(a,bench.exact_prompt(TokenizerFixture(),2048,'same-case'))


if __name__=='__main__':unittest.main()
