"""CPU-only HTTP contract test for the public SM70 benchmark tools.

An in-process loopback server applies Flare's endpoint rules for the requests
these tools send, then each tool runs against it as a subprocess and must exit 0
without any 4xx/5xx response.

- POST /v1/completions rejects return_token_ids/return_token_timestamps with the
  literal HTTP 400 body of helios/src/server/protocol/decode.rs. The mock also
  requires model and prompt and rejects messages; decode.rs itself ignores
  messages there and does not require model, so these two are tool-contract rules.
- POST /v1/chat/completions requires model and messages and allows the extensions.
- GET /health, /v1/models and /metrics answer 200.

The canonical request bodies are tests/fixtures/http_requests/*.json. The Rust
test public_tool_request_fixtures_match_endpoint_rules in decode.rs feeds the
same files to the real validator, so the tools and the server cannot drift apart.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
FIXTURES = Path(__file__).with_name('fixtures') / 'http_requests'
MODEL = 'contract-fixture-model'
OBSERVATION_KEYS = ('return_token_ids', 'return_token_timestamps')
OBSERVATION_ERROR = 'token observation extensions require /v1/chat/completions'
VARIABLE = frozenset({'model', 'prompt', 'messages', 'seed', 'max_tokens'})


def error_body(message):
    """Protocol::error for an HTTP 400 on the OpenAI-compatible endpoints."""
    return {'error': {'type': 'invalid_request_error', 'message': message,
                      'param': None, 'code': 'invalid_request_error'}}


class ContractServer:
    def __init__(self, record_dir):
        self.lock, self.record_dir, self.requests = threading.Lock(), record_dir, []
        self.counters = dict(active=0, pending=0, prefill_tokens=0, generated_tokens=0, completed=0,
                             failed=0, reused_tokens=0)
        self.admitted = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value, body=None):
                raw = json.dumps(value).encode()
                owner.record(self.command, self.path, status, body)
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == '/health':
                    return self.reply(200, {'status': 'ok'})
                if self.path == '/v1/models':
                    return self.reply(200, {'object': 'list', 'data': [
                        {'id': MODEL, 'object': 'model', 'flare': {'max_top_logprobs': 1}}]})
                if self.path == '/metrics?format=json':
                    with owner.lock:
                        counters = dict(owner.counters)
                    return self.reply(200, counters)
                if self.path == '/metrics?format=prometheus':
                    with owner.lock:
                        raw = (f'flare_requests_admitted_total {owner.admitted}\n'
                               'flare_requests_cancelled_total 0\n').encode()
                    owner.record(self.command, self.path, 200, None)
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(raw)))
                    self.end_headers()
                    return self.wfile.write(raw)
                return self.reply(404, error_body('not found'))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/v1/completions':
                    if any(body.get(key) is True for key in OBSERVATION_KEYS):
                        return self.reply(400, error_body(OBSERVATION_ERROR), body)
                    if 'messages' in body or not body.get('model') or 'prompt' not in body:
                        return self.reply(400, error_body('contract: /v1/completions takes model and prompt, not messages'), body)
                elif self.path == '/v1/chat/completions':
                    if not body.get('model') or not isinstance(body.get('messages'), list):
                        return self.reply(400, error_body('contract: /v1/chat/completions requires model and messages'), body)
                else:
                    return self.reply(404, error_body('not found'), body)
                if body.get('stream') is not True:
                    return self.reply(400, error_body('contract: the public tools stream every request'), body)
                owner.record(self.command, self.path, 200, body)
                self.stream(body)

            def send_event(self, event):
                event = {'id': 'contract', 'created': 0, 'model': MODEL, **event}
                self.wfile.write(b'data: ' + json.dumps(event).encode() + b'\n\n')
                self.wfile.flush()

            def stream(self, body):
                chat = self.path == '/v1/chat/completions'
                prompt = len(body['prompt']) if isinstance(body.get('prompt'), list) else 0
                output = body['max_tokens']
                kind = 'chat.completion.chunk' if chat else 'text_completion'

                def usage(generated):
                    return {'prompt_tokens': prompt, 'completion_tokens': generated,
                            'total_tokens': prompt + generated, 'prompt_tokens_details': {'cached_tokens': 0}}

                def choice(text, finish=None):
                    if chat:
                        return {'index': 0, 'delta': {'content': text} if text else {}, 'finish_reason': finish}
                    return {'index': 0, 'text': text, 'logprobs': None, 'finish_reason': finish}

                with owner.lock:
                    owner.admitted += 1
                    owner.counters['active'] += 1
                    owner.counters['prefill_tokens'] += prompt
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    if chat:
                        self.send_event({'object': kind, 'usage': usage(0), 'choices': [
                            {'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})
                    for generated in range(1, output + 1):
                        time.sleep(.002)
                        if chat and body.get('return_token_ids'):
                            observed = {'index': 0, 'delta': {}, 'finish_reason': None, 'token_ids': [1000 + generated]}
                            if body.get('return_token_timestamps'):
                                observed['token_timestamps_ns'] = [time.monotonic_ns()]
                            self.send_event({'object': kind, 'usage': usage(generated), 'choices': [observed]})
                        self.send_event({'object': kind, 'usage': usage(generated), 'choices': [choice('x')]})
                    final = {'object': kind, 'usage': usage(output), 'choices': [choice('', 'length')]}
                    if chat and body.get('return_token_ids'):
                        final['reused_prompt_tokens'] = 0
                    self.send_event(final)
                    self.send_event({'object': kind, 'usage': usage(output), 'choices': []})
                finally:
                    with owner.lock:
                        owner.counters['active'] -= 1
                        owner.counters['generated_tokens'] += output
                        owner.counters['completed'] += 1
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_port
        self.base = f'http://127.0.0.1:{self.port}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def record(self, method, path, status, body):
        with self.lock:
            index = len(self.requests)
            self.requests.append({'method': method, 'path': path, 'status': status, 'body': body})
        if body is not None:
            (self.record_dir / f'request-{index:04d}.json').write_text(json.dumps(body))

    def errors(self):
        with self.lock:
            return [(r['method'], r['path'], r['status']) for r in self.requests if r['status'] >= 400]

    def bodies(self, path):
        with self.lock:
            return [r['body'] for r in self.requests if r['method'] == 'POST' and r['path'] == path]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def server(tmp_path):
    record_dir = tmp_path / 'requests'
    record_dir.mkdir()
    instance = ContractServer(record_dir)
    yield instance
    instance.close()


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def assert_matches_fixture(body, name):
    expected = fixture(name)['body']
    assert set(body) == set(expected), name
    assert ({k: v for k, v in body.items() if k not in VARIABLE}
            == {k: v for k, v in expected.items() if k not in VARIABLE}), name


def run(command, cwd):
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=300)


def post(base, path, body):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def write_prompts(path, prefix):
    rows = [{'id': f'{prefix}-{index:02d}', 'thinking': False, 'input_set': 'contract-test',
             'messages': [{'role': 'user', 'content': f'Contract prompt {index}: merge two sorted lists.'}]}
            for index in range(4)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return [row['id'] for row in rows]


def test_fixtures_follow_the_endpoint_rules(server):
    names = sorted(path.name for path in FIXTURES.glob('*.json'))
    assert len(names) >= 4
    for name in names:
        value = fixture(name)
        status, text = post(server.base, value['endpoint'], value['body'])
        assert status == value['expect']['status'], name
        if status == 400:
            assert json.loads(text) == error_body(value['expect']['error']), name
        else:
            assert text.rstrip().endswith('data: [DONE]'), name


def test_mock_rejects_completions_contract_violations(server):
    good = fixture('completions-benchmark-prefill.json')['body']
    for body in ({**good, 'return_token_ids': True}, {**good, 'return_token_timestamps': True},
                 {**good, 'messages': [{'role': 'user', 'content': 'x'}]},
                 {k: v for k, v in good.items() if k != 'prompt'}, {k: v for k, v in good.items() if k != 'model'}):
        assert post(server.base, '/v1/completions', body)[0] == 400
    chat = fixture('chat-benchmark-mtp.json')['body']
    assert post(server.base, '/v1/chat/completions', {k: v for k, v in chat.items() if k != 'messages'})[0] == 400
    assert post(server.base, '/v1/chat/completions', chat)[0] == 200


def test_prefill_client_stays_within_the_completions_contract(server, tmp_path):
    tokenizers = pytest.importorskip('tokenizers', reason='benchmark_prefill.py loads the served tokenizer.json')
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(vocab={s: i for i, s in enumerate(alphabet)}, merges=[]))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    output = tmp_path / 'prefill.json'
    result = run([sys.executable, str(TOOLS / 'benchmark_prefill.py'), '--base-url', server.base, '--model', MODEL,
                  '--tokenizer', str(tmp_path / 'tokenizer.json'), '--output', str(output), '--run-id', 'contract',
                  '--exclusive-endpoint', '--warmups', '1', '--repetitions', '3', '--output-tokens', '2',
                  '--timeout', '60'], tmp_path)
    assert server.errors() == []
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(output.read_text())['status'] == 'complete'
    bodies = server.bodies('/v1/completions')
    assert len(bodies) == 80
    for body in bodies:
        assert_matches_fixture(body, 'completions-benchmark-prefill.json')


@pytest.mark.parametrize('client,prefix,concurrencies,name', [
    ('benchmark_mtp.py', 'code', ['1', '4'], 'chat-benchmark-mtp.json'),
    ('benchmark_long.py', 'long', ['4', '1'], 'chat-benchmark-long.json')])
def test_decode_clients_run_from_the_native_dry_run_plan(server, tmp_path, client, prefix, concurrencies, name):
    prompts = tmp_path / 'prompts.jsonl'
    ids = write_prompts(prompts, prefix)
    output_dir = tmp_path / 'native'
    output_dir.mkdir()
    # native.py serves depth d on --port + (d - 3), so depth 4 lands on the mock.
    dry = run([sys.executable, str(TOOLS / 'native.py'), '--mode', 'native', '--depths', '4',
               '--package', str(tmp_path / 'model.safetensors'), '--executable', str(tmp_path / 'flare'),
               '--output-dir', str(output_dir), '--gpu-uuid', 'GPU-contract',
               '--publisher-keys', str(tmp_path / 'publisher.json'), '--trust-store', str(tmp_path / 'trust.toml'),
               '--port', str(server.port - 1), '--tokens', '4', '--repetitions', '1',
               '--client', str(TOOLS / client), '--prompts', str(prompts), '--prompt-ids', *ids,
               '--concurrencies', *concurrencies, '--dry-run'], tmp_path)
    assert dry.returncode == 0, dry.stderr[-2000:]
    plan = json.loads(dry.stdout)
    command = [MODEL if part == '<native model ID>' else part for part in plan[0]['client']]
    assert command[command.index('--base-url') + 1] == server.base
    assert server.requests == []
    result = run(command, tmp_path)
    assert server.errors() == []
    assert result.returncode == 0, result.stderr[-2000:]
    report = json.loads(Path(command[command.index('--output') + 1]).read_text())
    assert report['status'] == 'complete'
    bodies = server.bodies('/v1/chat/completions')
    assert len(bodies) == 8
    for body in bodies:
        assert_matches_fixture(body, name)
