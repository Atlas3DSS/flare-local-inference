"""Announcement privacy, durable cursors, cooldown, and delivery recovery."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import tempfile
import unittest
import sys

if sys.platform != 'linux':
    raise unittest.SkipTest('announcement state uses the Linux runner file-lock contract')
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('announce_changes', Path(__file__).resolve().parents[2] / 'scripts/announce_changes.py')
a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(a)
A, B, C = 'a' * 40, 'b' * 40, 'c' * 40
NOW = a.parse_time('2026-10-20T12:00:00Z')
STAMP = '2026-10-20T12:00:00+00:00'
REPO, CHANNEL = 'local-inference-lab/flare', '1560667258232115220'


class GitHub:
    def __init__(self):
        self.calls = []
    def changes(self, base, head):
        self.calls.append((base, head))
        if base == head:
            return []
        if (base, head) not in ((A, B), (A, C), (B, C)):
            raise a.Error('divergent history')
        shas = [B, C] if (base, head) == (A, C) else [head]
        return [{'sha': s, 'category': 'Fix', 'title': 'Fix image ordering'} for s in shas]


class Model:
    def __init__(self):
        self.calls = 0
    def summarize(self, changes):
        self.calls += 1
        return ['Correct image ordering in conversations.']


class Discord:
    def __init__(self):
        self.posts = []
        self.recovered = None
        self.error = None
    def own_id(self):
        return '123'
    def reconcile(self, pending):
        return self.recovered
    def post(self, pending):
        self.posts.append(dict(pending))
        if self.error:
            raise self.error
        return {'id': '456', 'timestamp': STAMP}


class StateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = a.Store(self.tmp.name)
        self.github, self.model, self.discord = GitHub(), Model(), Discord()
    def run_job(self, **kw):
        return a.announce(self.store, self.github, self.model, self.discord,
            repository=REPO, channel=CHANNEL, now=kw.pop('now', NOW), **kw)
    def bootstrap(self, age=a.INTERVAL):
        self.store.save({'version': 1, 'repository': REPO, 'channel': CHANNEL,
            'last_announced_sha': A, 'last_announced_at': NOW - age, 'latest_head': A, 'pending': None})
    def test_missing_baseline_refuses_publication(self):
        with self.assertRaises(a.Error): self.run_job(head=B)
        self.assertEqual(self.discord.posts, [])
    def test_explicit_bootstrap_requires_timestamp(self):
        with self.assertRaises(a.Error): self.run_job(head=B, bootstrap_sha=A)
    def test_explicit_bootstrap(self):
        result = self.run_job(head=B, bootstrap_sha=A, bootstrap_time='2026-10-20T06:00:00Z')
        self.assertEqual(result['status'], 'announced')
    def test_cooldown_persists_pending_head_without_model_call(self):
        self.bootstrap(age=1)
        result = self.run_job(head=B)
        self.assertEqual(result['status'], 'throttled')
        self.assertEqual(self.store.read()['latest_head'], B)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(self.discord.posts, [])
    def test_scheduled_flush_without_further_push(self):
        self.bootstrap(age=1)
        self.run_job(head=B)
        result = self.run_job(now=NOW + a.INTERVAL)
        self.assertEqual(result['status'], 'announced')
        self.assertEqual(self.store.read()['last_announced_sha'], B)
    def test_cooldown_boundary_and_no_duplicate_unchanged_head(self):
        self.bootstrap()
        self.assertEqual(self.run_job(head=B)['status'], 'announced')
        self.assertEqual(self.run_job(head=B, now=NOW + a.INTERVAL - 1)['status'], 'throttled')
        self.assertEqual(self.run_job(head=B, now=NOW + a.INTERVAL)['status'], 'unchanged')
        self.assertEqual(len(self.discord.posts), 1)
    def test_pending_post_is_persisted_before_network(self):
        self.bootstrap()
        def post(pending):
            state = self.store.read()
            self.assertEqual(state['pending']['status'], 'posting')
            self.assertEqual(state['pending']['content'], pending['content'])
            return {'id': '456', 'timestamp': STAMP}
        self.discord.post = post
        self.run_job(head=B)
    def test_timeout_is_not_blindly_retried(self):
        self.bootstrap(); self.discord.error = a.Error('transport failure')
        with self.assertRaises(a.Error): self.run_job(head=B)
        self.assertEqual(self.store.read()['pending']['status'], 'uncertain')
        self.assertEqual(self.store.read()['last_announced_sha'], A)
        self.discord.error = None
        with self.assertRaises(a.Error): self.run_job(now=NOW + 2 * a.INTERVAL)
        self.assertEqual(len(self.discord.posts), 1)
    def test_unknown_outcome_reconciles_and_preserves_later_commits(self):
        self.bootstrap(); self.discord.error = a.Error('timeout')
        with self.assertRaises(a.Error): self.run_job(head=B)
        self.discord.recovered = {'id': '789', 'timestamp': STAMP}
        result = self.run_job(head=C)
        self.assertEqual(result['message_id'], '789')
        state = self.store.read()
        self.assertEqual(state['last_announced_sha'], B)
        self.assertEqual(state['latest_head'], C)
        self.assertEqual(len(self.discord.posts), 1)
    def test_known_rejection_retries_frozen_payload(self):
        self.bootstrap(); self.discord.error = a.HttpError(429)
        with self.assertRaises(a.HttpError): self.run_job(head=B)
        first = self.discord.posts[0]
        self.assertEqual(self.store.read()['pending']['status'], 'ready')
        self.discord.error = None
        self.run_job(head=C)
        self.assertEqual(self.discord.posts[-1]['content'], first['content'])
        self.assertEqual(self.discord.posts[-1]['nonce'], first['nonce'])
        self.assertEqual(self.model.calls, 1)
        self.assertEqual(self.store.read()['latest_head'], C)
    def test_server_error_is_uncertain(self):
        self.bootstrap(); self.discord.error = a.HttpError(500)
        with self.assertRaises(a.HttpError): self.run_job(head=B)
        self.assertEqual(self.store.read()['pending']['status'], 'uncertain')
    def test_uncertain_guard_starts_at_post_not_before_slow_model(self):
        self.bootstrap(); self.discord.error = a.Error('timeout')
        current = [NOW]
        def summarize(changes):
            current[0] += 300
            return ['Correct image ordering.']
        self.model.summarize = summarize
        with patch.object(a.time, 'time', side_effect=lambda: current[0]):
            with self.assertRaises(a.Error):
                a.announce(self.store, self.github, self.model, self.discord,
                    repository=REPO, channel=CHANNEL, head=B)
        pending = self.store.read()['pending']
        self.assertEqual(pending['attempt_at'], NOW + 300)
        self.discord.error = None
        with self.assertRaises(a.Error):
            self.run_job(retry_nonce=pending['nonce'], retry_head=B, now=NOW + a.INTERVAL)
        self.assertEqual(len(self.discord.posts), 1)
    def test_operator_retry_requires_exact_batch_and_six_hour_guard(self):
        self.bootstrap(); self.discord.error = a.Error('timeout')
        with self.assertRaises(a.Error): self.run_job(head=B)
        pending = self.store.read()['pending']
        self.discord.error = None
        with self.assertRaises(a.Error):
            self.run_job(retry_nonce='wrong', retry_head=B, now=NOW + a.INTERVAL)
        with self.assertRaises(a.Error):
            self.run_job(retry_nonce=pending['nonce'], retry_head=C, now=NOW + a.INTERVAL)
        with self.assertRaises(a.Error):
            self.run_job(retry_nonce=pending['nonce'], retry_head=B, now=NOW + a.INTERVAL - 1)
        self.assertEqual(len(self.discord.posts), 1)
        self.run_job(retry_nonce=pending['nonce'], retry_head=B, now=NOW + a.INTERVAL)
        self.assertEqual(len(self.discord.posts), 2)
        self.assertEqual(self.discord.posts[0]['content'], self.discord.posts[1]['content'])
        self.assertEqual(self.store.read()['last_announced_sha'], B)
        self.assertEqual(self.run_job(head=C, now=NOW + a.INTERVAL + 1)['status'], 'throttled')
    def test_operator_retry_recovers_without_post_when_message_exists(self):
        self.bootstrap(); self.discord.error = a.Error('timeout')
        with self.assertRaises(a.Error): self.run_job(head=B)
        pending = self.store.read()['pending']
        self.discord.recovered = {'id': '789', 'timestamp': STAMP}
        self.run_job(retry_nonce=pending['nonce'], retry_head=B)
        self.assertEqual(len(self.discord.posts), 1)
    def test_retry_without_uncertain_batch_is_rejected(self):
        self.bootstrap()
        with self.assertRaises(a.Error): self.run_job(head=B, retry_nonce='nonce', retry_head=B)
        self.assertEqual(self.discord.posts, [])
    def test_state_cannot_move_between_channels(self):
        self.bootstrap(); s = self.store.read(); s['channel'] = '999'; self.store.save(s)
        with self.assertRaises(a.Error): self.run_job(head=B)
    def test_history_regression_is_rejected(self):
        self.bootstrap(); s = self.store.read(); s['latest_head'] = C; self.store.save(s)
        with self.assertRaises(a.Error): self.run_job(head=B)
        self.assertEqual(self.store.read()['latest_head'], C)
    def test_backward_clock_never_bypasses_cooldown(self):
        self.bootstrap(age=0)
        self.assertEqual(self.run_job(head=B, now=NOW - 100)['status'], 'throttled')


class Http:
    def __init__(self, response):
        self.response, self.calls = response, []
    def json(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


class PrivacyTests(unittest.TestCase):
    def test_source_snippets_secrets_and_paths_are_not_sent(self):
        for text in ('Fix `fn main()`', 'fix: return secret', 'fix: x = 42',
                     'Fix ignition/decoder.py', 'Use https://private.example/code',
                     'Fix ghp_secret', '```rust source```', 'import os', '@everyone update',
                     'raise RuntimeError', 'while True:', 'await hidden_result',
                     'PreparedContract.private_field', 'Fix PreparedContract behavior', 'x plus y',
                     'Fix return secret', 'add eax, ebx'):
            with self.subTest(text=text):
                label, safe = a.safe_subject(text)
                self.assertNotEqual(safe, text)
                self.assertFalse(a.UNSAFE.search(safe))
    def test_commit_body_is_never_forwarded(self):
        class CommitHttp(Http):
            def json(self, method, url, **kwargs):
                self.calls.append((method, url, kwargs))
                base = [{'sha': A, 'commit': {'message': 'baseline'}}]
                return base if 'sha=' + A in url else [
                    {'sha': B, 'commit': {'message': 'Fix image ordering\n\nSECRET SOURCE: fn main() {}'}}, *base]
        http = CommitHttp(None)
        changes = a.GitHub(http, REPO, 'redacted').changes(A, B)
        self.assertEqual(changes[0]['title'], 'Fix image ordering')
        self.assertNotIn('SECRET', json.dumps(changes))
        self.assertIn('/commits?', http.calls[0][1])
        self.assertNotIn('/compare/', http.calls[0][1])
    def response(self, text='Correct image ordering.', ids=None):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            'bullets': [{'text': text, 'commits': ids or ['C1']}]})}}]}
    def test_model_receives_only_sanitized_subject_metadata(self):
        http = Http(self.response())
        changes = [{'sha': B, 'category': 'Fix', 'title': 'Fix image ordering'}]
        self.assertEqual(a.Model(http, 'http://model', 'model').summarize(changes), ['Correct image ordering.'])
        data = json.loads(http.calls[0][2]['payload']['messages'][1]['content'])
        self.assertEqual(data, {'changes': [{'id': 'C1', 'category': 'Fix', 'title': 'Fix image ordering'}]})
    def test_model_boundary_sanitizes_direct_callers(self):
        http = Http(self.response())
        a.Model(http, 'http://model', 'model').summarize([
            {'sha': B, 'category': 'private source', 'title': 'fix: return SECRET_SOURCE'}])
        payload = http.calls[0][2]['payload']
        self.assertNotIn('SECRET_SOURCE', json.dumps(payload))
        self.assertNotIn('private source', json.dumps(payload))
    def test_summary_schema_excludes_rejected_opening_verbs(self):
        changes = [{'sha': B, 'category': 'Maintenance', 'title': 'Maintenance update'}]
        rejected = 'Applied maintenance updates.'
        http = Http(self.response(rejected))
        with self.assertRaisesRegex(a.Error, 'Summary contains disallowed content'):
            a.Model(http, 'http://model', 'model').summarize(changes)
        schema = http.calls[0][2]['payload']['response_format']['json_schema']['schema']
        pattern = schema['properties']['bullets']['items']['properties']['text']['pattern']
        self.assertIsNone(re.search(pattern, rejected))
        for opening in a.SUMMARY_OPENINGS:
            self.assertIsNone(re.fullmatch(pattern, opening + ' '))
            text = opening + ' maintenance updates.'
            self.assertRegex(text, pattern)
            self.assertTrue(a.safe_prose(text))
        self.assertEqual(a.Model(Http(self.response('Updated maintenance updates.')),
            'http://model', 'model').summarize(changes), ['Updated maintenance updates.'])
    def test_code_urls_mentions_and_unknown_evidence_are_rejected(self):
        changes = [{'sha': B, 'category': 'Fix', 'title': 'Fix image ordering'}]
        for text in ('return x', 'Use `foo()`', 'Visit https://example.com', '@everyone update', 'Changed source.rs', 'raise RuntimeError', 'while True:',
                     'await hidden_result', 'PreparedContract.private_field', 'Fix return secret', 'add eax, ebx'):
            with self.subTest(text=text), self.assertRaises(a.Error):
                a.Model(Http(self.response(text)), 'http://model', 'model').summarize(changes)
        with self.assertRaises(a.Error):
            a.Model(Http(self.response(ids=['C2'])), 'http://model', 'model').summarize(changes)
    def test_embedded_statements_and_assembly_never_reach_model(self):
        for text in ('Fix return secret', 'add eax, ebx'):
            with self.subTest(text=text):
                http = Http(self.response())
                a.Model(http, 'http://model', 'model').summarize([
                    {'sha': B, 'category': 'Fix', 'title': text}])
                payload = json.dumps(http.calls[0][2]['payload']['messages'][1])
                self.assertNotIn(text, payload)
    def test_omitted_commit_is_rejected(self):
        changes = [{'sha': s, 'category': 'Fix', 'title': 'Fix images'} for s in (B, C)]
        with self.assertRaises(a.Error):
            a.Model(Http(self.response()), 'http://model', 'model').summarize(changes)
    def test_discord_nonce_and_mentions(self):
        http = Http({'id': '456'})
        a.Discord(http, 'redacted', CHANNEL).post({'content': 'Safe update', 'nonce': 'a' * 24})
        payload = http.calls[0][2]['payload']
        self.assertTrue(payload['enforce_nonce'])
        self.assertEqual(payload['allowed_mentions'], {'parse': []})
    def test_reconcile_requires_own_author_and_exact_content(self):
        class History(Http):
            def json(self, method, url, **kw):
                if url.endswith('/users/@me'): return {'id': '123'}
                return [{'id': '456', 'timestamp': STAMP, 'author': {'id': '999'}, 'content': 'wanted'},
                        {'id': '457', 'timestamp': STAMP, 'author': {'id': '123'}, 'content': 'wanted'}]
        found = a.Discord(History(None), 'redacted', CHANNEL).reconcile({'content': 'wanted', 'attempt_at': NOW})
        self.assertEqual(found['id'], '457')


class CommitHistoryTests(unittest.TestCase):
    def test_merge_includes_older_feature_commits_after_baseline(self):
        feature = 'd' * 40
        histories = {B: [B, A], C: [C, B, feature, A]}
        class CommitHttp:
            def json(self, method, url, **kwargs):
                from urllib.parse import urlsplit, parse_qs
                head = parse_qs(urlsplit(url).query)['sha'][0]
                return [{'sha': sha, 'commit': {'message': 'Fix request handling'}} for sha in histories[head]]
        changes = a.GitHub(CommitHttp(), REPO, 'redacted').changes(B, C)
        self.assertEqual([c['sha'] for c in changes], [feature, C])
    def test_unrelated_baseline_is_rejected(self):
        class CommitHttp:
            def json(self, method, url, **kwargs):
                from urllib.parse import urlsplit, parse_qs
                head = parse_qs(urlsplit(url).query)['sha'][0]
                return [{'sha': head, 'commit': {'message': 'Fix request handling'}}]
        with self.assertRaises(a.Error): a.GitHub(CommitHttp(), REPO, 'redacted').changes(A, B)


class RegistryTests(unittest.TestCase):
    def registry(self, arm_sha=B, corrupt=False):
        objects = {}
        def add(value):
            data = json.dumps(value).encode(); digest = 'sha256:' + hashlib.sha256(data).hexdigest()
            objects[digest] = data; return digest
        descriptors = []
        for arch, sha in [('amd64', B), ('arm64', arm_sha)]:
            config = add({'config': {'Labels': {'org.opencontainers.image.revision': sha}}})
            manifest = add({'config': {'digest': config}})
            descriptors.append({'platform': {'os': 'linux', 'architecture': arch}, 'digest': manifest})
        index = json.dumps({'manifests': descriptors}).encode()
        class Registry:
            def json(self, *args, **kwargs): return {'token': 'redacted'}
            def request(self, method, url, **kwargs):
                key = url.rsplit('/', 1)[1]
                data = index if key == 'latest' else objects[key]
                return (data + b' ' if corrupt and key != 'latest' else data), {}
        return Registry()
    def test_both_architectures_must_match(self):
        sha, digest = a.published_revision(self.registry(), 'ghcr.io/local-inference-lab/flare:latest')
        self.assertEqual(sha, B); self.assertTrue(digest.startswith('sha256:'))
        with self.assertRaises(a.Error):
            a.published_revision(self.registry(arm_sha=C), 'ghcr.io/local-inference-lab/flare:latest')
    def test_bad_digest_is_rejected(self):
        with self.assertRaises(a.Error):
            a.published_revision(self.registry(corrupt=True), 'ghcr.io/local-inference-lab/flare:latest')
    def test_other_repository_is_rejected(self):
        with self.assertRaises(a.Error):
            a.published_revision(self.registry(), 'ghcr.io/other/repo:latest')


if __name__ == '__main__':
    unittest.main()
