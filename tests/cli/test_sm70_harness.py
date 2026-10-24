"""Portable input/CLI and gate admission tests; no CUDA/model qualification claimed."""
import argparse
from contextlib import redirect_stderr
import copy
import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import benchmark_mtp, benchmark_long, generate_prompts, native, prepare_replay

ROOT=Path(__file__).resolve().parents[2]


class HarnessTests(unittest.TestCase):
    def test_target_length_uses_explicit_token_list_and_bounded_search(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kwargs):
                ids=[7]*(20+messages[0]['content'].count('def example_'))
                return ids if kwargs.get('return_dict') is False else {'input_ids':ids,'attention_mask':[1]*len(ids)}
        rows=generate_prompts.make_rows('short',32,tokenizer=Tokenizer(),target_tokens=40)
        self.assertTrue(all(row['rendered_prompt_tokens']==40 for row in rows))
        class ConstantTokenizer:
            def apply_chat_template(self,*args,**kwargs):return [7]
        with self.assertRaisesRegex(ValueError,'did not grow'):
            generate_prompts.make_rows('short',32,tokenizer=ConstantTokenizer(),target_tokens=40)

    def test_public_prompts_are_deterministic_and_consumed_by_both_clients(self):
        rows=generate_prompts.make_rows('short',32)
        self.assertEqual(rows,generate_prompts.make_rows('short',32))
        self.assertEqual(len(rows),6)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prompts.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            for client in (benchmark_mtp,benchmark_long):
                selected=client.load_prompts(path,['code-00','code-01','code-02','code-03'])
                self.assertEqual(len(selected),4)
                request=client.request_body('model',selected[0],128,7035,logprobs=True)
                self.assertTrue(request['ignore_eos']);self.assertEqual(request['max_tokens'],128)
                self.assertEqual(request['messages'],rows[0]['messages'])
            path.write_text(path.read_text()+json.dumps(rows[0])+'\n')
            with self.assertRaisesRegex(ValueError,'unique'):benchmark_mtp.load_prompts(path)

    def test_export_cannot_omit_any_release_gate(self):
        required=('validate','check_invariance','check_commit_growth','check_consumers',
                  'measure_components','measure_accuracy','measure','signing_dir','short_reference','long_reference')
        options=dict(output='package.safetensors',**{name:True for name in required})
        prepare_replay.require_export_gates(SimpleNamespace(**options))
        for name in required:
            with self.subTest(missing=name),self.assertRaisesRegex(ValueError,'every'):
                prepare_replay.require_export_gates(SimpleNamespace(**{**options,name:False}))

    def test_skip_reference_gates_drops_only_the_accuracy_gate(self):
        structural=('validate','check_invariance','check_commit_growth','check_consumers',
                    'measure_components','measure','signing_dir')
        options=dict(output='package.safetensors',skip_reference_gates=True,measure_accuracy=False,
                     short_reference=None,long_reference=None,**{name:True for name in structural})
        prepare_replay.require_export_gates(SimpleNamespace(**options))
        for name in structural:
            with self.subTest(missing=name),self.assertRaisesRegex(ValueError,'still requires'):
                prepare_replay.require_export_gates(SimpleNamespace(**{**options,name:False}))
        record={'reference_gates':'skipped','reason':'--skip-reference-gates'}
        self.assertEqual(prepare_replay.qualification(SimpleNamespace(**options)),record)
        stamp=prepare_replay.export_qualification(SimpleNamespace(**options))['qualification']
        self.assertEqual((stamp.reference_gates,stamp.reason),('skipped','--skip-reference-gates'))
        # Without the option nothing is recorded or stamped, as before.
        self.assertIsNone(prepare_replay.qualification(SimpleNamespace(output=None)))
        self.assertEqual(prepare_replay.export_qualification(SimpleNamespace(output=None)),{})

    def parse_prepare(self,*extra):
        """Run prepare_replay's argument checks; a missing --root stops it before any import or GPU."""
        captured=[];parse=argparse.ArgumentParser.parse_args
        def spy(parser,*args,**kwargs):
            captured.append(parse(parser,*args,**kwargs));return captured[-1]
        argv=['prepare_replay.py','--root','/nonexistent/flare-root','--prompts','p','--kernel-cache','k',
              '--checkpoint','c','--compile-workers','1','--report','r',*extra]
        stderr=io.StringIO()
        with mock.patch.object(argparse.ArgumentParser,'parse_args',spy),mock.patch.object(sys,'argv',argv),\
                redirect_stderr(stderr),self.assertRaises(SystemExit) as stop:
            prepare_replay.main()
        self.assertEqual(stop.exception.code,2)
        return captured[0],stderr.getvalue()

    def test_skip_reference_gates_flag_parsing(self):
        args,error=self.parse_prepare()
        self.assertNotIn('skip_reference_gates',vars(args))  # default reports keep their exact keys
        self.assertIn('--root must be an existing output directory',error)
        args,error=self.parse_prepare('--skip-reference-gates')
        self.assertIs(args.skip_reference_gates,True)
        self.assertIn('--root must be an existing output directory',error)
        for given in (['--measure-accuracy'],['--short-reference','s.safetensors'],['--long-reference','l.safetensors']):
            with self.subTest(given=given):
                _,error=self.parse_prepare('--skip-reference-gates',*given)
                self.assertIn('omit --measure-accuracy',error)
        _,error=self.parse_prepare('--skip-reference-gates','--output','o.safetensors','--measure','t.json',
                                   '--signing-dir','keys','--check-invariance','--check-commit-growth',
                                   '--check-consumers','--measure-components')
        self.assertIn('still requires',error)  # --validate missing

    def test_shell_helper_default_arguments_are_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            echo=Path(tmp)/'python';echo.write_text('#!/bin/sh\nshift\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
            echo.chmod(0o755)
            run=lambda *extra:subprocess.run(['bash',str(ROOT/'tools/prepare_replay.sh'),str(echo),*extra],
                capture_output=True,text=True,timeout=15,check=True).stdout.splitlines()
            helper=('--context 32768 --concurrency 4 --depth 4 --lease-gib 30 --chunk 512 --validate '
                    '--check-invariance --check-commit-growth --check-consumers --measure-components').split()
            tail='--measure-batches 1 4 --measure-prompt-ids code-00 code-01 code-02 code-03 --root /r'.split()
            self.assertEqual(run('--root','/r'),helper+['--measure-accuracy']+tail)
            self.assertEqual(run('--root','/r','--skip-reference-gates'),helper+tail+['--skip-reference-gates'])

    def test_native_refuses_skipped_reference_package_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            package=Path(tmp)/'package.safetensors';package.write_bytes(b'synthetic signed primary')
            gates=('validation','long_sequence','consumer_identity','commit_capacity_growth')
            report=dict(status='exported',package=str(package),components={'status':'complete'},
                        qualification={'reference_gates':'skipped','reason':'--skip-reference-gates'},
                        **{key:{'passed':True} for key in gates},
                        measurement={'status':'complete','qualification_passed':True,
                                     'runs':[{'status':'complete'}],'summary':[{'tau':2.0}]},
                        package_primary_sha256=hashlib.sha256(package.read_bytes()).hexdigest())
            with self.assertRaisesRegex(RuntimeError,'--allow-unqualified'):
                native.require_qualification(report,package)
            self.assertEqual(native.require_qualification(report,package,True),
                {'reference_gates':'skipped','reason':'--skip-reference-gates','allowed_by':'--allow-unqualified'})
            for field in (*gates,'components','measurement'):
                changed=copy.deepcopy(report);changed.pop(field)
                with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'qualification'):
                    native.require_qualification(changed,package,True)
            changed=copy.deepcopy(report);changed['qualification']={'reference_gates':'skipped','reason':'other'}
            with self.assertRaisesRegex(RuntimeError,'unknown package qualification'):
                native.require_qualification(changed,package,True)
            # A fully qualified report is accepted with or without the option and records nothing.
            full=copy.deepcopy(report);full.pop('qualification');full['accuracy']={'status':'complete'}
            self.assertIsNone(native.require_qualification(full,package))
            self.assertIsNone(native.require_qualification(full,package,True))
            full.pop('accuracy')
            with self.assertRaisesRegex(RuntimeError,'qualification'):
                native.require_qualification(full,package,True)

    def test_native_requires_all_qualification_results_and_exact_primary(self):
        with tempfile.TemporaryDirectory() as tmp:
            package=Path(tmp)/'package.safetensors';package.write_bytes(b'synthetic signed primary')
            gates=('validation','long_sequence','consumer_identity','commit_capacity_growth')
            report=dict(status='exported',package=str(package),components={'status':'complete'},
                        accuracy={'status':'complete'},**{key:{'passed':True} for key in gates},
                        measurement={'status':'complete','qualification_passed':True,
                                     'runs':[{'status':'complete'}],'summary':[{'tau':2.0}]},
                        package_primary_sha256=hashlib.sha256(package.read_bytes()).hexdigest())
            native.require_qualification(report,package)
            for field in (*gates,'components','accuracy','measurement','package','status'):
                changed=copy.deepcopy(report);changed.pop(field)
                with self.subTest(field=field),self.assertRaisesRegex(RuntimeError,'qualification'):
                    native.require_qualification(changed,package)
            for field,value in [('status','failed'),('qualification_passed',False),('runs',[]),
                                ('runs',[{'status':'failed'}]),('summary',[])]:
                changed=copy.deepcopy(report);changed['measurement'][field]=value
                with self.subTest(tau_field=field),self.assertRaisesRegex(RuntimeError,'qualification'):
                    native.require_qualification(changed,package)
            changed=copy.deepcopy(report);changed.pop('package_primary_sha256')
            with self.assertRaisesRegex(RuntimeError,'primary hash'):native.require_qualification(changed,package)
            package.write_bytes(b'different signed primary at same pathname')
            with self.assertRaisesRegex(RuntimeError,'primary hash'):native.require_qualification(report,package)

    def test_native_dry_run_uses_explicit_paths_and_does_not_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd=[sys.executable,str(ROOT/'tools/native.py'),'--dry-run','--mode','native',
                 '--depths','4','--package','/input/package.safetensors','--executable','/input/flare',
                 '--publisher-keys','/input/publisher.json','--trust-store','/input/trust.toml',
                 '--prompts','/input/prompts.jsonl','--gpu-uuid','GPU-test','--output-dir',tmp,
                 '--memory-limit','31G','--scheduler-policy','agent']
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=15,check=True)
            row=json.loads(result.stdout)[0]
            self.assertIn('/input/publisher.json',row['server']);self.assertIn('31G',row['server'])
            self.assertIn('agent',row['server']);self.assertIn('/input/prompts.jsonl',row['client'])
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_native_prefill_srpt_flag_reaches_the_server_command_only_when_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=[sys.executable,str(ROOT/'tools/native.py'),'--dry-run','--mode','native',
                  '--depths','4','--package','/input/package.safetensors','--executable','/input/flare',
                  '--publisher-keys','/input/publisher.json','--trust-store','/input/trust.toml',
                  '--prompts','/input/prompts.jsonl','--gpu-uuid','GPU-test','--output-dir',tmp]
            off=json.loads(subprocess.run(base,capture_output=True,text=True,timeout=15,check=True).stdout)[0]
            on=json.loads(subprocess.run(base+['--prefill-srpt'],capture_output=True,text=True,timeout=15,check=True).stdout)[0]
            self.assertNotIn('--prefill-srpt',off['server']);self.assertIn('--prefill-srpt',on['server'])
            self.assertEqual(list(Path(tmp).iterdir()),[])
        self.assertNotIn('--prefill-srpt',native.server_command(SimpleNamespace(mode='native',executable='/x',package='/p',
            publisher_keys='/k',trust_store='/t',prefix_cache='8G',memory_limit='31G',scheduler_policy='balanced',port=8383),4))

    def test_prompt_generator_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'prompts.jsonl';output.write_text('preserved')
            result=subprocess.run([sys.executable,str(ROOT/'tools/generate_prompts.py'),
                                   '--output',str(output)],capture_output=True,text=True,timeout=15)
            self.assertNotEqual(result.returncode,0);self.assertEqual(output.read_text(),'preserved')


if __name__=='__main__':unittest.main()
