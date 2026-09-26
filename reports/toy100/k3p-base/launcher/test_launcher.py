"""Offline integration checks: real worktrees, fake agent and GPU preflight.

Run: python3 -m unittest discover -s reports/toy100/k3p-base/launcher -v
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stream', SOURCE / 'claude-stream.py')
stream = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stream)


class StreamTests(unittest.TestCase):
    def render(self, events, engine='opencode'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            renderer = stream.Renderer(root, engine)
            for event in events:
                renderer.line(json.dumps(event))
            renderer.close()
            final = root / 'final.md'
            return (json.loads((root / f'{engine}-usage.json').read_text()),
                    final.read_text() if final.exists() else '')

    def test_opencode_multiturn_final_usage_and_duplicate_parts(self):
        tool = dict(type='tool_use', part=dict(id='tool1', tool='bash',
                    state=dict(status='completed', input={'command': 'true'}, output='ok')))
        finish = dict(type='step_finish', part=dict(id='finish1', reason='tool-calls',
                      cost=0.01, tokens=dict(input=20, output=5, reasoning=2, cache={'read': 4})))
        usage, final = self.render([
            dict(type='step_start', sessionID='session1'),
            dict(type='text', part=dict(id='text1', messageID='m1', text='Working')),
            tool, tool, finish, finish,
            dict(type='step_start'),
            dict(type='text', part=dict(id='text2', messageID='m2', text='Final')),
            dict(type='text', part=dict(id='text3', messageID='m2', text='Details')),
            dict(type='step_finish', part=dict(reason='stop', cost=0.02,
                 tokens=dict(input=30, output=10, reasoning=3, cache={'write': 2}))),
        ])
        self.assertEqual(final, 'Final\n\nDetails\n')
        self.assertTrue(usage['completed'])
        self.assertEqual(usage['tool_calls'], {'bash': 1})
        self.assertEqual(usage['num_turns'], 2)
        self.assertEqual(usage['session_id'], 'session1')
        self.assertAlmostEqual(usage['total_cost_usd'], 0.03)
        self.assertEqual(usage['usage'], dict(input=50, output=15, reasoning=5, cache_read=4, cache_write=2))

    def test_opencode_interruption_is_not_completion(self):
        usage, final = self.render([
            dict(type='text', part=dict(messageID='m1', text='Partial evidence')),
            dict(type='step_finish', part=dict(reason='tool-calls', cost=0.1)),
        ])
        self.assertFalse(usage['completed'])
        self.assertEqual(final, 'Partial evidence\n')
        self.assertEqual(usage['total_cost_usd'], 0.1)

    def test_opencode_api_error(self):
        usage, final = self.render([dict(type='error', error={'name': 'APIError'})])
        self.assertFalse(usage['completed'])
        self.assertTrue(usage['is_error'])
        self.assertEqual(usage['error']['name'], 'APIError')
        self.assertEqual(final, '')

    def test_existing_claude_and_grok_streams(self):
        for engine in ('claude', 'grok'):
            with self.subTest(engine=engine):
                usage, final = self.render([
                    dict(type='assistant', message={'content': [{'type': 'text', 'text': 'Interim'}]}),
                    dict(type='result', subtype='success', result='Done', total_cost_usd=0.2),
                ], engine)
                self.assertTrue(usage['completed'])
                self.assertEqual(final, 'Done\n')
                self.assertEqual(usage['total_cost_usd'], 0.2)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='gan-launcher-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('try-gan.sh', 'claude-stream.py', 'opencode-config.json'):
            shutil.copy2(SOURCE / name, self.root / name)
        self.repo = self.root / 'source'
        self.repo.mkdir()
        self.git('init', '-q')
        (self.repo / 'README.md').write_text('Fixture repository\n')
        self.git('add', '.')
        self.git('-c', 'user.name=Launcher Test', '-c', 'user.email=test@example.invalid',
                 '-c', 'commit.gpgsign=false', 'commit', '-qm', 'Fixture')
        self.brief = self.root / 'brief.md'
        self.brief.write_text('INTEGRATION_TEST_PROMPT\n')
        agent = self.root / 'fake-agent'
        agent.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
run = pathlib.Path.cwd().parent
config = json.loads(os.environ['OPENCODE_CONFIG_CONTENT'])
assert config['permission']['task'] == 'deny'
assert config['permission']['*'] == 'allow'
assert 'apiKey' not in config['provider']['nano-gpt']['options']
assert os.environ['OPENCODE_DISABLE_PROJECT_CONFIG'] == 'true'
assert os.environ['OPENCODE_DISABLE_CLAUDE_CODE'] == 'true'
assert os.environ['OPENCODE_DISABLE_EXTERNAL_SKILLS'] == 'true'
assert os.environ['XDG_CONFIG_HOME'] == str(run / 'opencode-config')
assert os.environ['NANO_GPT_API_KEY'] == 'TEST_ONLY_SECRET'
assert 'OPENCODE_PERMISSION' not in os.environ
(run / 'received.json').write_text(json.dumps(dict(argv=sys.argv[1:], prompt=sys.stdin.read())))
if os.environ.get('FAKE_EXIT'):
    print(json.dumps(dict(type='error', error=dict(name='APIError'))))
    sys.exit(int(os.environ['FAKE_EXIT']))
print(json.dumps(dict(type='text', part=dict(messageID='m1', text='SMOKE_OK'))))
print(json.dumps(dict(type='step_finish', part=dict(reason='stop', cost=0.01))))
''')
        agent.chmod(0o755)
        gpu = self.root / 'fake-python'
        gpu.write_text('#!/bin/sh\nprintf "GPU: test fixture\\n"\n')
        gpu.chmod(0o755)
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ('NANO_GPT_API_KEY', 'GAN_ENGINE', 'FAKE_EXIT')}
        self.env.update(OPENCODE_BIN=str(agent), GAN_PYTHON=str(gpu),
                        NANOGPT_API_KEY='TEST_ONLY_SECRET', OPENCODE_PERMISSION='{"task":"allow"}')

    def git(self, *args):
        return subprocess.run(['git', '-C', str(self.repo), *args], check=True, capture_output=True)

    def launch(self, *args):
        return subprocess.run(['bash', str(self.root / 'try-gan.sh'), '--engine', 'opencode',
                               '--repo', str(self.repo), '--prompt-file', str(self.brief),
                               '--runs-dir', str(self.root / 'runs'), *args],
                              env=self.env, text=True, capture_output=True, timeout=20)

    def test_launch_routes_prompt_model_config_and_artifacts(self):
        result = self.launch('--model', 'meta/muse-spark-1.3-contributor')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run, = (self.root / 'runs').iterdir()
        received = json.loads((run / 'received.json').read_text())
        self.assertEqual(received['argv'], ['run', '--pure', '--model',
                         'nano-gpt/meta/muse-spark-1.3-contributor', '--agent', 'build',
                         '--format', 'json', '--dir', str(run / 'repo')])
        self.assertIn('INTEGRATION_TEST_PROMPT', received['prompt'])
        self.assertEqual((run / 'final.md').read_text(), 'SMOKE_OK\n')
        self.assertEqual((run / 'exit-code.txt').read_text(), '0\n')
        self.assertTrue(json.loads((run / 'opencode-usage.json').read_text())['completed'])
        self.assertIn('SMOKE_OK', (run / 'opencode.log').read_text())
        for path in run.iterdir():
            if path.is_file():
                self.assertNotIn('TEST_ONLY_SECRET', path.read_text(), str(path))

    def test_agent_failure_reaches_launcher_status(self):
        self.env['FAKE_EXIT'] = '17'
        result = self.launch()
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
        run, = (self.root / 'runs').iterdir()
        self.assertIn('failed', (run / 'status.txt').read_text())
        self.assertFalse(json.loads((run / 'opencode-usage.json').read_text())['completed'])

    def test_pause_allows_dry_run_but_blocks_real_attempts(self):
        stop = self.root / 'gan-attempts' / 'STOP'
        stop.parent.mkdir()
        stop.write_text('Paused\n')
        result = self.launch('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('nano-gpt/meta/muse-spark-1.3-contributor', result.stdout)
        self.assertNotIn('TEST_ONLY_SECRET', result.stdout)
        self.assertFalse((self.root / 'runs').exists())
        self.assertEqual(self.launch().returncode, 75)
        self.assertEqual(self.launch('--help').returncode, 0)
        self.assertFalse((self.root / 'runs').exists())

    def test_budget_is_still_claude_only(self):
        result = self.launch('--budget-usd', '1')
        self.assertEqual(result.returncode, 2)
        self.assertIn('claude only', result.stderr)
        self.assertFalse((self.root / 'runs').exists())

    def test_pool_uses_opencode_default_without_claude_budget(self):
        lanes = self.root / 'lanes.py'
        lanes.write_text("LANES = {'probe': 'A test lane'}\nREPO = 'unused-in-dry-run'\n")
        result = subprocess.run(['python3', str(SOURCE / 'run-gan-pool.py'),
                                 '--engine', 'opencode', '--lanes-from', str(lanes), '--dry-run'],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['engine'], 'opencode')
        self.assertEqual(plan['model'], '')
        self.assertEqual(plan['budget_usd'], 0)

    def test_dashboard_finds_opencode_log_and_cost(self):
        spec = importlib.util.spec_from_file_location('monitor', SOURCE / 'monitor-gan.py')
        monitor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(monitor)
        self.assertIn('opencode.log', monitor.AGENT_LOGS)
        (self.root / 'opencode-usage.json').write_text('{"total_cost_usd": 0.0123}')
        self.assertEqual(monitor.run_cost(self.root), 0.0123)


if __name__ == '__main__':
    unittest.main()
