#!/usr/bin/env python3
"""Offline launch checks: no model request, GPU training, or worktree creation."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
launcher = types.ModuleType('continuous_api_launcher')
launcher.__file__ = str(HERE / 'launch.py')
exec(compile((HERE / 'launch.py').read_text(), launcher.__file__, 'exec'), launcher.__dict__)
CONFIG = json.loads((HERE / 'config.json').read_text())


class LaunchChecks(unittest.TestCase):
    def test_three_lanes_and_exact_driver_dry_run(self):
        launcher.validate_config(CONFIG)
        with patch.object(launcher, 'live_drivers', return_value=[]):
            _, plans = launcher.build_plan(CONFIG, list(CONFIG['lanes']), [], '')
        self.assertEqual(len(plans), 3)
        self.assertEqual(len({p['lane'] for p in plans}), 3)
        with tempfile.TemporaryDirectory() as temporary:
            for plan in plans:
                command = plan['command'].copy()
                command[command.index('--runs-dir') + 1] = str(Path(temporary) / plan['lane'])
                # Driver dry-run needs an existing prompt; it never starts Codex.
                command[command.index('--prompt-file') + 1] = str(HERE / 'COMMON.md')
                result = subprocess.run(command + ['--dry-run'], check=True,
                                        text=True, capture_output=True)
                self.assertIn('gpt-6-astra', result.stdout)
                self.assertIn('model_reasoning_effort=', result.stdout)
                self.assertIn('max', result.stdout)
                self.assertIn('Budget: no hard timeout', result.stdout)
                self.assertIn('at most 3 candidates; 1 workers', result.stdout)
                self.assertNotIn('timeout --signal', result.stdout)
                self.assertIn('NO fixed 81/81', plan['prompt'])
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_global_worker_capacity(self):
        with patch.object(launcher, 'live_drivers', return_value=[
                dict(pid=100, directory='/other/search', workers=2)]):
            _, plans = launcher.build_plan(CONFIG, list(CONFIG['lanes']), [], '')
        self.assertEqual(len(plans), 1)
        with patch.object(launcher, 'live_drivers', return_value=[
                dict(pid=100, directory='/other/search', workers=3)]):
            _, plans = launcher.build_plan(CONFIG, list(CONFIG['lanes']), [], '')
        self.assertEqual(plans, [])

    def test_active_lane_is_not_duplicated(self):
        name = next(iter(CONFIG['lanes']))
        record = dict(pid=100, lane=name, directory='/active/search')
        with patch.object(launcher, 'live_drivers', return_value=[
                dict(pid=100, directory='/active/search', workers=1)]):
            _, plans = launcher.build_plan(CONFIG, list(CONFIG['lanes']), [record], '')
        self.assertEqual(len(plans), 2)
        self.assertNotIn(name, [p['lane'] for p in plans])

    def test_replenishment_requires_review(self):
        name = next(iter(CONFIG['lanes']))
        record = dict(pid=100, lane=name, directory='/nonexistent/finished/search')
        with patch.object(launcher, 'live_drivers', return_value=[]):
            _, plans = launcher.build_plan(CONFIG, [name], [record], '')
            self.assertTrue(plans[0]['requires_review'])
            _, plans = launcher.build_plan(CONFIG, [name], [record], 'Measured failure; try the specified successor.')
            self.assertFalse(plans[0]['requires_review'])

    def test_stop_prevents_spawn(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(CONFIG)
            config['workspace'] = temporary
            config['runs_dir'] = str(Path(temporary) / 'gan-attempts/search')
            stop = Path(temporary) / 'gan-attempts/STOP'
            stop.parent.mkdir()
            stop.write_text('Stop for offline verification.\n')
            original = launcher.read_json

            def read_json(path, default):
                return config if path == HERE / 'config.json' else original(path, default)

            with patch.object(launcher, 'read_json', side_effect=read_json), \
                    patch.object(launcher, 'live_drivers', return_value=[]), \
                    patch.object(launcher.subprocess, 'Popen') as spawn, \
                    patch.object(launcher, 'validate_config'), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    launcher.main(['--launch-after-compaction'])
                self.assertEqual(raised.exception.code, 2)
                spawn.assert_not_called()
            self.assertTrue(stop.exists())
            self.assertFalse(Path(config['runs_dir']).exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
