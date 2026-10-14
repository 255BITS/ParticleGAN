"""No model calls or training: concurrency, crash fencing, leases and dry-run controls."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, MagicMock

from experiments.forge import codex_runner as runner
from experiments.forge.contracts import atomic_json, read_json

ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.ledger = self.directory / 'old-ledger.json'
        self.baseline = dict(remaining_queue_seconds=2337.4746000149316,
                             remaining_wall_seconds=746.1211251119094, reservations=0,
                             historical_receipts=['old-error', 'old-negative', 'old-positive'])
        atomic_json(self.ledger, self.baseline)
        self.spending = runner.Spending(self.directory / 'shared-budget.json')
        self.queue = self.directory / 'queue'
        self.spending.initialize(self.ledger, queue_root=self.queue, agent_seconds=60, agent_sessions=2)
        self.study = dict(id='distinct-study', hypothesis='A specified restoring gradient response reduces drift.',
                          falsifier='Original stationary/shift gates remain false.', cohort='clean-original-mog',
                          semantic_factors={'law': 'fixed-response', 'coefficient': 0.875},
                          task_ids=['gaussian1d_smoke', 'gaussian1d_stability'], request_path='/not-executed/request.json')
        self.config = dict(schema_version=1, slots=12, max_studies=1, studies=[self.study],
                           budget_state=str(self.spending.path), queue_root=str(self.queue),
                           runs_dir=str(self.directory / 'runs'), lease_root=str(self.directory / 'leases'),
                           stop_files=[], source_commit='a' * 40)
        self.path = self.directory / 'config.json'

    def plan(self):
        atomic_json(self.path, self.config)
        return runner.plan(self.path, root=ROOT)[1]

    def test_original_full_pair_cannot_be_shortened_to_fit(self):
        result = self.plan()
        self.assertEqual(result['total_full_reservation'], dict(queue=720, supervisor=760, agent=0))
        self.assertIn('aggregate supervisor reservation exceeds remaining balance', result['blockers'])
        self.assertEqual(result['launches'], 0)

    def test_original_shared_execution_group_is_reserved_once(self):
        frozen = runner.contract(ROOT)
        for task in self.study['task_ids']:
            frozen['tasks'][task]['execution']['execution_group'] = 'shared-software-fixture'
        atomic_json(self.path, self.config)
        with patch.object(runner, 'contract', return_value=frozen):
            report = runner.plan(self.path, root=ROOT)[1]
        self.assertEqual(report['total_full_reservation']['queue'], 600)
        self.assertEqual(report['total_full_reservation']['supervisor'], 620)

    def test_dry_run_does_not_launch_or_write_history(self):
        before = self.spending.path.read_bytes()
        atomic_json(self.path, self.config)
        with patch.object(runner, 'execute', side_effect=AssertionError('must not launch')):
            with patch.object(runner.subprocess, 'Popen', side_effect=AssertionError('must not spawn')):
                self.assertEqual(runner.main(['--root', str(ROOT), 'pool', '--config', str(self.path), '--execute', '--dry-run']), 0)
        self.assertEqual(before, self.spending.path.read_bytes())

    def test_aliases_do_not_create_distinct_studies(self):
        alias = {**self.study, 'id': 'alias', 'hypothesis': 'new wording', 'source_commit': 'b' * 40,
                 'seed': 5, 'cohort': 'same-law-new-label'}
        self.assertEqual(runner.semantic_key(alias), runner.semantic_key(self.study))
        self.config.update(max_studies=2, studies=[self.study, alias])
        with self.assertRaisesRegex(ValueError, 'duplicate study'):
            self.plan()

    def test_finite_list_and_slot_caps(self):
        for slots in (0, 13, True):
            self.config['slots'] = slots
            with self.assertRaises(ValueError):
                self.plan()
        self.config['slots'] = 12
        self.config['max_studies'] = 0
        with self.assertRaises(ValueError):
            self.plan()

    def test_no_path_traversal_or_borrowed_checkpoint(self):
        self.study['id'] = '../other-owner'
        with self.assertRaises(ValueError):
            self.plan()
        self.study['id'] = 'safe'
        self.study['task_ids'] = ['gaussian1d_stability']
        with self.assertRaisesRegex(ValueError, 'prerequisite'):
            self.plan()

    def test_state_cannot_reset_or_duplicate_the_same_queue_balance(self):
        old = self.ledger.read_bytes()
        for state in (self.spending, runner.Spending(self.directory / 'second-budget.json')):
            with self.assertRaises(ValueError):
                state.initialize(self.ledger, queue_root=self.queue)
        self.assertEqual(old, self.ledger.read_bytes())
        with self.assertRaises(ValueError):
            runner.Spending(self.directory / 'another.json').initialize(self.ledger, queue_root=self.directory / 'other-queue')

    def test_atomic_competing_reservations_and_crash_fencing(self):
        def reserve(index):
            try:
                self.spending.reserve(str(index), str(index), dict(queue=1, supervisor=500, agent=0))
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(reserve, range(2)))
        self.assertEqual(sum(outcomes), 1)
        self.assertEqual(len(read_json(self.spending.path)['reservations']), 1)
        self.assertIn('existing reservations require owner reconciliation; no automatic retry', self.plan()['blockers'])

    def test_full_cohort_is_reserved_atomically_before_streaming(self):
        entries = [dict(key=str(n), semantic=str(n), amounts=dict(queue=120, supervisor=140, agent=0)) for n in range(2)]
        self.spending.reserve_cohort(entries)
        state = read_json(self.spending.path)
        self.assertEqual(len(state['reservations']), 2)
        self.assertEqual(state['events'][-1]['amounts']['supervisor'], 280)
        before = self.spending.path.read_bytes()
        with self.assertRaises(ValueError):
            self.spending.reserve_cohort([dict(key='new', semantic='new', amounts=dict(queue=1, supervisor=1, agent=0)), entries[0]])
        self.assertEqual(before, self.spending.path.read_bytes())

    def test_explicit_six_six_mapping_uses_exact_existing_tokens(self):
        self.config['gpu_allocation'] = {'0': 6, '1': 6}
        layout = runner.deployment_layout(self.config)
        self.assertEqual([(r[0], r[1]) for r in layout], runner.SLOT_LAYOUT)
        self.assertEqual([(r[2], r[3]) for r in layout], [(0,n) for n in range(6)] + [(1,n) for n in range(6)])
        for allocation in ({'0':6,'1':5}, {'0':-1,'1':13}, {'0':True,'1':11}, {'0':6,'2':6}):
            self.config['gpu_allocation'] = allocation
            with self.assertRaises(ValueError): runner.deployment_layout(self.config)

    def test_occupied_remapped_fleet_refuses_before_any_budget_or_job(self):
        self.config['gpu_allocation'] = {'0':6,'1':6}
        lease_root = Path(self.config['lease_root']); lease_root.mkdir()
        for gpu, slot in runner.SLOT_LAYOUT:
            (lease_root / f'gpu{gpu}-slot{slot}.lock').write_bytes(b'preserved')
        before = self.spending.path.read_bytes()
        with (lease_root / 'gpu0-slot3.lock').open('r+b') as occupied:
            fcntl.flock(occupied, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fake = MagicMock(); fake.inspect.return_value = {'jobs': {}}
            fake.submit.side_effect = AssertionError('no partial fleet launch')
            with patch('experiments.forge.queue.Queue', return_value=fake):
                runs = runner.execute(self.config, self.plan(), ROOT)
        self.assertEqual(read_json(runs / 'catalog.json')['status'], 'BLOCKED_EXISTING_LEASE_OWNER')
        self.assertEqual(before, self.spending.path.read_bytes())

    def test_surviving_child_keeps_every_remapped_token_exclusive(self):
        from contextlib import ExitStack
        paths = []
        for gpu, slot in runner.SLOT_LAYOUT:
            path = self.directory / f'gpu{gpu}-slot{slot}.lock'; path.write_bytes(b'preserved'); paths.append(path)
        before = [(p.stat().st_ino, p.stat().st_size, p.stat().st_mode) for p in paths]
        with ExitStack() as fleet:
            owned = [fleet.enter_context(runner.Leases(self.directory).acquire(gpu, slot)) for gpu, slot in runner.SLOT_LAYOUT]
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.4)'],
                                       pass_fds=tuple(v['handle'].fileno() for v in owned))
        for path in paths:
            with path.open('r+b') as check:
                with self.assertRaises(BlockingIOError): fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process.wait(timeout=5)
        self.assertEqual(before, [(p.stat().st_ino, p.stat().st_size, p.stat().st_mode) for p in paths])
        for path in paths:
            with path.open('r+b') as check: fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_explicit_zero_admission_repair_preserves_history_and_fences_live_work(self):
        entry = dict(key='prelaunch', semantic='same-unit', amounts=dict(queue=120, supervisor=140, agent=0))
        self.spending.reserve_cohort([entry])
        with self.assertRaisesRegex(ValueError, 'alive'):
            self.spending.reconcile_prelaunch_refusal({'fixture': 'live owner'})
        with self.spending.state() as state:
            state['reservations']['prelaunch']['owner'] = '2147483647:absent'
        self.spending.reconcile_prelaunch_refusal({'fixture':'verified no jobs'})
        state = read_json(self.spending.path)
        self.assertEqual(state['events'][-1]['actual_new_attempts'],0)
        self.assertEqual(state['events'][-1]['previous_reservations']['prelaunch']['semantic'],'same-unit')
        self.spending.reserve_cohort([{**entry,'key':'repaired'}])
        atomic_json(self.queue/'queue/state.json',{'jobs':{'existing':{'status':'terminal'}},'submissions':{},'charges':[]})
        with self.assertRaisesRegex(ValueError,'empty queue'):
            self.spending.reconcile_prelaunch_refusal({'fixture':'must not release scientific work'})

    def test_actual_charges_and_negative_evidence_remain(self):
        self.spending.reserve('a', 'negative', dict(queue=100, supervisor=120, agent=10))
        self.spending.settle('a', dict(queue=21, supervisor=23, agent=4), {'status': 'FAIL', 'reason': 'original gate'})
        state = read_json(self.spending.path)
        self.assertEqual(state['initial_ledger'], self.baseline)
        self.assertEqual(state['spent'], dict(queue=21, supervisor=23, agent=4))
        self.assertEqual(state['events'][-1]['receipt']['status'], 'FAIL')
        self.assertEqual(state['events'][-1]['previous'], state['events'][-2]['sha256'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.spending.reserve('b', 'negative', dict(queue=1, supervisor=1, agent=0))

    def test_overrun_is_charged_and_blocks_further_work(self):
        self.spending.reserve('a', 'overrun', dict(queue=0, supervisor=5, agent=0))
        self.spending.settle('a', dict(queue=0, supervisor=800, agent=0), {'status': 'timeout'})
        self.assertLess(runner.Spending.remaining(read_json(self.spending.path))['supervisor'], 0)
        self.assertTrue(read_json(self.spending.path)['events'][-1]['exceeded']['supervisor'])

    def test_model_cap_is_separate_explicit_and_bounded(self):
        for index in range(2):
            self.spending.reserve(str(index), str(index), dict(queue=0, supervisor=0, agent=10), agent=True)
            self.spending.settle(str(index), dict(queue=0, supervisor=0, agent=1), {'status': 'turn_failed'})
        with self.assertRaisesRegex(ValueError, 'model session cap'):
            self.spending.reserve('3', '3', dict(queue=0, supervisor=0, agent=1), agent=True)

    def test_standard_command_has_no_permission_or_speed_bypass(self):
        command = runner.codex_command('codex', self.directory, self.directory)
        self.assertIn('gpt-6.1-sol', command)
        self.assertIn('model_reasoning_effort="xhigh"', command)
        self.assertIn('--approve-for-me', command)
        self.assertIn('workspace-write', command)
        self.assertNotIn('--ignore-rules', command)
        self.assertNotIn('--ask-for-approval', command)
        self.assertFalse(any('approval_policy=' in value for value in command))
        self.assertIn('service_tier="default"', command)
        self.assertFalse(any('service_tier="fast"' in item or 'danger' in item for item in command))
        (self.directory / '.codex').mkdir()
        (self.directory / '.codex/config.toml').write_text('service_tier="fast"\n')
        with self.assertRaisesRegex(ValueError, 'speed override'):
            runner.codex_command('codex', self.directory, self.directory)

    def test_installed_cli_identifiers_are_checked_without_generation(self):
        catalog = {'models': [{'slug': runner.MODEL, 'supported_reasoning_levels': [{'effort': 'xhigh'}]}]}
        help_text = '--approve-for-me --ignore-user-config --ephemeral --sandbox --json'
        with patch.object(runner.subprocess, 'check_output', side_effect=[help_text, json.dumps(catalog)]) as mocked:
            self.assertEqual(runner.verify_cli('codex')['generation_sessions'], 0)
            self.assertIn('--bundled', mocked.call_args_list[-1].args[0])
        with patch.object(runner.subprocess, 'check_output', side_effect=[help_text, '{"models":[]}']):
            with self.assertRaises(ValueError):
                runner.verify_cli('codex')

    def test_missing_lease_is_not_created(self):
        lease = runner.Leases(self.directory)
        with self.assertRaises(FileNotFoundError):
            with lease.acquire(0, 0):
                pass
        self.assertFalse((self.directory / 'gpu0-slot0.lock').exists())

    def test_child_holds_preserved_lease_after_coordinator_closes_fd(self):
        path = self.directory / 'gpu0-slot0.lock'
        path.write_bytes(b'preserve')
        before = path.stat()
        with runner.Leases(self.directory).acquire(0, 0) as lease:
            process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.4)'], pass_fds=(lease['handle'].fileno(),))
        with path.open('r+b') as check:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
            process.wait(timeout=5)
            fcntl.flock(check, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.assertEqual((before.st_ino, before.st_size, before.st_mode), (path.stat().st_ino, path.stat().st_size, path.stat().st_mode))
        self.assertEqual(path.read_bytes(), b'preserve')

    def test_queue_launch_passes_both_leases_without_changing_legacy_default(self):
        from experiments.forge.queue import Queue
        queue = Queue(self.directory / 'test-queue')
        directory = self.directory / 'attempt'
        directory.mkdir()
        worker = dict(directory=str(directory), token='fence', device='0')
        resolved = {'worker': worker, 'job': {'compatibility_key': 'key'}, 'request': {'source': {'snapshot_path': str(ROOT)}}}
        state = {'jobs': {'key': {'status': 'running', 'worker': dict(worker)}}}
        @contextmanager
        def state_context():
            yield state
        with patch.object(queue, 'state', state_context), patch('experiments.forge.queue.subprocess.Popen') as popen:
            popen.return_value.pid = os.getpid()
            with (self.directory / 'shared.lock').open('w+b') as fd:
                queue.launch(resolved, shared_slot_fd=fd.fileno())
                self.assertIn(fd.fileno(), popen.call_args.kwargs['pass_fds'])
                self.assertEqual(len(popen.call_args.kwargs['pass_fds']), 2)
                self.assertEqual(popen.call_args.kwargs['env']['FORGE_SHARED_SLOT_FD'], str(fd.fileno()))

    def test_worker_forwards_all_fleet_descriptors_and_rejects_stale_fd(self):
        from experiments.forge.worker import inherited_lease_fds
        with (self.directory / 'one').open('w+b') as one, (self.directory / 'two').open('w+b') as two:
            env = dict(FORGE_LEASE_FD=str(one.fileno()), FORGE_SHARED_SLOT_FD=str(two.fileno()),
                       FORGE_SHARED_FLEET_FDS=f'{one.fileno()},{two.fileno()}')
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(inherited_lease_fds(), (one.fileno(), two.fileno()))
            stale = two.fileno()
        with patch.dict(os.environ, {'FORGE_LEASE_FD':str(stale)}, clear=True):
            with self.assertRaises(OSError): inherited_lease_fds()

    def test_six_six_capacity_uses_original_memory_and_reports_shortfall(self):
        self.config.update(gpu_allocation={'0':6,'1':6}, required_concurrent_workers=12)
        request_path = self.directory / 'capacity-request.json'
        atomic_json(request_path, {'jobs':[{'resources':{'memory_mb':2048, 'gpu_model':'A6000'}}]})
        self.study['request_path'] = str(request_path)
        def slots(devices, workers_per_gpu, **kwargs):
            self.assertEqual(workers_per_gpu,6)
            return [{'device':devices[0], 'slot':n, 'model':'A6000', 'memory_mb':2048} for n in range(6)]
        with patch('experiments.forge.queue.device_slots', side_effect=slots):
            result = runner.concurrency_capacity(self.config)
        self.assertEqual(result['available_under_current_native_partition'],12)
        self.assertEqual([r['required_free_memory_mb'] for r in result['devices']],[12800,12800])
        def short(devices, workers_per_gpu, **kwargs):
            return [{**s, 'memory_mb':1900 if devices[0]=='0' else 2048} for s in slots(devices,workers_per_gpu)]
        with patch('experiments.forge.queue.device_slots', side_effect=short):
            self.assertEqual(runner.concurrency_capacity(self.config)['available_under_current_native_partition'],6)

    def test_original_contract_sections_and_seed_are_enforced(self):
        frozen = runner.contract(ROOT)
        request = {'view': frozen['view'], 'protocol': {'seed': 0}, 'jobs': [{'task_ids': self.study['task_ids'], 'task_id': self.study['task_ids'][0]}],
                   'tasks': frozen['tasks'], 'study_admission': {'bound': True}, 'study_review': {'status': 'READY'},
                   'source': {'digest': 'bound'}, 'candidate_revision': 'revision'}
        runner.validate_request(request, self.study, frozen)
        for section in ('execution', 'evaluation', 'resources', 'dependencies'):
            altered = deepcopy(request)
            altered['tasks']['gaussian1d_smoke'][section] = {'changed': True}
            with self.assertRaisesRegex(ValueError, 'changed'):
                runner.validate_request(altered, self.study, frozen)
        request['protocol']['seed'] = 1
        with self.assertRaisesRegex(ValueError, 'seed'):
            runner.validate_request(request, self.study, frozen)

    def test_revision9_view_archive_is_byte_exact(self):
        data = subprocess.check_output(['git', '-C', str(ROOT), 'show', 'dacf96749a4effc0648a7a0945266d508804110e:configs/forge/views/discriminator_stability.json'])
        self.assertEqual(data, (ROOT / 'configs/forge/view-history/discriminator_stability-v9.json').read_bytes())

    def test_stop_file_prevents_replenishment(self):
        stop = self.directory / 'STOP'
        self.config['stop_files'] = [str(stop)]
        self.assertFalse(runner.stopped(self.config))
        stop.touch()
        self.assertTrue(runner.stopped(self.config))

    def test_queue_claim_is_fenced_to_our_submission(self):
        from experiments.forge.queue import Queue
        from tests.test_forge_queue import request, campaign, grade, SLOTS
        queue = Queue(self.directory / 'shared-queue', grader=grade)
        foreign = request(self.directory, candidate='other-owner', cap=1)
        queue.submit(foreign, campaign())
        ours = deepcopy(foreign)
        ours['candidate']['id'] = 'our-declared-study'
        for job in ours['jobs']:
            job['compatibility_key'] += '-ours'
        admitted = queue.submit(ours, campaign())
        claim = queue.claim(SLOTS, request_filter=admitted['request']['request_id'])
        self.assertEqual(claim['request']['candidate']['id'], 'our-declared-study')
        self.assertEqual(queue.inspect()['jobs'][foreign['jobs'][0]['compatibility_key']]['status'], 'pending')

    def test_mock_end_to_end_records_supervisor_cost_not_fictitious_training(self):
        frozen = runner.contract(ROOT)
        self.study['task_ids'] = ['gaussian1d_smoke']
        request_path = self.directory / 'resolved.json'
        self.study['request_path'] = str(request_path)
        directory = self.directory / 'attempt'
        directory.mkdir()
        req = dict(view=frozen['view'], protocol={'seed': 0}, jobs=[{'task_ids': ['gaussian1d_smoke'], 'task_id': 'gaussian1d_smoke'}],
                   tasks=frozen['tasks'], study_admission={'bound': True}, study_review={'status': 'READY'},
                   candidate_revision='candidate', source={'digest': 'source', 'snapshot_path': str(ROOT),
                   'files': {f'experiments/forge/{name}': runner.digest(ROOT / f'experiments/forge/{name}') for name in ['worker.py', 'queue.py']}},
                   study={'campaign': {'id': 'ours'}})
        atomic_json(request_path, req)
        lease_root = Path(self.config['lease_root'])
        lease_root.mkdir()
        for gpu, slot in runner.SLOT_LAYOUT:
            (lease_root / f'gpu{gpu}-slot{slot}.lock').write_bytes(b'preserved')
        class FakeQueue:
            def __init__(self):
                self.launched = False
                self.claimed = False
                self.entry = {'request': {'request_id': 'ours'}, 'status': 'queued'}
            def inspect(self):
                return {'jobs': {'job': {'status': 'terminal' if self.launched else 'pending'}},
                        'submissions': {'ours': self.entry},
                        'charges': [{'attempt_id': 'fake', 'seconds': 2}] if self.launched else []}
            def submit(self, *args):
                return self.entry
            def claim(self, *args, **kwargs):
                if self.claimed:
                    return None
                assert kwargs['request_filter'] == 'ours'
                self.claimed = True
                return {'worker': {'directory': str(directory), 'attempt': 'fake'}, 'job': {'budget_seconds': 120, 'task_id': 'gaussian1d_smoke'}}
            def launch(self, *args, **kwargs):
                assert kwargs['shared_slot_fd'] >= 0
                self.launched = True
                process = MagicMock(pid=os.getpid(), returncode=0)
                process.poll.return_value = 0
                return process
            def collect(self):
                if self.launched:
                    self.entry['status'] = 'completed'
        fake = FakeQueue()
        report = self.plan()
        self.assertFalse(report['blockers'])
        with patch('experiments.forge.queue.Queue', return_value=fake), patch('experiments.forge.sources.verify_snapshot'), \
             patch('experiments.forge.queue.device_slots', return_value=[{'slot': n} for n in range(10)]), \
             patch('experiments.forge.test_stages.generated_reports', return_value={}):
            runs = runner.execute(self.config, report, ROOT)
        catalog = read_json(runs / 'catalog.json')
        self.assertEqual(catalog['training_starts'], [])
        self.assertFalse(catalog['additional_qualification'])
        self.assertEqual(catalog['receipts'][0]['attempts'][0]['attempt'], 'fake')
        self.assertEqual(read_json(self.spending.path)['spent']['queue'], 2)
        self.assertFalse(read_json(self.spending.path)['reservations'])

    def mock_proposal(self, events):
        runs = self.directory / 'proposal-runs'
        runs.mkdir()
        self.study.pop('request_path')
        self.study['agent_seconds'] = 10
        row = {'semantic': 'unique', 'amounts': {'agent': 10}}
        frozen = runner.contract(ROOT)
        def worktree(command, **kwargs):
            Path(command[-2]).mkdir()
        def fake_process(command, **kwargs):
            for event in events:
                kwargs['stdout'].write((json.dumps(event) + '\n').encode())
            process = MagicMock(pid=os.getpid(), returncode=0)
            return process
        with patch.object(runner.subprocess, 'check_output', return_value=self.config['source_commit']), \
             patch.object(runner.subprocess, 'run', side_effect=worktree), \
             patch.object(runner.subprocess, 'Popen', side_effect=fake_process), \
             patch.object(runner, 'contract', return_value=frozen):
            return runner.propose(self.config, self.study, row, ROOT, runs)

    def test_prepared_work_launches_before_last_study_admission(self):
        frozen = runner.contract(ROOT)
        order, entries, launched, claimed = [], {}, set(), set()
        studies = []
        for index in range(2):
            rid = f'prepared-{index}'
            path = self.directory / f'{rid}.json'
            study = {**self.study, 'id': rid, 'task_ids': ['gaussian1d_smoke'],
                     'semantic_factors': {'contrast': index}, 'request_path': str(path)}
            atomic_json(path, dict(view=frozen['view'], protocol={'seed': 0},
                jobs=[{'task_id': 'gaussian1d_smoke', 'task_ids': ['gaussian1d_smoke'], 'compatibility_key': rid}],
                tasks=frozen['tasks'], study_admission={'bound': True}, study_review={'status': 'READY'},
                candidate_revision='candidate', source={'digest': 'source', 'snapshot_path': str(ROOT),
                    'files': {f'experiments/forge/{name}': runner.digest(ROOT / f'experiments/forge/{name}') for name in ['queue.py', 'worker.py']}},
                study={'campaign': {'id': rid}}))
            (self.directory / rid).mkdir()
            studies.append(study)
        self.config.update(studies=studies, max_studies=2)
        lease_root = Path(self.config['lease_root']); lease_root.mkdir()
        for gpu, slot in runner.SLOT_LAYOUT:
            (lease_root / f'gpu{gpu}-slot{slot}.lock').write_bytes(b'preserved')
        class FakeQueue:
            def inspect(self):
                return {'jobs': {rid: {'status': 'terminal' if rid in launched else 'pending'} for rid in entries},
                        'submissions': entries, 'charges': [{'attempt_id': rid, 'seconds': 1} for rid in launched]}
            def submit(self, request, campaign):
                rid = campaign['id']; order.append(('submit', rid))
                entries[rid] = {'request': {'request_id': rid}, 'status': 'queued'}
                return entries[rid]
            def claim(self, slots, **filters):
                rid = filters['request_filter']
                if rid in claimed: return None
                claimed.add(rid)
                return {'worker': {'attempt': rid, 'directory': str(self_directory / rid)},
                        'job': {'budget_seconds': 120, 'task_id': 'gaussian1d_smoke'}}
            def launch(self, claim, **kwargs):
                rid = claim['worker']['attempt']; order.append(('launch', rid)); launched.add(rid)
                process = MagicMock(pid=os.getpid(), returncode=0); process.poll.return_value = 0
                return process
            def collect(self):
                for rid in launched: entries[rid]['status'] = 'completed'
        self_directory = self.directory
        with patch('experiments.forge.queue.Queue', return_value=FakeQueue()), \
             patch('experiments.forge.sources.verify_snapshot'), \
             patch('experiments.forge.queue.device_slots', return_value=[{'slot': n} for n in range(10)]), \
             patch('experiments.forge.test_stages.generated_reports', return_value={}):
            runs = runner.execute(self.config, self.plan(), ROOT)
        self.assertLess(order.index(('launch', 'prepared-0')), order.index(('submit', 'prepared-1')))
        self.assertEqual(read_json(runs / 'catalog.json')['peak_concurrent_training'], 0)
        self.assertFalse(read_json(self.spending.path)['reservations'])

    def test_capacity_blocker_precedes_admission_reservation_and_launch(self):
        self.config['required_concurrent_workers'] = 12
        before = self.spending.path.read_bytes()
        capacity = {'required': 12, 'available_under_current_native_partition': 2}
        with patch.object(runner, 'concurrency_capacity', return_value=capacity), \
             patch('experiments.forge.queue.Queue', side_effect=AssertionError('no admission')), \
             patch('experiments.forge.queue.subprocess.Popen', side_effect=AssertionError('no launch')):
            runs = runner.execute(self.config, self.plan(), ROOT)
        self.assertEqual(read_json(runs / 'catalog.json')['status'], 'BLOCKED_EXTERNAL_GPU_CAPACITY')
        self.assertEqual(before, self.spending.path.read_bytes())

    def test_mock_codex_proposal_records_cli_and_observed_turns_separately(self):
        path, elapsed = self.mock_proposal([{'type': 'turn.started'}, {'type': 'turn.completed', 'usage': {'input_tokens': 4}}])
        terminal = read_json(path.parent.parent / 'agent-terminal.json')
        start = read_json(path.parent.parent / 'agent-start.json')
        self.assertEqual(start['kind'], 'CODEX_CLI_PROCESS_STARTED_NOT_TRAINING')
        self.assertEqual(terminal['observed_model_turn_starts'], 1)
        self.assertTrue(terminal['completion_observed'])
        self.assertEqual(terminal['training_starts'], 0)
        self.assertIsNone(terminal['dollar_cost'])

    def test_mock_failed_turn_is_not_eligible_even_if_cli_exit_is_zero(self):
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            self.mock_proposal([{'type': 'turn.started'}, {'type': 'turn.failed', 'error': {'message': 'fixture'}}])
        terminal = read_json(self.directory / 'proposal-runs' / self.study['id'] / 'agent-terminal.json')
        self.assertFalse(terminal['completion_observed'])
        self.assertGreaterEqual(terminal['seconds'], 0)


if __name__ == '__main__':
    unittest.main()
