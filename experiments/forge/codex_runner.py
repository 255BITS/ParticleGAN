"""Bounded Codex proposals and ordinary Forge attempts; planning never launches either."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
from copy import deepcopy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .contracts import atomic_json, file_lock, identifier, read_json, stable_hash

MODEL = 'gpt-6.1-sol'
EFFORT = 'xhigh'
SLOT_LAYOUT = [(0, n) for n in range(10)] + [(1, n) for n in range(2)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, label, *, zero=True):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (not zero and not value):
        raise ValueError(f'{label} must be a finite {"nonnegative" if zero else "positive"} number')
    return value


def process_identity(pid):
    try:
        return str(pid) + ':' + Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


class Spending:
    """One shared, immutable baseline with append-only event history and fenced reservations.

    Unsettled reservations survive a crash. A new process never guesses them free.
    All users must share this state and the queue; it is not a per-pool allowance.
    """
    def __init__(self, path):
        self.path = Path(path).resolve()

    @contextmanager
    def state(self):
        with file_lock(self.path.with_suffix('.lock')):
            value = read_json(self.path)
            before = deepcopy(value)
            yield value
            if before != value:
                atomic_json(self.path, value)

    @staticmethod
    def event(state, kind, **fields):
        row = dict(kind=kind, unix=time.time(), **fields)
        row['previous'] = state['events'][-1]['sha256'] if state['events'] else None
        row['sha256'] = stable_hash(row)
        state['events'].append(row)

    def initialize(self, ledger, *, queue_root, agent_seconds=0, agent_sessions=0, history=()):
        ledger = Path(ledger).resolve()
        baseline = read_json(ledger)
        if baseline.get('reservations') != 0:
            raise ValueError('reconcile existing reservations before importing a baseline')
        chain, seen, current = [], set(), ledger
        while current not in seen:
            seen.add(current)
            value = read_json(current)
            chain.append(dict(path=str(current), sha256=digest(current)))
            previous = value.get('preceding_ledger')
            if not previous:
                break
            current = Path(previous['path'])
            if digest(current) != previous['sha256']:
                raise ValueError('historical ledger hash mismatch')
        limits = dict(queue=number(baseline['remaining_queue_seconds'], 'queue balance'),
                      supervisor=number(baseline['remaining_wall_seconds'], 'supervisor balance'),
                      agent=number(agent_seconds, 'explicit model-session seconds'))
        if type(agent_sessions) is not int or agent_sessions < 0:
            raise ValueError('model session limit must be a nonnegative integer')
        history = list(history)
        if any(not isinstance(key, str) or not key for key in history):
            raise ValueError('semantic history must explicitly list normalized keys, including negatives')
        queue_root = Path(queue_root).resolve()
        # A second state for this same queue must not mint the same balance twice.
        ledger_binding = ledger.with_suffix(ledger.suffix + '.runner-owner.json')
        with file_lock(ledger.with_suffix(ledger.suffix + '.runner-owner.lock')), file_lock(queue_root / 'runner-budget-binding.lock'), file_lock(self.path.with_suffix('.lock')):
            if self.path.exists():
                raise ValueError('shared spending state already exists; never reset historical charges')
            binding = queue_root / 'runner-budget-binding.json'
            if binding.exists() or ledger_binding.exists():
                raise ValueError('this shared queue already has a budget owner; reuse its existing state')
            value = dict(schema_version=1, initial_ledger=deepcopy(baseline), ledger_chain=chain,
                         queue_root=str(Path(queue_root).resolve()), limits=limits,
                         spent=dict(queue=0., supervisor=0., agent=0.), reservations={},
                         seen_semantics={key: {'status': 'historical'} for key in history},
                         imported_semantic_history=history, semantic_history_sha256=stable_hash(history),
                         agent_session_limit=agent_sessions, agent_sessions_started=0, events=[])
            self.event(value, 'baseline_imported', ledger=chain[0], new_training_funds=0,
                       explicit_model_seconds=agent_seconds, explicit_model_sessions=agent_sessions)
            atomic_json(binding, dict(state=str(self.path), baseline=chain[0]))
            atomic_json(ledger_binding, dict(state=str(self.path), queue_root=str(queue_root), baseline=chain[0]))
            atomic_json(self.path, value)

    @staticmethod
    def remaining(state):
        return {k: state['limits'][k] - state['spent'][k] - sum(r['amounts'][k] for r in state['reservations'].values())
                for k in state['limits']}

    @staticmethod
    def semantic_available(state, semantic):
        return semantic not in state['seen_semantics'] or state['seen_semantics'][semantic].get('status') == 'prelaunch_refusal_reconciled'

    def reserve(self, key, semantic, amounts, *, agent=False):
        with self.state() as state:
            if not self.semantic_available(state, semantic):
                raise ValueError('duplicate semantic factors, including terminal negatives/uncertain attempts')
            if agent and state['agent_sessions_started'] >= state['agent_session_limit']:
                raise ValueError('aggregate model session cap exhausted')
            for kind in state['limits']:
                number(amounts[kind], kind)
                if amounts[kind] > self.remaining(state)[kind]:
                    raise ValueError(f'cannot reserve full {kind} allowance')
            state['reservations'][key] = dict(amounts=amounts, semantic=semantic,
                                             owner=process_identity(os.getpid()), started_at=time.time())
            state['seen_semantics'][semantic] = dict(key=key, status='reserved')
            if agent:
                state['agent_sessions_started'] += 1
            self.event(state, 'reserved', key=key, semantic=semantic, amounts=amounts)

    def settle(self, key, actual, receipt):
        with self.state() as state:
            reservation = state['reservations'].pop(key)
            for kind in state['limits']:
                state['spent'][kind] += number(actual[kind], f'actual {kind}')
            state['seen_semantics'][reservation['semantic']]['status'] = 'terminal'
            self.event(state, 'settled', key=key, actual=actual, receipt=receipt,
                       exceeded={k: actual[k] > reservation['amounts'][k] for k in actual})

    def reserve_cohort(self, entries):
        """One atomic full-cohort admission, including every chain's continuation."""
        with self.state() as state:
            semantics = [entry['semantic'] for entry in entries]
            if len(set(semantics)) != len(semantics) or any(not self.semantic_available(state, s) for s in semantics):
                raise ValueError('duplicate semantic unit in full cohort, including uncertain attempts')
            sessions = sum(bool(e['amounts']['agent']) for e in entries)
            if state['agent_sessions_started'] + sessions > state['agent_session_limit']:
                raise ValueError('aggregate model session cap exhausted')
            for kind in state['limits']:
                total = sum(number(e['amounts'][kind], kind) for e in entries)
                if total > self.remaining(state)[kind]:
                    raise ValueError(f'cannot reserve full cohort {kind} allowance')
            for entry in entries:
                key, semantic, amounts = entry['key'], entry['semantic'], entry['amounts']
                if key in state['reservations']:
                    raise ValueError('reservation key already exists')
                state['reservations'][key] = dict(amounts=amounts, semantic=semantic,
                    owner=process_identity(os.getpid()), started_at=time.time())
                state['seen_semantics'][semantic] = dict(key=key, status='reserved')
                self.event(state, 'reserved', key=key, semantic=semantic, amounts=amounts)
            state['agent_sessions_started'] += sessions
            self.event(state, 'full_cohort_reserved', units=len(entries),
                       amounts={k: sum(e['amounts'][k] for e in entries) for k in state['limits']})

    def reconcile_prelaunch_refusal(self, receipt):
        """Explicit zero-job repair only; terminal science/uncertain jobs stay fenced."""
        from .queue import Queue
        with self.state() as state:
            queued = Queue(Path(state['queue_root'])).inspect()
            if (queued.get('jobs') or queued.get('submissions') or queued.get('charges')
                    or state['agent_sessions_started'] or any(state['spent'].values())):
                raise ValueError('prelaunch repair requires independently empty queue and zero model/actual spend')
            for reservation in state['reservations'].values():
                owner = reservation['owner']
                if owner and process_identity(int(owner.split(':')[0])) == owner:
                    raise ValueError('prelaunch reservation owner is still alive')
            prior = deepcopy(state['reservations'])
            for key, reservation in prior.items():
                state['seen_semantics'][reservation['semantic']].update(status='prelaunch_refusal_reconciled',
                    previous_reservation=key, reason='Explicit empty-queue/no-live-owner admission repair; no scientific attempt occurred')
            state['reservations'] = {}
            self.event(state, 'prelaunch_refusal_reconciled', previous_reservations=prior,
                       receipt=receipt, actual_new_attempts=0, new_funds=0, charged=dict(state['spent']))


class Leases:
    def __init__(self, root):
        self.root = Path(root)

    @contextmanager
    def acquire(self, gpu, slot):
        path = self.root / f'gpu{gpu}-slot{slot}.lock'
        # Existing files only: never create, truncate or replace these shared inodes.
        handle = path.open('r+b')
        before = path.stat()
        if not (os.fstat(handle.fileno()).st_dev, os.fstat(handle.fileno()).st_ino) == (before.st_dev, before.st_ino):
            handle.close()
            raise ValueError('lease path changed during open')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            yield None
            return
        try:
            yield dict(handle=handle, path=str(path), inode=before.st_ino, gpu=gpu, slot=slot)
        finally:
            # Unlocking a shared open-file description would unlock a surviving child.
            # Close our descriptor instead; the child keeps ownership until it exits.
            after = path.stat()
            handle.close()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mode) != (after.st_dev, after.st_ino, after.st_size, after.st_mode):
                raise RuntimeError('shared lease metadata changed')


def contract(root):
    root = Path(root)
    view_path = root / 'configs/forge/views/discriminator_stability.json'
    view = read_json(view_path)
    if view['revision'] not in (9, 10):
        raise ValueError('unsupported frozen current view revision')
    stages = None
    files = {'configs/forge/views/discriminator_stability.json': digest(view_path)}
    if view['revision'] == 10:
        stages = read_json(root / 'configs/forge/test-stages.json')
        if stages['qualification_input'] or stages['execution_policy'] != 'shared_original_runs_no_additional_updates':
            raise ValueError('stage readout must not invent qualification or training')
        files.update({'configs/forge/test-stages.json': digest(root / 'configs/forge/test-stages.json'),
                      'configs/forge/view-history/discriminator_stability-v9.json': digest(root / 'configs/forge/view-history/discriminator_stability-v9.json')})
    names = {a['task'] for a in view['assignments']}
    tasks = {}
    for name in names:
        path = root / 'configs/forge/tasks' / f'{name}.json'
        files[str(path.relative_to(root))] = digest(path)
        tasks[name] = read_json(path)
    for pin in (stages['sources'].values() if stages else ()):
        if digest(root / pin['path']) != pin['sha256']:
            raise ValueError('original stage source task changed')
    return dict(view=view, stages=stages, tasks=tasks, files=files)


def semantic_key(study):
    # Names/arbitrary seed edits remain aliases. Only a source-bound registered
    # experimental unit can declare a paired replication or a control role.
    factors = study.get('semantic_factors')
    if not isinstance(factors, dict) or not factors:
        raise ValueError('each study needs explicit normalized semantic_factors')
    identity = dict(factors=factors, tasks=sorted(study['task_ids']))
    if 'experimental_unit' in study:
        unit = study['experimental_unit']
        fields = ('role', 'replication_role', 'seed', 'protocol_semantics_sha256',
                  'comparison_sha256', 'source_semantics_sha256')
        if (not isinstance(unit, dict) or any(k not in unit for k in fields)
                or unit['role'] not in ('control', 'candidate') or unit['replication_role'] != 'paired_seed'):
            raise ValueError('registered experimental unit identity is incomplete')
        identity = dict(tasks=sorted(study['task_ids']), unit={k: unit[k] for k in fields})
    return stable_hash(identity)


def deployment_layout(cfg):
    """Map preserved lease tokens explicitly; remapping requires the whole fleet."""
    allocation = cfg.get('gpu_allocation')
    if allocation is None:
        return [(gpu, slot, gpu, slot) for gpu, slot in SLOT_LAYOUT]
    if (not isinstance(allocation, dict) or set(allocation) != {'0', '1'}
            or any(type(n) is not int or n < 0 for n in allocation.values())
            or not 1 <= sum(allocation.values()) <= 12 or cfg['slots'] != sum(allocation.values())
            or any(not s.get('request_path') for s in cfg['studies'])):
        raise ValueError('GPU remapping needs an explicit prepared-work allocation within the twelve-token fleet')
    physical = [(int(gpu), slot) for gpu in ('0', '1') for slot in range(allocation[gpu])]
    return [(*lease, *device) for lease, device in zip(SLOT_LAYOUT, physical)]


def plan(config_path, *, root):
    config_path = Path(config_path).resolve()
    cfg = read_json(config_path)
    if cfg.get('schema_version') != 1:
        raise ValueError('unknown runner configuration')
    if any(key in cfg for key in ('engine', 'budget_usd', 'max_budget_usd', 'service_tier')):
        raise ValueError('legacy engine/dollar/speed settings are unsupported; use explicit current caps')
    number(cfg.get('dispatch_window_seconds', 300), 'dispatch window', zero=False)
    frozen = contract(root)
    if cfg.get('expected_view_revision', frozen['view']['revision']) != frozen['view']['revision']:
        raise ValueError('explicit configured tier revision differs from current source')
    if cfg.get('model', MODEL) != MODEL or cfg.get('effort', EFFORT) != EFFORT or cfg.get('speed', 'standard') != 'standard':
        raise ValueError('this verified runner contract is Sol6.1/xhigh/standard only')
    slots, maximum = cfg['slots'], cfg['max_studies']
    if type(slots) is not int or not 1 <= slots <= 12 or type(maximum) is not int or not 1 <= maximum <= 12:
        raise ValueError('slots and finite max_studies must be integers in1..12')
    required = cfg.get('required_concurrent_workers', 0)
    if type(required) is not int or not 0 <= required <= slots:
        raise ValueError('required concurrent workers must be an integer within the slot cap')
    if required and any(not s.get('request_path') for s in cfg['studies']):
        raise ValueError('a concurrency capacity requirement needs prepared requests')
    deployment_layout(cfg)
    if not cfg['studies'] or len(cfg['studies']) > maximum:
        raise ValueError('provide a finite study list within max_studies; no replenishment beyond it')
    seen, names, rows = set(), set(), []
    for study in cfg['studies']:
        if not all(isinstance(study.get(k), str) and study[k].strip() for k in ('id', 'hypothesis', 'falsifier', 'cohort')):
            raise ValueError('study identity, hypothesis, falsifier and cohort are required')
        identifier(study['id'], 'study')
        if study['id'] in names:
            raise ValueError('duplicate study identity')
        names.add(study['id'])
        ids = study['task_ids']
        if not ids or len(ids) != len(set(ids)) or set(ids) - frozen['tasks'].keys():
            raise ValueError('study has duplicate or non-current task ids')
        for name in ids:
            for dependency in frozen['tasks'][name].get('dependencies', []):
                parent = dependency['task'] if isinstance(dependency, dict) else dependency
                if parent not in ids:
                    raise ValueError('include own-checkpoint/gate prerequisites; no borrowed prefix')
        semantic = semantic_key(study)
        if semantic in seen:
            raise ValueError('duplicate study factors; twelve slots are an upper bound')
        seen.add(semantic)
        # Forge groups uninterrupted tasks into one original execution; do not pay twice.
        groups = {}
        for name in ids:
            task = frozen['tasks'][name]
            group = task['execution'].get('execution_group', name)
            groups[group] = max(groups.get(group, 0), task['resources']['timeout_seconds'])
        queue = sum(groups.values())
        supervisor = queue + 20 * len(groups)
        agent = number(study.get('agent_seconds', 0), 'study agent_seconds')
        if bool(agent) == bool(study.get('request_path')):
            raise ValueError('choose either a bounded Codex proposal or an already resolved request')
        rows.append(dict(id=study['id'], semantic=semantic, amounts=dict(queue=queue, supervisor=supervisor, agent=agent),
                         scoped_claim=sorted(ids), status='PLANNED_NOT_STARTED'))
    spending = Spending(cfg['budget_state'])
    if not spending.path.exists():
        raise ValueError('initialize one explicit shared budget state before planning')
    state = read_json(spending.path)
    if Path(cfg['queue_root']).resolve() != Path(state['queue_root']):
        raise ValueError('all cohorts must use the budget-bound shared queue')
    balances = Spending.remaining(state)
    total = {k: sum(row['amounts'][k] for row in rows) for k in balances}
    blocked = [f'aggregate {k} reservation exceeds remaining balance' for k in balances if total[k] > balances[k]]
    for row in rows:
        if not Spending.semantic_available(state, row['semantic']):
            blocked.append(f'{row["id"]}: historical, active or terminal duplicate')
    for study in cfg['studies']:
        if study.get('request_path') and not Path(study['request_path']).is_file():
            blocked.append(f'{study["id"]}: resolved request file unavailable')
    if state['reservations']:
        blocked.append('existing reservations require owner reconciliation; no automatic retry')
    if state['agent_sessions_started'] + sum(bool(r['amounts']['agent']) for r in rows) > state['agent_session_limit']:
        blocked.append('aggregate model session cap exceeded')
    return cfg, dict(schema_version=1, view_revision=frozen['view']['revision'], contract_sha256=stable_hash(frozen['files']),
                     frozen_files=frozen['files'], studies=rows, total_full_reservation=total,
                     remaining=balances, blockers=blocked, model=MODEL, effort=EFFORT, speed='standard',
                     service_tier='default', qualification=False, launches=0)


def verify_cli(binary):
    help_text = subprocess.check_output([binary, '--no-daemon', 'exec', '--help'], text=True, timeout=30)
    for flag in ('--approve-for-me', '--ignore-user-config', '--ephemeral', '--sandbox', '--json'):
        if flag not in help_text:
            raise ValueError(f'installed Codex CLI does not expose {flag}')
    catalog = json.loads(subprocess.check_output([binary, '--no-daemon', '-c', 'service_tier="default"',
                                                '-c', f'model_reasoning_effort="{EFFORT}"', 'debug', 'models', '--bundled'],
                                                text=True, timeout=30))
    selected = next((m for m in catalog['models'] if m['slug'] == MODEL), None)
    if selected is None or EFFORT not in {r['effort'] for r in selected['supported_reasoning_levels']}:
        raise ValueError('installed Codex CLI does not advertise requested Sol6.1/xhigh')
    return dict(model=MODEL, effort=EFFORT, speed='standard', service_tier='default',
                catalog_sha256=stable_hash(catalog), generation_sessions=0)


def codex_command(binary, checkout, directory):
    # No bypass, never-approval, ignored-rules or full-access flags.
    for parent in [checkout, *checkout.parents]:
        path = parent / '.codex/config.toml'
        if path.is_file() and any(word in path.read_text() for word in ('service_tier', 'fast_mode', 'ultrafast')):
            raise ValueError('project speed override conflicts with verified Standard contract')
    return [binary, '--no-daemon', '--approve-for-me', 'exec', '--ignore-user-config', '--ephemeral',
            '--sandbox', 'workspace-write', '--model', MODEL, '-c', f'model_reasoning_effort="{EFFORT}"',
            '-c', 'service_tier="default"',
            '--disable', 'multi_agent', '--json', '--cd', str(checkout), '--color', 'never',
            '--output-last-message', str(directory / 'final.md'), '-']


def stop_process(process, directory):
    """Cancel only a process we launched and its exact recorded training child."""
    child_path = directory / 'child.json'
    if child_path.exists():
        child = read_json(child_path)
        if process_identity(child['pid']) == child['process_identity']:
            try:
                os.killpg(child['pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    if child_path.exists():
        child = read_json(child_path)
        if process_identity(child['pid']) == child['process_identity']:
            try:
                os.killpg(child['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass


def propose(cfg, study, row, root, runs):
    directory = runs / study['id']
    directory.mkdir()
    checkout = directory / 'repo'
    base = cfg['source_commit']
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', base], text=True).strip() != base:
        raise ValueError('source_commit must be an exact verified full commit')
    subprocess.run(['git', '-C', str(root), 'worktree', 'add', '--detach', str(checkout), base], check=True, capture_output=True)
    # Exact committed source/declarations are required; historical results retain their recorded revision.
    frozen = contract(checkout)
    command = codex_command(cfg.get('codex_bin', 'codex'), checkout, directory)
    prompt = (f'Read AGENTS.md, EXPERIMENTATION.md and compiled experiment memory. Study {study["id"]}: '
              f'{study["hypothesis"]}\nFalsifier: {study["falsifier"]}\nCohort: {study["cohort"]}\n'
              'Prepare one declared meaningful candidate and normal Forge v3 admitted request.json in this worktree. '
              f'Its tasks are exactly {study["task_ids"]}; preserve their original sampler, initialization, gates, '
              'cadence, horizons and own-state dependencies. Read frozen revision10 test-stages.json; revision9 results '
              'retain original provenance. Declare every formulation delta. Do not train, launch any job or nested agent, '
              'invoke another model CLI, publish, merge, or change budget/lease/evaluator/task files. The coordinator '
              'alone admits and executes requests. No winner can be inferred from an admitted turn.\n')
    (directory / 'prompt.txt').write_text(prompt)
    atomic_json(directory / 'command.json', command)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    started = time.monotonic()
    with (directory / 'codex.jsonl').open('wb') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=log, env=env,
                                   start_new_session=True, cwd=checkout)
        atomic_json(directory / 'agent-start.json', dict(kind='CODEX_CLI_PROCESS_STARTED_NOT_TRAINING', pid=process.pid,
                    process_identity=process_identity(process.pid), unix=time.time(), semantic=row['semantic']))
        try:
            process.communicate(prompt.encode(), timeout=row['amounts']['agent'])
        except subprocess.TimeoutExpired:
            stop_process(process, directory)
    elapsed = time.monotonic() - started
    usage = []
    with (directory / 'codex.jsonl').open(errors='replace') as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') in ('turn.started', 'turn.completed', 'turn.failed', 'error'):
                usage.append(event)
    completed = any(e.get('type') == 'turn.completed' for e in usage) and not any(
        e.get('type') in ('turn.failed', 'error') for e in usage)
    atomic_json(directory / 'agent-terminal.json', dict(exit_code=process.returncode, seconds=elapsed,
                events=usage, dollar_cost=None, dollar_cost_reason='CLI has no verified hard dollar cap or cost receipt',
                completion_observed=completed,
                observed_model_turn_starts=sum(e.get('type') == 'turn.started' for e in usage),
                training_starts=0, model=MODEL, effort=EFFORT, speed='standard'))
    if process.returncode or not completed:
        raise RuntimeError(f'Codex turn failed; preserve {directory}; no retry')
    if frozen['files'] != contract(checkout)['files']:
        raise ValueError('proposal changed frozen gauntlet declarations')
    return checkout / 'request.json', elapsed


def validate_request(request, study, frozen):
    if request.get('preflight_blockers'):
        raise ValueError('unsupported request remains BLOCKED before allocation')
    if request['view']['id'] == frozen['view']['id']:
        if request['view']['revision'] != frozen['view']['revision'] or request['view']['assignments'] != frozen['view']['assignments']:
            raise ValueError('request must bind the explicitly frozen current view revision')
    elif study.get('scope') != 'scoped' or request['view']['id'] != study.get('view_id'):
        raise ValueError('a partial claim needs its own explicitly declared registered scoped view')
    tiers = {a['task']: a['qualification_tier'] for a in request['view']['assignments']}
    current_tiers = {a['task']: a['qualification_tier'] for a in frozen['view']['assignments']}
    if not study.get('experimental_unit') and any(current_tiers.get(name) != tier for name, tier in tiers.items()):
        raise ValueError('scoped views preserve current tier labels and original tasks')
    if study.get('experimental_unit'):
        from .research_protocols import validate_unit
        validate_unit(request, study['experimental_unit'])
    elif request['protocol'].get('id', 'screening') != 'screening' or request['protocol']['seed'] != 0:
        raise ValueError('alternate seeds need an explicit registered experimental unit')
    members = {name for job in request['jobs']
               if all(tiers[name] <= request.get('through_tier', 3) for name in job.get('task_ids', [job['task_id']]))
               for name in job.get('task_ids', [job['task_id']])}
    if members != set(study['task_ids']):
        raise ValueError('resolved request task scope differs from declared study')
    for name in members:
        for section in ('execution', 'evaluation', 'resources', 'dependencies'):
            if request['tasks'][name].get(section) != frozen['tasks'][name].get(section):
                raise ValueError(f'original {name}.{section} changed')
    for key in ('study_admission', 'study_review', 'source', 'candidate_revision'):
        if not request.get(key):
            raise ValueError('normal source-bound study admission is required')
    if request['study_review']['status'] != 'READY':
        raise ValueError('normal study review is not READY')


def stopped(cfg):
    return any(Path(path).exists() for path in
               [str(Path(cfg['runs_dir']) / 'STOP'), *cfg.get('stop_files', ['/ml2/hypergan/gan-attempts/STOP'])])


def observe_training(active, prior):
    """Require a new completed update and the exact live training child."""
    directory = active['directory']
    child_path, log = directory / 'child.json', directory / 'run.log'
    if not child_path.exists() or not log.exists() or any(t['attempt'] == active['attempt'] for t in prior):
        return None
    child = read_json(child_path)
    child.pop('supervisor_pid', None)  # authoritative supervisor is recorded below, once
    if child.get('phase') in ('evaluation', 'grading') or process_identity(child['pid']) != child['process_identity']:
        return None
    prefix = active['prefix_steps']
    with log.open(errors='replace') as stream:
        stream.seek(active['log_offset'])
        lines = stream.readlines()
        active['log_offset'] = stream.tell()
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('event') == 'observation' and type(event.get('step')) is int and event['step'] > prefix:
            owner = active['owner']
            return dict(kind='OBSERVED_REAL_TRAINING_UPDATES', unix=time.time(), attempt=active['attempt'],
                        **child, completed_steps=event['step'], new_updates_confirmed=event['step'] - prefix,
                        supervisor_pid=active['process'].pid, lease=active['lease'],
                        source_digest=owner['request']['source']['digest'],
                        candidate_revision=owner['request']['candidate_revision'],
                        task_id=active['task_id'],
                        study=owner['study']['id'], experimental_unit=owner['study'].get('experimental_unit'),
                        log=str(log))
    return None


def concurrency_capacity(cfg):
    """Native per-slot reservations, using unchanged prepared task resources."""
    from .queue import device_slots
    resources = [job['resources'] for study in cfg['studies']
                 for job in read_json(study['request_path'])['jobs']]
    memory = max(r['memory_mb'] for r in resources)
    devices = []
    total = 0
    counts = {gpu: sum(row[2] == gpu for row in deployment_layout(cfg)) for gpu in (0, 1)}
    for gpu, count in counts.items():
        if not count:
            continue
        slots = [s for s in device_slots([str(gpu)], workers_per_gpu=count, allow_sharing=True)
                 if s['device'] == str(gpu)]
        compatible = [s for s in slots if s['memory_mb'] >= memory
                      and all(not r.get('gpu_model') or s['model'] == r['gpu_model'] for r in resources)]
        total += len(compatible)
        devices.append(dict(gpu=gpu, lease_slots=count, native_slots=slots,
                            required_memory_mb_per_worker=memory, compatible_slots=len(compatible),
                            required_free_memory_mb=count * memory + 512))
    return dict(unix=time.time(), required=cfg['required_concurrent_workers'],
                available_under_current_native_partition=total, devices=devices,
                original_task_resources_unchanged=True, deployment_layout=deployment_layout(cfg))


def coordinated_window_blocks(cfg,horizon,*,now=None):
    from datetime import datetime
    now=time.time() if now is None else now
    for window in cfg.get('blocked_UTC_windows',[]):
        begin=datetime.fromisoformat(window['start'].replace('Z','+00:00')).timestamp()
        end=datetime.fromisoformat(window['end'].replace('Z','+00:00')).timestamp()
        if now < end and now+horizon > begin:return True
    return False


def execute(cfg, report, root):
    from .queue import Queue, device_slots
    from .sources import verify_snapshot
    spending = Spending(cfg['budget_state'])
    runs = Path(cfg['runs_dir']).resolve() / uuid.uuid4().hex
    runs.mkdir(parents=True)
    leases = Leases(cfg['lease_root'])
    atomic_json(runs / 'plan.json', report)
    if cfg.get('required_concurrent_workers'):
        capacity = concurrency_capacity(cfg)
        atomic_json(runs / 'capacity.json', capacity)
        if capacity['available_under_current_native_partition'] < capacity['required']:
            atomic_json(runs / 'catalog.json', dict(status='BLOCKED_EXTERNAL_GPU_CAPACITY',
                        capacity=capacity, receipts=[], training_starts=[], peak_concurrent_training=0,
                        unknown_or_unsettled=[], unadmitted_studies=[s['id'] for s in cfg['studies']],
                        new_reservations=0, paid_model_sessions=0, additional_qualification=False))
            return runs
    if any(s.get('agent_seconds') for s in cfg['studies']):
        atomic_json(runs / 'cli-contract.json', verify_cli(cfg.get('codex_bin', 'codex')))
    admitted, running, receipts, training = [], [], [], []
    attempt_receipts = []
    stop = False
    peak_training = []

    def drain(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)
    def admit(study, row, key):
        if stop or stopped(cfg):
            return
        model_seconds = 0.
        submission_started = False
        try:
            if study.get('request_path'):
                request_path = Path(study['request_path']).resolve()
            else:
                request_path, model_seconds = propose(cfg, study, row, root, runs)
            request = read_json(request_path)
            validate_request(request, study, contract(root))
            verify_snapshot(Path(request['source']['snapshot_path']), request['source'])
            for name in ('queue.py', 'worker.py'):
                relative = 'experiments/forge/' + name
                if request['source']['files'].get(relative) != digest(root / relative):
                    raise ValueError('executed source must retain current queue/worker lease propagation')
            source_root = Path(request['source']['snapshot_path'])
            if contract(source_root)['files'] != report['frozen_files']:
                raise ValueError('executed source drifted from the explicitly frozen global gauntlet')
            # Admission uses the authoritative declarations; durable attempt
            # projections must never be written into the immutable snapshot.
            queue = Queue(Path(cfg['queue_root']), report_root=Path(root) / 'reports/forge')
            before = queue.inspect()
            submission_started = True
            submission = queue.submit(request, request['study']['campaign'])
            admitted.append(dict(key=key, study=study, row=row, queue=queue, request=request,
                                 submission=submission, model_seconds=model_seconds, attempts=[]))
        except Exception as error:
            # A paid turn that errored retains its actual cost; never charge zero by assumption.
            terminal = runs / study['id'] / 'agent-terminal.json'
            if terminal.exists():
                model_seconds = read_json(terminal)['seconds']
            if submission_started:
                atomic_json(runs / 'uncertain-submission.json', dict(key=key, reason=str(error),
                            reservation_retained=True, explicit_reconciliation_required=True))
                raise
            spending.settle(key, dict(queue=0., supervisor=0., agent=model_seconds), dict(status='ERROR', reason=str(error)))
            receipts.append(dict(study=study['id'], status='ERROR', reason=str(error)))

    with file_lock(spending.path.with_suffix('.coordinator.lock'), blocking=False), file_lock(Path(cfg['queue_root']) / 'coordinator.lock', blocking=False), ExitStack() as fleet_stack:
        existing = Queue(Path(cfg['queue_root'])).inspect()
        if any(j['status'] == 'running' for j in existing['jobs'].values()):
            raise ValueError('existing shared queue workers require owner reconciliation')
        layout = deployment_layout(cfg)
        fleet = {}
        if cfg.get('gpu_allocation') is not None:
            targets = {(a, b): (c, d) for a, b, c, d in layout}
            for lease_gpu, lease_slot in SLOT_LAYOUT:
                gpu, slot = targets.get((lease_gpu, lease_slot), (None, None))
                lease = fleet_stack.enter_context(leases.acquire(lease_gpu, lease_slot))
                if lease is None:
                    atomic_json(runs / 'catalog.json', dict(status='BLOCKED_EXISTING_LEASE_OWNER',
                        occupied_lease=str(leases.root / f'gpu{lease_gpu}-slot{lease_slot}.lock'),
                        receipts=[], training_starts=[], unknown_or_unsettled=[],
                        new_reservations=0, additional_qualification=False))
                    return runs
                fleet[(lease_gpu, lease_slot)] = {**lease, 'lease_gpu': lease_gpu, 'lease_slot': lease_slot,
                                                 'gpu': gpu, 'slot': slot}
            atomic_json(runs / 'fleet-binding.json', dict(unix=time.time(), entire_fleet_exclusive=True,
                all_children_inherit_entire_fleet=True, leases=[{k: v for k, v in lease.items() if k != 'handle'} for lease in fleet.values()]))
        if stopped(cfg):
            atomic_json(runs / 'catalog.json', dict(status='STOPPED_BEFORE_ADMISSION', receipts=[],
                        training_starts=[], unknown_or_unsettled=[], new_reservations=0, additional_qualification=False))
            return runs
        pending = [(study, row, uuid.uuid4().hex) for study, row in zip(cfg['studies'], report['studies'])]
        spending.reserve_cohort([dict(key=key, semantic=row['semantic'], amounts=row['amounts']) for _, row, key in pending])
        atomic_json(runs / 'full-cohort-reservation.json', dict(units=len(pending), amounts=report['total_full_reservation'],
                    keys=[key for _, _, key in pending], reserved_before_first_launch=True))
        dispatch_started = time.monotonic()
        dispatch_window = number(cfg.get('dispatch_window_seconds', 300), 'dispatch window', zero=False)
        while pending or admitted:
            if stopped(cfg) or time.monotonic() - dispatch_started >= dispatch_window:
                stop = True
            if pending and not stop:
                admit(*pending.pop(0))
            available_by_gpu = {}
            if not stop:
                for gpu in {row[2] for row in layout}:
                    count = sum(row[2] == gpu for row in layout)
                    available_by_gpu[gpu] = device_slots([str(gpu)], workers_per_gpu=count, allow_sharing=True)
            if cfg.get('overall_deadline_unix') and time.time() >= cfg['overall_deadline_unix']:
                stop = True
            for owner in admitted:
                owner['queue'].collect()
            for active in running[:]:
                elapsed = time.monotonic() - active['clock']
                if active['process'].poll() is None:
                    if cfg.get('overall_deadline_unix') and time.time() >= cfg['overall_deadline_unix']:
                        atomic_json(active['directory'] / 'cancel.json', dict(reason='bounded loop deadline'))
                        if time.time() >= cfg['overall_deadline_unix'] + 5:
                            stop_process(active['process'], active['directory'])
                    if elapsed >= active['allowance'] - 10:
                        atomic_json(active['directory'] / 'cancel.json', dict(reason='full supervisor allowance ceiling'))
                    if elapsed >= active['allowance'] - 5:
                        stop_process(active['process'], active['directory'])
                    observed = observe_training(active, training)
                    if observed:
                        training.append(observed)
                        atomic_json(runs / 'training-starts.json', training)
                    continue
                active['context'].__exit__(None, None, None)
                running.remove(active)
                active['owner']['queue'].collect()
                receipt = dict(attempt=active['attempt'], supervisor_seconds=elapsed,
                    exit_code=active['process'].returncode, lease=active['lease'], log=str(active['directory'] / 'run.log'),
                    supervisor_pid=active['process'].pid, task_id=active['task_id'], terminal_observed_unix=time.time())
                active['owner']['attempts'].append(receipt)
                attempt_receipts.append(receipt)
                atomic_json(runs / 'attempt-receipts.json', attempt_receipts)
            if not stop:
                for owner in admitted:
                    for lease_gpu, lease_slot, gpu, slot in layout:
                        if len(running) >= cfg['slots']:
                            break
                        if fleet and any(a['lease']['path'] == fleet[(lease_gpu, lease_slot)]['path'] for a in running):
                            continue
                        context = nullcontext(fleet[(lease_gpu, lease_slot)]) if fleet else leases.acquire(gpu, slot)
                        lease = context.__enter__()
                        if lease is None:
                            context.__exit__(None, None, None)
                            continue
                        try:
                            available = available_by_gpu[gpu]
                            if cfg.get('blocked_UTC_windows'):
                                tiers = {a['task']:a['qualification_tier'] for a in owner['request']['view']['assignments']}
                                state_now = owner['queue'].inspect()
                                pending_jobs = [j for j in owner['request']['jobs'] if state_now['jobs'][j['compatibility_key']]['status']=='pending']
                                if pending_jobs:
                                    current_tier = min(tiers[j['task_id']] for j in pending_jobs)
                                    horizon = max(j['budget_seconds']+20 for j in pending_jobs if tiers[j['task_id']]==current_tier)
                                    if coordinated_window_blocks(cfg,horizon):
                                        stop=True
                                        context.__exit__(None,None,None)
                                        break
                            claimed = owner['queue'].claim([s for s in available if s['slot'] == slot],
                                    campaign_filter=owner['request']['study']['campaign']['id'],
                                    request_filter=owner['submission']['request']['request_id'])
                            if claimed is None:
                                context.__exit__(None, None, None)
                                continue
                            clock = time.monotonic()
                            process = owner['queue'].launch(claimed, shared_slot_fd=lease['handle'].fileno(),
                                shared_fleet_fds=tuple(v['handle'].fileno() for v in fleet.values()))
                            directory = Path(claimed['worker']['directory'])
                            binding = {k: lease[k] for k in ('path', 'inode', 'gpu', 'slot')}
                            binding.update(lease_gpu=lease_gpu, lease_slot=lease_slot)
                            atomic_json(directory / 'runner-start.json', dict(kind='FORGE_SUPERVISOR_STARTED',
                                        pid=process.pid, process_identity=process_identity(process.pid),
                                        unix=time.time(), lease=binding, attempt=claimed['worker']['attempt']))
                            running.append(dict(owner=owner, process=process, clock=clock, directory=directory,
                                allowance=claimed['job']['budget_seconds'] + 20, context=context, lease=binding,
                                attempt=claimed['worker']['attempt'], log_offset=0,
                                task_id=claimed['job']['task_id'],
                                prefix_steps=owner['request']['tasks'][claimed['job']['task_id']]['execution'].get('preserve_prefix_steps', 0)))
                        except Exception:
                            context.__exit__(*sys.exc_info())
                            # Keep the whole cohort reservation fenced after uncertain launch.
                            raise
            for owner in admitted[:]:
                state = owner['queue'].inspect()
                submission = state['submissions'][owner['submission']['request']['request_id']]
                active = [a for a in running if a['owner'] is owner]
                if active or submission['status'] not in ('completed', 'concluded', 'blocked', 'cancelled'):
                    continue
                ids = {a['attempt'] for a in owner['attempts']}
                queue_seconds = sum(c['seconds'] for c in state['charges'] if c['attempt_id'] in ids)
                outcomes = []
                for job in owner['request']['jobs']:
                    result = state['jobs'].get(job.get('compatibility_key'), {}).get('result')
                    if result:
                        for task_row in result['task_results']:
                            if task_row['task_id'] in owner['study']['task_ids']:
                                outcomes.append({k: task_row.get(k) for k in
                                                 ('task_id', 'gate_status', 'raw_status', 'metrics', 'reason', 'compatibility_key')})
                receipt = dict(study=owner['study']['id'], status=submission['status'], attempts=owner['attempts'],
                               source_digest=owner['request']['source']['digest'],
                               candidate_revision=owner['request']['candidate_revision'], view_revision=owner['request']['view']['revision'],
                               semantic=owner['row']['semantic'], task_results=outcomes,
                               actual_new_attempts=len(owner['attempts']), qualification=False)
                spending.settle(owner['key'], dict(queue=queue_seconds,
                    supervisor=sum(a['supervisor_seconds'] for a in owner['attempts']), agent=owner['model_seconds']), receipt)
                receipts.append(receipt)
                admitted.remove(owner)
            live_training = [t for t in training if any(a['attempt'] == t['attempt'] for a in running)
                             and process_identity(t['pid']) == t['process_identity']]
            if len(live_training) > len(peak_training):
                peak_training = list(live_training)
                atomic_json(runs / 'peak-training.json', dict(unix=time.time(), count=len(peak_training), workers=peak_training))
            atomic_json(runs / 'live.json', dict(supervisors=len(running), observed_training_starts=len(training),
                        supervisor_receipts=[dict(pid=a['process'].pid, process_identity=process_identity(a['process'].pid),
                          attempt=a['attempt'], task_id=a['task_id'], lease=a['lease']) for a in running],
                        current_training_workers=len(live_training), training_workers=live_training,
                        admitted_studies=len(admitted), agent_turns_are_not_experiments=True))
            if stop and not running:
                # Preserve any queued work/reservations for explicit reconciliation; never replenish after STOP.
                break
            time.sleep(.1)
        atomic_json(runs / 'catalog.json', dict(schema_version=1, receipts=receipts, training_starts=training,
                    peak_concurrent_training=len(peak_training), peak_training_workers=peak_training,
                    unknown_or_unsettled=[o['key'] for o in admitted] + [key for _, _, key in pending],
                    unadmitted_studies=[study['id'] for study, _, _ in pending], additional_qualification=False))
        # Existing report readers retain cohort binding; no new scored evidence is invented.
        if not admitted and contract(root)['view']['revision']==10:
            try:
                from .test_stages import generated_reports
                for name, content in generated_reports(root).items():
                    destination = runs / Path(name).name
                    destination.write_text(content)
            except (FileNotFoundError, ValueError) as error:
                atomic_json(runs / 'stage-readout-unavailable.json', dict(reason=str(error), status='UNMEASURED'))
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init-budget', help='explicitly import a reconciled balance; never overwrite a state')
    init.add_argument('--state', type=Path, required=True)
    init.add_argument('--ledger', type=Path, required=True)
    init.add_argument('--queue-root', type=Path, required=True)
    init.add_argument('--agent-seconds', type=float, default=0)
    init.add_argument('--agent-sessions', type=int, default=0)
    init.add_argument('--semantic-history', type=Path, required=True, help='explicit JSON list of known semantic keys, including negatives')
    for name in ('study', 'pool'):
        command = commands.add_parser(name)
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--execute', action='store_true', help='explicit paid/compute action within the frozen caps')
        command.add_argument('--dry-run', action='store_true', help='plan only; never import launchers or call models')
    args = parser.parse_args(argv)
    if args.command == 'init-budget':
        Spending(args.state).initialize(args.ledger, queue_root=args.queue_root,
            agent_seconds=args.agent_seconds, agent_sessions=args.agent_sessions,
            history=read_json(args.semantic_history) if args.semantic_history else ())
        return 0
    cfg, report = plan(args.config, root=args.root)
    if args.command == 'study' and len(cfg['studies']) != 1:
        parser.error('try-gan requires exactly one distinct study')
    print(json.dumps(report, indent=2))
    if args.execute and not args.dry_run:
        if report['blockers']:
            parser.error('; '.join(report['blockers']))
        print(execute(cfg, report, args.root.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
