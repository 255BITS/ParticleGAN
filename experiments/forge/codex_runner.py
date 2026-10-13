"""Bounded Codex proposals and ordinary Forge attempts; planning never launches either."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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

    def reserve(self, key, semantic, amounts, *, agent=False):
        with self.state() as state:
            if semantic in state['seen_semantics']:
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
    if view['revision'] != 10:
        raise ValueError('runner requires the explicit revision-10 view; no historical reinterpretation')
    stages = read_json(root / 'configs/forge/test-stages.json')
    if stages['qualification_input'] or stages['execution_policy'] != 'shared_original_runs_no_additional_updates':
        raise ValueError('stage readout must not invent qualification or training')
    names = {a['task'] for a in view['assignments']}
    files = {'configs/forge/views/discriminator_stability.json': digest(view_path),
             'configs/forge/test-stages.json': digest(root / 'configs/forge/test-stages.json'),
             'configs/forge/view-history/discriminator_stability-v9.json': digest(root / 'configs/forge/view-history/discriminator_stability-v9.json')}
    tasks = {}
    for name in names:
        path = root / 'configs/forge/tasks' / f'{name}.json'
        files[str(path.relative_to(root))] = digest(path)
        tasks[name] = read_json(path)
    for pin in stages['sources'].values():
        if digest(root / pin['path']) != pin['sha256']:
            raise ValueError('original stage source task changed')
    return dict(view=view, stages=stages, tasks=tasks, files=files)


def semantic_key(study):
    # Names, seed and source-only edits do not make a fresh experiment.
    factors = study.get('semantic_factors')
    if not isinstance(factors, dict) or not factors:
        raise ValueError('each study needs explicit normalized semantic_factors')
    return stable_hash(dict(factors=factors, tasks=sorted(study['task_ids'])))


def plan(config_path, *, root):
    config_path = Path(config_path).resolve()
    cfg = read_json(config_path)
    if cfg.get('schema_version') != 1:
        raise ValueError('unknown runner configuration')
    if any(key in cfg for key in ('engine', 'budget_usd', 'max_budget_usd', 'service_tier')):
        raise ValueError('legacy engine/dollar/speed settings are unsupported; use explicit current caps')
    number(cfg.get('dispatch_window_seconds', 300), 'dispatch window', zero=False)
    frozen = contract(root)
    if cfg.get('model', MODEL) != MODEL or cfg.get('effort', EFFORT) != EFFORT or cfg.get('speed', 'standard') != 'standard':
        raise ValueError('this verified runner contract is Sol6.1/xhigh/standard only')
    slots, maximum = cfg['slots'], cfg['max_studies']
    if type(slots) is not int or not 1 <= slots <= 12 or type(maximum) is not int or not 1 <= maximum <= 12:
        raise ValueError('slots and finite max_studies must be integers in1..12')
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
        if row['semantic'] in state['seen_semantics']:
            blocked.append(f'{row["id"]}: historical, active or terminal duplicate')
    for study in cfg['studies']:
        if study.get('request_path') and not Path(study['request_path']).is_file():
            blocked.append(f'{study["id"]}: resolved request file unavailable')
    if state['reservations']:
        blocked.append('existing reservations require owner reconciliation; no automatic retry')
    if state['agent_sessions_started'] + sum(bool(r['amounts']['agent']) for r in rows) > state['agent_session_limit']:
        blocked.append('aggregate model session cap exceeded')
    return cfg, dict(schema_version=1, view_revision=10, contract_sha256=stable_hash(frozen['files']),
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
    # A committed revision10 source is required; uncommitted runner work is not a source binding.
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
        if request['view']['revision'] != 10 or request['view']['assignments'] != frozen['view']['assignments']:
            raise ValueError('request must bind the current revision10 view')
    elif study.get('scope') != 'scoped' or request['view']['id'] != study.get('view_id'):
        raise ValueError('a partial claim needs its own explicitly declared registered scoped view')
    tiers = {a['task']: a['qualification_tier'] for a in request['view']['assignments']}
    current_tiers = {a['task']: a['qualification_tier'] for a in frozen['view']['assignments']}
    if any(current_tiers.get(name) != tier for name, tier in tiers.items()):
        raise ValueError('scoped views preserve current tier labels and original tasks')
    if request['protocol']['seed'] != 0:
        raise ValueError('seed-only studies are not allowed')
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
    if request['study_review']['status'] != 'PASS':
        raise ValueError('normal study review did not PASS')


def stopped(cfg):
    return any(Path(path).exists() for path in
               [str(Path(cfg['runs_dir']) / 'STOP'), *cfg.get('stop_files', ['/ml2/hypergan/gan-attempts/STOP'])])


def execute(cfg, report, root):
    from .queue import Queue, device_slots
    from .test_stages import generated_reports
    from .sources import verify_snapshot
    spending = Spending(cfg['budget_state'])
    runs = Path(cfg['runs_dir']).resolve() / uuid.uuid4().hex
    runs.mkdir(parents=True)
    leases = Leases(cfg['lease_root'])
    atomic_json(runs / 'plan.json', report)
    if any(s.get('agent_seconds') for s in cfg['studies']):
        atomic_json(runs / 'cli-contract.json', verify_cli(cfg.get('codex_bin', 'codex')))
    admitted, running, receipts, training = [], [], [], []
    stop = False

    def drain(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, drain)
    signal.signal(signal.SIGINT, drain)
    with file_lock(spending.path.with_suffix('.coordinator.lock'), blocking=False), file_lock(Path(cfg['queue_root']) / 'coordinator.lock', blocking=False):
        for study, row in zip(cfg['studies'], report['studies']):
            if stop or stopped(cfg):
                break
            key = uuid.uuid4().hex
            spending.reserve(key, row['semantic'], row['amounts'], agent=bool(row['amounts']['agent']))
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
                    raise ValueError('executed source drifted from the frozen global revision10 gauntlet')
                queue = Queue(Path(cfg['queue_root']), report_root=source_root / 'reports/forge')
                before = queue.inspect()
                if any(j['status'] == 'running' for j in before['jobs'].values()):
                    raise ValueError('existing shared queue workers require owner reconciliation')
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
        dispatch_started = time.monotonic()
        dispatch_window = number(cfg.get('dispatch_window_seconds', 300), 'dispatch window', zero=False)
        while admitted:
            if stopped(cfg) or time.monotonic() - dispatch_started >= dispatch_window:
                stop = True
            for owner in admitted:
                owner['queue'].collect()
            for active in running[:]:
                elapsed = time.monotonic() - active['clock']
                if active['process'].poll() is None:
                    if elapsed >= active['allowance'] - 10:
                        atomic_json(active['directory'] / 'cancel.json', dict(reason='full supervisor allowance ceiling'))
                    if elapsed >= active['allowance'] - 5:
                        stop_process(active['process'], active['directory'])
                    child_path = active['directory'] / 'child.json'
                    if child_path.exists():
                        child = read_json(child_path)
                        marker = (active['attempt'], child['process_identity'])
                        if child.get('phase') not in ('evaluation', 'grading') and marker not in {(t['attempt'], t['process_identity']) for t in training} and process_identity(child['pid']) == child['process_identity']:
                            training.append(dict(kind='OBSERVED_TRAINING_START', attempt=active['attempt'], **child))
                            atomic_json(runs / 'training-starts.json', training)
                    continue
                active['context'].__exit__(None, None, None)
                running.remove(active)
                active['owner']['queue'].collect()
                active['owner']['attempts'].append(dict(attempt=active['attempt'], supervisor_seconds=elapsed,
                    exit_code=active['process'].returncode, lease=active['lease'], log=str(active['directory'] / 'run.log')))
                atomic_json(runs / 'attempt-receipts.json', [a for o in admitted for a in o['attempts']])
            if not stop:
                for owner in admitted:
                    for gpu, slot in SLOT_LAYOUT:
                        if len(running) >= cfg['slots']:
                            break
                        context = leases.acquire(gpu, slot)
                        lease = context.__enter__()
                        if lease is None:
                            context.__exit__(None, None, None)
                            continue
                        try:
                            available = device_slots([str(gpu)], workers_per_gpu=10 if gpu == 0 else 2, allow_sharing=True)
                            claimed = owner['queue'].claim([s for s in available if s['slot'] == slot],
                                    campaign_filter=owner['request']['study']['campaign']['id'],
                                    request_filter=owner['submission']['request']['request_id'])
                            if claimed is None:
                                context.__exit__(None, None, None)
                                continue
                            clock = time.monotonic()
                            process = owner['queue'].launch(claimed, shared_slot_fd=lease['handle'].fileno())
                            directory = Path(claimed['worker']['directory'])
                            binding = {k: lease[k] for k in ('path', 'inode', 'gpu', 'slot')}
                            atomic_json(directory / 'runner-start.json', dict(kind='FORGE_SUPERVISOR_STARTED',
                                        pid=process.pid, process_identity=process_identity(process.pid),
                                        unix=time.time(), lease=binding, attempt=claimed['worker']['attempt']))
                            running.append(dict(owner=owner, process=process, clock=clock, directory=directory,
                                allowance=claimed['job']['budget_seconds'] + 20, context=context, lease=binding,
                                attempt=claimed['worker']['attempt']))
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
                               candidate_revision=owner['request']['candidate_revision'], view_revision=10,
                               semantic=owner['row']['semantic'], task_results=outcomes,
                               actual_new_attempts=len(owner['attempts']), qualification=False)
                spending.settle(owner['key'], dict(queue=queue_seconds,
                    supervisor=sum(a['supervisor_seconds'] for a in owner['attempts']), agent=owner['model_seconds']), receipt)
                receipts.append(receipt)
                admitted.remove(owner)
            atomic_json(runs / 'live.json', dict(supervisors=len(running), observed_training_starts=len(training),
                        admitted_studies=len(admitted), agent_turns_are_not_experiments=True))
            if stop and not running:
                # Preserve any queued work/reservations for explicit reconciliation; never replenish after STOP.
                break
            time.sleep(.1)
        atomic_json(runs / 'catalog.json', dict(schema_version=1, receipts=receipts, training_starts=training,
                    unknown_or_unsettled=[o['key'] for o in admitted], additional_qualification=False))
        # Existing report readers retain cohort binding; no new scored evidence is invented.
        if not admitted:
            try:
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
