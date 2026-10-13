"""Resume the admitted B snapshot with one report-only collector topology fix.

No preparation, submission, recipe/source rewrite or retry is provided. Root
reviews and commits this coordinator separately; scientific attribution stays B.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EXECUTION = '67920d9090727bd74adbd331187f0e6f22533e48'
WORKFLOW_SHA = '991e573c137327b8dd00c129f8ac916dff1f95f790171ea78bc77ee94b189b96'
REGISTRATION_SHA = '848667950e78d79675c208f3d903ca14fd7a047794c22abe6712bc0c21547e2c'
REQUESTS = {'energy_u':'5e165fc7ac28e8791e044bee', 'non_saturating':'a7918d6e2a6a708d8316364b'}
ADMIN_FILES = {str((OUT / name).relative_to(ROOT)) for name in
               ('administrative_drain.py', 'test_administrative_drain.py')}
ADMIN_RECEIPT = str((OUT / 'administrative-drain-receipt.json').relative_to(ROOT))

if hashlib.sha256((OUT / 'workflow.py').read_bytes()).hexdigest() != WORKFLOW_SHA:
    raise ValueError('The admitted workflow source must remain unchanged')
spec = importlib.util.spec_from_file_location('frozen_energy_admin_workflow', OUT / 'workflow.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
shared = frozen.shared


class AdministrativeQueue(frozen.ScopedQueue):
    @contextmanager
    def state(self):
        if getattr(self._scope, 'mode', None) != 'collect':
            with super().state() as state:
                yield state
            return
        # Keep the original request/projection guards and all global charges.
        # This distinct mode prevents the frozen collector's 14-key truncation.
        with self.scoped('administrative_collect'), super().state() as state:
            keys = {j['compatibility_key'] for s in state['submissions'].values() for j in s['request']['jobs']}
            authorized = {j['compatibility_key'] for s in state['submissions'].values() for j in s['request']['jobs']
                          if shared.Queue._authorized(s['request'], j)}
            for key in keys - authorized:
                job = state['jobs'][key]
                shared.require(not job['attempts'] and job.get('result') is None and not job.get('subscribers')
                               and not job.get('reserved_seconds', 0) and job['status'] == 'pending',
                               'Higher-tier bookkeeping must remain unattempted and unsubscribed')
            yield {**state, 'jobs': {k: state['jobs'][k] for k in keys}}


def git_text(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def git_files(root, commit, expected):
    """Compare exact committed blobs in one read-only Git batch."""
    paths = sorted(expected)
    result = subprocess.run(['git', 'cat-file', '--batch'], cwd=root,
        input=''.join(f'{commit}:{p}\n' for p in paths).encode(), stdout=subprocess.PIPE, check=True).stdout
    offset = 0
    for path in paths:
        end = result.index(b'\n', offset)
        header = result[offset:end].split()
        shared.require(len(header) == 3 and header[1] == b'blob', f'Missing committed blob: {commit}:{path}')
        length = int(header[2]); data = result[end + 1:end + 1 + length]
        shared.require(hashlib.sha256(data).hexdigest() == expected[path], f'Committed bytes changed: {commit}:{path}')
        offset = end + 1 + length + 1
    shared.require(offset == len(result), 'Unexpected Git blob batch output')


def verify(root, coordinator):
    shared.require(len(coordinator) == 40 and all(c in '0123456789abcdef' for c in coordinator)
                   and coordinator != EXECUTION and git_text(root, 'rev-parse', 'HEAD') == coordinator,
                   'Use the explicit reviewed administrative commit C, distinct from admitted science B')
    subprocess.run(['git', 'merge-base', '--is-ancestor', EXECUTION, coordinator], cwd=root, check=True)
    changed = set(git_text(root, 'diff', '--name-only', EXECUTION, coordinator).splitlines())
    shared.require(ADMIN_FILES <= changed <= ADMIN_FILES | {ADMIN_RECEIPT},
                   'Coordinator C may add only the narrow administrative driver, guards and separate receipt')
    git_files(root, coordinator, {p: shared.file_hash(root / p) for p in changed})
    registration_path = OUT / 'registration.json'
    registration = shared.read_json(registration_path)
    progress_path = frozen.ARCHIVE / f'{frozen.CAMPAIGN}-progress.json'
    progress = shared.read_json(progress_path)
    shared.require(shared.file_hash(registration_path) == REGISTRATION_SHA
                   and progress['registration_sha256'] == REGISTRATION_SHA
                   and progress['source_commit'] == EXECUTION and progress['requests'] == REQUESTS
                   and progress['phase'] == 'admitted', 'Preserve original admitted B progress, request IDs and registration')
    shared.require(len(registration['declarations']) == 36
                   and all(shared.file_hash(root / p) == h for p, h in registration['declarations'].items()),
                   'All 36 admitted declarations remain immutable')
    git_files(root, EXECUTION, {**registration['declarations'],
                              str(registration_path.relative_to(root)): REGISTRATION_SHA})
    frozen.storage(frozen.ARCHIVE)
    state = shared.read_json(frozen.ARCHIVE / 'queue/queue/state.json')
    planned, authorized, sources = set(), set(), []
    for role, rid in REQUESTS.items():
        request = state['submissions'][rid]['request']
        shared.require(request['request_id'] == rid and request['campaign_id'] == frozen.CAMPAIGN
                       and shared.stable_hash({k:v for k,v in request.items() if k != 'request_id'})[:24] == rid
                       and shared.read_json(frozen.ARCHIVE / 'queue/queue/requests' / f'{rid}.json') == request,
                       'Frozen admission identity or archived request changed')
        shared.require(shared.request_projection(request) == {k:v for k,v in registration['candidates'][role].items() if k != 'card'}
                       and request['study']['campaign'] == registration['manifest']['campaign'],
                       'Actual original source, recipe, study, task and protocol projections changed')
        sources.append(request['source'])
        shared.require(request['source']['origin_commit'] == EXECUTION
                       and request['source']['digest'] == registration['source_digest']
                       and Path(request['source']['snapshot_path']) == frozen.ORIGINAL / 'queue/snapshots' / registration['source_digest'],
                       'Workers must use the original admitted scientific snapshot B')
        planned.update(j['compatibility_key'] for j in request['jobs'])
        authorized.update(j['compatibility_key'] for j in request['jobs'] if shared.Queue._authorized(request, j))
    shared.require(len(planned) == 58 and len(authorized) == 14 and sources[0] == sources[1],
                   'Retain all 58 planned records and only fourteen authorized Tier1 jobs')
    from experiments.forge.sources import SOURCE_DIRS, inspect_source, verify_snapshot
    source = sources[0]
    extra = [p for p in source['files'] if Path(p).parts[0] not in SOURCE_DIRS]
    current = inspect_source(root, extra_paths=extra)
    shared.require(current['files'] == source['files'] and current['digest'] == source['digest'],
                   'Administrative C cannot change scientific source bytes or add a scientific source file')
    git_files(root, EXECUTION, source['files'])
    git_files(root, coordinator, source['files'])
    verify_snapshot(Path(source['snapshot_path']), source)
    publication, audit, _ = frozen.predecessor(root)
    charges, _ = frozen.historical_charges(state, publication, audit)
    shared.require(math_paid(charges) + sum(j.get('reserved_seconds', 0) for j in state['jobs'].values()) <= 8000,
                   'Preserve all historical charges and concurrent full reservations under paid8000')
    return dict(coordinator_commit=coordinator, scientific_execution_commit=EXECUTION,
        source_digest=source['digest'], source_files=len(source['files']), declarations=36,
        original_workflow_sha256=WORKFLOW_SHA, registration_sha256=REGISTRATION_SHA,
        requests=REQUESTS, planned_jobs=58, authorized_tier1_jobs=14,
        paid_attempts_before=len(charges), paid_seconds_before=math_paid(charges), qualification_input=False)


def math_paid(charges):
    return frozen.math.fsum(c['seconds'] for c in charges.values())


def resume(root, coordinator, gpus):
    receipt = verify(root, coordinator)
    shared.require(sys.executable == shared.read_json(OUT / 'manifest.json')['scientific_python_executable'],
                   'Use the original declared scientific Python environment')
    queue = AdministrativeQueue(frozen.ARCHIVE / 'queue', report_root=root / 'reports/forge',
                                on_completion=None, paid_cap_seconds=8000)
    stop, errors = threading.Event(), []
    def guard():
        while not stop.wait(20):
            try:
                frozen.storage(frozen.ARCHIVE)
            except ValueError as error:
                errors.append(str(error)); queue.pause(frozen.CAMPAIGN, True)
                shared.emit('storage_stop', reason=str(error))
                return
    watcher = threading.Thread(target=guard, daemon=True)
    watcher.start()
    shared.emit('administrative_resume', **receipt)
    try:
        frozen.drain(queue, gpus.split(','), workers_per_gpu=1, allow_sharing=False, watch=False, poll_seconds=1)
    finally:
        stop.set(); watcher.join(timeout=2)
    state = queue.inspect()
    path = frozen.ARCHIVE / f'{frozen.CAMPAIGN}-progress.json'
    progress = shared.read_json(path)
    shared.require(progress['source_commit'] == EXECUTION and progress['requests'] == REQUESTS
                   and progress['registration_sha256'] == REGISTRATION_SHA, 'Original immutable execution bindings changed')
    progress['phase'] = 'paused' if errors or any(state['submissions'][rid]['status'] in shared.ACTIVE for rid in REQUESTS.values()) else 'complete'
    progress['storage_errors'] = errors
    shared.atomic_json(path, progress)
    shared.emit('administrative_execution_state', **receipt, phase=progress['phase'], storage_errors=errors,
                paid_attempts_after=len(state['charges']), paid_seconds_after=sum(c['seconds'] for c in state['charges']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'drain'))
    parser.add_argument('--administrative-commit', required=True)
    parser.add_argument('--gpus', default='0,1')
    args = parser.parse_args()
    if args.action == 'check':
        shared.emit('administrative_verified', **verify(ROOT, args.administrative_commit))
    else:
        resume(ROOT, args.administrative_commit, args.gpus)
