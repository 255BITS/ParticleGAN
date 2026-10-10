#!/usr/bin/env python
"""Apply resource guards, then invoke the original, unmodified screen.py API."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

sys.dont_write_bytecode = True
from lane import (CONFIG, ENV, GPU0_UUID, HARNESS, OPTIONS, PACKAGE, ROOT, SCREEN, TASKS, UNSET,
    command, now, read, sha, task_plan, verify_frozen, write)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=PACKAGE)
    parser.add_argument('--overrides', type=Path, default=CONFIG)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--candidate-options', type=Path, default=ROOT / 'candidate-options.json')
    parser.add_argument('--cand', default='CB64-RA6')
    parser.add_argument('--check-only', action='store_true', help='verify static receipt; never import torch or train')
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / 'runs' / args.task
    if args.package_root.resolve() != PACKAGE or args.overrides.resolve() != CONFIG:
        parser.error('only the frozen CB64-RA6 package and config are authorized')
    if args.output.resolve() != ROOT / 'runs' / args.task:
        parser.error('output must be this lane\'s runs/TASK directory')
    if args.device != 'cuda:0' or args.cand != 'CB64-RA6':
        parser.error('device cuda:0 and candidate CB64-RA6 are required')
    if args.candidate_options.resolve() != ROOT / 'candidate-options.json' or read(args.candidate_options) != OPTIONS:
        parser.error('candidate options must equal the frozen four-option file')
    integrity = verify_frozen()
    if args.check_only:
        print(json.dumps(dict(event='static_check', task=args.task, integrity=integrity,
            plan=task_plan(args.task), command=command(args.task))), flush=True)
        return 0
    with (ROOT / '.screen-lane.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another screen owns this lane; coordinator must serialize jobs')
        out = args.output.resolve()
        out.mkdir(parents=True, exist_ok=True)
        previous = [p.name for p in out.iterdir() if p.name != 'run.log']
        if previous:
            raise RuntimeError(f'one authorized run per task; output already contains {previous}')
        for key in UNSET:
            os.environ.pop(key, None)
        os.environ.update(ENV)
        started = time.monotonic()
        receipt = dict(status='RUNNING', started_at=now(), task=args.task,
            exact_command=[sys.executable, *sys.argv], original_screen=str(SCREEN),
            source_integrity_before=integrity, source_freeze_sha256=sha(ROOT / 'source-freeze.json'),
            ready_sha256=sha(ROOT / 'READY.json'), expected=task_plan(args.task),
            environment={k: os.environ[k] for k in ENV}, unset_environment={k: k not in os.environ for k in UNSET},
            resources=dict(physical_gpu=0, device='cuda:0', expected_gpu_uuid=GPU0_UUID,
                cuda_memory_fraction=.2, numeric_threads=1),
            fixture_policy='original CUDA checks stay active; mismatches are not bypassed',
            canonical_gpu_acceptance='PENDING', verdict_scope='original frozen CUDA screen, live noisy primary')
        write(out / 'execution-receipt.json', receipt)
        print(json.dumps(dict(event='screen_start', task=args.task, expected=receipt['expected'],
            resources=receipt['resources'], source_integrity=integrity)), flush=True)
        error = None
        code = 0
        torch = None
        try:
            if any(n == 'particlegan' or n.startswith('particlegan.') for n in sys.modules):
                raise RuntimeError('candidate requires a fresh process')
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable; canonical GPU run cannot proceed')
            if torch.cuda.device_count() != 1:
                raise RuntimeError('CUDA_VISIBLE_DEVICES=0 did not expose exactly one device')
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(.2, device=0)
            props = torch.cuda.get_device_properties(0)
            raw_uuid = str(props.uuid)
            actual_uuid = 'GPU-' + raw_uuid.removeprefix('GPU-').lower()
            receipt['resources'].update(gpu_name=props.name, gpu_uuid=actual_uuid, cuda_property_uuid=raw_uuid,
                total_gpu_bytes=props.total_memory,
                allocator_limit_bytes=int(.2 * props.total_memory))
            if actual_uuid != GPU0_UUID:
                raise RuntimeError(f'physical GPU0 UUID {actual_uuid} != frozen authorized UUID {GPU0_UUID}')
            receipt.update(torch=str(torch.__version__), cuda=torch.version.cuda)
            write(out / 'execution-receipt.json', receipt)
            sys.path.insert(0, str(HARNESS))
            sys.argv = [str(SCREEN), '--package-root', str(PACKAGE), '--overrides', str(CONFIG),
                '--task', args.task, '--output', str(out), '--device', 'cuda:0',
                '--candidate-options', str(args.candidate_options), '--cand', 'CB64-RA6']
            receipt['original_api_argv'] = list(sys.argv)
            runpy.run_path(str(SCREEN), run_name='__main__')
        except SystemExit as exc:
            code = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
            if code:
                error = dict(error=repr(exc), traceback=traceback.format_exc())
        except Exception as exc:
            code = 1
            error = dict(error=repr(exc), traceback=traceback.format_exc())
            print(error['traceback'], file=sys.stderr, flush=True)
        finally:
            if torch is not None and torch.cuda.is_initialized():
                receipt['peak_allocated_gpu_mib'] = round(torch.cuda.max_memory_allocated(0) / 2**20, 3)
                receipt['peak_reserved_gpu_mib'] = round(torch.cuda.max_memory_reserved(0) / 2**20, 3)
            try:
                receipt['source_integrity_after'] = verify_frozen()
            except Exception as exc:
                code = 1
                receipt['source_integrity_after'] = dict(status='INVALID', error=repr(exc))
            if error:
                write(out / 'wrapper-error.json', error)
            if not (out / 'result.json').exists():
                code = 1
                write(out / 'result.json', dict(status='ERROR', task=args.task, cand='CB64-RA6',
                    result_origin='wrapper failure; original harness produced no result',
                    **(error or {'error': 'original harness produced no result.json'})))
            result = read(out / 'result.json')
            receipt.update(status='COMPLETE' if code == 0 else 'ERROR', finished_at=now(),
                wall_seconds=round(time.monotonic() - started, 3), process_exit_code=code,
                primary_status=result.get('status'), result_sha256=sha(out / 'result.json'))
            write(out / 'execution-receipt.json', receipt)
            print(json.dumps(dict(event='screen_end', task=args.task, primary_status=result.get('status'),
                process_exit_code=code, peak_allocated_gpu_mib=receipt.get('peak_allocated_gpu_mib'),
                peak_reserved_gpu_mib=receipt.get('peak_reserved_gpu_mib'), wall_seconds=receipt['wall_seconds'])), flush=True)
        return code

if __name__ == '__main__':
    raise SystemExit(main())
