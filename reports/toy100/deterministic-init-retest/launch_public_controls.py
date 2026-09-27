#!/usr/bin/env python3
"""Launch exactly three reviewed, existing controls through external Codex."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
WORKSPACE = Path('/ml2/hypergan')
DRIVER = WORKSPACE / 'try-gan.sh'
DRIVER_SHA = 'b6da05b968b22216588aea85ac99ebb2b56abbb4b896b10c5f78f135512b18c2'
SEAL_SHA = '15ad9cf18d5fd0387d1f724bbd3ceec45f2474858d51fded318f3fa43f6ddc76'
API = WORKSPACE / 'ParticleGAN-ka2-default'
BASE = '25751c0864dd8259b00c5804f600cd41cce6e4cf'
RUNS = WORKSPACE / 'gan-attempts/deterministic-init-retest-20260927'
GPUS = ['GPU-72c1b506-891d-b8bc-b353-e020585e1c47',
        'GPU-cb4ce47d-d968-bffd-5646-e830a9fa1c69',
        'GPU-72c1b506-891d-b8bc-b353-e020585e1c47']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    assert sha(DRIVER) == DRIVER_SHA
    assert sha(HERE / 'harness-sha256.json') == SEAL_SHA
    seal = json.loads((HERE / 'harness-sha256.json').read_text())
    for name, expected in seal['files'].items():
        assert sha(HERE / name) == expected, name
    review = json.loads((HERE / 'port-api-review.json').read_text())['harness_review']
    assert review['status'] == 'NO_SOURCE_LAUNCH_BLOCKER_FOR_PUBLIC_THREE_CONTROLS'
    assert review['seal_sha256'] == SEAL_SHA
    spec = importlib.util.spec_from_file_location('prior_launcher', HERE.parent / 'continuous-eligibility/launch/launch.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    controls = []
    for name, gpu in zip(('public-k3p', 'public-ka2', 'public-ka2-constant'), GPUS):
        declaration = json.loads((HERE / f'{name}-declaration.json').read_text())
        source = Path(declaration['algorithm_source']['path'])
        for path, expected in declaration['package_sha256'].items():
            assert sha(source / path) == expected, path
        controls.append(dict(name=name, gpu=gpu, source=str(source), declaration=declaration))
    if not args.launch:
        print(json.dumps(dict(status='REVIEWED_READY', controls=[c['name'] for c in controls],
                              live=launcher.live_drivers(DRIVER), harness_sha256=SEAL_SHA), indent=2))
        return
    RUNS.parent.mkdir(exist_ok=True)
    with (RUNS.parent / 'continuous-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not (RUNS.parent / 'STOP').exists() and not (RUNS / 'STOP').exists()
        assert not launcher.live_drivers(DRIVER), 'Public three need all three worker reservations'
        assert not (RUNS / 'batch.json').exists(), 'Already launched; do not repeat the same controls'
        RUNS.mkdir(exist_ok=True)
        frozen = RUNS / 'public-controls-source'
        frozen.mkdir()
        for name in (*seal['files'], 'harness-sha256.json'):
            destination = frozen / 'harness' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / name, destination)
        write_json(frozen / 'harness-review.json', review)
        records = []
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        for control in controls:
            name, gpu = control['name'], control['gpu']
            for path in control['declaration']['package_sha256']:
                destination = frozen / name / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(control['source']) / path, destination)
                assert sha(destination) == control['declaration']['package_sha256'][path]
            directory = RUNS / stamp / name
            directory.mkdir(parents=True)
            brief = directory / 'brief.md'
            brief.write_text(f'''Rerun the existing {name} control using the new deterministic initialization. This is one fixed configuration, not a new mechanism search. Read AGENTS.md, then begin the already-reviewed test promptly.

Read-only sealed inputs: {frozen}. Copy harness/ and the complete {name}/ package into your own checkout under reports/fixed-init-control-inputs/ without modifying any bytes. Verify all hashes in harness/harness-sha256.json and the package hashes in harness/{name}-declaration.json. Independent complete-source review is in harness-review.json. CPU zero-step initialization checks already passed; no repeat test suites are needed before this fixed CUDA test.

Execute exactly the supplied mode_hold_harness.py, with --package-root pointing at the copied {name} package root, --declaration pointing at the copied harness/{name}-declaration.json, and --output pointing to a new reports/fixed-init-mode-hold directory in your checkout. Use /tmp/pr38-default-env/bin/python with CUDA_VISIBLE_DEVICES={gpu}, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and NUMEXPR_NUM_THREADS=1. Capture stdout/stderr to a readable log. The script runs the entire 1200-draw CUDA sampling preflight before updates, then 1200 public API updates and all 24 observations. Do not execute a separate duplicate CUDA preflight.

Use actual new deterministic network AND recipe-created prior initialization. No old weight restore, caller random-prior substitution, source edits, seed changes, schedule changes, hyperparameter sweep, new candidate or metric changes. Preserve the declared policy horizon/noise. Constant KA2 retains its historical4600 recipe horizon and360/720 noise; scheduled controls use1200/120/240, so this is not an LR-only ablation.

The native noncapturable Adam scalar step counters are CPU metadata; parameter tensors, moments, gradients and all updates are CUDA. The generic driver wording must not cause moving/repairing these counters. The harness validates declared device ownership and full-step serial scope itself.

If any assertion/runtime error occurs, preserve the output/error/source/log artifacts and report ERROR with the exact cause; do not fix or rerun without supervisor review. On completion report quality pass/fail, all24 observations, final-five suffix, mode/HQ extrema, time, source/declaration hashes and exact evidence directory in result.md and tests.jsonl. A quick-screen pass does not establish indefinite eligibility or release qualification. Do not run any broader/long test yet. Exit after this one result. No GPU training outside this external worker, no nested agents, no pushes or PR changes.
''')
            assert not (RUNS.parent / 'STOP').exists() and not (RUNS / 'STOP').exists()
            command = [str(DRIVER), '--engine', 'codex', '--model', 'gpt-6-astra',
                       '--repo', str(API), '--base', BASE, '--gpu', gpu,
                       '--minutes', '0', '--candidates', '1', '--workers', '1',
                       '--runs-dir', str(directory), '--prompt-file', str(brief), '--focus', name]
            env = dict(os.environ, GAN_PYTHON='/tmp/pr38-default-env/bin/python')
            with (directory / 'launcher.log').open('wb') as log:
                child = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record = dict(lane=name, directory=str(directory), command=command, gpu=gpu,
                          pid=child.pid, base=BASE, engine='codex', model='gpt-6-astra',
                          reasoning_effort='max', minutes=0, workers=1,
                          brief_sha256=sha(brief), source_sha256=sha(Path(__file__)),
                          driver_sha256=DRIVER_SHA, harness_sha256=SEAL_SHA,
                          harness_review_sha256=sha(frozen / 'harness-review.json'))
            records.append(record)
            write_json(RUNS / 'batch.json', records)
            print(json.dumps({'started': record}), flush=True)
        registry = RUNS.parent / 'active-batches.txt'
        previous = registry.read_text().splitlines() if registry.exists() else []
        if str(RUNS) not in previous:
            registry.write_text('\n'.join(previous + [str(RUNS)]) + '\n')


if __name__ == '__main__':
    main()
