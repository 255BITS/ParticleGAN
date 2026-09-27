#!/usr/bin/env python3
"""Launch a reviewed list of existing API ports in one external Astra/max lane."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

from launch_public_controls import HERE, WORKSPACE, DRIVER, DRIVER_SHA, SEAL_SHA, API, BASE, RUNS, GPUS, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', required=True, help='Reviewed port directory name')
    parser.add_argument('--lane', required=True)
    parser.add_argument('--gpu-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    assert args.lane.replace('-', '').replace('_', '').isalnum()
    assert len(set(args.candidate)) == len(args.candidate)
    assert sha(DRIVER) == DRIVER_SHA and sha(HERE / 'harness-sha256.json') == SEAL_SHA
    seal = json.loads((HERE / 'harness-sha256.json').read_text())
    for name, expected in seal['files'].items():
        assert sha(HERE / name) == expected, name
    candidates = []
    for name in args.candidate:
        assert name.replace('-', '').isalnum()
        source = HERE / 'port-source' / name
        declaration = json.loads((source / 'candidate-declaration.json').read_text())
        cpu = HERE / f'{name}-cpu-preflight.json'
        review = HERE / f'{name}-independent-init-audit.json'
        c, r = json.loads(cpu.read_text()), json.loads(review.read_text())
        assert c['status'] == c['cpu_initialization']['status'] == 'PASS'
        assert r['status'] == 'PASS_INITIALIZATION_ONLY'
        digest = sha(source / 'candidate-declaration.json')
        assert c['declaration_sha256'] == r['declaration_sha256'] == digest
        assert r['cpu_receipt_sha256'] == sha(cpu)
        assert r['port_manifest_sha256'] == sha(source / 'port-manifest.json')
        for path, expected in declaration['package_sha256'].items():
            assert sha(source / 'package' / path) == expected, path
        candidates.append(dict(name=name, source=source, declaration=declaration,
                               cpu=cpu, review=review))
    spec = importlib.util.spec_from_file_location('prior_launcher', HERE.parent / 'continuous-eligibility/launch/launch.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    live = launcher.live_drivers(DRIVER)
    if not args.launch:
        print(json.dumps(dict(status='REVIEWED_READY', candidates=args.candidate, live=live)))
        return
    with (RUNS.parent / 'continuous-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not (RUNS.parent / 'STOP').exists() and not (RUNS / 'STOP').exists()
        live = launcher.live_drivers(DRIVER)
        assert len(live) < 3 and sum(row['workers'] for row in live) < 3, 'No worker reservation'
        records = json.loads((RUNS / 'batch.json').read_text())
        gpu = GPUS[args.gpu_index]
        live_ids = {row['pid'] for row in live}
        on_gpu = [row for row in records if row['pid'] in live_ids and row['gpu'] == gpu]
        assert len(on_gpu) < (2 if args.gpu_index == 0 else 1), 'Assigned GPU lane full'
        previous_candidates = {name for row in records for name in row.get('candidates', [])}
        assert not previous_candidates.intersection(args.candidate), 'No unreviewed duplicate attempt'
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory = RUNS / stamp / args.lane
        directory.mkdir(parents=True, exist_ok=False)
        frozen = directory / 'reviewed-inputs'
        for name in (*seal['files'], 'harness-sha256.json'):
            destination = frozen / 'harness' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / name, destination)
        for candidate in candidates:
            destination = frozen / candidate['name']
            destination.mkdir()
            for path, expected in candidate['declaration']['package_sha256'].items():
                target = destination / 'package' / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(candidate['source'] / 'package' / path, target)
                assert sha(target) == expected
            for name in ('candidate-declaration.json', 'port-manifest.json', 'initialization-port.patch'):
                shutil.copyfile(candidate['source'] / name, destination / name)
            for name in ('cpu', 'review'):
                shutil.copyfile(candidate[name], destination / candidate[name].name)
        brief = directory / 'brief.md'
        brief.write_text(f'''Retest ONLY these already reviewed existing candidates with the new deterministic initialization, in this order: {', '.join(args.candidate)}. This is a finite replay batch, not a new mechanism search. Read AGENTS.md and promptly start the first GPU test; source and CPU initialization reviews are complete.

Read-only reviewed inputs: {frozen}. Copy them into your own checkout under reports/fixed-init-inputs/ with byte-exact files. Each candidate directory contains complete package/, candidate-declaration.json, port manifest/diff, CPU zero-step preflight and independent initialization-only audit. Verify the declared package hashes, audit/declaration hash binding and harness/harness-sha256.json. No full test suite or additional source implementation is required.

For each listed candidate, execute the copied harness/mode_hold_harness.py through /tmp/pr38-default-env/bin/python, --package-root reports/fixed-init-inputs/CANDIDATE/package, --declaration reports/fixed-init-inputs/CANDIDATE/candidate-declaration.json, --output reports/fixed-init-mode-hold/CANDIDATE (new directory). Set CUDA_VISIBLE_DEVICES={gpu}, CUBLAS_WORKSPACE_CONFIG=:4096:8 and all OMP/MKL/OPENBLAS/NUMEXPR thread limits1 explicitly. One GPU worker; run candidates sequentially and capture readable per-candidate logs. Full1200 CUDA sampling receipt validation is already part of the harness before update1; do not run a duplicate preflight.

Preserve the exact reviewed package, actual deterministic network and recipe-prior initialization, prior-derived geometry, declared recipe policy, sample streams, all24 observations and final5 rule. No old weights, seed variants, coefficients, source repair, schedule changes or new mechanisms. Keep native optimizer counters on their explicitly declared CPU/parameter device; the driver generic CUDA wording does not override candidate-owned counter placement. All gradients, parameters, moments and training updates stay CUDA. Full-step restoring serial scope is supplied by the harness.

On ERROR preserve all sources/partial state/logs, report the exact failure, and continue only to the next independently listed candidate. Never alter a failing package or harness, or retry without supervisor review. On PASS/FAIL preserve every observation and report arrival, departures and final suffix, not just endpoint. Append each actual gate to tests.jsonl, keep result.md updated with exact evidence paths and hashes. A screen pass is not qualification. Do not run broader or long tests. Exit after all {len(candidates)} specified configurations complete or error. No nested sessions, pushes or PR edits.
''')
        command = [str(DRIVER), '--engine', 'codex', '--model', 'gpt-6-astra', '--repo', str(API),
                   '--base', BASE, '--gpu', gpu, '--minutes', '0', '--candidates', str(len(candidates)),
                   '--workers', '1', '--runs-dir', str(directory), '--prompt-file', str(brief), '--focus', args.lane]
        with (directory / 'launcher.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=WORKSPACE,
                env=dict(os.environ, GAN_PYTHON='/tmp/pr38-default-env/bin/python'), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(lane=args.lane, candidates=args.candidate, directory=str(directory), command=command,
            gpu=gpu, pid=child.pid, base=BASE, engine='codex', model='gpt-6-astra', reasoning_effort='max',
            minutes=0, workers=1, brief_sha256=sha(brief), source_sha256=sha(Path(__file__)),
            driver_sha256=DRIVER_SHA, harness_sha256=SEAL_SHA,
            candidate_reviews={row['name']: sha(row['review']) for row in candidates})
        records.append(record)
        write_json(RUNS / 'batch.json', records)
        print(json.dumps(dict(started=record)), flush=True)


if __name__ == '__main__':
    main()
