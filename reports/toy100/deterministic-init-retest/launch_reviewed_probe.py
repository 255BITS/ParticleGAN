#!/usr/bin/env python3
"""Launch an explicitly reviewed research, diagnostic, or continuation batch."""
import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess

from launch_public_controls import HERE, WORKSPACE, DRIVER, DRIVER_SHA, API, BASE, RUNS, GPUS, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--gpu-index', type=int, choices=(0, 1), required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert spec['status'] == 'REVIEWED_READY_FOR_EXTERNAL_RUN'
    assert spec['lane'].replace('-', '').isalnum()
    assert spec['cases'] and len({case['id'] for case in spec['cases']}) == len(spec['cases'])
    assert sha(DRIVER) == DRIVER_SHA
    for group, source in spec['sources'].items():
        assert group.replace('-', '').isalnum()
        for name, digest in source['files'].items():
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts
            assert sha(Path(source['root']) / name) == digest, name
    for proof in spec['independent_reviews']:
        assert sha(Path(proof['path'])) == proof['sha256']
        assert json.loads(Path(proof['path']).read_text())['status'] == proof['required_status']
    assert spec['independent_reviews']
    module_spec = importlib.util.spec_from_file_location('prior_launcher', HERE.parent / 'continuous-eligibility/launch/launch.py')
    launcher = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(launcher)
    if not args.launch:
        print(json.dumps(dict(status='REVIEWED_READY', cases=[case['id'] for case in spec['cases']],
                              live=launcher.live_drivers(DRIVER))))
        return
    with (RUNS.parent / 'continuous-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not (RUNS.parent / 'STOP').exists() and not (RUNS / 'STOP').exists()
        live = launcher.live_drivers(DRIVER)
        assert len(live) < 3 and sum(row['workers'] for row in live) < 3
        records = json.loads((RUNS / 'batch.json').read_text())
        digest = sha(args.spec)
        assert digest not in {row.get('probe_spec_sha256') for row in records}, 'Already launched this exact probe'
        gpu = GPUS[args.gpu_index]
        live_ids = {row['pid'] for row in live}
        assert sum(row['pid'] in live_ids and row['gpu'] == gpu for row in records) < (2 if args.gpu_index == 0 else 1)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        directory = RUNS / stamp / spec['lane']
        directory.mkdir(parents=True, exist_ok=False)
        inputs = directory / 'reviewed-inputs'
        for group, source in spec['sources'].items():
            for name, expected in source['files'].items():
                target = inputs / group / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(source['root']) / name, target)
                assert sha(target) == expected
        for index, proof in enumerate(spec['independent_reviews']):
            shutil.copyfile(proof['path'], inputs / f'independent-review-{index}.json')
        shutil.copyfile(args.spec, directory / 'probe-spec.json')
        commands = '\n'.join('- ' + case['id'] + ': ' + shlex.join(case['argv']) for case in spec['cases'])
        brief = directory / 'brief.md'
        brief.write_text(f'''Execute only this reviewed fixed batch: {spec['lane']}. Read AGENTS.md and start the first test promptly. No new mechanisms, seeds, coefficient sweeps, host/budget changes, repairs or unlisted experiments.

Exact reviewed source inputs: {inputs}. Copy the full directory into your own checkout as reports/reviewed-probe-inputs and verify every file against {directory / 'probe-spec.json'}. Expected membership includes each sources/GROUP/files entry plus top-level independent-review-N.json copies, whose hashes are spec.independent_reviews[N].sha256. Source/CPU checks are already complete and their independent receipts are included. Preserve all bytes. {{inputs}} in commands means the absolute path to your copied inputs. {{output}} means a fresh directory under your checkout reports/reviewed-probe-output. Use these commands exactly after substituting those two paths:
{commands}

Set CUDA_VISIBLE_DEVICES={gpu}, CUBLAS_WORKSPACE_CONFIG=:4096:8, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and NUMEXPR_NUM_THREADS=1 explicitly on each subprocess. One GPU worker, sequential commands only. The benchmark Python is /tmp/pr38-default-env/bin/python. Native CPU Adam scalar counters or explicitly declared historical eager counters retain their reviewed placement; all parameters, gradients, moments and updates are CUDA. Do not normalize or repair state.

Scope and interpretation: {spec['instructions']}

Preserve all raw sources, initial/final/error checkpoints, sample/RNG receipts, rates, controller diagnostics, every observation and exceptions. Save readable per-case logs, append actual executed gates to tests.jsonl and update result.md with exact artifact paths and scores. On ERROR preserve it and continue only to the next listed independent command; no repair or retry. No broader follow-up beyond this list, extra agents, detached jobs, pushes or PR edits. Exit after these exact commands and concise result recording. Read supervisor.md before each new case.
''')
        command = [str(DRIVER), '--engine', 'codex', '--model', 'gpt-6-astra', '--repo', str(API),
            '--base', BASE, '--gpu', gpu, '--minutes', '0', '--candidates', str(len(spec['cases'])),
            '--workers', '1', '--runs-dir', str(directory), '--prompt-file', str(brief), '--focus', spec['lane']]
        with (directory / 'launcher.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=WORKSPACE,
                env=dict(os.environ, GAN_PYTHON='/tmp/pr38-default-env/bin/python'), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(lane=spec['lane'], candidates=[case['id'] for case in spec['cases']],
            directory=str(directory), command=command, gpu=gpu, pid=child.pid, base=BASE,
            engine='codex', model='gpt-6-astra', reasoning_effort='max', minutes=0, workers=1,
            probe_spec_sha256=digest, brief_sha256=sha(brief), source_sha256=sha(Path(__file__)),
            driver_sha256=DRIVER_SHA)
        records.append(record)
        write_json(RUNS / 'batch.json', records)
        print(json.dumps(dict(started=record)))


if __name__ == '__main__':
    main()
