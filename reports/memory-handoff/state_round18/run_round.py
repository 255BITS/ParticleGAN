"""Append a declared scout set, drain both GPUs, then evaluate completed models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
BASE = ROOT/'runs/memory_path/principles_round12/runs/match_shuffle25'
BASE5 = ROOT/'runs/memory_path/principles_round12_followup/runs/match_shuffle25_5k'


def run(command, **kwargs):
    return subprocess.run([PYTHON, *map(str, command)], cwd=ROOT, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', required=True)
    parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    report = ROOT/'reports/memory-handoff'/args.round
    queue = ROOT/'runs/memory_path'/args.round
    folder = ROOT/'experiments/configs/memory_handoff'/args.round
    names = json.loads((report/'scouts.json').read_text())
    queue.mkdir(exist_ok=True)
    assert not (queue/'SEALED').exists(), 'Use a fresh queue segment after sealing'
    if not (queue/'train.log').exists():
        (queue/'train.log').symlink_to('../core_round1/train.log')
    if not (queue/'report_baselines.json').exists():
        (queue/'report_baselines.json').write_text(json.dumps(list(map(str, (BASE, BASE5))))+'\n')
    dispatch = ROOT/'experiments/memory_dispatch.py'
    run([dispatch, 'add', '--queue', queue, '--trainer', ROOT/'experiments/memory_handoff_scout.py',
         '--configs', *[folder/f'{n}.json' for n in names]])
    run([dispatch, 'seal', '--queue', queue])
    run([dispatch, 'drain', '--queue', queue, '--devices', 'cuda:0', 'cuda:1',
         '--reporter', ROOT/'experiments/analyze_memory_handoff.py', '--report-out', report])
    for state in ('pending', 'running', 'failed'):
        assert not list((queue/state).glob('*.json')), state
    jobs = [json.loads(p.read_text()) for p in (queue/'done').glob('*.json')]
    assert sorted(j['name'] for j in jobs) == sorted(names)
    runs = [queue/'runs'/n for n in names]
    sources = [json.loads((p/'provenance.json').read_text())['sources'] for p in runs]
    assert all(s == sources[0] for s in sources), 'Training sources changed during round'
    execution = dict(done=len(jobs), failed=0, devices=sorted({j['device'] for j in jobs}),
        queue_wall_seconds=max(j['finished'] for j in jobs)-min(j['started'] for j in jobs),
        train_gpu_seconds=sum(json.loads((p/'summary.json').read_text())['train_seconds'] for p in runs),
        sources=sources[0])
    (report/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    if args.diagnostics:
        recurrent_runs = [p for p in runs if json.loads((p/'input.json').read_text()).get('g_state_dim', 0)]
        specs = [
            ('diagnose_memory_information.py', 'information', ['--device', 'cuda:1'], runs),
            ('diagnose_memory_information.py', 'g_information', ['--device', 'cuda:0', '--memory', 'Mg'], recurrent_runs),
            ('diagnose_memory_process.py', 'process', ['--recompute-original'], [BASE, BASE5]+runs),
        ]
        workers, streams = [], []
        try:
            for script, name, extra, paths in specs:
                stream = (report/f'{name}.log').open('w')
                streams.append(stream)
                command = [PYTHON, str(ROOT/'experiments'/script), '--runs', *map(str, paths),
                           '--out', str(report/f'{name}.json'), *extra]
                workers.append(subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
            codes = [worker.wait() for worker in workers]
            assert not any(codes), f'Diagnostic failure {codes}; inspect completed diagnostic logs'
        finally:
            for stream in streams:
                stream.close()
        print(json.dumps(dict(event='diagnostics_complete', report=str(report))), flush=True)
    print(json.dumps(dict(event='round_complete', **execution)), flush=True)


if __name__ == '__main__':
    main()
