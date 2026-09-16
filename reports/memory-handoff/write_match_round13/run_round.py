"""One-shot round13 pipeline: completion-only drain, fixed selection, diagnostics."""
import json
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[3]
report = Path(__file__).resolve().parent
queue = root/'runs/memory_path/write_match_round13'
python = sys.executable
dispatch = [python, str(root/'experiments/memory_dispatch.py')]
trainer = str(root/'experiments/memory_handoff_scout.py')
baseline = root/'runs/memory_path/principles_round12/runs/match_shuffle25'
baseline5 = root/'runs/memory_path/principles_round12_followup/runs/match_shuffle25_5k'


def run_queue(path, configs, destination):
    subprocess.run(dispatch+['add', '--queue', str(path), '--trainer', trainer,
                            '--configs', *map(str, configs)], cwd=root, check=True)
    subprocess.run(dispatch+['seal', '--queue', str(path)], cwd=root, check=True)
    subprocess.run(dispatch+['drain', '--queue', str(path), '--devices', 'cuda:0', 'cuda:1',
        '--reporter', str(root/'experiments/analyze_memory_handoff.py'),
        '--report-out', str(destination)], cwd=root, check=True)


def diagnostics(runs, destination):
    # Both commands require completed summaries. No in-progress checkpoints/logs.
    jobs = []
    with (destination/'diagnostics.log').open('w') as log:
        for script, filename, extra in (
            ('diagnose_memory_process.py', 'process.json', ['--recompute-original']),
            ('diagnose_memory_local_signal.py', 'signal.json', []),
        ):
            jobs.append(subprocess.Popen([python, str(root/'experiments'/script),
                '--runs', *map(str, runs), '--out', str(destination/filename), *extra],
                cwd=root, stdout=log, stderr=subprocess.STDOUT))
        codes = [job.wait() for job in jobs]
        assert not any(codes), f'Diagnostic failure: {codes}; see {destination}/diagnostics.log'
    print(json.dumps(dict(event='diagnostics_complete', report=str(destination))), flush=True)


configs = root/'experiments/configs/memory_handoff/write_match_round13'
order = ('mixed_mild', 'explored_mild', 'mixed_full', 'explored_full', 'mixed_mild_headonly')
run_queue(queue, [configs/f'{name}.json' for name in order], report)
subprocess.run([python, str(report/'select_extensions.py')], cwd=root, check=True,
               stdout=subprocess.DEVNULL)
selected = json.loads((report/'extension_decision.json').read_text())['selected']
print(json.dumps(dict(event='selection_complete', selected=selected)), flush=True)

if selected:
    followup = queue.with_name('write_match_round13_followup')
    followup.mkdir(exist_ok=False)
    followup_report = report/'followup'
    followup_report.mkdir(exist_ok=True)
    (followup/'train.log').symlink_to('../core_round1/train.log')
    sources = [baseline, baseline5]+[queue/'runs'/name for name in selected]
    (followup/'report_baselines.json').write_text(json.dumps(list(map(str, sources)))+'\n')
    folder = configs.with_name('write_match_round13_followup')
    folder.mkdir(exist_ok=True)
    paths = []
    for name in selected:
        cfg = json.loads((configs/f'{name}.json').read_text())
        cfg.update(name=name+'_5k', steps=5000, resume=str(queue/'runs'/name/'model.pt'))
        path = folder/f'{name}_5k.json'
        path.write_text(json.dumps(cfg, indent=2)+'\n')
        paths.append(path)
    run_queue(followup, paths, followup_report)
    diagnostics(sorted((followup/'runs').iterdir()), followup_report)

diagnostics([baseline, baseline5]+[queue/'runs'/name for name in order], report)
print(json.dumps(dict(event='round_finished', selected=selected)), flush=True)
