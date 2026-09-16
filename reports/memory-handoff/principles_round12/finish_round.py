"""Wait for completion without polling, apply fixed gates, and drain selected extensions."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[3]
queue = root/'runs/memory_path/principles_round12'
report = Path(__file__).resolve().parent
python = str(root/'.venv/bin/python')

# Drain owns this lock until all workers and their completed-only reports finish.
with (queue/'drain.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    assert (queue/'SEALED').exists()
    assert not list((queue/'running').glob('*.json'))
    assert not list((queue/'pending').glob('*.json'))
    assert not list((queue/'failed').glob('*.json'))
    assert len(list((queue/'done').glob('*.json'))) == 12

subprocess.run([python, str(report/'select_extensions.py')], cwd=root, check=True,
               stdout=subprocess.DEVNULL)
selected = json.loads((report/'extension_decision.json').read_text())['selected']
print(json.dumps(dict(event='selection_complete', selected=selected)), flush=True)
followup = queue.with_name('principles_round12_followup')
followup_report = report/'followup'
process = None
if selected:
    followup.mkdir(exist_ok=False)
    followup_report.mkdir(exist_ok=True)
    (followup/'train.log').symlink_to('../core_round1/train.log')
    baselines = [root/'runs/memory_path/recovery_round10/runs/proposal_mixed_pair25']
    baselines += [queue/'runs'/name for name in selected]
    (followup/'report_baselines.json').write_text(json.dumps([str(p) for p in baselines])+'\n')
    configs = root/'experiments/configs/memory_handoff/principles_round12_followup'
    configs.mkdir(exist_ok=True)
    paths = []
    for name in selected:
        config = json.loads((queue/'runs'/name/'input.json').read_text())
        config.update(name=name+'_5k', steps=5000, resume=str(queue/'runs'/name/'model.pt'))
        path = configs/(config['name']+'.json')
        path.write_text(json.dumps(config, indent=2)+'\n')
        paths.append(str(path))
    cmd = [python, str(root/'experiments/memory_dispatch.py')]
    subprocess.run(cmd+['add', '--queue', str(followup), '--trainer',
        str(root/'experiments/memory_handoff_scout.py'), '--configs', *paths], cwd=root, check=True)
    subprocess.run(cmd+['seal', '--queue', str(followup)], cwd=root, check=True)
    process = subprocess.Popen(cmd+['drain', '--queue', str(followup), '--devices', 'cuda:0', 'cuda:1',
        '--reporter', str(root/'experiments/analyze_memory_handoff.py'), '--report-out',
        str(followup_report)], cwd=root)

scouts = sorted(queue.joinpath('runs').iterdir())
subprocess.run([python, str(root/'experiments/diagnose_memory_local_signal.py'), '--runs',
    *map(str, scouts), '--out', str(report/'signal.json')], cwd=root, check=True)
if process is not None:
    assert process.wait() == 0
    extensions = sorted(followup.joinpath('runs').iterdir())
    subprocess.run([python, str(root/'experiments/diagnose_memory_local_signal.py'), '--runs',
        *map(str, extensions), '--out', str(followup_report/'signal.json')], cwd=root, check=True)
print(json.dumps(dict(event='round_finished', selected=selected)), flush=True)
