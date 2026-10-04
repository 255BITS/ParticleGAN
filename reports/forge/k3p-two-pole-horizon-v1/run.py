"""Run only the four reviewed diagnostic arms, through Forge's budgeted queue."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.forge.contracts import read_json
from experiments.forge.planning import resolve_idea
from experiments.forge.queue import Queue, drain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != args.expected_commit or sys.executable != '/usr/bin/python':
        raise ValueError('Use the exact reviewed commit and declared system scientific Python')
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=ROOT, check=True)
    plan = read_json(ROOT / 'reports/forge/k3p-two-pole-horizon-v1/plans.json')
    study = plan['study']
    queue_root = ROOT / 'runs/forge' / study / 'queue'
    queue = Queue(queue_root, report_root=ROOT / 'reports/forge', on_completion=None)
    if queue.inspect().get('campaigns', {}).get(study):
        raise ValueError('Campaign already admitted; inspect existing work instead of rerunning')
    campaign = read_json(ROOT / 'configs/forge/campaigns' / (study + '.json'))
    for row in plan['candidate_plans']:
        declaration = read_json(ROOT / row['declaration'])
        request = resolve_idea(ROOT, row['candidate_id'], declaration=declaration,
            view_id=plan['view'], through_tier=1, execution_backend='cpu',
            freeze_source=True, queue_root=queue_root)
        assert request['source']['digest'] == plan['source_digest']
        assert request['source']['origin_commit'] == head
        assert request['candidate_revision'] == row['candidate_revision']
        assert request['decision_review']['status'] == 'READY'
        queue.submit(request, campaign)
    print(json.dumps({'event': 'diagnostic_enqueued', 'source_commit': head,
        'source_digest': plan['source_digest'], 'arms': 4, 'maximum_workers': 1,
        'events': str(queue_root / 'events.jsonl')}, sort_keys=True), flush=True)
    drain(queue, ['cpu'], workers_per_gpu=1, allow_sharing=False, campaign=study)
    state = queue.inspect()
    attempts = [a for job in state['jobs'].values() for a in job.get('attempts', [])]
    print(json.dumps({'event': 'diagnostic_complete', 'submissions':
        {k: v['status'] for k, v in state['submissions'].items()}, 'attempts': len(attempts)}, sort_keys=True), flush=True)
    assert all(s['status'] == 'completed' for s in state['submissions'].values()), state['submissions']


if __name__ == '__main__':
    main()
