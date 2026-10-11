"""Read-only, tail-friendly required-gate progress for the external campaign."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('/mnt/ml7tb/ParticleGAN-forge/bcap-tier2-round6-20261011'))
    options = parser.parse_args()
    while True:
        try:
            progress = json.loads((options.artifacts / 'progress.json').read_text())
            state = json.loads((options.artifacts / 'queue/queue/state.json').read_text())
            arms = {}
            for role, rid in progress['requests'].items():
                sub = state['submissions'][rid]
                assignments = {a['task']: a for a in sub['request']['view']['assignments']}
                jobs = [j for j in state['jobs'].values() if rid in j.get('subscribers', [])]
                rows = [r for j in jobs if j.get('result') for r in j['result']['task_results']]
                arms[role] = {'status': sub['status'],
                    'running': [j['definition']['task_id'] for j in jobs if j['status'] == 'running'],
                    'required': {str(t): dict(Counter(r['gate_status'] for r in rows
                        if assignments[r['task_id']]['importance'] == 'required'
                        and assignments[r['task_id']]['qualification_tier'] == t)) for t in (1, 2)}}
            print(json.dumps({'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'phase': progress['phase'], 'arms': arms,
                'charged_seconds': round(sum(c['seconds'] for c in state['charges']), 3)}), flush=True)
            if progress['phase'] == 'complete':
                return
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
            print(json.dumps({'monitor_wait': str(error)}), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
