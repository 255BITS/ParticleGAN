"""Fill at most three external Codex slots from a finite list of reviewed specs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from launch_public_controls import GPUS, RUNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('specs', type=Path, nargs='+')
    args = parser.parse_args()
    launcher = Path(__file__).with_name('launch_reviewed_probe.py')
    for spec in args.specs:
        while True:
            if (RUNS.parent / 'STOP').exists() or (RUNS / 'STOP').exists():
                raise SystemExit('STOP present; no more launches')
            command = [sys.executable, str(launcher), '--spec', str(spec)]
            preview = subprocess.run(command + ['--gpu-index', '0'], check=True, capture_output=True, text=True)
            live = json.loads(preview.stdout)['live']
            records = json.loads((RUNS / 'batch.json').read_text())
            live_ids = {row['pid'] for row in live}
            if len(live) < 3 and sum(row['workers'] for row in live) < 3:
                for gpu_index in (1, 0):
                    used = sum(row['pid'] in live_ids and row['gpu'] == GPUS[gpu_index] for row in records)
                    if used < (1 if gpu_index == 1 else 2):
                        # The launcher rechecks all capacity, source hashes and reviews under its lock.
                        subprocess.run(command + ['--gpu-index', str(gpu_index), '--launch'], check=True)
                        break
                else:
                    raise RuntimeError('Unaccounted external worker allocation')
                break
            time.sleep(10)
    print(json.dumps(dict(status='ALL_REVIEWED_BATCHES_LAUNCHED', count=len(args.specs))), flush=True)


if __name__ == '__main__':
    main()
