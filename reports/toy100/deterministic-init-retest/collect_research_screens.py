"""Audit and archive complete prepared research screens without learner execution."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BATCH = Path('/ml2/hypergan/gan-attempts/deterministic-init-retest-20260927/batch.json')


def main():
    index = ROOT / 'research-screen-queue/prepared-index.json'
    rows = {Path(row['directory']).name: row for row in json.loads(index.read_text())['rows']}
    table = json.loads((ROOT / 'research-results.json').read_text())
    recorded = {row['source_directory'] for row in table['results']}
    added, errors = [], []
    for lane in json.loads(BATCH.read_text()):
        for result in Path(lane['directory']).glob('*/repo/reports/reviewed-probe-output/**/result.json'):
            source = result.parent
            name = source.name
            if name not in rows or str(source) in recorded:
                continue
            data = json.loads(result.read_text())
            # Some original mechanisms have no atexit receipt writer. Their
            # in-result receipt is authoritative; allow process teardown first.
            if data['status'] not in ('PASS', 'FAIL') or time.time() - result.stat().st_mtime < 2:
                continue
            audit = ROOT / 'research-mode-hold-review/completed' / (name + '-runtime-audit.json')
            command = [sys.executable, str(ROOT / 'research-mode-hold-review/runtime_audit.py'),
                       '--prepared-index', str(index), '--queue-row', rows[name]['queue_row'],
                       '--source', str(source), '--output', str(audit)]
            outcome = subprocess.run(command, capture_output=True, text=True)
            if outcome.returncode:
                errors.append(dict(candidate=name, error=outcome.stderr[-4000:]))
                continue
            reviewed = json.loads(audit.read_text())
            assert reviewed['status'] == 'PASS' and isinstance(reviewed['historical_eligibility'], str)
            subprocess.run([sys.executable, str(ROOT / 'archive_research_screen.py'), '--audit', str(audit),
                            '--id', 'research-' + name + '-new-init', '--eligibility', reviewed['historical_eligibility']], check=True)
            recorded.add(str(source))
            added.append(name)
    print(json.dumps(dict(archived=added, audit_errors=errors)))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
