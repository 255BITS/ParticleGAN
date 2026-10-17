"""Stop only the explicitly owned CPU monitor after the recorded toy failure."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
AREA = Path(__file__).resolve().parent
OUT = ROOT / 'integration/review/validation-cb64-ra7-monitor'
AUDIT = ROOT / 'performance/training-regression/count-review/ra7-prospective'
PID, START = 672040, 165052997
EXPECTED = ['/tmp/pr38-default-env/bin/python', '-u', '-B',
    str(ROOT / 'integration/review/monitor_validation.py'), '--validation',
    str(ROOT / 'validation-cb64-ra7'), '--output', str(OUT), '--watch']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p, v: p.write_text(json.dumps(v, indent=2) + '\n')


def identity():
    proc = Path(f'/proc/{PID}')
    stat = proc.joinpath('stat').read_text()
    fields = stat.rsplit(')', 1)[1].split()
    children = sorted({int(child) for task in proc.joinpath('task').iterdir()
        for child in task.joinpath('children').read_text().split()})
    return dict(pid=PID, startticks=int(fields[19]), process_state=fields[0],
        cmdline=[v.decode() for v in proc.joinpath('cmdline').read_bytes().split(b'\0') if v],
        children=children, stat=stat)


assert not (AREA / 'STOPPED-AT-TOY-FAILURE.json').exists()
summary_path = OUT / 'summary.json'
summary = json.loads(summary_path.read_text())
assert summary['status'] == 'PENDING' and summary['completed'] == 0 and summary['total'] == 16
assert len(summary['records']) == 16 and all(row['primary_status'] == row['acceptance_status'] == 'PENDING'
    and row['canonical_fixture_validity'] == 'UNVERIFIED' for row in summary['records'])
assert summary['counts'] == {'portability': {'PENDING': 13}, 'native': {'PENDING': 3}}
assert summary['source_integrity']['status'] == 'VALID'
final = json.loads((AUDIT / 'FINAL-RECEIPT.json').read_text())
assert final['status'] == 'VALID' and final['original_saved_metric_gate']['status'] == 'FAIL'
assert final['original_saved_metric_gate']['checks'] == dict(precision=False, coverage=True, mass_tv=False)
assert all(sha(Path(path)) == digest for path, digest in final['verified_hashes'].items())
result_path = ROOT / 'validation-cb64-ra7/learned/training/toy/CB64-RA7/result.json'
result = json.loads(result_path.read_text())
assert result['status'] == 'COMPLETE' and result['steps'] == 2000
assert result['final']['metrics'] == final['final_metrics']
first = identity()
assert first['startticks'] == START and first['cmdline'] == EXPECTED and not first['children']
assert first['process_state'] in {'R', 'S'}
write(AREA / 'identity-before-stop.json', first)
# Recheck identity immediately before sending the one authorized signal.
second = identity()
assert second['startticks'] == START and second['cmdline'] == EXPECTED and not second['children']
os.kill(PID, signal.SIGTERM)
terminal = None
for _ in range(30):
    if not Path(f'/proc/{PID}').exists():
        terminal = 'gone'; break
    stat = Path(f'/proc/{PID}/stat').read_text().rsplit(')', 1)[1].split()
    assert int(stat[19]) == START, 'PID was reused; no further signal permitted'
    if stat[0] == 'Z': terminal = 'zombie'; break
    time.sleep(.1)
assert terminal is not None, 'owned watcher did not acknowledge SIGTERM; no other signal sent'
# Preserve stable final bytes after the owned writer has exited.
for original, name in [(summary_path, 'summary-at-stop.json'),
        (AUDIT / 'FINAL-RECEIPT.json', 'FINAL-RECEIPT.json'),
        (AUDIT / 'FINAL-FROZEN.json', 'FINAL-FROZEN.json')]:
    (AREA / name).write_bytes(original.read_bytes())
assert json.loads((AREA / 'summary-at-stop.json').read_text()) == summary
source_integrity = summary['source_integrity']
receipt = dict(status='STOPPED_AT_TOY_FAILURE', utc=datetime.now(timezone.utc).isoformat(),
    authorization='Root explicitly requested stop of this owned RA7 CPU metadata watcher after final negative audit.',
    owned_pid=PID, owned_startticks=START, exact_cmdline=EXPECTED, no_children_before_signal=True,
    signal='SIGTERM', terminal=terminal, watcher_source_sha256=sha(Path(EXPECTED[3])),
    source_integrity=source_integrity, final_saved_artifact_status='VALID',
    verified_final_artifact_source_files=len(final['verified_hashes']),
    final_verified_hashes=final['verified_hashes'],
    quality=final['original_saved_metric_gate'], final_metrics=final['final_metrics'],
    canonical_screens=dict(completed=0, total=16, all_pending=True, fixture_validity='UNVERIFIED',
        unrun_quality_verdict=None), parent_original_ra4_processes_signaled=False,
    gpu_processes_signaled=False, numerical_jobs_started=0, torch_imported=False,
    numerical_inputs_sources_modified=False,
    evidence_sha256={str(path): sha(path) for path in [summary_path,
        AUDIT / 'FINAL-RECEIPT.json', AUDIT / 'FINAL-FROZEN.json', result_path,
        AREA / 'summary-at-stop.json', AREA / 'identity-before-stop.json', Path(__file__)]})
write(AREA / 'STOPPED-AT-TOY-FAILURE.json', receipt)
print(json.dumps(dict(status=receipt['status'], pid=PID, terminal=terminal,
    verified_files=len(final['verified_hashes']), canonical_completed=0, canonical_pending=16)), flush=True)
