"""Post-exit closed-log/result seal for the one authorized descriptive run."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from bindings import HERE, sha, read, write_new, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--startticks', type=int, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    if args.attempt.resolve().parent != HERE or (args.attempt/'FROZEN.json').exists():
        raise SystemExit('new completed attempt required; preserve old seals')
    process_path = Path('/proc')/str(args.pid)/'stat'
    if process_path.exists():
        fields = process_path.read_text().rsplit(')', 1)[1].split()
        same_process = int(fields[19]) == args.startticks
        if same_process and fields[0] != 'Z':
            raise SystemExit('run process still active; log is not yet closed')
    preseal = read(HERE/'SOURCE-FROZEN.json')
    verify(preseal['source_and_input_sha256'])
    result = read(args.attempt/'result.json')
    assert result['status'] == 'COMPLETE'
    assert result['source_seal_sha256'] == sha(HERE/'SOURCE-FROZEN.json')
    assert result['original_grid_fixture_status'] == 'VALID' and result['original_grid_quality'] == 'FAIL'
    assert result['checks']['chart_fits'] == 1 and result['checks']['PT_objects_loaded'] == 1
    assert args.log.is_file() and args.log.resolve().parent in (HERE, args.attempt.resolve())
    local = {str(path): sha(path) for path in args.attempt.rglob('*') if path.is_file()}
    local[str(args.log.resolve())] = sha(args.log)
    local[str(HERE/'SOURCE-FROZEN.json')] = sha(HERE/'SOURCE-FROZEN.json')
    receipt = dict(status='PASS', scope='closed fixed descriptive diagnostic evidence audit',
        utc=datetime.now(timezone.utc).isoformat(), result_sha256=sha(args.attempt/'result.json'),
        source_and_input_sha256=preseal['source_and_input_sha256'], local_sha256=local,
        original_grid_fixture_status='VALID', original_grid_quality='FAIL',
        descriptive_measurements_not_quality_acceptance=True,
        process_identity=dict(pid=args.pid, startticks=args.startticks, exited=True),
        log_closed=True, reruns=0, numerical_law_changed=False)
    write_new(args.attempt/'receipt.json', receipt)
    local[str(args.attempt/'receipt.json')] = sha(args.attempt/'receipt.json')
    verify(preseal['source_and_input_sha256'])
    write_new(args.attempt/'FROZEN.json', dict(status='PASS', scope='AUTHORITATIVE_POST_EXIT_DIAGNOSTIC_SEAL',
        utc=datetime.now(timezone.utc).isoformat(), receipt_sha256=sha(args.attempt/'receipt.json'),
        source_and_input_sha256=preseal['source_and_input_sha256'], local_sha256=local,
        process_exited=True, log_closed=True, original_grid_quality='FAIL',
        production_qualification=False))
    print('CLOSED_PASS', sha(args.attempt/'receipt.json'), sha(args.attempt/'FROZEN.json'), flush=True)


if __name__ == '__main__':
    main()
