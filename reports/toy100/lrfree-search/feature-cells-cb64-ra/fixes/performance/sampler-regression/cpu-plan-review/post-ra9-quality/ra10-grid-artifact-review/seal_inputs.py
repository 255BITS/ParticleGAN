"""Raw-bind a completed original grid before saved PT storage inspection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
LANE = ROOT / 'validation-cb64-ra10'
RUN = LANE / 'screens/runs/grid100'
CANONICAL = ROOT / 'integration/review/validation-cb64-ra10-monitor/canonical-receipts/screens/runs/grid100/acceptance-receipt.json'
STEPS = [0,1,10,25,50,100] + list(range(250,7001,250))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(HERE) and not args.output.exists()
    checker = read(HERE/'CHECKER-FROZEN.json')
    files = {}
    def bind(p, expected=None):
        p = Path(p).resolve()
        actual = sha(p)
        assert expected is None or actual == expected, p
        assert str(p) not in files or files[str(p)] == actual
        files[str(p)] = actual
    for field in ('source_and_input_sha256', 'helper_sha256'):
        for p,h in checker[field].items(): bind(p,h)
    bind(HERE/'CHECKER-FROZEN.json')
    execution, result, canonical = read(RUN/'execution-receipt.json'), read(RUN/'result.json'), read(CANONICAL)
    assert execution['status'] == 'COMPLETE' and type(execution['process_exit_code']) is int and execution['process_exit_code'] == 0
    assert result['status'] in ('PASS','FAIL') and result['completed_steps'] == 7000 and result['observations'] == len(STEPS)
    assert canonical['canonical_fixture_validity'] == 'VALID' and not canonical['validity_reasons']
    assert canonical['source_integrity']['status'] == 'VALID'
    assert canonical['primary_status'] == canonical['acceptance_status'] == result['status']
    assert canonical['result_sha256'] == execution['result_sha256'] == sha(RUN/'result.json')
    assert canonical['execution_receipt_sha256'] == sha(RUN/'execution-receipt.json')
    journal = (LANE/'run.log').read_bytes()
    prefix = bytearray()
    event = None
    for line in journal.splitlines(keepends=True):
        prefix.extend(line)
        try: item = json.loads(line)
        except json.JSONDecodeError: continue
        if item.get('event') == 'job_complete' and item.get('name') == 'screen-grid100':
            event = item
            break
    assert event is not None and event['returncode'] == 0 and event['status'] == result['status']
    assert event['result_sha256'] == sha(RUN/'result.json')
    assert Path(event['result']).resolve() == (RUN/'result.json').resolve()
    bind(event['log'])
    for p in sorted(RUN.rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts: bind(p)
    args.output.mkdir()
    (args.output/'journal-prefix.log').write_bytes(bytes(prefix))
    (args.output/'grid-job-event.json').write_text(json.dumps(event,indent=2)+'\n')
    (args.output/'canonical-grid-acceptance-receipt.json').write_bytes(CANONICAL.read_bytes())
    bind(CANONICAL)
    for name in ('journal-prefix.log','grid-job-event.json','canonical-grid-acceptance-receipt.json'): bind(args.output/name)
    value = dict(status='COMPLETED_GRID_INPUTS_FROZEN', utc=datetime.now(timezone.utc).isoformat(),
        validation=str(LANE), package_root=str(ROOT/'pkg-CB64-RA10'),
        package_sha256=checker['package_sha256'], config_sha256=checker['config_sha256'],
        checker_freeze_sha256=sha(HERE/'CHECKER-FROZEN.json'), steps=STEPS,
        source_and_input_sha256=files, final_state_sha256=sha(RUN/'final-state.pt'),
        canonical_receipt_sha256=sha(CANONICAL), journal_prefix_sha256=sha(args.output/'journal-prefix.log'),
        live_queue_tail_not_required_immutable=True, raw_PT_hashes_only=True,
        PT_objects_loaded_before_seal=0, model_forwards=0, new_scoring_calls=0, quality_verdict=None)
    (args.output/'INPUTS-FROZEN.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],guards=len(files),freeze_sha256=sha(args.output/'INPUTS-FROZEN.json'))))


if __name__ == '__main__': main()
