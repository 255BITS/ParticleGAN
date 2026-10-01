"""Post-exit seal of completed grid validity evidence and its closed log."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-freeze', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--closed-log', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(HERE)
    receipt = json.loads((args.output/'receipt.json').read_text())
    assert receipt['status'] == receipt['evidence_status'] == 'VALID'
    files = dict(receipt['source_and_input_sha256'])
    files[str(args.input_freeze.resolve())] = sha(args.input_freeze)
    files[str(args.closed_log.resolve())] = sha(args.closed_log)
    for p in sorted(args.output.rglob('*')):
        if p.is_file(): files[str(p.resolve())] = sha(p)
    for p,h in files.items(): assert sha(p) == h, p
    report = args.output/'REPORT.md'
    assert not report.exists()
    report.write_text('# RA11 grid saved-state validity\n\n'
        +f'Artifact evidence VALID; original quality {receipt["quality_verdict"]}. '
        +f'All {len(receipt["observations"])} observation metadata rows and original native artifact checks pass. '
        +'The saved final trainer5/backend10 state, RNG placement, fourth phase, bounded lineage and cumulative own evidence reset totals agree.\n\n'
        + '\n'.join('- '+limit for limit in receipt['limits'])+'\n')
    files[str(report.resolve())] = sha(report)
    value = dict(status='PASS',scope='FROZEN_COMPLETED_GRID_CPU_ARTIFACT_VALIDITY',utc=datetime.now(timezone.utc).isoformat(),
        files=files,receipt_sha256=sha(args.output/'receipt.json'),closed_log_sha256=sha(args.closed_log),
        input_freeze_sha256=sha(args.input_freeze),artifact_validity='VALID',quality_verdict=receipt['quality_verdict'],
        CPU_only=True,cuda_initialized=False,new_scoring_calls=0,model_forwards=0,training_updates=0)
    assert not (args.output/'FROZEN.json').exists()
    (args.output/'FROZEN.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',receipt_sha256=value['receipt_sha256'],freeze_sha256=sha(args.output/'FROZEN.json'),guards=len(files))))


if __name__ == '__main__': main()
