"""Write the post-exit descriptive report and immutable artifact seal."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--log',type=Path,required=True)
    args=parser.parse_args()
    args.input_freeze=args.input_freeze.resolve();args.output=args.output.resolve();args.log=args.log.resolve()
    assert args.output.is_relative_to(HERE) and args.log.is_relative_to(HERE)
    assert not (args.output/'FROZEN.json').exists()
    inputs=read(args.input_freeze)
    for name,digest in inputs['source_and_input_sha256'].items():assert sha(name)==digest,name
    receipt=read(args.output/'receipt.json')
    assert receipt['status']==receipt['evidence_status']=='VALID' and len(receipt['checkpoints'])==10
    assert receipt['quality_verdict'] is None and not receipt['RA9_training_parity_asserted']
    lines=['# RA10 completed toy saved-artifact audit','',
        'All ten original CUDA toy checkpoints are VALID under the unchanged artifact authority and the backend9 metadata law.',
        'This was CPU deserialization only: no model construction, forward, planner, training, emitted sample or CUDA context.',
        '', '| Update | Mean witness | Last mean moves | Cumulative mean moves | Serving lease |',
        '|---:|---|---:|---:|---|']
    for row in receipt['checkpoints']:
        lines.append(f"| {row['step']} | {row['mean']['status']} | {row['mean']['moves']} | {row['mean_counters']['mean_moves']} | {row['derived_served_view']} |")
    lines += ['', 'Checked actual3K+3/finite-fit resolution, typed scalar witness and fourth-phase budget/row accounting, ',
        'lineage symmetry, complete reset counters, population state, serving stamp expiry and JSON metadata.',
        'At saved reaction boundaries, current mean-copy history/moments and own reset/participation/link evidence were checked.',
        'Other checkpoints do not reconstruct historical offspring categories or incarnations. Isolation row lists are unsaved.',
        '', 'The original auditor\'s three-threshold toy subgate is retained separately in receipt.json. ',
        'Root owns the full strict toy verdict and canonical Grid100/replay gates; this report asserts no RA9 training parity.', '']
    with (args.output/'REPORT.md').open('x') as stream:stream.write('\n'.join(lines))
    protected=dict(inputs['source_and_input_sha256']);protected[str(args.input_freeze)]=sha(args.input_freeze)
    files={str(path):sha(path) for path in sorted(args.output.rglob('*')) if path.is_file()}
    files[str(args.log)]=sha(args.log)
    frozen=dict(status='POST_EXIT_FROZEN_VALID_CPU_TOY_ARTIFACT_AUDIT',frozen_UTC=datetime.now(timezone.utc).isoformat(),
        files=files,protected_file_sha256=protected,receipt_sha256=sha(args.output/'receipt.json'),
        checkpoints=10,quality_verdict=None,RA9_training_parity_asserted=False)
    with (args.output/'FROZEN.json').open('x') as stream:stream.write(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps(dict(status=frozen['status'],receipt_sha256=frozen['receipt_sha256'],
        freeze_sha256=sha(args.output/'FROZEN.json'),protected_files=len(protected))))

if __name__=='__main__':main()
