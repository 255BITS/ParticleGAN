"""Rebuild the complete accuracy-search ledger, retaining failed/partial runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100.accuracy import audit_npz
from benchmarks.toy100.gate import evaluate_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'artifacts/toy100-accuracy')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/toy100/accuracy-search-ledger')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.rglob('summary.json')):
        summary = json.loads(path.read_text())
        if 'problem' not in summary or 'eval_steps' not in summary:
            continue
        directory = path.parent
        row = dict(run=str(directory.relative_to(args.input)),
                   status=summary['status'], problem=summary['problem'],
                   steps=summary.get('completed_steps', 0),
                   total_seconds=summary.get('total_seconds'),
                   config=summary['config'], coverage='INCOMPLETE',
                   final_accuracy=None,
                   source_sha256=summary['provenance']['source_sha256'])
        if summary['status'] == 'complete':
            gate = evaluate_suite(directory.parent, problem=summary['problem'], write=False)
            row['coverage'] = gate['problems'][summary['problem']]['status']
            row['final_accuracy'] = audit_npz(directory / 'final_samples.npz', summary['problem'])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    lines = [
        '# Accuracy search ledger', '',
        'Every discovered 100-mode trial is retained, including failures and interrupted runs. '
        'The final fidelity column scores one saved 20,000-sample draw; it does **not** '
        'certify five-checkpoint/100,000-sample accuracy or the common 22-toy gate. '
        'All trials use the frozen training seed; no seed sweep is performed.', '',
        '| Run | Steps | Coverage gate | Final modes | HQ | Mass TV | Center / σ | Width bias | Radial KS | Final fidelity |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---|',
    ]
    def number(value):
        return '—' if value is None else f'{value:.4f}'
    for row in rows:
        accuracy = row['final_accuracy'] or {}
        summary = json.loads((args.input / row['run'] / 'summary.json').read_text())
        metrics = summary.get('final', {}).get('live', {})
        lines.append('| ' + ' | '.join([
            f"`{row['run']}`", str(row['steps']), row['coverage'],
            str(metrics.get('modes', '—')), number(metrics.get('hq')),
            number(accuracy.get('mass_tv')), number(accuracy.get('center_rms_sigma')),
            number(accuracy.get('cov_trace_bias')), number(accuracy.get('radial_ks')),
            ('PASS' if accuracy.get('accuracy_pass') else 'FAIL') if accuracy else row['status'],
        ]) + ' |')
    args.output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(trials=len(rows), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
