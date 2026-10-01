"""Summarize the actual candidate's original final metrics without substitution."""
import json
from pathlib import Path
import common
from contracts import toy_gate

inputs = common.verify_inputs()
rows = []
for problem in common.PROBLEMS:
    paths = {name: Path(root) / problem / name for name, root in inputs['comparative_control_training_roots'].items()}
    paths.update({name: common.ROOT / 'training' / problem / name for name in common.VARIANTS})
    for variant, directory in paths.items():
        path = directory / 'result.json'
        if not path.exists():
            rows.append(dict(problem=problem, variant=variant, status='NOT_RUN'))
            continue
        result = json.loads(path.read_text())
        metrics, diagnostics = result['final']['metrics'], result['final']['diagnostics']
        rows.append(dict(problem=problem, variant=variant, status=result['status'], steps=result['steps'],
                         original_quality_gate=('PASS' if toy_gate(metrics) else 'FAIL') if problem == 'toy' else None,
                         metrics=metrics, diagnostics=diagnostics,
                         training_seconds=result['training_seconds'], whole_run_seconds=result['whole_run_seconds'],
                         peak_gpu_allocated_bytes=result['peak_gpu_allocated_bytes'],
                         peak_gpu_reserved_bytes=result['peak_gpu_reserved_bytes'],
                         result=str(path), result_sha256=common.sha(path)))
summary = dict(mnist_quality_gate=None, original_numerical_failures_preserved=True, rows=rows,
               source_freeze_sha256=common.sha(common.ROOT / 'SOURCE-FREEZE.json'))
common.write_json(common.ROOT / 'summary.json', summary)
print(json.dumps({'rows': [{k: value for k, value in row.items() if k in ('problem', 'variant', 'status', 'original_quality_gate')}
                          for row in rows]}, indent=2), flush=True)
