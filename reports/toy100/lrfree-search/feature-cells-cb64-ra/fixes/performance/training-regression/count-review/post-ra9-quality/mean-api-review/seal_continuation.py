"""Raw guard binding for only the two unexecuted CPU continuation updates."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    output = HERE / 'CONTINUATION-INPUTS-FROZEN.json'
    assert not output.exists()
    first_path = HERE / 'API-INPUTS-FROZEN.json'
    first = json.loads(first_path.read_text())
    assert sha(first_path) == 'c8e4b765804566ef00d9af7f71e146f40cd3b7226d49a4d47ca6de2075ada230'
    files = dict(first['protected_sha256'])
    for name, digest in files.items():
        assert sha(name) == digest, name
    failure_path = HERE / 'attempt1/FAILURE.json'
    failure = json.loads(failure_path.read_text())
    assert failure['status'] == 'FAIL' and 'NoneType' in failure['error']
    assert len(failure['completed_records']) == 2
    assert all(r['status'] == 'PASS' for r in failure['completed_records'])
    assert sum(len(r['rejected_controls']) for r in failure['completed_records']) == 17
    skeleton = ast.parse((HERE / 'check_api_skeleton.py').read_text())
    function = next(n for n in skeleton.body if isinstance(n, ast.FunctionDef) and n.name == 'two_updates')
    guard = next(n for n in function.body if isinstance(n, ast.Assert)
                 and ast.unparse(n.test) == "state['completed_steps'] < state['recipe']['total_steps']")
    calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'step']
    assert len(calls) == 2 and all(n.lineno > guard.lineno for n in calls)
    assert f'line {guard.lineno}, in two_updates' in failure['traceback']
    extras = [first_path, HERE / 'attempt1.log', HERE / 'seal-api-inputs.log',
        HERE / 'seal-api-inputs-attempt2.log', HERE / 'check_continuation.py',
        HERE / 'CONTINUATION-ADDENDUM.md', Path(__file__)]
    extras.extend(p for p in sorted((HERE / 'attempt1').rglob('*')) if p.is_file())
    for path in extras:
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=str(path))
        files[str(path.resolve())] = sha(path)
    result = dict(status='PRE_EXECUTION_CONTINUATION_ONLY_FROZEN', created_UTC=datetime.now(timezone.utc).isoformat(),
        previous_input_seal=str(first_path), previous_failure=str(failure_path),
        package_root=first['package_root'], toy_after=first['fixtures']['toy']['after'],
        two_updates_guard_error_before_both_step_calls=True, completed_prior_optimizer_updates=0,
        prior_complete_cases=2, prior_atomic_controls=17, API_samples_to_repeat=0,
        next_updates_total=2, protected_sha256=dict(sorted(files.items())),
        scope='Raw bytes/AST/source and closed metadata only; no Torch/PT/import/forward/sample/test/update')
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], guarded_files=len(files), sha256=sha(output))))


if __name__ == '__main__':
    main()
