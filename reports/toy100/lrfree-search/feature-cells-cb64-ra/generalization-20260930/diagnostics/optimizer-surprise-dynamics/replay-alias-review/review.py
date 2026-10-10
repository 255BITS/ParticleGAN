"""Restoration-only source bridge; stdlib/raw JSON, no checkpoint imports."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prep = json.loads((HERE / 'INPUTS-FROZEN.json').read_text())
    for p, h in prep['sha256'].items():
        assert sha(Path(p)) == h, p
    base, candidate = ROOT / 'pkg-RA13-settled/particlegan', ROOT / 'pkg-RA14-replay/particlegan'
    files = lambda p: {str(x.relative_to(p)): sha(x) for x in p.rglob('*.py')}
    bm, cm = files(base), files(candidate)
    assert len(bm) == len(cm) == 28 and bm.keys() == cm.keys()
    assert [k for k in bm if bm[k] != cm[k]] == ['policy.py']
    manifest = lambda m: hashlib.sha256(json.dumps(m, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    owner = json.loads((ROOT / 'portability/replay-alias/COMPLETION.json').read_text())
    bridge = json.loads((ROOT / 'portability/replay-alias/SOURCE-BRIDGE.json').read_text())
    assert manifest(bm) == owner['base_package_sha256'] == bridge['base_package_sha256']
    assert manifest(cm) == owner['package_sha256'] == bridge['candidate_package_sha256']
    assert cm == owner['source_sha256'] == bridge['source_sha256']
    before, after = (base / 'policy.py').read_text(), (candidate / 'policy.py').read_text()
    bt, ct = ast.parse(before), ast.parse(after)
    bn = next(n for n in bt.body if isinstance(n, ast.FunctionDef) and n.name == '_state_to_device')
    cn = next(n for n in ct.body if isinstance(n, ast.FunctionDef) and n.name == '_state_to_device')
    original_lines, changed_lines = before.splitlines(keepends=True), after.splitlines(keepends=True)
    inverse = ''.join(changed_lines[:cn.lineno - 1] + original_lines[bn.lineno - 1:bn.end_lineno]
                      + changed_lines[cn.end_lineno:])
    assert inverse == before
    assert ast.dump(ast.parse(inverse)) == ast.dump(bt)
    calls = []
    class Calls(ast.NodeVisitor):
        def __init__(self, file):
            self.file, self.stack = file, []
        def visit_ClassDef(self, node):
            self.stack.append(node.name); self.generic_visit(node); self.stack.pop()
        def visit_FunctionDef(self, node):
            self.stack.append(node.name); self.generic_visit(node); self.stack.pop()
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_state_to_device':
                calls.append({'file': self.file, 'line': node.lineno, 'owner': '.'.join(self.stack)})
            self.generic_visit(node)
    for p in candidate.rglob('*.py'):
        Calls(str(p.relative_to(candidate))).visit(ast.parse(p.read_text()))
    allowed = {'_state_to_device', 'UpdatePolicy._check_state', 'UpdatePolicy.load_state_dict', 'GANTrainer._load_state_dict'}
    assert all(c['owner'] in allowed for c in calls)
    assert sorted(calls, key=lambda c: (c['file'], c['line'])) == sorted(bridge['helper_callsites'], key=lambda c: (c['file'], c['line']))
    to_calls = [n for n in ast.walk(cn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'to']
    assert len(to_calls) == 1 and [k.arg for k in to_calls[0].keywords] == ['device']
    assert 'key = id(value)' in after and 'memo = {} if memo is None else memo' in after
    assert (ROOT / 'configs/RA14-replay.json').read_bytes() == (ROOT / 'configs/RA13-settled.json').read_bytes()
    assert sha(ROOT / 'configs/RA14-replay.json') == owner['config_sha256']
    training = ROOT / 'mnist/ra13-settled/run_training.py'
    for n in ast.walk(ast.parse(training.read_text())):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'load_state_dict':
            assert ast.unparse(n.func.value) == 'self.model', 'unexpected learned trainer restore'
    for kind in ('toy', 'mnist'):
        path = ROOT / ('mnist/ra13-settled/COMPLETION-' + kind + '.json')
        record = json.loads(path.read_text())
        assert record['status'] == 'COMPLETE' and record['returncode'] == 0
        assert sha(Path(record['result'])) == record['result_sha256']
        assert sha(Path(record['log'])) == record['log_sha256']
        result = json.loads(Path(record['result']).read_text())
        assert result['status'] == 'COMPLETE' and result['steps'] == 2000
    assert owner['cpu_tests']['new_alias_contracts'] == 4 and owner['cpu_tests']['combined_pass'] == 58
    assert sha(Path(owner['test_log'])) == owner['test_log_sha256']
    assert sha(Path(owner['test_path'])) == owner['test_sha256']
    assert '58 passed' in Path(owner['test_log']).read_text()
    full = json.loads((ROOT / 'validation-ra13-r2/FULL-TESTS.json').read_text())
    assert full['status'] == 'PASS' and full['returncode'] == 0
    assert sha(Path(full['log'])) == full['log_sha256']
    assert '1404 passed, 12 skipped' in Path(full['log']).read_text()
    for p, h in prep['sha256'].items():
        assert sha(Path(p)) == h, p
    receipt = dict(status='PASS', scope='independent restoration-only source bridge; no PT/model/numerical execution',
        base_package_sha256=manifest(bm), candidate_package_sha256=manifest(cm),
        package_files=28, unchanged_modules=27, only_changed_definition='_state_to_device',
        whole_policy_byte_and_ast_inverse_exact=True, checkpoint_only_callsites=calls,
        config_byte_identical=True, schema_and_state_fields_unchanged=True,
        ownership='one transfer-local memo keyed by original Tensor object identity; distinct owners not coalesced',
        dtype='only device is passed to original Tensor.to; shape/dtype/bits policy unchanged',
        existing_cpu_evidence='4 allocation/alias/rebase controls within58 PASS; reviewed, not repeated',
        fresh_quality_carry_forward='valid with explicit RA13 result/source/input provenance and this RA14 restore-only bridge; no relabeling',
        original_replay_failures='remain FAIL; corrected native/CPU-map CUDA replays pending',
        historical_full_suite='RA13 only:1404 PASS/12skip/18subtests; not relabeled as RA14 suite',
        limitations=['no reconstruction of distinct Tensor view/storage aliases or shared Python containers',
                     'no healing of previously saved alias-broken endpoints', 'no arbitrary cyclic/custom state-tree claim'],
        remaining_source_blockers=[], no_torch_pt_model_forward_draw_update_scoring_cuda=True)
    with (HERE / 'receipt.json').open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
    print(json.dumps({'status': 'PASS', 'receipt_sha256': sha(HERE / 'receipt.json')}))


if __name__ == '__main__':
    main()
