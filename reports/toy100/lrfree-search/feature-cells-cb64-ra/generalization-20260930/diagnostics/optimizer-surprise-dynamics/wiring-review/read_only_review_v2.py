"""Raw hashes, JSON and source ASTs only; no numerical/package imports."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(node):
    return ast.dump(node, include_attributes=False)


ALLOWED = {
    'continuous.py': {'OptimizerSurprise': {'decide'}},
    'recipes.py': {'Recipe': {'__post_init__', 'to_dict'}},
    'policy.py': {'UpdatePolicy': {'__init__', 'begin_step', 'after_critic_step',
                                 'state_dict', '_check_state', 'load_state_dict'}},
    'training.py': {'GANTrainer': {'_state_dict', '_load_state_dict'}},
}


def scoped(tree, filename, candidate):
    tree = deepcopy(tree)
    if filename == 'continuous.py' and candidate:
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.ClassDef) and n.name == 'SettledReopenGuard')]
    for node in tree.body:
        if filename == 'policy.py' and isinstance(node, ast.ImportFrom):
            node.names = [n for n in node.names if n.name != 'SettledReopenGuard']
        if not isinstance(node, ast.ClassDef):
            continue
        body = []
        for child in node.body:
            if filename == 'recipes.py' and isinstance(child, ast.AnnAssign) and (
                    isinstance(child.target, ast.Name) and child.target.id == 'reopen_guard'):
                continue
            if filename == 'policy.py' and candidate and isinstance(child, ast.FunctionDef) and (
                    child.name in ('_contracted_network', '_loss_epoch')):
                continue
            if isinstance(child, ast.FunctionDef) and child.name in ALLOWED.get(filename, {}).get(node.name, set()):
                child = ast.parse('def ' + child.name + '():\n    pass').body[0]
            body.append(child)
        node.body = body
    return dump(tree)


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    prep = json.loads((HERE / 'INPUTS-FROZEN-v2.json').read_text())
    for p, h in prep['sha256'].items():
        assert sha(Path(p)) == h, p
    closed = json.loads((ROOT / 'RA13-SOURCE-CLOSURE.json').read_text())
    base = ROOT / 'pkg-RA12-cpu-fix'
    proposal = ROOT / 'pkg-RA13-settled'
    files = lambda p: {str(x.relative_to(p)): sha(x) for x in p.rglob('*.py')}
    bm, pm = files(base), files(proposal)
    assert len(bm) == len(pm) == 28 and bm.keys() == pm.keys()
    digest = lambda m: hashlib.sha256(json.dumps({Path(k).name: v for k, v in m.items()}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest(pm) == closed['package_sha256']
    assert digest(bm) == closed['base_manifest_sha256']
    changed = sorted(Path(k).name for k in pm if pm[k] != bm[k])
    assert changed == sorted(ALLOWED)
    for name in ALLOWED:
        old = ast.parse((base / 'particlegan' / name).read_text())
        new = ast.parse((proposal / 'particlegan' / name).read_text())
        assert scoped(old, name, False) == scoped(new, name, True), name
    # Inverse the exact optional branches, not the complete decide method.
    new = (proposal / 'particlegan/continuous.py').read_text()
    assert new.count('\n\nclass SettledReopenGuard:') == 1
    inverse = new.split('\n\nclass SettledReopenGuard:')[0]
    for old, restored in (
        ('def decide(self, step=None, *, guard=None, network=None):', 'def decide(self, step=None):'),
        ('        permitted = True if guard is None else guard.observe(ratio, self.CALM, step, network)\n', ''),
        ('if calm or not self.armed or not permitted:', 'if calm or not self.armed:'),
        ('if permitted and ratio > self.RISE and abrupt else 0', 'if ratio > self.RISE and abrupt else 0'),
        ('                if guard is not None:\n                    guard.after_fire()\n', ''),
    ):
        inverse = replace_once(inverse, old, restored)
    assert dump(ast.parse(inverse)) == dump(ast.parse((base / 'particlegan/continuous.py').read_text()))
    a = json.loads((ROOT / 'configs/RA12-auto.json').read_text())
    b = json.loads((ROOT / 'configs/RA13-settled.json').read_text())
    assert {k: v for k, v in b.items() if a.get(k) != v} == {'reopen_guard': 'settled'}
    assert a.keys() <= b.keys() and all(b[k] == v for k, v in a.items())
    assert sha(ROOT / 'configs/RA13-settled.json') == closed['config_sha256']
    owner = json.loads((ROOT / 'portability/settled-guard/COMPLETION.json').read_text())
    assert owner == closed
    assert owner['cpu_test_count'] == 58
    for name, check in owner['cpu_checks'].items():
        p = ROOT / 'portability/settled-guard' / name
        assert sha(p) == check['sha256']
        assert str(check['pass']) + ' passed' in p.read_text()
    policy = (proposal / 'particlegan/policy.py').read_text()
    assert 'role in SettledReopenGuard.NETWORK_ROLES and tester.s < 1.' in policy
    assert 'self.reopen_guard.observe_epoch(self._loss_epoch(), self.surprise)\n        if self.lr_settle is not None:\n            self._settle_observe(1)' in policy
    assert 'anchor_started=self._loss_epoch(state["optimizers"][1])' in policy
    recipes = (proposal / 'particlegan/recipes.py').read_text()
    assert 'if self.reopen_guard is None:\n            result.pop("reopen_guard")' in recipes
    for p, h in prep['sha256'].items():
        assert sha(Path(p)) == h, p
    receipt = {
        'status': 'PASS', 'scope': 'independent read-only source/JSON review; no runtime repetition',
        'package_sha256': digest(pm), 'base_package_sha256': digest(bm),
        'package_files': 28, 'unchanged_from_cpu_fix': 24,
        'changed_modules': changed, 'outside_declared_nodes_ast_equal': True,
        'whole_continuous_module_ast_inverse_exact': True,
        'config_only_delta': {'reopen_guard': 'settled'},
        'default_schema_and_dict_compatibility': 'guard None omits recipe and checkpoint key; original schemas1/4 retained',
        'epoch_timing': 'after actual critic optimizer/penalty epoch change, before changed-loss q queued',
        'guard_checkpoint_validation': 'strict scalar/role/scale/time checks after optimizer/control validation and before live release/writes',
        'generic_roles': ['generator', 'encoder', 'router', 'critic'],
        'routed_and_encoder_ownership': 'same existing homogeneous group ownership; table/noise roles excluded; routing module unchanged',
        'owner_runtime_evidence': '19 affected +35 existing +4 default R1 PASS; reviewed, not rerun',
        'concrete_remaining_source_defects': [],
        'remaining_limits': ['not an external-target classifier', 'full guarded quality and GPU replay require original fresh validation',
                            'custom caller loss phases without an attached KA2 optimizer record are not inferred'],
        'torch_imports': 0, 'pt_reads': 0, 'models': 0, 'forwards': 0, 'updates': 0,
        'draws': 0, 'scorers': 0, 'cuda_calls': 0,
    }
    with (HERE / 'receipt.json').open('x') as f:
        json.dump(receipt, f, sort_keys=True, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'PASS', 'receipt_sha256': sha(HERE / 'receipt.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
