"""Read-only RA10 composed-source audit; stdlib only, after composition."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(node):
    if isinstance(node, (list, tuple)):
        return repr([dump(value) for value in node])
    return ast.dump(node, include_attributes=False)


def tree(path):
    return ast.parse(Path(path).read_text(), filename=str(path))


def maps(package):
    return {str(path.relative_to(package / 'particlegan')): sha(path)
            for path in sorted((package / 'particlegan').rglob('*.py'))}


def package_digest(package, mapping):
    h = hashlib.sha256()
    for name in sorted(mapping):
        h.update(name.encode() + b'\0' + (package / 'particlegan' / name).read_bytes() + b'\0')
    return h.hexdigest()


def guard_nested(value, base, protected):
    """Guard absolute maps and existing ready-relative maps; package map explicit."""
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(name, str) and isinstance(item, str) and len(item) == 64:
                path = Path(name)
                if path.is_absolute() or (base / path).is_file():
                    path = path if path.is_absolute() else base / path
                    assert sha(path) == item, str(path)
                    assert str(path) not in protected or protected[str(path)] == item
                    protected[str(path)] = item
            guard_nested(item, base, protected)
    elif isinstance(value, list):
        for item in value:
            guard_nested(item, base, protected)


def scope_inverse(original, proposal, allowed, additions, class_assignments=()):
    """Restore named reviewed edits; require every remaining AST byte-equivalent."""
    result = copy.deepcopy(proposal)
    original_classes = {node.name: node for node in original.body if isinstance(node, ast.ClassDef)}
    changed = []
    added = []
    for node in list(result.body):
        if isinstance(node, ast.ImportFrom) and node.module == 'mean_transport':
            assert node.level == 1
            result.body.remove(node)
            added.append('import mean_transport')
        elif isinstance(node, ast.ClassDef):
            base = original_classes[node.name]
            methods = {item.name: item for item in base.body if isinstance(item, ast.FunctionDef)}
            for index, item in list(enumerate(node.body)):
                if isinstance(item, ast.FunctionDef):
                    key = (node.name, item.name)
                    if item.name not in methods:
                        assert key in additions, key
                        added.append('.'.join(key))
                    elif dump(item) != dump(methods[item.name]):
                        assert key in allowed, key
                        node.body[index] = copy.deepcopy(methods[item.name])
                        changed.append('.'.join(key))
                elif (isinstance(item, ast.Assign) and len(item.targets) == 1
                      and isinstance(item.targets[0], ast.Name)
                      and (node.name, item.targets[0].id) in class_assignments):
                    match = [value for value in base.body if isinstance(value, ast.Assign)
                             and dump(value.targets) == dump(item.targets)]
                    assert len(match) == 1
                    node.body[index] = copy.deepcopy(match[0])
            node.body = [item for item in node.body
                         if not (isinstance(item, ast.FunctionDef) and (node.name, item.name) in additions)]
    assert dump(result) == dump(original), 'undeclared feature_cells module AST delta'
    assert set(changed) == {'.'.join(key) for key in allowed}
    assert set(added) == {'import mean_transport', *('.'.join(key) for key in additions)}
    return changed, added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    owner = root / 'integration/review/training-regression/post-ra9-quality/mean-category-production'
    base = root / 'pkg-CB64-RA9'
    candidate = root / 'pkg-CB64-RA10'
    composed_path = root / 'quality/ra10/COMPOSITION.json'
    composed = json.loads(composed_path.read_text())
    ready = json.loads((owner / 'READY.json').read_text())
    assert ready['status'] == 'FROZEN_CPU_QUALIFIED'
    assert ready['backend_schema'] == composed['backend_schema'] == 9
    assert ready['trainer_schema'] == composed['trainer_schema'] == 5
    protected = {}
    for path in (owner / 'READY.json', owner / 'FROZEN.json', owner / 'SOURCE-FROZEN.json',
                 composed_path, root / 'quality/ra9/READY.json'):
        guard_nested(json.loads(path.read_text()), path.parent, protected)
        protected[str(path)] = sha(path)
    proposal = Path(ready['package_root'])
    baseline, own, actual = maps(base), maps(proposal), maps(candidate)
    assert len(baseline) == 29 and len(actual) == 30
    assert actual == own == ready['package_source_sha256'] == composed['source_sha256']
    assert set(actual) - set(baseline) == {'mean_transport.py'}
    assert set(baseline) - set(actual) == set()
    changed = [name for name in baseline if actual[name] != baseline[name]]
    assert changed == ['birth_phase.py', 'feature_cells.py'], changed
    identity = package_digest(candidate, actual)
    assert identity == ready['package_sha256'] == composed['package_sha256']
    for package, mapping in ((base, baseline), (proposal, own), (candidate, actual)):
        protected.update({str(package / 'particlegan' / name): value for name, value in mapping.items()})
    configs = (root / 'configs/overrides-CB64-RA9.json',
               Path(ready['config_path']), root / 'configs/overrides-CB64-RA10.json')
    assert all(path.read_bytes() == configs[0].read_bytes() for path in configs)
    assert sha(configs[0]) == ready['config_sha256'] == composed['config_sha256']
    assert sha(configs[0]) == 'b3656ea7494413106484c556e53877900dd5ccebf2597b3f80b7d0e7d5ba4437'
    protected.update({str(path): sha(path) for path in configs})

    allowed = {('FeatureCellSnapshot', 'cell_comparison'),
               ('FeatureCellSnapshot', 'ordinary_transport'),
               ('FeatureCellBirthDeath', '__init__'),
               ('FeatureCellBirthDeath', 'maybe_apply'),
               ('FeatureCellBirthDeath', 'check_state'),
               ('FeatureCellBirthDeath', '_check_paired_average_state')}
    additions = {('FeatureCellBirthDeath', '_check_mean_actions')}
    edits, added = scope_inverse(tree(base / 'particlegan/feature_cells.py'),
        tree(candidate / 'particlegan/feature_cells.py'), allowed, additions,
        {('FeatureCellBirthDeath', 'BACKEND_SCHEMA')})

    old_birth = tree(base / 'particlegan/birth_phase.py')
    new_birth = tree(candidate / 'particlegan/birth_phase.py')
    inverse_birth = copy.deepcopy(new_birth)
    assert isinstance(inverse_birth.body[0], ast.Expr)
    inverse_birth.body[0] = copy.deepcopy(old_birth.body[0])
    old_functions = {node.name: node for node in old_birth.body if isinstance(node, ast.FunctionDef)}
    birth_edits = []
    for index, node in enumerate(inverse_birth.body):
        if isinstance(node, ast.FunctionDef) and dump(node) != dump(old_functions[node.name]):
            assert node.name == 'global_certificate_residual'
            birth_edits.append(node.name)
            inverse_birth.body[index] = copy.deepcopy(old_functions[node.name])
    assert birth_edits == ['global_certificate_residual']
    assert dump(inverse_birth) == dump(old_birth), 'undeclared birth module AST delta'

    helper = tree(candidate / 'particlegan/mean_transport.py')
    forbidden = {'grid', 'grid100', 'toy', 'mnist', 'mode_hold', 'img_blobs4',
                 'vector_unequal_mass', 'holdout', 'oracle', 'centers_true'}
    decisions = [node for node in ast.walk(helper) if isinstance(node, (ast.If, ast.IfExp, ast.While))]
    for decision in decisions:
        names = {item.id.lower() for item in ast.walk(decision.test) if isinstance(item, ast.Name)}
        strings = {item.value.lower() for item in ast.walk(decision.test)
                   if isinstance(item, ast.Constant) and isinstance(item.value, str)}
        assert not forbidden & (names | strings), (names, strings)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and any(
        name.name.split('.')[0] in {'subprocess', 'requests'} for name in node.names)
        for node in helper.body)

    # Only the reviewer helper itself and its closed output are new artifacts.
    protected[str(Path(__file__).resolve())] = sha(Path(__file__))
    receipt = dict(status='PASS', scope='Read-only full-module source/config/provenance review; no Torch/PT/model/CUDA work',
        package_sha256=identity, package_source_sha256=actual, config_sha256=sha(configs[0]),
        unchanged_original_modules=27, added_modules=['mean_transport.py'],
        feature_cells_ast_edits=edits, feature_cells_ast_additions=added,
        feature_cells_class_assignments=['FeatureCellBirthDeath.BACKEND_SCHEMA'],
        birth_ast_edits=birth_edits, whole_module_ast_inverses=True,
        no_benchmark_oracle_decision_branches=True, trainer_schema=5, backend_schema=9,
        quality_verdict=None, protected_sha256=protected)
    assert not args.output.exists()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status':'PASS','package_sha256':identity,'receipt_sha256':sha(args.output)}))


if __name__ == '__main__':
    main()
