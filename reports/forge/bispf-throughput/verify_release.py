"""Verify the 0.9.1 default against frozen measured source without training."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from particlegan import Recipe, get_recipe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts', type=Path, required=True)
args = parser.parse_args()
frozen = '72db8af31586937045e24723fc17bea153fca652'

def original(path):
    return subprocess.check_output(['git', 'show', f'{frozen}:{path}'], cwd=ROOT)

previous = types.ModuleType('_frozen_release_recipes')
sys.modules[previous.__name__] = previous
exec(compile(original('particlegan/recipes.py'), 'frozen-recipes.py', 'exec'), previous.__dict__)
old, new = previous.get_recipe_fields(), asdict(get_recipe())
delta = {key: {'before': old[key], 'after': new[key]} for key in old if old[key] != new[key]}
assert delta == {
    'optimizer_svd_backend': {'before': 'native', 'after': 'batched'},
    'training_precision': {'before': 'float32', 'after': 'bf16_channels_last'},
}
assert asdict(Recipe(**old)) == old
candidate = json.loads((ROOT / 'configs/forge/ideas/bispf-throughput-v1.json').read_text())
resolved = json.loads(json.dumps(new))
assert all(resolved[key] == value for key, value in candidate['recipe_overrides'].items() if key != 'name')
for name in ('bcap', 'bispf', 'ka2', 'e22', 'e22_routed', 'atlas', 'halloween'):
    assert asdict(get_recipe(name)) == previous.get_recipe_fields(name)

def unchanged_training_body(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name == '_convolution_autocast')]
    return ast.dump(tree, include_attributes=False)

assert unchanged_training_body(original('particlegan/training.py')) == unchanged_training_body(
    (ROOT / 'particlegan/training.py').read_bytes())
state = json.loads((args.artifacts / 'performance-queue/queue/state.json').read_text())
source = next(iter(state['submissions'].values()))['request']['source']
changed = [name for name, digest in source['files'].items()
           if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
assert set(changed) == {'particlegan/recipes.py', 'particlegan/training.py'}
checks = []
for name, count in [('tests-release-091-final.log', 121), ('tests-release-091-compat-final.log', 66),
                    ('tests-release-091-readme.log', 3)]:
    log = args.artifacts / name
    assert f'{count} passed' in log.read_text(), name
    checks.append({'path': str(log), 'sha256': hashlib.sha256(log.read_bytes()).hexdigest(), 'passed': count})
receipt = {
    'schema': 'bispf_throughput_release_verification_v1', 'version': '0.9.1',
    'measured_publication_commit': frozen, 'measured_source_digest': source['digest'],
    'default_matches_measured_candidate_except_label': True, 'recipe_delta': delta,
    'previous_concrete_recipe_restores': True, 'other_named_trainers_unchanged': True,
    'changed_measured_source_files': changed,
    'training_body_unchanged_except_native_bf16_capability_gate': True,
    'qualification_regraded': False, 'training_launched': False, 'software_checks': checks,
}
(ROOT / 'reports/forge/bispf-throughput/release-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
