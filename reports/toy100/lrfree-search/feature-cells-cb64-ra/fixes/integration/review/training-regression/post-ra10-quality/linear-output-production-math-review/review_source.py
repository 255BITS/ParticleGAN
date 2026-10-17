"""Raw-byte/AST production review only; never import the candidate or tensor tools."""
from pathlib import Path
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'integration/review/training-regression/post-ra10-quality/linear-output-production'
BASE = ROOT / 'pkg-CB64-RA10/particlegan'
PACKAGE = OWNER / 'pkg-OUTPUT-MEAN/particlegan'
SELECTED = ROOT / 'performance/sampler-regression/cpu-plan-review/post-ra10-quality/linear-output-mean-prototype/linear_mean.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dumps(value):
    return json.dumps(value, sort_keys=True, indent=2) + '\n'

def methods(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    result[node.name + '.' + child.name] = ast.dump(child)
    return result

class ResolveSelectedImports(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'mt' and node.attr in ('group_means', 'Q'):
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node

def main():
    if (HERE / 'receipt.json').exists():
        raise RuntimeError('retain closed review; refusing overwrite')
    preseal = OWNER / 'SOURCE-FROZEN.json'
    seal = json.loads(preseal.read_text())
    guards = dict(seal['source_and_input_sha256'])
    if not str(seal['status']).startswith('FROZEN'):
        raise ValueError('owner source not frozen')
    for path, digest in guards.items():
        if not Path(path).is_absolute() or sha(path) != digest:
            raise ValueError('owner raw guard mismatch: ' + path)
    guards[str(preseal)] = sha(preseal)
    current = {p.name: sha(p) for p in PACKAGE.glob('*.py')}
    baseline = {p.name: sha(p) for p in BASE.glob('*.py')}
    assert all(guards.get(str(p)) == current[p.name] for p in PACKAGE.glob('*.py'))
    changed = sorted(name for name in current if name in baseline and current[name] != baseline[name])
    added = sorted(set(current) - set(baseline))
    assert len(current) == 31 and len(baseline) == 30
    assert changed == ['feature_cells.py', 'mean_transport.py'] and added == ['output_moments.py']
    assert set(baseline).issubset(current)
    assert sha(OWNER / 'overrides.json') == sha(ROOT / 'configs/overrides-CB64-RA9.json') == sha(ROOT / 'configs/overrides-CB64-RA10.json')
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_text())
    before, after = methods(BASE / 'feature_cells.py'), methods(PACKAGE / 'feature_cells.py')
    assert set(before) == set(after)
    assert {name for name in before if before[name] != after[name]} == {
        'FeatureCellBirthDeath.__init__', 'FeatureCellBirthDeath.state_dict',
        'FeatureCellBirthDeath.check_state', 'FeatureCellBirthDeath.load_state_dict'}
    old_fc = ast.parse((BASE / 'feature_cells.py').read_text())
    new_fc = ast.parse((PACKAGE / 'feature_cells.py').read_text())
    old_class = next(n for n in old_fc.body if isinstance(n, ast.ClassDef) and n.name == 'FeatureCellBirthDeath')
    new_class = next(n for n in new_fc.body if isinstance(n, ast.ClassDef) and n.name == 'FeatureCellBirthDeath')
    schema = next(n for n in new_class.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BACKEND_SCHEMA' for t in n.targets))
    assert ast.literal_eval(schema.value) == 10
    old_members = {n.name: n for n in old_class.body if isinstance(n, ast.FunctionDef)}
    restored = []
    for node in new_class.body:
        if isinstance(node, ast.FunctionDef) and 'FeatureCellBirthDeath.' + node.name in {
            'FeatureCellBirthDeath.__init__', 'FeatureCellBirthDeath.state_dict',
            'FeatureCellBirthDeath.check_state', 'FeatureCellBirthDeath.load_state_dict'}:
            node = copy.deepcopy(old_members[node.name])
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BACKEND_SCHEMA' for t in node.targets):
            node.value = ast.Constant(value=9)
        restored.append(node)
    new_class.body = restored
    for node in new_fc.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'mean_transport':
            node.names = [name for name in node.names if name.name not in ('PROJECTION_POLICY', 'FIT_CHUNK')]
    assert ast.dump(old_fc) == ast.dump(new_fc), 'unexpected feature-cell AST change'
    trees = []
    for path in (SELECTED, PACKAGE / 'output_moments.py'):
        trees.append({n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))})
    core = ('OutputProjection', 'fit_projection', 'FixedOutputMoment', 'OutputObservation', 'freeze_moment', 'odd_witness')
    for name in core:
        nodes = [ResolveSelectedImports().visit(copy.deepcopy(tree[name])) for tree in trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1]), name
    for name in ('group_means', 'neutral_observation', '_version', 'epoch', 'tensor_digest', 'copy_state'):
        assert methods(BASE / 'mean_transport.py')[name] == methods(PACKAGE / 'mean_transport.py')[name], name
    for path in (SELECTED, ROOT / 'quality/RA11-PLAN.md', ROOT / 'quality/results/RA11-selection.json', Path(__file__).resolve()):
        guards[str(path)] = sha(path)
    for path, digest in guards.items():
        if sha(path) != digest:
            raise ValueError('raw guard changed during source review: ' + path)
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(),
        scope='SOURCE_ONLY_PREEXECUTION_BACKEND10_OUTPUT_MOMENT_MATH_AND_PLUMBING',
        owner_source_freeze_sha256=sha(preseal), owner_raw_guards=len(seal['source_and_input_sha256']),
        source_and_input_sha256=guards, package_source_sha256=current,
        original_modules_byte_unchanged=28, changed_original_modules=changed, added_modules=added,
        config_exact_RA9_RA10=True, backend_schema=10, trainer_schema=5,
        exact_selected_core_AST=list(core), original_FC_action_methods_AST_unchanged=True,
        complete_FC_AST_inverse_exact_RA10=True,
        original_owned_observer_epoch_digest_copy_state_AST_unchanged=True,
        reviewed_math='even-only output frame; all odd Xi including zero directions; ddof1; norm psi<=R, norm u<=1, range4R; R=sqrt(moment_rank/Q); alpha=Q/(3*actual_chart_K+3)',
        reviewed_plumbing='critic chart legality separate from output objective; pre-prefix frozen directions; post-prefix fresh means/counts; one shared jitter draw; CPU float64 sequential objective; exact coordinate/output packet; inherited copy history; complete existing reset/rebase and final lease',
        reviewed_state='strict backend10/mean schema2; chart/moment ranks separated; axes/output dimension/fitted rows typed and bound; old9 rejection; selected_axes ownership at save/load',
        limitations=['conditional iid algebra only; trained/shared chart and FIFO are empirical evidence',
            'coordinate orientation/units and top-variance sketch can omit semantic axes',
            'radial clipping and group conditioning remain nonlinear; latent jitter through G remains',
            'clean objective progress does not certify emitted spread, repeated significance, population stationarity or quality'],
        Torch_NumPy_imports_PT_objects_array_values=0, reviewer_numerical_execution=False,
        production_runtime_or_quality_qualified=False, root_GO_required_before_mechanics=True)
    (HERE / 'receipt.json').write_text(dumps(receipt))
    print(dumps({k: receipt[k] for k in ('status', 'scope', 'owner_source_freeze_sha256', 'owner_raw_guards', 'original_modules_byte_unchanged')}), end='')

if __name__ == '__main__':
    main()
