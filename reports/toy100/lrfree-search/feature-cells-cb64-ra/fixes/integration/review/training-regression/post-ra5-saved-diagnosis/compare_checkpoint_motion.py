"""CPU saved-coordinate counterfactuals; row indices do not imply birth continuity."""
import os
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')
MAIN_SHA = 'bf3e2001a703b1801f718bbacc185e24ea7bd631f9cf61997c32a817f505a735'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transitions(old, changed, oracle):
    _, old_modes, old_supported = oracle(old)
    _, new_modes, new_supported = oracle(changed)
    both = old_supported & new_supported
    return dict(old_supported=int(old_supported.sum()), new_supported=int(new_supported.sum()),
        retained_raw_support=int(both.sum()), lost_raw_support=int((old_supported & ~new_supported).sum()),
        gained_raw_support=int((~old_supported & new_supported).sum()),
        retained_raw_support_same_mode=int((both & (old_modes == new_modes)).sum()),
        old_supported_nearest_mode_transition_counts=torch.bincount(
            old_modes[old_supported] * 25 + new_modes[old_supported], minlength=25 * 25).reshape(25, 25).tolist())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--old-step', type=int, required=True)
parser.add_argument('--new-step', type=int, required=True)
parser.add_argument('--package-root', type=Path, required=True)
parser.add_argument('--source-ready', type=Path, required=True)
parser.add_argument('--saved-diagnosis', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert args.old_step < args.new_step
if args.output.exists():
    raise SystemExit('Existing output; choose a new path.')
package = args.package_root.resolve()
sys.path.insert(0, str(package))
sys.path.insert(0, str(PREV))
sys.path.insert(0, str(HERE.parent / 'post-ra4-quality'))
from particlegan.feature_cells import FeatureCellSnapshot
from models_metrics import oracle_centres
from measure_saved_utils import tensor_state_hash

main = HERE / 'analyze_saved.py'
assert sha(main) == MAIN_SHA
source_paths = [*sorted(package.rglob('*.py')), Path(__file__), main, args.source_ready,
    PREV / 'models_metrics.py', HERE.parent / 'post-ra4-quality/measure_saved_utils.py']
source_before = {str(p): sha(p) for p in source_paths}
ready = json.loads(args.source_ready.read_text())
assert {str(p.relative_to(package / 'particlegan')): sha(p)
    for p in package.rglob('*.py')} == ready['package_source_sha256']
input_paths = [args.run_dir / f'checkpoint-{step:04d}.pt'
    for step in (args.old_step, args.new_step)] + [args.saved_diagnosis]
inputs_before = {str(p): sha(p) for p in input_paths}
diagnosis = json.loads(args.saved_diagnosis.read_text())
record_by_step = {r['step']: r for r in diagnosis['records']}
for step, path in zip((args.old_step, args.new_step), input_paths):
    assert record_by_step[step]['checkpoint_sha256'] == sha(path)
centres = oracle_centres()
namespace = dict(torch=torch, F=F, centres=centres, NMODES=len(centres))
tree = ast.parse(main.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
    and n.name in {'forward', 'oracle', 'score'}]
assert {n.name for n in functions} == {'forward', 'oracle', 'score'}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(main), 'exec'), namespace)
forward, oracle, score = [namespace[k] for k in ('forward', 'oracle', 'score')]
rng_before = torch.get_rng_state().clone()
old, new = [torch.load(p, map_location='cpu', weights_only=False)['trainer'] for p in input_paths[:2]]
state_before = [tensor_state_hash(s) for s in (old, new)]
assert (old['completed_steps'], new['completed_steps']) == (args.old_step, args.new_step)
results = {}
cohort = {}
rows = torch.as_tensor(old['birth_death']['last'].get('novel_birth_children', []), dtype=torch.long)
with torch.no_grad():
    bd = old['birth_death']
    real_features = forward(bd['reservoir'], old['models']['D'], head=True).double()
    snapshot = FeatureCellSnapshot.fit(real_features,
        generator=torch.Generator().set_state(old['cpu_rng']), cells=bd['settings']['cells'],
        rank=bd['settings']['rank'], chunk=bd['settings']['chunk'])
    for kind, prior_key, generator_key in [('fast', 'prior', 'G'), ('ema', 'ema_prior', 'ema_G')]:
        old_z, new_z = [s['models'][prior_key]['z'] for s in (old, new)]
        points = dict(old_G_old_z=forward(old_z, old['models'][generator_key]),
            new_G_old_z=forward(old_z, new['models'][generator_key]),
            old_G_new_z=forward(new_z, old['models'][generator_key]),
            new_G_new_z=forward(new_z, new['models'][generator_key]))
        scores = {name: score(value) for name, value in points.items()}
        assert scores['old_G_old_z'] == record_by_step[args.old_step][kind]
        assert scores['new_G_new_z'] == record_by_step[args.new_step][kind]
        results[kind] = dict(scores=scores,
            fixed_old_coordinates_generator_change=transitions(points['old_G_old_z'], points['new_G_old_z'], oracle),
            fixed_old_generator_table_change=transitions(points['old_G_old_z'], points['old_G_new_z'], oracle),
            generator_change_on_new_coordinates=transitions(points['old_G_new_z'], points['new_G_new_z'], oracle))
        cohort[kind] = []
        for name in ('old_G_old_z', 'new_G_old_z'):
            feature = forward(points[name][rows], old['models']['D'], head=True).double()
            flags, pvalues, _ = snapshot.support(feature)
            category = snapshot.count_categories(feature)
            distance, modes, accepted = oracle(points[name][rows])
            for i, row in enumerate(rows.tolist()):
                cohort[kind].append(dict(saved_row_index=row, counterfactual=name,
                    fixed_saved_coordinates=True, generator_step=args.old_step if name == 'old_G_old_z' else args.new_step,
                    nearest_mode=int(modes[i]), raw_supported=bool(accepted[i]), distance=float(distance[i]),
                    old_fixed_cpu_head_pvalue=float(pvalues[i]), old_fixed_cpu_head_inside=bool(category[i] % 2 == 0),
                    old_fixed_cpu_head_eligible=bool((pvalues[i] > .05) & (category[i] % 2 == 0)),
                    old_fixed_cpu_head_flag=bool(flags[i])))
assert source_before == {str(p): sha(p) for p in source_paths}
assert inputs_before == {str(p): sha(p) for p in input_paths}
assert state_before == [tensor_state_hash(s) for s in (old, new)]
assert torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized()
receipt = dict(status='COMPLETE_READ_ONLY_FIXED_COORDINATE_DIAGNOSIS', old_step=args.old_step,
    new_step=args.new_step, results=results, old_observed_birth_coordinate_cohort=cohort,
    old_observed_birth_reaction=old['birth_death']['last'].get('step'),
    source_sha256=source_before, input_sha256=inputs_before,
    all_inputs_sources_and_checkpoint_tensors_unchanged=True, global_rng_unchanged=True,
    cpu_only=True, cuda_initialized=False, new_seeds=0, new_training_steps=0, new_proposals=0, new_emissions=0,
    limits=['Nonlinear saved-coordinate counterfactuals are descriptive, not an additive causal decomposition.',
        'Table changes combine optimizer motion, copies and births; intervening actions are not fully logged.',
        'Fixed old coordinates are offline values; no claim that those birth incarnations remained in the later table.',
        'The fixed old learned head uses a saved CPU refit, not the historical GPU birth geometry.',
        'Raw mode labels annotate diagnostics and do not enter production policy.'])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(receipt, indent=2) + '\n')
for kind, result in results.items():
    print(json.dumps(dict(table=kind, scores={name: {key: score[key] for key in ('precision', 'coverage')}
        for name, score in result['scores'].items()}, fixed_old_coordinates=result['fixed_old_coordinates_generator_change']
        | {'old_supported_nearest_mode_transition_counts': 'in receipt'})), flush=True)
