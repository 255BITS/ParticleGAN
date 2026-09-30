"""CPU virtual G steps at two root-specified rates using recorded gradients."""
import os
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PACKAGE = ROOT / 'pkg-CB64-RA6'
DIAG = HERE.parents[1] / 'post-ra5-saved-diagnosis'
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation')
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PREV))
sys.path.insert(0, str(HERE.parents[1] / 'post-ra4-quality'))
from particlegan.feature_cells import FeatureCellSnapshot
from models_metrics import oracle_centres
from measure_saved_utils import tensor_state_hash

output = HERE / 'result.json'
if output.exists():
    raise SystemExit('Existing result; preserve it and choose a fresh attempt directory.')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
main = DIAG / 'analyze_saved.py'
assert sha(main) == 'bf3e2001a703b1801f718bbacc185e24ea7bd631f9cf61997c32a817f505a735'
ready_path = ROOT / 'quality/ra6/READY.json'
ready = json.loads(ready_path.read_text())
package_map = {str(p.relative_to(PACKAGE / 'particlegan')): sha(p) for p in PACKAGE.rglob('*.py')}
assert package_map == ready['package_source_sha256']
sources = [*sorted(PACKAGE.rglob('*.py')), main, Path(__file__), HERE / 'PROTOCOL.md', ready_path,
    PREV / 'models_metrics.py', HERE.parents[1] / 'post-ra4-quality/measure_saved_utils.py']
source_before = {str(p): sha(p) for p in sources}
paths = [ROOT / f'validation-cb64-ra6/learned/training/toy/CB64-RA6/checkpoint-{step:04d}.pt'
    for step in (1000, 2000)]
inputs = paths + [DIAG / 'ra6-step2000/receipt.json', ROOT / 'configs/overrides-CB64-RA6.json']
input_before = {str(p): sha(p) for p in inputs}
saved_diagnosis = json.loads(inputs[2].read_text())
diagnosis_by_step = {r['step']: r for r in saved_diagnosis['records']}
centres = oracle_centres()
namespace = dict(torch=torch, F=F, centres=centres, NMODES=len(centres))
definitions = [n for n in ast.parse(main.read_text()).body if isinstance(n, ast.FunctionDef)
    and n.name in {'forward', 'oracle', 'score'}]
assert {n.name for n in definitions} == {'forward', 'oracle', 'score'}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(main), 'exec'), namespace)
forward, raw_annotation = namespace['forward'], namespace['score']


def virtual_step(weights, optimizer, factor):
    group = optimizer['param_groups'][0]
    assert len(group['params']) == len(weights) == 6 and group['betas'][0] == 0
    assert group['weight_decay'] == 0 and group['amsgrad'] and not group['maximize']
    assert optimizer['regularizer']['direct'] is None
    updated = {}; total_delta_squared = 0.; total_gradient_squared = 0.; reference_parameters = []
    for name, pid in zip(weights, group['params']):
        weight, state = weights[name], optimizer['state'][pid]
        assert state['exp_avg'].shape == weight.shape
        gradient = state['exp_avg'].clone()  # beta1=0 and no G-specific gradient modifier.
        second = state['exp_avg_sq'].clone().mul_(group['betas'][1])
        second.addcmul_(gradient, gradient, value=1 - group['betas'][1])
        second = torch.maximum(state['max_exp_avg_sq'], second)
        step = float(state['step']) + 1
        denominator = second.sqrt().div_(math.sqrt(1 - group['betas'][1] ** step)).add_(group['eps'])
        updated[name] = weight.clone().addcdiv_(gradient, denominator, value=-group['lr'] * factor)
        delta = updated[name].double() - weight.double()
        total_delta_squared += float(delta.square().sum())
        total_gradient_squared += float(gradient.double().square().sum())
        parameter = torch.nn.Parameter(weight.clone())
        parameter.grad = gradient
        reference_parameters.append(parameter)
    reference_group = deepcopy(group); reference_group['lr'] *= factor
    reference = torch.optim.Adam(reference_parameters)
    reference.load_state_dict(dict(state={pid: deepcopy(optimizer['state'][pid]) for pid in group['params']},
        param_groups=[reference_group]))
    reference.step()
    max_error = max(float((updated[name] - value.detach()).abs().max())
        for name, value in zip(weights, reference_parameters))
    assert all(torch.equal(updated[name], value.detach()) for name, value in zip(weights, reference_parameters))
    return updated, dict(parameter_delta_l2=math.sqrt(total_delta_squared),
        recorded_gradient_l2=math.sqrt(total_gradient_squared), cpu_adam_reference_max_error=max_error,
        virtual_rate=group['lr'] * factor, recorded_gradient_step=float(optimizer['state'][group['params'][0]]['step']))


def measurement(snapshot, features, old=None):
    flags, pvalues, scores = snapshot.support(features)
    category = snapshot.count_categories(features)
    cell = torch.div(category, 2, rounding_mode='floor')
    eligible = (pvalues > .05) & (category.remainder(2) == 0)
    projected = snapshot.transform(features)
    result = dict(eligible_pQ=int((pvalues > .05).sum()), eligible_inside=int(eligible.sum()),
        flags=int(flags.sum()), categories=torch.bincount(category, minlength=2 * snapshot.cells).tolist())
    if old is not None:
        previous, initial, old_cell, old_category = old
        displacement = (projected - previous).norm(dim=1)
        normalized = displacement / snapshot.cell_scale[old_cell]
        result.update(initial_eligible_inside=int(initial.sum()),
            retained_eligible_inside=int((initial & eligible).sum()),
            retained_eligible_inside_same_cell=int((initial & eligible & (old_cell == cell)).sum()),
            lost_initial_eligible_inside=int((initial & ~eligible).sum()),
            changed_cell_rows=int((old_cell != cell).sum()), changed_category_rows=int((old_category != category).sum()),
            projected_displacement_mean=float(displacement.mean()),
            projected_displacement_rms=float(displacement.square().mean().sqrt()),
            displacement_in_old_cell_scale_mean=float(normalized.mean()),
            displacement_in_old_cell_scale_max=float(normalized.max()),
            categorical_TV_from_initial=float((torch.bincount(category, minlength=2 * snapshot.cells)
                - torch.bincount(old_category, minlength=2 * snapshot.cells)).abs().sum()) / (2 * len(features)))
    return result, (projected, eligible, cell, category)


rng_before = torch.get_rng_state().clone()
records = []
for path in paths:
    state = torch.load(path, map_location='cpu', weights_only=False)['trainer']
    before = tensor_state_hash(state)
    assert diagnosis_by_step[state['completed_steps']]['checkpoint_sha256'] == sha(path)
    weights, optimizer, bd = state['models'], state['optimizers'][0], state['birth_death']
    assert list(weights['G']) == ['0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias']
    assert all(float(optimizer['state'][pid]['step']) == state['completed_steps']
        for pid in optimizer['param_groups'][0]['params'])
    real_features = forward(bd['reservoir'], weights['D'], head=True).double()
    snapshot = FeatureCellSnapshot.fit(real_features, generator=torch.Generator().set_state(state['cpu_rng']),
        cells=bd['settings']['cells'], rank=bd['settings']['rank'], chunk=bd['settings']['chunk'])
    z = weights['prior']['z']
    with torch.no_grad():
        original = forward(z, weights['G'])
        features = forward(original, weights['D'], head=True).double()
        initial_measure, initial_tuple = measurement(snapshot, features)
        assert raw_annotation(original) == diagnosis_by_step[state['completed_steps']]['fast']
        candidates = []
        for factor in (1., .25):
            candidate_weights, adam_detail = virtual_step(weights['G'], optimizer, factor)
            candidate_points = forward(z, candidate_weights)
            candidate_features = forward(candidate_points, weights['D'], head=True).double()
            measured, _ = measurement(snapshot, candidate_features, initial_tuple)
            candidates.append(dict(rate_factor=factor, learned_measurements=measured, adam=adam_detail,
                output_displacement_rms=float((candidate_points - original).double().square().mean().sqrt()),
                raw_oracle_annotation=raw_annotation(candidate_points)))
    assert before == tensor_state_hash(state)
    records.append(dict(checkpoint=state['completed_steps'], initial_learned=initial_measure,
        initial_raw_oracle_annotation=raw_annotation(original), candidates=candidates,
        available_optimizer_information=dict(beta1=optimizer['param_groups'][0]['betas'][0],
            beta2=optimizer['param_groups'][0]['betas'][1], amsgrad=True, latest_gradient_available=True,
            gradient_basis='saved exp_avg equals latest G gradient because beta1=0',
            next_gradient_available=False, historical_step_reconstructed=False,
            gpu_stream_bytes={k: v.numel() for k, v in state['streams'].items()},
            gpu_streams_installed_on_cpu=False),
        initial_rates=state['initial_lrs'], current_generator_rate=optimizer['param_groups'][0]['lr'],
        snapshot=dict(valid_metric=snapshot.valid_metric, rank=snapshot.rank, cells=snapshot.cells,
            count_boundary=float(snapshot.count_boundary), support_q=.05,
            scope='saved current FIFO/critic CPU refit held fixed for both virtual candidates'),
        all_checkpoint_tensors_unchanged=True))
assert source_before == {str(p): sha(p) for p in sources}
assert input_before == {str(p): sha(p) for p in inputs}
assert torch.equal(rng_before, torch.get_rng_state()) and not torch.cuda.is_initialized()
output.write_text(json.dumps(dict(status='COMPLETE_FIXED_INPUT_G_RATE_PROBE', records=records,
    source_sha256=source_before, input_sha256=input_before,
    all_sources_inputs_tensors_and_rng_unchanged=True, cuda_initialized=False, cpu_only=True,
    new_training_steps=0, new_seeds=0, new_data=0, new_emissions=0, new_birth_proposals=0,
    virtual_private_adam_calls=4, production_changed=False, quality_verdict=None,
    proposed_coupled_config=None,
    limits=['Virtual next Adam step repeats a recorded old gradient; no actual next or historical gradient replay.',
        'G-only virtual candidates hold sigma fixed; prospective config also quarters learned-sigma optimization.',
        'Learned measurements use a fixed saved CPU refit, not the historical GPU partition.',
        'Oracle annotations never choose candidate levels or acceptance.',
        'One-step reduced motion does not establish cumulative learned quality or canonical stability.']), indent=2) + '\n')
for record in records:
    print(json.dumps(dict(checkpoint=record['checkpoint'], initial_eligible=record['initial_learned']['eligible_inside'],
        candidates=[dict(factor=c['rate_factor'], **c['learned_measurements']) for c in record['candidates']])), flush=True)
