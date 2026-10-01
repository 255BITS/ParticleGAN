"""Read-only current learned FAST/EMA coherence; no emissions or training."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import ast
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
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
PACKAGE = ROOT / 'pkg-CB64-RA7'
RUN = ROOT / 'validation-cb64-ra7/learned/training/toy/CB64-RA7'
READY = ROOT / 'quality/ra7/READY.json'
MAIN = ROOT / 'integration/review/training-regression/post-ra5-saved-diagnosis/analyze_saved.py'
UTIL = ROOT / 'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py'
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(UTIL.parent))
from particlegan.feature_cells import FeatureCellSnapshot, Q
from measure_saved_utils import tensor_state_hash

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(MAIN) == 'bf3e2001a703b1801f718bbacc185e24ea7bd631f9cf61997c32a817f505a735'
package_map = {str(p.relative_to(PACKAGE/'particlegan')): sha(p) for p in PACKAGE.rglob('*.py')}
assert package_map == json.loads(READY.read_text())['package_source_sha256']
out = HERE / 'receipt.json'
if out.exists():
    raise SystemExit('Existing receipt; use a separate attempt directory.')
namespace = dict(torch=torch, F=F)
definitions = [n for n in ast.parse(MAIN.read_text()).body
    if isinstance(n, ast.FunctionDef) and n.name == 'forward']
assert len(definitions) == 1
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(MAIN), 'exec'), namespace)
forward = namespace['forward']
source_paths = [*sorted(PACKAGE.rglob('*.py')), READY, MAIN, UTIL, Path(__file__), HERE/'PROTOCOL.md']
sources = {str(p): sha(p) for p in source_paths}
checkpoints = [RUN/f'checkpoint-{step:04d}.pt' for step in (500, 1000, 2000)]
inputs = {str(p): sha(p) for p in checkpoints}
global_rng = torch.get_rng_state().clone()


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def tv(a, b):
    return float((a.double()/a.sum()-b.double()/b.sum()).abs().sum()/2)


def measurement(snapshot, features):
    flags, pvalues, _ = snapshot.support(features)
    categories = snapshot.count_categories(features)
    cells = categories // 2
    group = snapshot._mass_topology()[cells]
    eligible = ~flags & (pvalues>Q) & (categories%2==0)
    comparison = snapshot.cell_comparison(features)
    coarse_counts = comparison['mass']['fake_counts']
    group_counts = snapshot._group_counts(coarse_counts)
    group_real = snapshot._group_counts(snapshot.real_calibration_counts)
    detail = dict(rows=len(features), flags=int(flags.sum()), eligible_pQ=int((pvalues>Q).sum()),
        inside_rows=int((categories%2==0).sum()), eligible_inside=int(eligible.sum()),
        group_counts=group_counts, group_real_counts=group_real, group_TV=tv(group_counts,group_real),
        coarse_cell_TV=tv(coarse_counts,comparison['mass']['real_counts']),
        refined_category_TV=tv(comparison['support']['fake_counts'],comparison['support']['real_counts']),
        global_outside_fraction=float((categories%2).double().mean()),
        real_global_outside_fraction=float(comparison['global_support']['real_counts'][1]/snapshot.calibration_rows),
        conditional_formula_descriptive_only=True, conditional_comparison=comparison)
    for law in ('mass', 'support', 'global_support'):
        sub = comparison[law]
        detail[law+'_excess_categories'] = sub['excess'].nonzero().flatten()
        detail[law+'_deficit_categories'] = sub['deficit'].nonzero().flatten()
    return detail, dict(flags=flags,pvalues=pvalues,eligible=eligible,categories=categories,cells=cells,group=group)


def contingency(a, b):
    # Rows: FAST false/true; columns: EMA false/true.
    return torch.bincount(a.long()*2+b.long(), minlength=4).reshape(2,2)


records = []
for path in checkpoints:
    state = torch.load(path, map_location='cpu', weights_only=False)['trainer']
    before = tensor_state_hash(state)
    weights, bd = state['models'], state['birth_death']
    private_rng = torch.Generator().set_state(state['cpu_rng'])
    with torch.no_grad():
        real_features = forward(bd['reservoir'],weights['D'],head=True).double()
        snapshot = FeatureCellSnapshot.fit(real_features, generator=private_rng,
            cells=bd['settings']['cells'],rank=bd['settings']['rank'],chunk=bd['settings']['chunk'])
        fast_points = forward(weights['prior']['z'],weights['G'])
        ema_points = forward(weights['ema_prior']['z'],weights['ema_G'])
        fast_features = forward(fast_points,weights['D'],head=True).double()
        ema_features = forward(ema_points,weights['D'],head=True).double()
        fast, a = measurement(snapshot,fast_features)
        ema, b = measurement(snapshot,ema_features)
        n = len(fast_features)
        same_group = a['group']==b['group']
        same_cell = a['cells']==b['cells']
        paired_groups = torch.bincount(a['group']*snapshot.mass_groups+b['group'],
            minlength=snapshot.mass_groups**2).reshape(snapshot.mass_groups,snapshot.mass_groups)
        requirement = n-math.floor(Q*n)
        pair = dict(finite_coordinates=bool(torch.isfinite(fast_points).all() & torch.isfinite(ema_points).all()),
            required_same_group_rows=requirement, same_group_rows=int(same_group.sum()),
            same_group_fraction=float(same_group.double().mean()),
            root_fixed_empirical_group_coherence=bool(int(same_group.sum())>=requirement),
            same_cell_rows=int(same_cell.sum()), same_refined_category_rows=int((a['categories']==b['categories']).sum()),
            group_contingency=paired_groups, eligible_inside_contingency=contingency(a['eligible'],b['eligible']),
            eligible_pQ_contingency=contingency(a['pvalues']>Q,b['pvalues']>Q),
            unflagged_contingency=contingency(~a['flags'],~b['flags']),
            inside_contingency=contingency(a['categories']%2==0,b['categories']%2==0),
            same_group_both_eligible_inside=int((same_group&a['eligible']&b['eligible']).sum()),
            same_group_ema_eligible_inside=int((same_group&b['eligible']).sum()),
            same_group_fast_eligible_inside=int((same_group&a['eligible']).sum()),
            empirical_group_TV_between_pairs=tv(fast['group_counts'],ema['group_counts']),
            output_coordinate_RMS=float((fast_points.double()-ema_points.double()).square().mean().sqrt()),
            output_coordinate_RMS_scope='same current row IDs; no historical row continuity assumed')
        quality_order = dict(group_TV_ema_no_worse=ema['group_TV']<=fast['group_TV'],
            coarse_cell_TV_ema_no_worse=ema['coarse_cell_TV']<=fast['coarse_cell_TV'],
            refined_category_TV_ema_no_worse=ema['refined_category_TV']<=fast['refined_category_TV'],
            global_outside_ema_no_more=ema['global_outside_fraction']<=fast['global_outside_fraction'],
            eligible_inside_ema_no_less=ema['eligible_inside']>=fast['eligible_inside'])
        group_real = fast['group_real_counts'].double()/snapshot.calibration_rows
        fg = fast['group_counts'].double()/n
        eg = ema['group_counts'].double()/n
        quality_order['groups_ema_absolute_mass_discrepancy_worse'] = ((eg-group_real).abs()>(fg-group_real).abs()).nonzero().flatten()
        fcat = fast['conditional_comparison']['support']['difference'].abs()
        ecat = ema['conditional_comparison']['support']['difference'].abs()
        quality_order['refined_categories_ema_absolute_mass_discrepancy_worse'] = (ecat>fcat).nonzero().flatten()
        cross = []
        rows = torch.arange(0,n,8)
        assert len(rows)==128
        for generator_name, prior_name in (('G','prior'),('ema_G','ema_prior'),('G','ema_prior'),('ema_G','prior')):
            points = forward(weights[prior_name]['z'][rows],weights[generator_name])
            features = forward(points,weights['D'],head=True).double()
            flags,pvalues,_ = snapshot.support(features)
            categories = snapshot.count_categories(features)
            group = snapshot._mass_topology()[categories//2]
            cross.append(dict(generator=generator_name,prior=prior_name,rows=rows,
                eligible_inside=int(((pvalues>Q)&(categories%2==0)).sum()), flags=int(flags.sum()),
                same_group_as_actual_fast=int((group==a['group'][rows]).sum()),
                same_group_as_actual_ema=int((group==b['group'][rows]).sum()),
                global_outside_rows=int((categories%2).sum())))
        table = state['lr_settle'][0][1]
        alpha = min(1.,table['s']/(state['recipe']['serve_average']*table['b']))
        testers = [[None if t is None else {k:t[k] for k in ('s','b','tau','last_decisive','last','windows','counts')}
            for t in row] for row in state['lr_settle']]
        averaging = dict(alpha=alpha,e_folding_window_updates=-1/math.log1p(-alpha),
            inverse_weight_window_updates=1/alpha,table_intrinsic_window=state['recipe']['serve_average']*table['b'],
            generator_intrinsic_window_at_current_scale=state['lr_settle'][0][0]['s']/alpha,
            sigma_intrinsic_window_at_current_scale=state['lr_settle'][0][2]['s']/alpha,
            actual_serving='EMA' if table['last_decisive']==-1 else 'FAST',
            serves_entire_matched_model_prior_pair=True,ema_does_not_average_sigma=True)
        # For fixed categorical probabilities and iid heldout reference rows:
        # TV(p_hat,p)=max_A[p_hat(A)-p(A)]. There are 2^G subsets.
        # Hoeffding + union bound gives <=sqrt(log(2^G/delta)/(2m)).
        radius = math.sqrt((snapshot.mass_groups*math.log(2)-math.log(Q))/(2*snapshot.calibration_rows))
        finite_power = dict(assumption='fixed even-fitted chart and iid odd reference rows, exact clean finite population only',
            confidence_delta=Q,heldout_rows=snapshot.calibration_rows,groups=snapshot.mass_groups,
            real_group_TV_radius=radius,
            fast_clean_real_group_TV_upper=min(1.,fast['group_TV']+radius),
            ema_clean_real_group_TV_upper=min(1.,ema['group_TV']+radius),
            formula='sqrt((G*log(2)-log(delta))/(2*m)) via all-subsets one-sided Hoeffding union bound',
            clean_to_emitted_error_bound_available=False, learned_head_training_dependence_not_certified=True,
            no_nondiscovery_equivalence_claim=True)
    assert tensor_state_hash(state)==before
    record = dict(step=state['completed_steps'],checkpoint_sha256=inputs[str(path)],
        snapshot=dict(valid_metric=snapshot.valid_metric,duplicate_fraction=snapshot.duplicate_fraction,
            rank=snapshot.rank,cells=snapshot.cells,groups=snapshot.mass_groups,
            topology=snapshot.mass_topology,reference_group_ids=snapshot.mass_group_ids,
            count_boundary=float(snapshot.count_boundary),count_partition=snapshot.count_partition,
            chart_scope='single current saved FIFO/current D CPU refit, not historical GPU chart'),
        fast=fast,ema=ema,pair=pair,empirical_quality_order=quality_order,crossed_pairs_fixed128=cross,
        averaging=averaging,initial_rates=state['initial_lrs'],
        applied_rates=[[g['lr'] for g in opt['param_groups']] for opt in state['optimizers']],
        settle_testers=testers,finite_power=finite_power,checkpoint_tensors_unchanged=True)
    records.append(record)
    print(json.dumps(dict(step=record['step'],groups=snapshot.mass_groups,
        same_group=pair['same_group_rows'],required=requirement,eligible_inside_fast=fast['eligible_inside'],
        eligible_inside_ema=ema['eligible_inside'],group_TV_fast=fast['group_TV'],group_TV_ema=ema['group_TV'],
        outside_fast=fast['global_outside_fraction'],outside_ema=ema['global_outside_fraction'])),flush=True)

assert sources == {str(p):sha(p) for p in source_paths}
assert inputs == {str(p):sha(p) for p in checkpoints}
assert torch.equal(global_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
receipt = dict(status='COMPLETE_READ_ONLY_CURRENT_PAIR_DIAGNOSIS',records=records,
    source_sha256=sources,input_sha256=inputs,all_sources_inputs_tensors_and_global_rng_unchanged=True,
    cpu_only=True,cuda_initialized=False,new_emissions=0,new_training_steps=0,new_optimizer_calls=0,
    new_seeds=0,production_changed=False,quality_verdict=None,
    limits=['Clean row counts are dependent, noiseless finite-population descriptions, not emitted iid draws.',
        'Existing conditional formulas are descriptive here; a new joint two-model decision needs common multiplicity.',
        'No rejection of mismatch cannot establish positive equivalence.',
        'Current CPU fit is not historical GPU chart replay; saved CUDA RNG streams are not consumed.',
        'Paired agreement at one endpoint is not temporal stationarity or per-incarnation survival.',
        'Finite confidence illustration does not certify adaptively learned critic/reference independence.'])
out.write_text(json.dumps(plain(receipt),indent=2)+'\n')
print(json.dumps(dict(event='current_pair_diagnosis_complete',output=str(out),status=receipt['status'])),flush=True)
