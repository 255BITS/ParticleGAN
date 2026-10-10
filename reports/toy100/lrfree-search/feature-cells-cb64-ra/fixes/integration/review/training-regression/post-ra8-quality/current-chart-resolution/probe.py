"""Single fixed64/128 current-D chart comparison; descriptive CPU only."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
AREA = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
PACKAGE = ROOT / 'pkg-CB64-RA8'
TOY_FORWARD = ROOT / 'integration/review/training-regression/post-ra5-saved-diagnosis/analyze_saved.py'
GRID_HEAD = ROOT / 'performance/training-regression/count-review/post-ra8-quality/grid-covariance/diagnose.py'
HASH_HELPER = ROOT / 'integration/review/training-regression/post-ra4-quality/measure_saved_utils.py'
GEOMETRY = Path('/ml2/hypergan/lrfree-20260926/harness/hosts/native100/problems.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    freeze = json.loads((AREA / 'SOURCE-FROZEN.json').read_text())
    for path, expected in freeze['source_and_input_sha256'].items():
        assert sha(path) == expected, path
    return freeze


def extract(path, names):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = dict(torch=torch, F=F, hashlib=hashlib, math=math,
                     PROBLEM_NAMES=('grid100', 'rotated100', 'staggered100'), DATA_STD=.03)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return [namespace[name] for name in names]


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def occupancy(counts):
    values = counts.double()
    return dict(bins=len(values), empty=int((values == 0).sum()), min=int(values.min()),
                median=float(values.median()), max=int(values.max()), mean=float(values.mean()))


def tv(a, b):
    return float((a.double()/a.sum()-b.double()/b.sum()).abs().sum()/2)


def annotate(points, centers):
    # Descriptive labels only; no caller can pass these to the production chart.
    return torch.cat([((block.double()[:, None]-centers[None])**2).sum(-1).argmin(1)
                      for block in points.split(256)])


def alias_summary(ids, labels, bins, modes):
    table = torch.bincount(ids*modes+labels, minlength=bins*modes).reshape(bins, modes)
    represented = (table > 0).sum(1)
    majority = table.argmax(1)
    mode_majority = table.argmax(0)
    return dict(contingency=table, nonempty_bins=int((table.sum(1) > 0).sum()),
                multi_mode_bins=int((represented > 1).sum()),
                oracle_modes_per_bin=represented, weighted_majority_purity=float(table.max(1).values.sum()/len(ids)),
                max_modes_per_bin=int(represented.max()), dominant_mode_per_bin=majority,
                dominant_bin_per_mode=mode_majority,
                distinct_dominant_bins_for_modes=int(torch.unique(mode_majority).numel()))


def measurement(snapshot, features, points, centers, Q, parent_limit):
    flags, pvalues, scores = snapshot.support(features)
    categories = snapshot.count_categories(features)
    ids = categories//2
    eligible = ~flags & (pvalues > Q)
    inside_eligible = eligible & (categories.remainder(2) == 0)
    eligible_counts = torch.bincount(ids[eligible], minlength=snapshot.cells)
    inside_counts = torch.bincount(ids[inside_eligible], minlength=snapshot.cells)
    comparison = snapshot.cell_comparison(features)
    targets = snapshot._mass_targets(len(features))
    supported = torch.bincount(ids[~flags], minlength=snapshot.cells)
    vacancies = (targets-supported).clamp_min(0)
    group_vacancies = (snapshot._group_counts(targets)-snapshot._group_counts(supported)).clamp_min(0)
    mass = comparison['mass']; refined = comparison['support']; collapsed = comparison['global_support']
    gross_mass_birth = torch.floor(len(features)*(-mass['difference']).clamp_min(0)+1e-10).long()*mass['deficit']
    gross_inside_birth = (torch.floor(len(features)*(-refined['difference'][0::2]).clamp_min(0)+1e-10).long()
                          *refined['deficit'][0::2])
    possible_mass = torch.minimum(torch.minimum(gross_mass_birth, vacancies), eligible_counts.clamp_max(parent_limit))
    possible_inside = torch.minimum(torch.minimum(gross_inside_birth, vacancies), inside_counts.clamp_max(parent_limit))
    labels = annotate(points, centers)
    modes = len(centers)
    mode_eligible = torch.bincount(labels[eligible], minlength=modes)
    mode_inside = torch.bincount(labels[inside_eligible], minlength=modes)
    laws = {}
    for name, law in (('mass', mass), ('support', refined), ('global_support', collapsed)):
        laws[name] = dict(real_counts=law['real_counts'], fake_counts=law['fake_counts'],
                         pvalues=law['pvalues'], difference=law['difference'],
                         excess=law['excess'].nonzero().flatten(), deficit=law['deficit'].nonzero().flatten())
    return dict(rows=len(features), flags=int(flags.sum()), eligible_pQ=int(eligible.sum()),
                inside_rows=int((categories.remainder(2) == 0).sum()), eligible_inside=int(inside_eligible.sum()),
                outside_fraction=float((categories.remainder(2)).double().mean()),
                scores_quantiles=torch.quantile(scores.double(), torch.tensor([0., .5, .95, 1.], dtype=torch.float64)),
                coarse_cell_TV=tv(mass['fake_counts'], mass['real_counts']),
                refined_category_TV=tv(refined['fake_counts'], refined['real_counts']),
                group_TV=tv(snapshot._group_counts(mass['fake_counts']), snapshot._group_counts(mass['real_counts'])),
                cells_with_eligible_parents=int((eligible_counts > 0).sum()),
                cells_with_inside_parents=int((inside_counts > 0).sum()),
                eligible_parent_counts=eligible_counts, inside_eligible_parent_counts=inside_counts,
                targets=targets, supported_counts=supported, cell_vacancies=vacancies, group_vacancies=group_vacancies,
                vacancy_rows_without_pQ_parents=int(vacancies[eligible_counts == 0].sum()),
                vacancy_rows_without_inside_parents=int(vacancies[inside_counts == 0].sum()),
                target_cells_without_pQ_parents=int(((targets > 0) & (eligible_counts == 0)).sum()),
                mass_birth_upper_bound_before_shared_ledger=int(possible_mass.sum()),
                local_inside_birth_upper_bound_before_shared_ledger=int(possible_inside.sum()),
                unchanged_shared_ordinary_budget=math.floor(Q*len(features)),
                annotation_only=dict(clean_modes=torch.bincount(labels, minlength=modes),
                                     pQ_parent_mode_counts=mode_eligible, inside_parent_mode_counts=mode_inside,
                                     modes_without_pQ_parent=(mode_eligible == 0).nonzero().flatten(),
                                     modes_without_inside_parent=(mode_inside == 0).nonzero().flatten()),
                family_sizes=comparison['family_sizes'], actual_multiplicity=comparison['multiplicity'],
                common_cutoff=comparison['cutoff'], count_laws= laws,
                count_interpretation='descriptive clean finite table; not emitted iid, no equivalence/action/quality claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), 'Preserve previous evidence; choose a new attempt path'
    freeze = verify()
    sys.path.insert(0, str(PACKAGE))
    from particlegan.feature_cells import FeatureCellSnapshot, Q, PARENT_RESERVOIR, paired_average_geometry
    forward, = extract(TOY_FORWARD, ['forward'])
    head, = extract(GRID_HEAD, ['head_features'])
    state_hash, = extract(HASH_HELPER, ['tensor_state_hash'])
    _, geometry = extract(GEOMETRY, ['_centers', 'evaluation_geometry'])
    global_rng = torch.get_rng_state().clone()
    assert not torch.cuda.is_initialized()
    cases = [
        ('toy', ROOT/'validation-cb64-ra8/learned/training/toy/CB64-RA8/checkpoint-2000.pt', 2000),
        ('grid100', ROOT/'validation-cb64-ra8/screens/runs/grid100/final-state.pt', 7000)]
    records = []
    for problem, checkpoint, expected_step in cases:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)['trainer']
        before = state_hash(state)
        assert state['completed_steps'] == expected_step
        weights, bd = state['models'], state['birth_death']
        assert bd['fill'] == state['recipe']['num_particles'] and bd['settings']['cells'] == 64
        assert bd['settings']['rank'] == 8 and bd['settings']['chunk'] == 256
        with torch.no_grad():
            if problem == 'toy':
                feature = lambda points: forward(points, weights['D'], head=True).double()
                fast_points = forward(weights['prior']['z'], weights['G'])
                ema_points = forward(weights['ema_prior']['z'], weights['ema_G'])
                centers = torch.cartesian_prod(torch.linspace(-1, 1, 5, dtype=torch.float64),
                                              torch.linspace(-1, 1, 5, dtype=torch.float64))
            else:
                assert set(weights['G']) == set(weights['ema_G']) == {'weight', 'bias'}
                feature = lambda points: head(points, weights['D']).double()
                fast_points = F.linear(weights['prior']['z'], weights['G']['weight'], weights['G']['bias'])
                ema_points = F.linear(weights['ema_prior']['z'], weights['ema_G']['weight'], weights['ema_G']['bias'])
                centers, _ = geometry('grid100', dtype=torch.float64)
            real_features = feature(bd['reservoir'])
            fast_features, ema_features = feature(fast_points), feature(ema_points)
            baseline_geometry = baseline_rng = None
            for cells in (64, 128):
                started = time.perf_counter()
                private = torch.Generator().set_state(state['cpu_rng'])
                snapshot = FeatureCellSnapshot.fit(real_features, generator=private, cells=cells, rank=8, chunk=256)
                geometry_hash = state_hash([snapshot.mean, snapshot.scale, snapshot.basis])
                rng_hash = state_hash(private.get_state())
                if cells == 64:
                    baseline_geometry, baseline_rng = geometry_hash, rng_hash
                else:
                    assert geometry_hash == baseline_geometry and rng_hash == baseline_rng
                snapshot.cache_queries(fast_features)
                finite = torch.isfinite(fast_points).all(1) & torch.isfinite(ema_points).all(1)
                serving = paired_average_geometry(snapshot, ema_features, step=expected_step,
                              snapshot_serial=bd['snapshot_serial'], finite_coordinates=finite)
                fast = measurement(snapshot, fast_features, fast_points, centers, Q, PARENT_RESERVOIR)
                ema = measurement(snapshot, ema_features, ema_points, centers, Q, PARENT_RESERVOIR)
                # Annotation is deliberately downstream of all chart/decision
                # calculations and receives no write access to the chart.
                reference_ids, _ = snapshot.assign(real_features[0::2])
                labels = annotate(bd['reservoir'][0::2], centers)
                groups = snapshot._mass_topology()
                aliases = dict(cell=alias_summary(reference_ids, labels, snapshot.cells, len(centers)),
                               group=alias_summary(groups[reference_ids], labels, snapshot.mass_groups, len(centers)))
                record = dict(problem=problem, saved_step=expected_step, cells=cells,
                    current_chart_scope='same final FIFO/currentD CPU refit; not historicalGPUchart',
                    chart=dict(rank=snapshot.rank, width=snapshot.width, fitted_rows=len(real_features[0::2]),
                        calibration_rows=snapshot.calibration_rows, groups=snapshot.mass_groups,
                        count_boundary=float(snapshot.count_boundary), count_partition=snapshot.count_partition,
                        topology=snapshot.mass_topology, duplicate_fraction=snapshot.duplicate_fraction,
                        fitted_cell_occupancy=occupancy(snapshot.reference_counts),
                        odd_cell_occupancy=occupancy(snapshot.real_calibration_counts),
                        odd_refined_occupancy=occupancy(snapshot.real_calibration_category_counts),
                        real_odd_outside_fraction=float(snapshot.real_calibration_category_counts[1::2].sum()/snapshot.calibration_rows),
                        actual_multiplicity=3*snapshot.cells+2, common_cutoff=Q/(3*snapshot.cells+2),
                        matched_even_metric_sha256=geometry_hash, matched_fit_rng_end_sha256=rng_hash),
                    actual_saved_stamp=bd['paired_average'], current_cpu_refit_stamp=serving,
                    fast=fast, ema=ema, annotation_only_real_reference_aliasing=aliases,
                    work=dict(snapshot.work), cpu_elapsed_seconds=time.perf_counter()-started)
                records.append(plain(record))
                print(json.dumps(dict(problem=problem, cells=cells, groups=snapshot.mass_groups,
                    eligible_fast=fast['eligible_inside'], eligible_ema=ema['eligible_inside'],
                    same_group=serving['same_group_rows'], joint=serving['coherent_rows'],
                    required=serving['required'], empirical_geometry_eligible=serving['eligible'],
                    real_group_purity=aliases['group']['weighted_majority_purity'],
                    cpu_seconds=record['cpu_elapsed_seconds'])), flush=True)
        assert state_hash(state) == before, 'loaded trainer state mutated'
    assert torch.equal(torch.get_rng_state(), global_rng), 'globalCPU RNG advanced'
    assert not torch.cuda.is_initialized()
    verify()
    result = dict(status='COMPLETE_FIXED_DESCRIPTIVE_PROBE', source_freeze_sha256=sha(AREA/'SOURCE-FROZEN.json'),
                  records=records, resolutions=[64,128], rank=8, Q=Q, conditional_family='3K+2 commonQ',
                  new_seeds=0, training_steps=0, emitted_samples=0, model_constructions=0,
                  private_fit_rng_source='separate clones of savedCPU_rng, same start for64/128',
                  global_and_trainer_rng_unchanged=True, loaded_trainer_tensors_unchanged=True,
                  cuda_initialized=False, production_changes=0, quality_verdict=None,
                  oracle_scope='downstream alias/parent annotation only, never chart/decision input',
                  finished_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable,*sys.argv])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
