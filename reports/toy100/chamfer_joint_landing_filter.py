"""Fixed-bank C+Q reallocation and joint-neural landing on saved GAN states.

This is an offline numerical feasibility filter. It copies G/prior, performs
no GAN or Adam update, and never selects a target with known ring centers.
"""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve()
SOURCE_FILES = (
    'reports/toy100/chamfer_joint_landing_filter.py',
    'reports/toy100/chamfer_discrete_reallocation.py',
    'reports/toy100/chamfer_pullback.py',
    'reports/toy100/joint_output_pullback.py',
    'reports/toy100/pr84_early_geometry.py',
    'reports/toy100/coverage_fixed_eval.py',
    'reports/toy100/pr84_critic_refinement_capture.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
    'tests/test_chamfer_discrete_reallocation.py',
    'tests/test_joint_output_pullback.py',
)
WARM_COMPACT_SHA = '77b24bbaf646befe70ac97f23a944d32717306e4ff0b1b8d8b6ff107dfef82f9'
WARM_STAGE_SHA = 'edc2137c32f98f00be91d9f7d3aedd25e52c85ee55591a8d3117ca75213eb502'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    sys.path.insert(0, str(source_root))
    from benchmarks.locked_shared import mode_hold
    from reports.toy100.pr84_early_geometry import _model
    from reports.toy100.pr84_critic_refinement_capture import _sha
    from reports.toy100.coverage_fixed_eval import fixed_draw, score_support
    from reports.toy100.chamfer_discrete_reallocation import _cost, greedy_real_reallocate
    from reports.toy100.chamfer_pullback import chamfer_targets
    from reports.toy100.joint_output_pullback import fit_output_targets

    source_hashes = {}
    for name in SOURCE_FILES:
        raw = HERE.read_bytes() if name.endswith('chamfer_joint_landing_filter.py') else (source_root/name).read_bytes()
        stored = args.output/'source'/name
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_bytes(raw)
        source_hashes[name] = digest(raw)

    archive = source_root/'reports/toy100/continuous-evidence/pr84-finite-cold-prefix100'
    cold_raw = gzip.decompress((archive/'prefix-states.pt.gz').read_bytes())
    expected_cold = json.loads((archive/'result.json').read_text())['state_file_sha256']
    if digest(cold_raw) != expected_cold:
        raise RuntimeError('cold state archive hash mismatch')
    cold = torch.load(io.BytesIO(cold_raw), weights_only=True, map_location='cpu')['selected']
    warm_archive = source_root/'reports/toy100/continuous-evidence/pr84-stationary-failure-diagnosis'
    warm_packed = (warm_archive/'compact-states.pt.gz').read_bytes()
    warm_manifest = json.loads((warm_archive/'manifest.json').read_text())
    if (digest(warm_packed) != WARM_COMPACT_SHA
            or digest(warm_packed) != warm_manifest['files']['compact-states.pt.gz']['sha256']):
        raise RuntimeError('tracked warm compact state archive hash mismatch')
    warm = torch.load(io.BytesIO(gzip.decompress(warm_packed)), weights_only=True, map_location='cpu')
    if _sha(warm[1324]['pre_step']) != WARM_STAGE_SHA:
        raise RuntimeError('warm stage differs from the verified full replay state')

    declaration = dict(
        scope='saved-state fixed-bank C+Q numerical landing only; no host training',
        cases=['cold1 pre_step', 'cold100 post_bounded_g', 'warm1324 pre_step'],
        source_sha256=source_hashes, cold_state_sha256=expected_cold,
        warm_compact_sha256=WARM_COMPACT_SHA, warm_stage_sha256=WARM_STAGE_SHA,
        warm_full_archive_sha256=warm_manifest['original_replay_sha256']['selected-states.pt'],
        real_batch=mode_hold.BATCH,
        objective='C=mean_real nearest_particle squared_distance; '
                  'Q=mean_particle nearest_real squared_distance; S=C+Q',
        output_rule='up to N strict best real-row relocations, then one exact '
                    'fixed-assignment C+Q target per round',
        rounds=8, solver=dict(max_iterations=20, max_halves=12, svd_rtol=1e-6,
                              relative_landing_tolerance=1e-5,
                              minimum_reduction_ratio=.1),
        gate='all targets converge; all eight warm rounds pass; final cold rounds pass',
        evaluation='fixed 4096-draw late .029 output noise, clocks 241..248',
        state_effect='copied G/prior only; no Adam, D, EMA, training or RNG update',
    )
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(event='DECLARED', declaration=declaration)), flush=True)

    def case(label, saved):
        saved_hash = _sha(saved)
        rng_before = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            generator = _model(saved, 'g')
        prior = torch.nn.Parameter(saved['prior']['z'].detach().clone())
        data = torch.Generator().set_state(saved['rng']['data'])
        real = mode_hold.sample_ring(mode_hold.ring_means(), mode_hold.BATCH,
                                     mode_hold.SIGMA, data)
        initial = generator(prior).detach()
        rows = []
        for round_index in range(1, 9):
            points = generator(prior).detach()
            before = _cost(real, points)
            relocated, relocation = greedy_real_reallocate(real, points)
            target, counts, real_assignment, nearest_real = chamfer_targets(real, relocated)
            target_cost = _cost(real, target)
            if target_cost > relocation['final_objective'] + 1e-9:
                raise RuntimeError('fixed-assignment C+Q target raised objective')
            fit = fit_output_targets(generator, prior, target.to(points))
            after = generator(prior).detach()
            actual = _cost(real, after)
            indices, noise = fixed_draw(240+round_index, after)
            grade = score_support(after, indices, noise, mode_hold.ring_means())
            row = dict(round=round_index, objective_before=before,
                       relocation=relocation, target_cost=target_cost,
                       target_max_output_motion=float((target-points.double()).norm(dim=1).max()),
                       chamfer_counts=counts.tolist(),
                       chamfer_real_assignment=real_assignment.tolist(),
                       chamfer_nearest_real=nearest_real.tolist(),
                       fit=fit, objective_after=actual,
                       whole_map_nonincrease=actual <= before + 1e-9,
                       grade=grade, points=after.tolist())
            rows.append(row)
            if fit['status'] != 'CONVERGED':
                break
        if _sha(saved) != saved_hash or not torch.equal(torch.get_rng_state(), rng_before):
            raise RuntimeError('saved state or global CPU RNG changed')
        passes = [row['grade']['modes'] == 8 and row['grade']['hq'] >= .9 for row in rows]
        return dict(label=label, saved_sha256=saved_hash, real=real.tolist(),
                    initial=initial.tolist(), records=rows,
                    all_landed=len(rows) == 8 and all(row['fit']['status'] == 'CONVERGED' for row in rows),
                    all_quality=len(rows) == 8 and all(passes),
                    final_quality=bool(passes and passes[-1]),
                    first_quality_pass=next((i+1 for i, value in enumerate(passes) if value), None),
                    whole_map_nonincrease=all(row['whole_map_nonincrease'] for row in rows),
                    final=rows[-1]['grade'], final_objective=rows[-1]['objective_after'])

    rows = []
    for label, saved in [('cold1', cold[1]['pre_step']),
                         ('cold100', cold[100]['post_bounded_g']),
                         ('warm1324', warm[1324]['pre_step'])]:
        row = case(label, saved)
        rows.append(row)
        (args.output/f'{label}.json').write_text(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        print(json.dumps(dict(event='CASE_DONE', label=label,
                              all_landed=row['all_landed'],
                              all_quality=row['all_quality'],
                              final_quality=row['final_quality'],
                              first_quality_pass=row['first_quality_pass'],
                              whole_map_nonincrease=row['whole_map_nonincrease'],
                              final=row['final'])), flush=True)
        if not row['all_landed']:
            break
    passed = (len(rows) == 3 and all(row['all_landed'] for row in rows)
              and all(row['final_quality'] for row in rows)
              and rows[-1]['all_quality'])
    for name, expected in source_hashes.items():
        current = HERE.read_bytes() if name.endswith('chamfer_joint_landing_filter.py') else (source_root/name).read_bytes()
        if digest(current) != expected:
            raise RuntimeError(f'source changed during saved-state filter: {name}')
    summary = dict(status='PASS' if passed else 'FAIL',
                   declaration=declaration, results=rows,
                   runtime=dict(torch=torch.__version__, threads=torch.get_num_threads(),
                                cpu_capability=torch.backends.cpu.get_cpu_capability()))
    (args.output/'summary.json').write_text(json.dumps(summary, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(event='DONE', status=summary['status'])), flush=True)


if __name__ == '__main__':
    main()
