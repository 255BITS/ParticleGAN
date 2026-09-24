"""Read-only geometry of the exact first-100 repaired-cold state captures."""

import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

ORIGINAL_COLD_GZ_SHA = '4859f577d5e5124a8672b0f95dceaee85bd1af3c2c5d659eb44e062da1d5f86d'

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from reports.toy100.pr84_critic_refinement_capture import _sha


def _model(state, kind):
    if kind == 'g':
        model = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
        mapping = state['generator']
    else:
        model = SimpleMLPDiscriminator(2, mode_hold.HIDDEN, mode_hold.N_HIDDEN, mode_hold.FOURIER)
        mapping = state['critic']
    model.load_state_dict({name.removeprefix('model.'): value for name, value in mapping.items()})
    model.eval()
    return model


def _flat_delta(first, second, key):
    a, b = first[key], second[key]
    names = [name for name in a if a[name].dtype.is_floating_point and name not in ('model.freqs', 'freqs')]
    return torch.cat([(b[name].double() - a[name].double()).flatten() for name in names]).norm().item()


def _rms(tensor):
    return float(tensor.double().square().sum(-1).mean().sqrt())


def _score_rms(first, second, points):
    with torch.no_grad():
        values = _model(second, 'd')(points) - _model(first, 'd')(points)
    return float(values.double().square().mean().sqrt())


def _support(points):
    means = mode_hold.ring_means()
    quality = mode_hold.diversity(points, means, detailed=True)
    angles = torch.remainder(torch.atan2(points[:, 1], points[:, 0]).double(), 2 * math.pi).sort().values
    gaps = torch.diff(torch.cat((angles, angles[:1] + 2 * math.pi)))
    return dict(centroid=points.double().mean(0).tolist(),
                spread_rms=_rms(points - points.mean(0)),
                radius_mean=float(points.double().norm(dim=-1).mean()),
                angular_span_degrees=float((2 * math.pi - gaps.max()) * 180 / math.pi),
                largest_empty_angle_degrees=float(gaps.max() * 180 / math.pi),
                clean_hq=quality['hq'], clean_modes=quality['modes'],
                nearest_counts=quality['nearest_counts'],
                nearest_mode=quality['nearest_mode'])


@torch.no_grad()
def analyze_step(stage):
    pre, accepted, fitted, after = [stage[key] for key in (
        'pre_step', 'post_accepted_d', 'post_refined_d', 'post_bounded_g')]
    for same in (('generator', pre, accepted), ('prior', pre, accepted),
                 ('generator', accepted, fitted), ('prior', accepted, fitted),
                 ('critic', fitted, after)):
        if any(not torch.equal(value, same[2][same[0]][name])
               for name, value in same[1][same[0]].items()):
            raise RuntimeError(f'non-owner weights changed between captured stages: {same[0]}')
    z0, z1 = pre['prior']['z'], after['prior']['z']
    g0, g1 = _model(pre, 'g'), _model(after, 'g')
    x0, x1 = g0(z0), g1(z1)
    network_only, prior_only = g1(z0), g0(z1)
    motion = x1 - x0
    translation = motion.double().mean(0)
    residual = motion.double() - translation
    total = _rms(motion)
    translation_norm = float(translation.norm())
    deformation = _rms(residual)
    centers = mode_hold.ring_means()
    probe = torch.cat((x0, centers), dim=0)
    dstep = _flat_delta(pre, accepted, 'critic')
    dfit = _flat_delta(accepted, fitted, 'critic')
    return dict(pre=_support(x0), after=_support(x1),
        accepted_g_motion=dict(output_rms=total, common_translation_norm=translation_norm,
            relative_deformation_rms=deformation,
            common_energy_fraction=(translation_norm/total)**2 if total else None,
            translation_vector=translation.tolist(),
            network_only_rms=_rms(network_only-x0),
            prior_only_rms=_rms(prior_only-x0),
            interaction_rms=_rms(x1-network_only-prior_only+x0),
            per_particle_displacement=motion.double().norm(dim=-1).tolist()),
        critic_changes=dict(ordinary_bounded_d_parameter_l2=dstep,
            later_fit_parameter_l2=dfit,
            fit_to_ordinary_parameter_ratio=dfit/dstep if dstep else None,
            fixed_pre_support_and_target_score_rms_ordinary=_score_rms(pre, accepted, probe),
            fixed_pre_support_and_target_score_rms_fit=_score_rms(accepted, fitted, probe)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--original-cold', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = json.loads((args.archive/'result.json').read_text())
    with gzip.open(args.archive/'prefix-states.pt.gz','rb') as stream:
        raw=stream.read()
    if hashlib.sha256(raw).hexdigest() != receipt['state_file_sha256']:
        raise RuntimeError('state sidecar hash differs from exact replay receipt')
    states=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu')
    for step,stages in states['selected'].items():
        for stage,value in stages.items():
            if _sha(value) != receipt['selected_capture']['state_sha256'][str(step)][stage]:
                raise RuntimeError('captured stage hash differs from replay receipt')
    if hashlib.sha256(args.original_cold.read_bytes()).hexdigest() != ORIGINAL_COLD_GZ_SHA:
        raise RuntimeError('archived original PR84 cold control changed')
    with gzip.open(args.original_cold,'rt') as stream:
        original=json.load(stream)
    if original['dynamics']['outer_steps'] != 1200 or len(original['result']['observations']) != 24:
        raise RuntimeError('original smoothed PR84 cold archive is incomplete')
    result=dict(scope='posthoc geometry only; no update uses target centers',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        replay_state_sha256=receipt['state_file_sha256'],
        original_cold_stored_sha256=hashlib.sha256(args.original_cold.read_bytes()).hexdigest(),
        original_original_function_sha256=original['dynamics']['host_source']['original_function_sha256'],
        original_generated_function_sha256=original['dynamics']['host_source']['generated_function_sha256'],
        first100_observations=dict(fitted=receipt['observations'], original=[
            dict(step=row['step'], modes=row['modes'], hq=row['hq'],
                 nearest_counts=row['support']['nearest_counts'])
            for row in original['result']['observations'][:2]]),
        fitted_steps={str(step):analyze_step(stage) for step,stage in states['selected'].items()},
        interpretation='four fitted captures permit geometry and D-displacement attribution; original arm has only archived sparse observations, so its within-step motion is not claimed')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(event='EARLY_GEOMETRY_DONE',output=str(args.output),
                          steps=list(result['fitted_steps']))),flush=True)


if __name__ == '__main__':
    main()
