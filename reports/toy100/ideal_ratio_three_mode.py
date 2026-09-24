"""Population-ratio field on a captured three-mode cloud and its centered copy.

This is an offline local-field calculation, not a GAN update. Target centers
are used to construct and grade the artificial centered cloud only. Both p
and q include the native input/output Gaussian noise at the captured clock;
the late zero-input-noise law is shown separately, not substituted into the
step-100 host. q support is frozen when differentiating each query point.
"""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator


ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def exact_ring_centers():
    return torch.tensor([[mode_hold.RADIUS * math.cos(2*math.pi*k/mode_hold.N_MODES),
                          mode_hold.RADIUS * math.sin(2*math.pi*k/mode_hold.N_MODES)]
                         for k in range(mode_hold.N_MODES)], dtype=torch.float64)


def log_density(x, means, sigma):
    squared = (x[:, None, :] - means[None, :, :]).square().sum(-1)
    return (torch.logsumexp(-squared/(2*sigma*sigma), dim=1)
            - math.log(len(means)) - x.shape[-1]*math.log(sigma*math.sqrt(2*math.pi)))


def ratio_score(x, p_means, q_means, sigma_p, sigma_q):
    return log_density(x, p_means, sigma_p) - log_density(x, q_means, sigma_q)


def ratio_gradient(x, p_means, q_means, sigma_p, sigma_q):
    """Stable mixture-score sum; subtract displacements before averaging."""
    p_offset = p_means[None, :, :] - x[:, None, :]
    q_offset = q_means[None, :, :] - x[:, None, :]
    p_weight = torch.softmax(-p_offset.square().sum(-1)/(2*sigma_p*sigma_p), dim=1)
    q_weight = torch.softmax(-q_offset.square().sum(-1)/(2*sigma_q*sigma_q), dim=1)
    return ((p_weight[:, :, None]*p_offset).sum(1)/(sigma_p*sigma_p)
            - (q_weight[:, :, None]*q_offset).sum(1)/(sigma_q*sigma_q))


def norm_without_underflow(row):
    return math.hypot(float(row[0]), float(row[1]))


def analyze_cloud(support, p_means, missing, sigma_p, sigma_q):
    nearest = torch.cdist(support, p_means).argmin(1)
    occupied = p_means[nearest]
    missing_means = p_means[missing]
    missing_for_each = missing_means[torch.cdist(occupied, missing_means).argmin(1)]
    to_occupied = occupied-support
    unit_occupied = to_occupied/to_occupied.norm(dim=1, keepdim=True)
    to_missing_from_occupied = missing_for_each-occupied
    unit_missing = to_missing_from_occupied/to_missing_from_occupied.norm(dim=1, keepdim=True)
    gradient = ratio_gradient(support, p_means, support.detach(), sigma_p, sigma_q)
    occupied_projection = (gradient*unit_occupied).sum(1)
    missing_projection = (gradient*unit_missing).sum(1)
    # Validate the hand-coded score formula against the exact autodiff of a
    # frozen-q log density on these actual clean support points.
    query = support.detach().clone().requires_grad_(True)
    autodiff = torch.autograd.grad(ratio_score(query, p_means, support.detach(),
                                              sigma_p, sigma_q).sum(), query)[0]
    max_formula_error = float((gradient-autodiff).abs().max())
    if max_formula_error > 1e-9:
        raise AssertionError('analytic mixture score disagrees with frozen-q autodiff')
    return dict(max_score_gradient_norm=max(norm_without_underflow(row) for row in gradient),
                mean_score_gradient_norm=sum(norm_without_underflow(row) for row in gradient)/len(gradient),
                toward_assigned_occupied_center=int((occupied_projection > 0).sum()),
                toward_nearest_missing_chord=int((missing_projection > 0).sum()),
                occupied_center_projection=occupied_projection.tolist(),
                missing_chord_projection=missing_projection.tolist(),
                gradient=gradient.tolist(), max_autodiff_error=max_formula_error)


def centered_field(centered, p_means, sigma_p, sigma_q):
    gradient = ratio_gradient(centered, p_means, centered.detach(), sigma_p, sigma_q)
    lengths = [norm_without_underflow(row) for row in gradient]
    # The supported mode at +45 degrees has an empty next mode (+90 degrees)
    # and an occupied previous mode (0 degrees). The p mixture is exactly
    # reflection symmetric in these two directions. q's occupied-neighbor
    # imbalance provides the only missing-specific tangent at its center.
    adjacent = float((p_means[2]-p_means[1]).norm())
    tangent_component = mode_hold.RADIUS*math.sin(math.pi/4)
    # For q occupied at modes 0,1,7, mode 0 is the nearest competitor to 1.
    # log10 of its leading tangential contribution is computable even when
    # exp(-d^2/(2 sigma_q^2)) is below float64's dynamic range.
    log10_tangent_leading = (math.log10(tangent_component/(sigma_q*sigma_q))
                             - adjacent*adjacent/(2*sigma_q*sigma_q*math.log(10)))
    hq_radius = 3*mode_hold.SIGMA
    log10_q_competitor_at_hq = (-(adjacent*adjacent-2*adjacent*hq_radius)
                                /(2*sigma_q*sigma_q*math.log(10)))
    # At the exact midpoint, q has two equally close occupied centers in
    # the occupied direction but only one in the empty direction, producing
    # about log(2) of nonlocal score contrast. This is already >5 HQ radii.
    midpoint = adjacent/2
    center = p_means[1]
    toward_empty = (p_means[2]-center)/adjacent
    toward_occupied = (p_means[0]-center)/adjacent
    points = torch.stack((center+midpoint*toward_empty,
                          center+midpoint*toward_occupied))
    midpoint_scores = ratio_score(points, p_means, centered.detach(), sigma_p, sigma_q)
    midpoint_delta = float(midpoint_scores[0]-midpoint_scores[1])
    center_scores = ratio_score(p_means[[1, 2]], p_means, centered.detach(),
                                sigma_p, sigma_q)
    # One real-data sigma along the mode-1 to empty-mode-2 chord stays inside
    # the HQ ball. Compare two *different* coordinate fields at the same
    # query: D frozen at the centered cloud versus D reoptimized for the
    # shifted q, followed by the ordinary partial G score ascent. Moving all
    # four duplicates together leaves their local q score zero at the new
    # center, so p pulls them back toward their original occupied mode.
    displacement = mode_hold.SIGMA
    shifted_query = (center+displacement*toward_empty)[None]
    shifted_support = centered.clone()
    members = torch.cdist(centered, p_means).argmin(1) == 1
    shifted_support[members] += displacement*toward_empty
    fixed_d_field = float(ratio_gradient(shifted_query, p_means, centered.detach(),
                                         sigma_p, sigma_q)[0]@toward_empty)
    retracked_d_partial_field = float(ratio_gradient(
        shifted_query, p_means, shifted_support.detach(),
        sigma_p, sigma_q)[0]@toward_empty)
    if not fixed_d_field > 0 or not retracked_d_partial_field < 0:
        raise AssertionError('one-sigma fixed-D and retracked-D field signs changed')
    return dict(max_clean_center_gradient_norm=max(lengths),
                leading_log10_missing_specific_tangent_at_mode1=log10_tangent_leading,
                leading_log10_q_neighbor_relative_mass_at_hq_radius=log10_q_competitor_at_hq,
                adjacent_mode_center_distance=adjacent, hq_radius=hq_radius,
                midpoint_distance=midpoint, midpoint_is_hq_radii=midpoint/hq_radius,
                log_ratio_empty_vs_occupied_at_midpoint=midpoint_delta,
                log_ratio_at_occupied_mode1=float(center_scores[0]),
                log_ratio_at_missing_mode2=float(center_scores[1]),
                one_real_sigma_displacement=displacement,
                one_sigma_fixed_d_score_ascent_toward_empty=fixed_d_field,
                one_sigma_retracked_d_partial_score_ascent_toward_empty=
                    retracked_d_partial_field,
                local_isolated_component_curvature=1/(sigma_q*sigma_q)-1/(sigma_p*sigma_p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((args.archive/'manifest.json').read_text())
    compressed_path = args.archive/'prefix-states.pt.gz'
    compressed = compressed_path.read_bytes()
    if sha(compressed) != manifest['files']['prefix-states.pt.gz']['sha256']:
        raise RuntimeError('archived first-100 states changed')
    raw = gzip.decompress(compressed)
    if sha(raw) != manifest['decompressed_state_sha256']:
        raise RuntimeError('decompressed first-100 states changed')
    states = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')
    saved = states['after_checkpoint100']
    if saved['noise']['step_calls'] != 100:
        raise RuntimeError('wrong native noise clock')
    generator = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN,
                                   mode_hold.N_HIDDEN, 2)
    generator.load_state_dict({name.removeprefix('model.'): value
                               for name, value in saved['generator'].items()})
    support = generator(saved['prior']['z']).detach().double()
    means = exact_ring_centers()
    nearest = torch.cdist(support, means).argmin(1)
    counts = torch.bincount(nearest, minlength=mode_hold.N_MODES).tolist()
    if counts != [4, 4, 0, 0, 0, 0, 0, 4]:
        raise RuntimeError(f'captured cloud is not the declared three-mode subset: {counts}')
    missing = torch.tensor([i for i, count in enumerate(counts) if not count])
    centered = means[nearest]
    input_sigma = saved['noise']['input_sigma']
    output_sigma = saved['noise']['output_sigma']
    laws = {
        'captured_step100': (math.hypot(mode_hold.SIGMA, input_sigma),
                             math.hypot(output_sigma, input_sigma)),
        'late_noise_law_on_same_frozen_step100_support': (mode_hold.SIGMA, .029),
    }
    analyzed = {}
    for name, (sigma_p, sigma_q) in laws.items():
        analyzed[name] = dict(effective_real_sigma=sigma_p,
                              effective_fake_sigma=sigma_q,
                              actual_clean_cloud=analyze_cloud(support, means, missing,
                                                               sigma_p, sigma_q),
                              centered_cloud=centered_field(centered, means,
                                                            sigma_p, sigma_q))
    result = dict(scope='read-only unrestricted unregularized population Rp ratio; '
                             'clean-point partial field with q frozen',
                  captured_update=100, state_archive_sha256=sha(compressed),
                  state_raw_sha256=sha(raw),
                  noise=dict(input_sigma=input_sigma, output_sigma=output_sigma,
                             configured_late_output_sigma=.029),
                  actual_support=support.tolist(), nearest_mode=nearest.tolist(),
                  nearest_counts=counts, centered_support=centered.tolist(),
                  laws=analyzed,
                  limitations=['not the finite b_cap-penalized critic optimum',
                               'not the noisy-pair expected G update or shared-network Adam step',
                               'mode centers are used only after training to construct and grade this copy',
                               'late noise law is applied to the same step-100 cloud, not a late trained state',
                               'no claim that the wrong-subset cloud is a stable host equilibrium'],
                  source={name: sha((ROOT/name).read_bytes()) for name in (
                      'reports/toy100/ideal_ratio_three_mode.py',
                      'benchmarks/locked_shared/mode_hold.py',
                      'benchmarks/locked_shared/mlp.py')},
                  runtime=dict(torch=torch.__version__, threads=torch.get_num_threads(),
                               cpu_capability=torch.backends.cpu.get_cpu_capability()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
