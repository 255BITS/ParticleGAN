"""Offline nonlocal support-coverage field from a saved native real minibatch.

This proposes an explicit *additional objective*, not an optimizer-only repair
or a production training adapter. It uses no target centers or configured mode
count. A minimum-spanning-tree edge gap groups real samples; one distinct
generated anchor is assigned to each group, with extra generated particles
free to sit near any group. The source tests only support-space geometry.
"""

import argparse
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy
import torch

from benchmarks.locked_shared import mode_hold
from benchmarks.locked_shared.mlp import SimpleMLPGenerator


ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def mst_groups(real):
    """Prim MST, cut at its unique largest additive edge-length gap."""
    x = real.detach().double().cpu().numpy()
    n = len(x)
    if n < 4:
        raise ValueError('MST gap diagnosis needs at least four real samples')
    distances = np.linalg.norm(x[:, None, :]-x[None, :, :], axis=2)
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    best = distances[0].copy()
    parent = np.zeros(n, dtype=np.int64)
    best[0] = np.inf
    edges = []
    for _ in range(n-1):
        j = min((float(best[k]), k) for k in range(n) if not selected[k])[1]
        edges.append((int(parent[j]), j, float(best[j])))
        selected[j] = True
        for k in range(n):
            if not selected[k] and (distances[j, k] < best[k] or
                                    (distances[j, k] == best[k] and j < parent[k])):
                best[k] = distances[j, k]
                parent[k] = j
    lengths = np.sort(np.array([edge[2] for edge in edges]))
    gaps = np.diff(lengths)
    cut = int(np.argmax(gaps))
    threshold = float((lengths[cut]+lengths[cut+1])/2)
    roots = list(range(n))

    def find(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    for a, b, length in edges:
        if length <= threshold:
            roots[find(b)] = find(a)
    groups = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    members = sorted(groups.values(), key=lambda row: min(row))
    centers = torch.stack([real[row].double().mean(dim=0) for row in members])
    return centers, dict(n_groups=len(members), member_indices=members,
                         member_sizes=[len(row) for row in members],
                         largest_within_edge=float(lengths[cut]),
                         smallest_between_edge=float(lengths[cut+1]),
                         cut_threshold=threshold,
                         largest_additive_gap=float(gaps[cut]),
                         second_largest_gap=float(np.partition(gaps, -2)[-2]))


def field(support, centers):
    """Piecewise gradient of a one-anchor-per-group plus precision objective."""
    if len(centers) > len(support):
        raise ValueError('more observed groups than generated particles')
    y = support.detach().double().clone().requires_grad_(True)
    c = centers.detach().double()
    distances = torch.cdist(c, y).square()
    groups, particles = linear_sum_assignment(distances.detach().cpu().numpy())
    if list(groups) != list(range(len(c))):
        raise AssertionError('not every real group received a distinct anchor')
    nearest = torch.cdist(y, c).square().argmin(1)
    coverage = (y[particles]-c[groups]).square().sum(-1).mean()
    precision = (y-c[nearest]).square().sum(-1).mean()
    total = coverage+precision
    gradient = torch.autograd.grad(total, y)[0]
    return dict(total=float(total.detach()), coverage=float(coverage.detach()),
                precision=float(precision.detach()),
                assignments=[dict(group=int(k), particle=int(j),
                                  distance=float(distances[k, j].detach().sqrt()))
                             for k, j in zip(groups, particles)],
                nearest_group=nearest.tolist(),
                gradient=gradient.detach().tolist(),
                negative_gradient=(-gradient).detach().tolist(),
                gradient_l2=float(gradient.square().sum().sqrt().detach()))


def group_stability(full, half):
    cost = torch.cdist(full, half).square().cpu().numpy()
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != len(full) or len(columns) != len(half):
        raise AssertionError('half-bank groups do not cover the full-bank groups')
    distances = [float((full[i]-half[j]).norm()) for i, j in zip(rows, columns)]
    return dict(mean_center_displacement=sum(distances)/len(distances),
                max_center_displacement=max(distances), matching=columns.tolist())


def output_mm_step(support, centers):
    """Minimize one active quadratic piece in output space, then rematch."""
    before = field(support, centers)
    n, k = len(support), len(centers)
    nearest = torch.as_tensor(before['nearest_group'])
    target = centers[nearest].double().clone()
    for assignment in before['assignments']:
        group, particle = assignment['group'], assignment['particle']
        target[particle] = (n * centers[group] + k * centers[nearest[particle]]) / (n+k)
    after = field(target, centers)
    if after['total'] > before['total'] + 1e-10:
        raise AssertionError('exact output-space quadratic minimization increased loss')
    return dict(before=before['total'], after=after['total'],
                target=target.tolist(), target_nearest_groups=after['nearest_group'],
                target_occupied_groups=sorted(set(after['nearest_group'])),
                max_output_displacement=float((target-support).norm(dim=1).max()),
                after_gradient_l2=after['gradient_l2'])


def controlled_fields(support, centers):
    actual = field(support, centers)
    nearest = torch.cdist(support, centers).argmin(1)
    occupied = sorted(set(nearest.tolist()))
    wrong_centered = centers[nearest]
    wrong = field(wrong_centered, centers)
    missing_groups = sorted(set(range(len(centers)))-set(occupied))
    wrong_donors = [row for row in wrong['assignments'] if row['group'] in missing_groups]
    if len(wrong_donors) != len(missing_groups):
        raise AssertionError('missing groups did not receive distinct nonlocal donors')
    # Good support has one generated particle per observed real group plus
    # N−K duplicates. This uses sample centroids only, not known ring means.
    extras = centers[torch.arange(len(support)-len(centers)) % len(centers)]
    good_support = torch.cat((centers, extras), dim=0)
    good = field(good_support, centers)
    if good['total'] != 0 or good['gradient_l2'] != 0:
        raise AssertionError('good all-group support does not rest exactly')
    # One real-data sigma perturbation of an extra good particle tests the
    # precision restoring term with coverage already satisfied by anchors.
    perturbed_support = good_support.clone()
    extra_index = len(centers)
    order = torch.cdist(centers[:1], centers)[0].argsort()
    toward_other = centers[int(order[1])]-centers[0]
    direction = toward_other/toward_other.norm()
    displacement = mode_hold.SIGMA
    perturbed_support[extra_index] += displacement*direction
    perturbed = field(perturbed_support, centers)
    restoring_component = -float(torch.tensor(
        perturbed['gradient'][extra_index], dtype=direction.dtype)@direction)
    if not restoring_component < 0:
        raise AssertionError('one-sigma good-support perturbation is not restored')
    return dict(actual=actual, actual_occupied_groups=occupied,
                actual_missing_groups=missing_groups,
                actual_output_mm=output_mm_step(support, centers),
                wrong_centered=wrong, wrong_centered_support=wrong_centered.tolist(),
                wrong_nonlocal_donors=wrong_donors,
                wrong_output_mm=output_mm_step(wrong_centered, centers),
                good=good, good_support=good_support.tolist(),
                good_perturbed=dict(field=perturbed,
                                    perturbed_particle=extra_index,
                                    displacement=displacement,
                                    restoring_component=restoring_component))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((args.archive/'manifest.json').read_text())
    compressed = (args.archive/'prefix-states.pt.gz').read_bytes()
    if sha(compressed) != manifest['files']['prefix-states.pt.gz']['sha256']:
        raise RuntimeError('saved cold-prefix archive changed')
    raw = gzip.decompress(compressed)
    if sha(raw) != manifest['decompressed_state_sha256']:
        raise RuntimeError('saved cold-prefix raw state changed')
    saved = torch.load(BytesIO(raw), weights_only=True, map_location='cpu')['after_checkpoint100']
    if saved['noise']['step_calls'] != 100:
        raise RuntimeError('not the declared update-100 state')
    with torch.random.fork_rng(devices=[]):
        generator = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN,
                                       mode_hold.N_HIDDEN, 2)
        generator.load_state_dict({name.removeprefix('model.'): value
                                   for name, value in saved['generator'].items()})
        support = generator(saved['prior']['z']).detach().double()
    data = torch.Generator()
    data.set_state(saved['rng']['data'])
    real = mode_hold.sample_ring(mode_hold.ring_means(), mode_hold.BATCH,
                                 mode_hold.SIGMA, data).detach()
    centers, full = mst_groups(real)
    half_results = []
    for half in (real[:len(real)//2], real[len(real)//2:]):
        half_centers, half_receipt = mst_groups(half)
        if len(half_centers) != len(centers):
            raise RuntimeError('single-bank half did not recover the full group count')
        half_results.append(dict(grouping=half_receipt,
                                 relative_to_full=group_stability(centers, half_centers)))
    if len(centers) > len(support):
        raise RuntimeError('sampled group count exceeds particle count')
    fields = controlled_fields(support, centers)
    result = dict(scope='sample-derived MST groups and one-distinct-anchor-per-group '
                        'support-space objective only; no training',
                  state_archive_sha256=sha(compressed), state_raw_sha256=sha(raw),
                  real_bank_sha256=sha(real.contiguous().numpy().tobytes()),
                  real_batch=len(real), n_particles=len(support),
                  mst_full=full, mst_halves=half_results,
                  empirical_centers=centers.tolist(),
                  support=support.tolist(), fields=fields,
                  loss='mean squared distinct group-anchor distance + '
                       'mean squared nearest-group precision distance; coefficients 1',
                  source={name: sha((ROOT/name).read_bytes()) for name in (
                      'reports/toy100/sample_group_anchor.py',
                      'benchmarks/locked_shared/mode_hold.py',
                      'benchmarks/locked_shared/mlp.py')},
                  runtime=dict(torch=torch.__version__, scipy=scipy.__version__,
                               threads=torch.get_num_threads(),
                               cpu_capability=torch.backends.cpu.get_cpu_capability()),
                  limitations=['additional support coverage objective, not the frozen Rp game',
                               'MST gap assumes well-separated sampled groups',
                               'finite real minibatches move group centroids and induce noise',
                               'support-space field may be distorted by shared G/prior Jacobian',
                               'does not match probability masses or within-mode variance'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(status='COMPLETE', group_count=len(centers),
                          support=len(support), missing=len(fields['actual_missing_groups']),
                          wrong_donors=len(fields['wrong_nonlocal_donors']),
                          good_gradient=fields['good']['gradient_l2'],
                          good_restoring=fields['good_perturbed']['restoring_component'])))


if __name__ == '__main__':
    main()
