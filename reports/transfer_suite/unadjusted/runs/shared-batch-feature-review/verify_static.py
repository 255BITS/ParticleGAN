"""Read-only source, episode and autograd audit of the batch-feature D winner.

Run from any checkout with the winning module, or pass --checkout. No optimizer
step, benchmark episode, behavioral selection, or source edit is performed.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import torch


def main(checkout, output, stage=None):
    checkout = checkout.resolve()
    sys.path.insert(0, str(checkout))
    from benchmarks.transfer_suite.shared_batch_feature_research import ARCHITECTURES, BatchFeatureCritic
    from benchmarks.transfer_suite.shared_batch_feature_search import recipe
    from benchmarks.transfer_suite.shared_variants import architecture_spec
    from benchmarks.transfer_suite.compare_defaults import effective_spec
    from benchmarks.transfer_suite.protocol import test_verdict
    from particlegan.grad_regularizers import GradientPenalty

    torch.set_num_threads(1)
    stage = (checkout/'reports/transfer_suite/unadjusted/runs/shared-batch-feature-search/screen'
             if stage is None else stage.resolve())
    index = json.loads((stage/'index.json').read_text())
    protocol = json.loads((stage/'protocol.json').read_text())
    card = next(c for c in ARCHITECTURES if c['name'] == 'batchfeat_center6_distance_head')
    row = next(r for r in index['records'] if r['architecture'] == card['name'])
    packed = (stage/row['artifact']).read_bytes()
    raw = gzip.decompress(packed)
    episode = json.loads(raw)
    normalize = lambda value: json.loads(json.dumps(value))

    with tarfile.open(stage/'source.tar.gz', 'r:gz') as archive:
        source_hashes = {item.name: hashlib.sha256(archive.extractfile(item).read()).hexdigest()
                         for item in archive if item.isfile()}
    expected_spec = normalize(effective_spec(
        architecture_spec(episode['original_spec'], episode['discriminator_variant']), recipe()))
    applied = {item['role']: item for item in episode['applied']}
    receipt = all(applied[role]['lr'] == lr and applied[role]['betas'] == [0., .99]
                  for role, lr in (('d', .00425), ('g', .00425), ('prior', .0085)))

    torch.manual_seed(0)
    critic = BatchFeatureCritic(2, 96, 3, 0, architecture=card)
    x = torch.tensor([[-1.7, -1.4], [-1.5, -1.5], [-1.4, -1.3],
                      [1.4, 1.3], [1.5, 1.5], [1.6, 1.4]], requires_grad=True)
    scores = critic(x)
    permutation = torch.tensor([5, 2, 0, 3, 1, 4])
    equivariance_error = float((critic(x[permutation])-scores[permutation]).abs().max().detach())
    summed = torch.autograd.grad(scores.sum(), x, retain_graph=True, create_graph=True)[0]
    diagonal = []
    cross_l1 = 0.
    for i in range(len(x)):
        gradient = torch.autograd.grad(scores[i], x, retain_graph=True)[0]
        diagonal.append(gradient[i])
        cross_l1 += float(gradient[torch.arange(len(x)) != i].abs().sum())
    diagonal = torch.stack(diagonal)
    cap = GradientPenalty('b_cap', coeff=6., kappa=1.25)(critic, x.detach(), x.detach())
    parameter_gradients = torch.autograd.grad(cap, list(critic.parameters()), allow_unused=True)

    checks = dict(
        source_archive_matches_manifest=source_hashes == protocol['source_sha256'],
        checkout_numerical_source_matches_archive=all(
            hashlib.sha256((checkout/name).read_bytes()).hexdigest() == digest
            for name, digest in source_hashes.items()),
        episode_hash_matches_index=hashlib.sha256(raw).hexdigest() == row['uncompressed_sha256'],
        recipe_matches=episode['recipe'] == normalize(recipe().to_dict()),
        discriminator_only_spec_matches=episode['spec'] == expected_spec,
        verdict_recomputes=test_verdict(episode['spec'], episode['result']) == episode['verdict'],
        original_budget_and_observations=(episode['result']['update_counts'] == {'d': 1200, 'g': 1200}
                                          and len(episode['result']['observations']) == 24),
        actual_optimizer_receipts_match=receipt,
        permutation_equivariant=equivariance_error == 0.,
        finite_first_input_gradient=bool(torch.isfinite(summed).all()),
        finite_native_cap_second_order_gradient=all(
            gradient is None or bool(torch.isfinite(gradient).all())
            for gradient in parameter_gradients),
    )
    data = dict(kind='read_only_batch_feature_review', checks=checks,
                architecture=card['name'], live_status=episode['verdict']['status'],
                live_passing_suffix=episode['verdict']['convergence']['passing_suffix'],
                ema_status=episode['ema_verdict']['status'],
                d_parameters=sum(p.numel() for p in critic.parameters()),
                static_graph=dict(permutation_error=equivariance_error,
                                  cross_sample_score_gradient_l1=cross_l1,
                                  native_cap_gradient_minus_own_logit_diagonal_l1=float(
                                      (summed-diagonal).abs().sum().detach()),
                                  native_cap_value=float(cap.detach()),
                                  native_cap_parameter_gradient_l1=sum(
                                      float(gradient.abs().sum().detach())
                                      for gradient in parameter_gradients if gradient is not None)),
                complexity=dict(training_batch=episode['spec']['batch'],
                                scales=len(card['kernel_scales']),
                                kernel_elements_per_forward=(episode['spec']['batch']**2
                                                             * len(card['kernel_scales'])),
                                kernel_bytes_float32=(episode['spec']['batch']**2
                                                      * len(card['kernel_scales'])*4)))
    if output is not None:
        output.write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')
    print(json.dumps(data, indent=2, sort_keys=True))
    if not all(checks.values()) or cross_l1 <= 0 or data['static_graph']['native_cap_parameter_gradient_l1'] <= 0:
        raise SystemExit('static batch-feature audit failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', type=Path, help='Episode output folder whose archived source matches this checkout')
    args = parser.parse_args()
    main(args.checkout, args.output, args.stage)
