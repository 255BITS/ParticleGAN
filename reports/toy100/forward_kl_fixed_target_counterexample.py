"""One source-bound fixed-target limitation of finite-GH9 EM plus current-bank donors.

This deterministic synthetic example is not sampled from the native ring host.
It disproves a universal acquisition claim; it does not estimate failure odds.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


EXPECTED_METHOD_SHA256 = 'adc5d6f6ded8e6cf297ee90aa0d540723becc632a601b93b5b9a4393cd344485'
WIDTH = 0.031286240422040236
SIGMA = 0.029


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    method_source = args.method_root/'reports/toy100/forward_kl_gh9_stress.py'
    method_sha = digest(method_source.read_bytes())
    if method_sha != EXPECTED_METHOD_SHA256:
        raise RuntimeError('counterexample requires the audited frozen GH9 method')
    # The frozen method imports its audited free-output fixture, whose native
    # state utilities live in this independent source tree.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(args.method_root))
    from reports.toy100.forward_kl_gh9_stress import optimize
    from reports.toy100.forward_kl_free_filter import quadrature, cross_entropy
    from benchmarks.locked_shared import mode_hold

    torch.set_num_threads(1)
    neg = torch.tensor([[-2., 0.]], dtype=torch.float32).repeat(128, 1)
    middle = torch.zeros((128, 2), dtype=torch.float32)
    pos = torch.tensor([[2., 0.]], dtype=torch.float32).repeat(128, 1)
    # The current 128-real donor bank is the last of these three banks.
    history = torch.cat((neg, pos, middle))
    current_bank = middle.clone()
    collapsed = torch.zeros((12, 2), dtype=torch.float64)
    allocated = torch.cat((neg[:4], middle[:4], pos[:4])).double()
    before_rng = torch.random.get_rng_state().clone()
    selected, row = optimize(history, current_bank, collapsed, WIDTH, SIGMA,
                             means=mode_hold.ring_means())
    after_rng = torch.random.get_rng_state()
    locations, weights = quadrature(history, WIDTH, 9)
    variance = WIDTH**2+SIGMA**2
    bad = float(cross_entropy(locations, weights, collapsed, variance))
    good = float(cross_entropy(locations, weights, allocated, variance))
    if (row['selected'] != 'EXACT_REST' or row['proposal_donors']
            or row['proposal_em'] or row['fallback_em']
            or not torch.equal(selected, collapsed)
            or not torch.equal(before_rng, after_rng)
            or abs(bad-row['initial_audit9']) > 1e-10
            or good >= bad):
        raise RuntimeError('frozen finite-GH9 counterexample did not reproduce')
    receipt = dict(status='COMPLETE', scope='synthetic fixed finite target, no native training',
        method_source_sha256=method_sha,
        script_sha256=digest(Path(__file__).read_bytes()),
        target_banks=[{'x': -2., 'count': 128}, {'x': 2., 'count': 128},
                      {'x': 0., 'count': 128, 'current_donor_bank': True}],
        equal_weight_particles=12, initial_particle_x=0., width=WIDTH, sigma=SIGMA,
        selected=row['selected'], gh9_before=bad, gh9_after=row['final_audit9'],
        gh9_explicit_4_4_4_allocation=good,
        gh9_gradient_norm=row['final_gradient_9_l2'],
        proposal_donors=len(row['proposal_donors']),
        proposal_em=len(row['proposal_em']),
        fallback_em=len(row['fallback_em']),
        exact_output_rest=torch.equal(selected, collapsed),
        torch_rng_unchanged=torch.equal(before_rng, after_rng),
        interpretation='symmetry and current-bank donor support block this local method; '
            'a different candidate bank or nonsymmetric perturbation can escape; '
            'this gives no native ring failure probability')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
