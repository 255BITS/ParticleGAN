"""Bounded post-rest remembered-donor diagnostic on one archived synthetic state.

No native learner source is changed. The search occurs only after the frozen
current-bank/GH9 method selects exact rest. It scans full remembered real
coordinates in deterministic candidate-major order, chunking candidates and
quadrature rows, and checks an actual finite-GH9 decrease before one GH9 EM.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


METHOD_SHA256 = 'adc5d6f6ded8e6cf297ee90aa0d540723becc632a601b93b5b9a4393cd344485'
DONOR_SHA256 = 'e0977838c7d5ae787f5194e653d1a8a09d5c7c0af7eb5f48b65af49a303b141f'
WIDTH = 0.031286240422040236
SIGMA = 0.029
CANDIDATE_CHUNK = 32
QUADRATURE_CHUNK = 2048


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def remembered_first_descent(history, points, locations, weights, variance,
                             *, donor_values, cross_entropy):
    before = float(cross_entropy(locations, weights, points, variance))
    tolerance = 64*torch.finfo(torch.float64).eps*max(1., abs(before))
    max_matrix = 0
    scanned = 0
    evaluated = 0
    for start in range(0, len(history), CANDIDATE_CHUNK):
        candidates = history[start:start+CANDIDATE_CHUNK]
        scores, accounting = donor_values(candidates, points, locations, weights,
            variance, chunk_rows=QUADRATURE_CHUNK)
        evaluated += len(candidates)
        max_matrix = max(max_matrix, accounting['maximum_candidate_matrix_elements'])
        for local_candidate in range(len(candidates)):
            scanned += 1
            for donor in range(len(points)):
                predicted = float(scores[donor, local_candidate])
                if predicted >= before-tolerance:
                    continue
                candidate = points.clone()
                candidate[donor] = candidates[local_candidate].double()
                actual = float(cross_entropy(locations, weights, candidate, variance))
                if actual < before-tolerance:
                    return candidate, dict(status='STRICT_FINITE_GH9_DONOR',
                        remembered_real_index=start+local_candidate, donor_index=donor,
                        inspected_candidates=scanned, evaluated_candidates=evaluated,
                        predicted_cost=predicted,
                        actual_cost=actual, before_cost=before,
                        max_candidate_matrix_elements=max_matrix,
                        candidate_chunk=CANDIDATE_CHUNK,
                        quadrature_chunk=QUADRATURE_CHUNK)
    return points.clone(), dict(status='NO_SINGLE_REMEMBERED_DONOR',
        inspected_candidates=scanned, evaluated_candidates=evaluated,
        before_cost=before,
        max_candidate_matrix_elements=max_matrix,
        candidate_chunk=CANDIDATE_CHUNK, quadrature_chunk=QUADRATURE_CHUNK)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {name: args.method_root/'reports/toy100'/name for name in
        ('forward_kl_gh9_stress.py', 'forward_kl_chunked.py')}
    hashes = {name: sha(path.read_bytes()) for name,path in paths.items()}
    if hashes != {'forward_kl_gh9_stress.py': METHOD_SHA256,
                  'forward_kl_chunked.py': DONOR_SHA256}:
        raise RuntimeError('frozen method or chunked donor source differs')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(args.method_root))
    from benchmarks.locked_shared import mode_hold
    from reports.toy100.forward_kl_gh9_stress import optimize
    from reports.toy100.forward_kl_chunked import donor_values
    from reports.toy100.forward_kl_free_filter import (
        quadrature, cross_entropy, em_centroids)

    torch.set_num_threads(1)
    neg = torch.tensor([[-2., 0.]], dtype=torch.float32).repeat(128, 1)
    middle = torch.zeros((128, 2), dtype=torch.float32)
    pos = torch.tensor([[2., 0.]], dtype=torch.float32).repeat(128, 1)
    history = torch.cat((neg, pos, middle))
    collapsed = torch.zeros((12, 2), dtype=torch.float64)
    before_rng = torch.random.get_rng_state().clone()
    old, old_row = optimize(history, middle, collapsed, WIDTH, SIGMA,
                            means=mode_hold.ring_means())
    if old_row['selected'] != 'EXACT_REST' or not torch.equal(old, collapsed):
        raise RuntimeError('frozen current-bank method no longer exactly rests')
    locations, weights = quadrature(history, WIDTH, 9)
    variance = WIDTH**2+SIGMA**2
    donor, donor_row = remembered_first_descent(history, collapsed, locations,
        weights, variance, donor_values=donor_values, cross_entropy=cross_entropy)
    if donor_row['status'] != 'STRICT_FINITE_GH9_DONOR':
        raise RuntimeError('remembered candidate failed to reveal strict descent')
    em, em_rows = em_centroids(donor, locations, weights, variance, limit=1,
                                audit=(locations, weights))
    final = em if em_rows else donor
    final_cost = float(cross_entropy(locations, weights, final, variance))
    if (final_cost > donor_row['actual_cost']+1e-12
            or final_cost >= old_row['initial_audit9']-old_row['gh9_strict_tolerance']
            or not torch.equal(before_rng, torch.random.get_rng_state())):
        raise RuntimeError('remembered donor/EM failed exact finite objective or RNG guard')
    receipt = dict(status='COMPLETE', scope='one synthetic fixed-target post-rest rescue; '
            'not a native learner update or global convergence proof',
        source_sha256=hashes, script_sha256=sha(Path(__file__).read_bytes()),
        current_bank_selection=old_row['selected'],
        current_bank_gh9_cost=old_row['initial_audit9'],
        remembered_donor=donor_row,
        gh9_em_steps=len(em_rows),
        gh9_after_donor=donor_row['actual_cost'],
        gh9_after_one_em=final_cost,
        output_after_one_em=final.tolist(),
        target_modes_with_atom_within_quarter=sum(
            float((final[:,0]-center).abs().min()) < .25
            for center in (-2., 0., 2.)),
        torch_rng_unchanged=torch.equal(before_rng, torch.random.get_rng_state()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, allow_nan=False))


if __name__ == '__main__':
    main()
