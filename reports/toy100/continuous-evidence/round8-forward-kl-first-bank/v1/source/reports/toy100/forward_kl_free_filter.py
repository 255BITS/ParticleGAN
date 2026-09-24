"""Pure free-output forward-KL filter with one frozen data-local bandwidth.

The target is the empirical native real bank convolved with N(0,h²I).
The generated density is an equal-weight Gaussian mixture with covariance
(actual_output_sigma²+h²)I at the clean particle locations. A fixed 5x5
Gauss-Hermite rule integrates the target side. Global observed-real donor
replacements and equal-weight/fixed-covariance EM steps strictly decrease
that *same* quadrature cross-entropy. A 9x9 rule only audits accepted steps.

This is a data/output diagnostic: no GAN, neural parameter, Adam state,
training noise stream, target mode label, or learned LR is touched.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from benchmarks.locked_shared import mode_hold
from reports.toy100.sample_anchor_free1200 import initial_support, load_states
from reports.toy100.sample_anchor_local_mmd_filter import local_width
from reports.toy100.sample_anchor_mmd_filter import quality


DONOR_LIMIT = 12
EM_LIMIT = 20
GH_UPDATE = 5
GH_AUDIT = 9
DATA_BATCH = 128
SOURCE_NAMES = (
    'reports/toy100/forward_kl_free_filter.py',
    'reports/toy100/sample_anchor_free1200.py',
    'reports/toy100/sample_anchor_local_mmd_filter.py',
    'reports/toy100/sample_anchor_mmd_filter.py',
    'reports/toy100/coverage_fixed_eval.py',
    'reports/toy100/pr84_early_geometry.py',
    'benchmarks/locked_shared/mode_hold.py',
    'benchmarks/locked_shared/mlp.py',
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def quadrature(real, width, order):
    """Frozen target samples and positive weights for E_{x+hZ}."""
    x = real.detach().double()
    if x.ndim != 2 or x.shape[1] != 2 or not len(x) or not bool(torch.isfinite(x).all()):
        raise ValueError('finite nonempty R2 target real bank required')
    if not math.isfinite(width) or width <= 0 or order not in (GH_UPDATE, GH_AUDIT):
        raise ValueError('invalid frozen width or declared GH order')
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    mesh = np.stack(np.meshgrid(nodes, nodes, indexing='ij'), axis=-1).reshape(-1, 2)
    noise = torch.as_tensor(math.sqrt(2)*width*mesh, dtype=torch.float64)
    w = torch.as_tensor((weights[:,None]*weights[None,:]).reshape(-1)/math.pi,
                        dtype=torch.float64)
    if (not bool((w>0).all()) or abs(float(w.sum())-1) > 1e-14
            or float((w[:,None]*noise).sum(0).abs().max()) > 1e-14
            or float(((w[:,None]*noise.square()).sum(0)-width**2).abs().max()) > 1e-14):
        raise AssertionError('Gauss-Hermite target rule is not normalized N(0,h²I)')
    locations = (x[:,None,:]+noise[None,:,:]).reshape(-1,2)
    all_weights = w.repeat(len(x))/len(x)
    if abs(float(all_weights.sum())-1) > 1e-13:
        raise AssertionError('target quadrature weights do not sum to one')
    return locations, all_weights


def log_kernels(locations, points, variance):
    if variance <= 0 or not math.isfinite(variance):
        raise ValueError('shared generated covariance must be positive')
    return -(locations[:,None,:]-points[None,:,:]).square().sum(-1)/(2*variance)


def cross_entropy(locations, weights, points, variance):
    points = points.double()
    if not len(points) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('nonempty equal-weight R2 particles required')
    logq = torch.logsumexp(log_kernels(locations, points, variance), dim=1)
    return torch.log(torch.tensor(float(len(points)), dtype=torch.float64)) + \
        math.log(2*math.pi*variance) - weights@logq


def global_donor(real, points, locations, weights, variance, *, limit, audit):
    """Choose the globally best strict cross-entropy donor→observed-real move."""
    x = real.double()
    current = points.double().clone()
    rows = []
    audit_locations, audit_weights = audit
    # Candidate real coordinates stay fixed within this native bank/history.
    candidate_logk = log_kernels(locations, x, variance).T.contiguous()
    for move in range(limit):
        old = float(cross_entropy(locations, weights, current, variance))
        old9 = float(cross_entropy(audit_locations, audit_weights, current, variance))
        matrix = log_kernels(locations, current, variance)
        constant = math.log(len(current)) + math.log(2*math.pi*variance)
        best = (old, None, None)
        for donor in range(len(current)):
            keep = torch.cat((matrix[:,:donor], matrix[:,donor+1:]), dim=1)
            logremain = torch.logsumexp(keep, dim=1)
            replacement_logq = torch.logaddexp(candidate_logk, logremain[None,:])
            values = constant - replacement_logq@weights
            index = int(values.argmin())
            candidate = float(values[index])
            if candidate < best[0]:
                best = (candidate, index, donor)
        predicted, real_index, donor_index = best
        tolerance = 64*torch.finfo(torch.float64).eps*max(1., abs(old))
        if real_index is None or predicted >= old-tolerance:
            break
        proposal = current.clone()
        proposal[donor_index] = x[real_index]
        exact = float(cross_entropy(locations, weights, proposal, variance))
        exact9 = float(cross_entropy(audit_locations, audit_weights, proposal, variance))
        if (not math.isfinite(exact) or abs(predicted-exact) > 1e-10
                or exact >= old-tolerance):
            raise RuntimeError('stable global donor calculation disagrees with exact quadrature')
        rows.append(dict(move=move+1, real_index=real_index, donor_index=donor_index,
                         before=old, after=exact, predicted=predicted,
                         audit9_before=old9, audit9_after=exact9,
                         audit9_delta=exact9-old9,
                         audit9_sign_flip=exact9>old9+1e-12,
                         max_output_displacement=float((proposal-points).norm(dim=1).max())))
        current = proposal
    return current, rows


def em_centroids(points, locations, weights, variance, *, limit, audit):
    """Fixed-weight/covariance EM center M-step; no weight or bandwidth update."""
    current = points.double().clone()
    rows = []
    audit_locations, audit_weights = audit
    for step in range(limit):
        old = float(cross_entropy(locations, weights, current, variance))
        old9 = float(cross_entropy(audit_locations, audit_weights, current, variance))
        logkernels = log_kernels(locations, current, variance)
        logresponsibilities = logkernels-torch.logsumexp(logkernels, dim=1, keepdim=True)
        logweighted = weights.log()[:,None]+logresponsibilities
        # Normalize each atom over fixed target rows in log space: an atom
        # with extremely small responsibility still has a valid EM centroid.
        normalized = torch.softmax(logweighted, dim=0)
        proposal = normalized.T@locations
        logmass = torch.logsumexp(logweighted, dim=0)
        exact = float(cross_entropy(locations, weights, proposal, variance))
        exact9 = float(cross_entropy(audit_locations, audit_weights, proposal, variance))
        tolerance = 64*torch.finfo(torch.float64).eps*max(1., abs(old))
        if not math.isfinite(exact) or exact > old+tolerance:
            raise RuntimeError('equal-weight fixed-covariance EM raised its declared objective')
        if exact >= old-tolerance:
            break
        rows.append(dict(step=step+1, before=old, after=exact,
                         audit9_before=old9, audit9_after=exact9,
                         audit9_delta=exact9-old9,
                         audit9_sign_flip=exact9>old9+1e-12,
                         minimum_log_component_mass=float(logmass.min()),
                         max_output_displacement=float((proposal-points).norm(dim=1).max())))
        current = proposal
    return current, rows


def gradient(locations, weights, points, variance):
    value = points.detach().double().clone().requires_grad_(True)
    loss = cross_entropy(locations, weights, value, variance)
    return torch.autograd.grad(loss, value)[0].detach()


def optimize(real, initial, width, sigma, *, means):
    x = real.detach().double()
    points = initial.detach().double().clone()
    if len(points) != DONOR_LIMIT:
        raise ValueError('this bounded diagnostic requires the native twelve particles')
    variance = width**2+sigma**2
    update = quadrature(x, width, GH_UPDATE)
    audit = quadrature(x, width, GH_AUDIT)
    before = float(cross_entropy(*update, points, variance))
    before9 = float(cross_entropy(*audit, points, variance))
    donated, donors = global_donor(x, points, *update, variance,
                                   limit=DONOR_LIMIT, audit=audit)
    after_donor = float(cross_entropy(*update, donated, variance))
    final, em_rows = em_centroids(donated, *update, variance,
                                  limit=EM_LIMIT, audit=audit)
    after = float(cross_entropy(*update, final, variance))
    after9 = float(cross_entropy(*audit, final, variance))
    grad5 = gradient(*update, final, variance)
    grad9 = gradient(*audit, final, variance)
    if after > before+1e-11:
        raise RuntimeError('whole pure output update raised cross-entropy')
    return final, dict(initial_cross_entropy=before,
        donor_cross_entropy=after_donor, final_cross_entropy=after,
        initial_audit9=before9, final_audit9=after9,
        accepted_donors=donors, accepted_em=em_rows,
        audit9_sign_flips=sum(row['audit9_sign_flip'] for row in donors+em_rows),
        audit9_max_objective_discrepancy=max([abs(before-before9), abs(after-after9)] +
            [abs(row['after']-row['audit9_after']) for row in donors+em_rows]),
        final_gradient_5_l2=float(grad5.norm()),
        final_gradient_9_l2=float(grad9.norm()),
        final_gradient_max_absolute_discrepancy=float((grad5-grad9).abs().max()),
        initial_quality=quality(points, means), final_quality=quality(final, means),
        final_points=final.tolist(),
        max_output_displacement=float((final-points).norm(dim=1).max()),
        actual_sigma=sigma, frozen_width=width,
        quadrature_update=GH_UPDATE, quadrature_audit=GH_AUDIT)


def first_native_bank(cold_state):
    stream = torch.Generator().set_state(cold_state['rng']['data'])
    bank = mode_hold.sample_ring(mode_hold.ring_means(), DATA_BATCH, mode_hold.SIGMA, stream)
    return bank, sha(stream.get_state().numpy().tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-root', type=Path, required=True,
                        help='source root for the already published free-output fixture')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cold, warm, input_hashes = load_states()
    bank, bank_rng = first_native_bank(cold)
    width, nearest = local_width(bank)
    source_paths = {name: (ROOT/name if (ROOT/name).exists() else args.code_root/name)
                    for name in SOURCE_NAMES}
    source = {name: sha(path.read_bytes()) for name,path in source_paths.items()}
    args.output.mkdir(parents=True)
    for name in SOURCE_NAMES:
        target = args.output/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_paths[name].read_bytes())
    declaration = dict(scope='one native D real bank, two saved clean output clouds; no training',
        source=source, input_hashes=input_hashes,
        native_real_bank_sha256=sha(bank.contiguous().numpy().tobytes()),
        data_rng_after_bank_sha256=bank_rng,
        target='empirical real bank convolved with N(0,h²I)',
        generated='equal-weight N Gaussian atoms with covariance (actual output sigma²+h²)I',
        width_rule='median strictly positive nearest-neighbor distance of first cold native D bank',
        frozen_width=width, fixed_quadrature='5x5 GH target nodes/positive weights',
        audit_quadrature='independent 9x9 GH at every accepted state; never selects an update',
        donor='global best real-coordinate replacement, at most12 strict cross-entropy decreases',
        em='at most20 fixed-equal-weight/covariance EM center M-steps, actual quadrature nonincrease',
        actual_output_sigma=dict(cold1=float(cold['noise_policy']['output_sigma']),
                                 warm1324=float(warm['noise']['output_sigma'])),
        cheap_gate='both strict 5x5 descent; no 9x9 accepted-step sign flip; warm final8/HQ>=.9; cold final at least one mode and HQ>initial',
        no_oracle_width_or_update=True, no_seed_or_budget_sweep=True,
        shared_gate_eligible=False)
    (args.output/'declaration.json').write_text(json.dumps(declaration, indent=2)+'\n')
    print(json.dumps(dict(event='FORWARD_KL_DECLARED', h=width,
                          source_sha256=source['reports/toy100/forward_kl_free_filter.py'])), flush=True)
    result = dict(status='INCOMPLETE', declaration=declaration,
                  common_bank=bank.tolist(), nearest_neighbor_distances=nearest.tolist(),
                  cases={}, global_training_rng_unchanged=None)
    before_rng = torch.random.get_rng_state().clone()
    try:
        for name,state in (('cold1',cold),('warm1324',warm)):
            final, row = optimize(bank, initial_support(state), width,
                                  declaration['actual_output_sigma'][name],
                                  means=mode_hold.ring_means())
            result['cases'][name] = row
            row['strict_descent'] = row['final_cross_entropy'] < row['initial_cross_entropy'] - 1e-12
            row['cheap_gate'] = (row['strict_descent'] and row['audit9_sign_flips']==0 and
                (row['final_quality']['modes']==8 and row['final_quality']['hq']>=.9
                 if name=='warm1324' else
                 row['final_quality']['modes']>=1 and
                 row['final_quality']['hq']>row['initial_quality']['hq']))
            print(json.dumps(dict(event='FORWARD_KL_CASE_DONE', case=name,
                initial=(row['initial_quality']['modes'],row['initial_quality']['hq']),
                final=(row['final_quality']['modes'],row['final_quality']['hq']),
                donors=len(row['accepted_donors']), em=len(row['accepted_em']),
                objective=(row['initial_cross_entropy'],row['final_cross_entropy']),
                audit9_sign_flips=row['audit9_sign_flips'], cheap_gate=row['cheap_gate'])), flush=True)
        result['status']='COMPLETE'
        result['all_cheap_gates_pass']=all(v['cheap_gate'] for v in result['cases'].values())
        result['global_training_rng_unchanged']=torch.equal(torch.random.get_rng_state(), before_rng)
        if not result['global_training_rng_unchanged']:
            raise RuntimeError('pure quadrature filter changed global training RNG')
        (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(event='FORWARD_KL_DONE', all_cheap_gates_pass=result['all_cheap_gates_pass'])), flush=True)
    except BaseException as error:
        (args.output/'error.json').write_text(json.dumps(dict(status='ERROR_INCOMPLETE',
            error=repr(error), completed_cases=list(result['cases'])))+'\n')
        raise


if __name__ == '__main__':
    main()
