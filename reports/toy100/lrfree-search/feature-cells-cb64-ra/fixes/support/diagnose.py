"""CPU real-only support decomposition on the existing frozen fixtures."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent
OLD = Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929')
PREV = Path('/ml2/hypergan/gan-attempts/scaling-portability-20260929')
sys.path.insert(0, str(OLD/'geometry'))
import run_validation as common
sys.path.insert(0, str(PREV/'scaling_a'))
import shared_toy as cost
sys.path.insert(0, str(PREV/'geometry_a'))
import toy_family as geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    common.namespace('support_baseline', OLD/'pkg-CB64-RA'/'particlegan')
    common.namespace('support_reference', common.REFERENCE)
    return SimpleNamespace(cb=importlib.import_module('support_baseline.feature_cells'),
        ref=importlib.import_module('support_reference.birth_death'),
        cb_recipe=importlib.import_module('support_baseline.recipes').Recipe,
        ref_recipe=importlib.import_module('support_reference.recipes').Recipe,
        geometry=geometry, cost=cost, bundle=geometry.load_bundle(PREV/'geometry_a'/'bundle.pt'),
        config=json.loads(common.CONFIG.read_text()), ref_config=json.loads(common.REF_CONFIG.read_text()))


def conformal(snapshot, query_scores, calibration_scores):
    null = calibration_scores.sort().values
    p = (1. + len(null)-torch.searchsorted(null, query_scores)) / (1.+len(null))
    ordered = p.sort().values
    passed = (ordered <= .05*torch.arange(1, len(p)+1, dtype=p.dtype)/len(p)).nonzero().flatten()
    flags = p <= ordered[passed[-1]] if len(passed) else torch.zeros(len(p), dtype=torch.bool)
    if snapshot.duplicate_fraction > .05 or not snapshot.valid_metric:
        flags.zero_()
    return flags, p


def summary(values, mask):
    values = values[mask]
    if not len(values):
        return None
    return dict(min=float(values.min()), median=float(values.median()),
                q95=float(torch.quantile(values, .95)), max=float(values.max()))


def covariance_metric(snapshot, ref):
    """Only projected even real rows; pooled within-cell shape prior of 4 rows."""
    x = snapshot.transform(ref)
    ids, _ = snapshot._assign_metric(x)
    residual = x-snapshot.centers[ids]
    pooled = residual.T@residual/len(x)
    # The floor only makes floating point singular covariance invertible.
    floor = torch.finfo(torch.float64).eps*max(1, snapshot.rank)*pooled.trace().clamp_min(1e-30)
    ridge = torch.eye(snapshot.rank, dtype=torch.float64)*floor
    precision = []
    for cell in range(snapshot.cells):
        delta = residual[ids == cell]
        covariance = (delta.T@delta + 4.*pooled)/(len(delta)+4.)
        precision.append(torch.linalg.inv(covariance+ridge))
    precision = torch.stack(precision)
    def score(features):
        result = []
        for block in snapshot.transform(features).split(snapshot.chunk):
            delta = block[:, None]-snapshot.centers[None]
            distance = torch.einsum('nkr,krh,nkh->nk', delta, precision, delta).clamp_min(0.)
            result.append(distance.min(1).values.sqrt())
        return torch.cat(result)
    return score, pooled


def full_diagonal_metric(snapshot, ref):
    """All observed active dimensions, cell diagonal spread with four-row prior."""
    t = (ref.double()-snapshot.mean)/snapshot.scale
    ids, _ = snapshot.assign(ref)
    means = torch.stack([t[ids == c].mean(0) if bool((ids == c).any()) else t.mean(0)
                         for c in range(snapshot.cells)])
    delta = t-means[ids]
    pooled = delta.square().mean(0)
    variance = torch.stack([(delta[ids == c].square().sum(0)+4.*pooled)/(int((ids == c).sum())+4.)
                            for c in range(snapshot.cells)])
    # Constant captured dimensions remain omitted in this diagnostic.
    active = pooled > pooled.max()*1e-16
    variance = variance[:, active].clamp_min(pooled[active]*1e-6)
    def score(features):
        tquery = ((features.double()-snapshot.mean)/snapshot.scale)[:, active]
        result = []
        for block in tquery.split(snapshot.chunk):
            distance = ((block[:, None]-means[None, :, active]).square()/variance[None]).sum(2)
            result.append(distance.min(1).values.sqrt())
        return torch.cat(result)
    return score


def inspect(ev, shared):
    trainer, bd = common.make_trainer(ev, 'cb64_ra', shared)
    rawq = bd._features(trainer, ev.G(ev.z))
    rawR = bd._features(trainer, ev.real_raw)
    snapshot = shared.cb.FeatureCellSnapshot.fit(rawR, generator=bd.stream, cells=64, rank=8, chunk=256)
    baseline_flags, baseline_p, baseline_scores = snapshot.support(rawq)
    bad = ev.initial_modes == ev.unsupported_bin
    rare = ev.initial_modes == ev.rare_bin
    legitimate = ~bad
    tR = (rawR.double()-snapshot.mean)/snapshot.scale
    tq = (rawq.double()-snapshot.mean)/snapshot.scale
    projectedR, projectedq = tR@snapshot.basis, tq@snapshot.basis
    residualR = tR-projectedR@snapshot.basis.T
    residualq = tq-projectedq@snapshot.basis.T
    residual_norm_R, residual_norm_q = residualR.norm(dim=1), residualq.norm(dim=1)
    residual_flags, residual_p = conformal(snapshot, residual_norm_q, residual_norm_R[1::2])
    shapescore, covariance = covariance_metric(snapshot, rawR[0::2])
    shapeq, shapeR = shapescore(rawq), shapescore(rawR[1::2])
    shapeflags, shapep = conformal(snapshot, shapeq, shapeR)
    diagscore = full_diagonal_metric(snapshot, rawR[0::2])
    diagq, diagR = diagscore(rawq), diagscore(rawR[1::2])
    diagflags, diagp = conformal(snapshot, diagq, diagR)
    qcell, _ = snapshot.assign(rawq)
    dropped = ~torch.isfinite(snapshot.scale)
    changed_dropped = ((rawq.double()-snapshot.mean).abs()[:, dropped] > 0).any(1)
    row_ids = baseline_flags.nonzero().flatten()
    details = []
    for index in row_ids[legitimate[row_ids]].tolist():
        c = int(qcell[index])
        details.append(dict(row=index, rare=bool(rare[index]), cell=c,
                            even_real_count=int(snapshot.reference_counts[c]),
                            odd_real_count=int(snapshot.real_calibration_counts[c]),
                            spherical_radius=float(snapshot.cell_scale[c]),
                            baseline_score=float(baseline_scores[index]), baseline_p=float(baseline_p[index]),
                            shape_score=float(shapeq[index]), shape_p=float(shapep[index]),
                            full_diagonal_score=float(diagq[index]), full_diagonal_p=float(diagp[index]),
                            residual_score=float(residual_norm_q[index]), residual_p=float(residual_p[index])))
    return dict(family=ev.family, n=ev.n, mechanism=ev.mechanism, dim=ev.dim, feature_map=ev.feature_map,
        width=snapshot.width, rank=snapshot.rank, dropped_dimensions=int(dropped.sum()),
        queries_changed_dropped_dimensions=int(changed_dropped.sum()),
        projected_within_covariance_eigenvalues=torch.linalg.eigvalsh(covariance).tolist(),
        residual_even_real=summary(residual_norm_R[0::2], torch.ones(len(rawR[0::2]), dtype=torch.bool)),
        residual_odd_real=summary(residual_norm_R[1::2], torch.ones(len(rawR[1::2]), dtype=torch.bool)),
        residual_bad=summary(residual_norm_q, bad), residual_legitimate=summary(residual_norm_q, legitimate),
        false_positive_details=details,
        scores=dict(spherical=ev.detector(baseline_flags), projected_shape=ev.detector(shapeflags),
                    orthogonal_residual=ev.detector(residual_flags), full_diagonal=ev.detector(diagflags)),
        row1745=dict(baseline_p=float(baseline_p[1745]), shape_p=float(shapep[1745]),
                     diagonal_p=float(diagp[1745]), residual_p=float(residual_p[1745])) if ev.n > 1745 else None)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    shared = setup()
    cases = [('geometry', 2048, 'fold', 2, 'trained600'),
             ('geometry', 2048, 'fold', 128, 'trained600'),
             ('cost', 2048, 'highdim', 128, 'frozen_toy_head')]
    if args.all:
        cases = [('geometry', n, 'fold', 128, fmap) for fmap in ('trained600', 'frozen_initialization')
                 for n in (1024, 2048, 4096)]
        cases += [('cost', n, mechanism, 128 if mechanism == 'highdim' else 8, 'frozen_toy_head')
                  for mechanism in cost.SCENARIOS for n in cost.SIZES]
    files = [Path(__file__), ROOT/'PROTOCOL.md', PREV/'geometry_a'/'bundle.pt',
             PREV/'geometry_a'/'toy_family.py', PREV/'scaling_a'/'shared_toy.py',
             OLD/'geometry'/'run_validation.py', OLD/'pkg-CB64-RA'/'particlegan'/'feature_cells.py',
             common.CONFIG, common.REF_CONFIG]
    hashes = {str(p): sha(p) for p in files}
    result = dict(scope='CPU fixed-fixture support decomposition, diagnostic only',
                  seeds=dict(geometry=geometry.SEED, cost=cost.SEED), source_sha256=hashes, cases=[])
    for case in cases:
        ev = common.Evaluator(*case, shared)
        row = inspect(ev, shared)
        assert ev.state_hash() == ev.initial_state_hash
        result['cases'].append(row)
        print(json.dumps(dict(event='case', fixture=case,
            scores={k:{f:v for f,v in m.items() if f!='flag_ids'} for k,m in row['scores'].items()},
            false_positive_details=row['false_positive_details'])), flush=True)
    assert hashes == {str(p): sha(p) for p in files}
    result['cuda_initialized'] = torch.cuda.is_initialized()
    assert not result['cuda_initialized']
    output = ROOT/('diagnosis-all.json' if args.all else 'diagnosis.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(event='complete', cases=len(result['cases']), cuda_initialized=False)), flush=True)


if __name__ == '__main__':
    main()
