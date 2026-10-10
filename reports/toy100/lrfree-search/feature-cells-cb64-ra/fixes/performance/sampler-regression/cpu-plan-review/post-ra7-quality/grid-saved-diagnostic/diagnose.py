"""Fixed saved-cloud diagnostic; no constructor, RNG draw or production writes."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
from pathlib import Path
import sys
import hashlib
import json
import math
from datetime import datetime, timezone
import numpy as np
import torch
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
RUN = ROOT / 'validation-ra4/screens/runs/grid100'
HOSTS = Path('/ml2/hypergan/lrfree-20260926/harness/hosts')
sys.path.insert(0, str(HOSTS))
from native100.problems import evaluation_geometry
from native100.accuracy import TARGET_PRECISION, TARGET_CONDITIONAL_VARIANCE

torch.set_num_threads(1)
CENTERS_T, SIGMA = evaluation_geometry('grid100', dtype=torch.float64)
CENTERS = CENTERS_T.numpy()
K = len(CENTERS)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    sealed = read(HERE / 'PREPARATION-FROZEN.json')
    for path, digest in sealed['source_sha256'].items():
        assert sha(path) == digest, path
    prep = read(HERE / 'PREPARATION.json')
    for path, digest in prep['source_and_input_sha256'].items():
        assert sha(path) == digest, path
    return prep


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    return dict(mean=float(np.mean(x)), median=float(np.median(x)),
                p05=float(np.quantile(x, .05)), p95=float(np.quantile(x, .95)),
                max=float(np.max(x)))


def norm_stats(x):
    values = np.linalg.norm(np.asarray(x), axis=1)
    return dict(rms=float(np.sqrt(np.mean(values ** 2))), **stats(values))


def assign(points):
    points = np.asarray(points, dtype=np.float64)
    assert points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all()
    ids = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 4096):
        block = points[start:start + 4096]
        delta = block[:, None, :] - CENTERS[None, :, :]
        distance = np.einsum('nkd,nkd->nk', delta, delta)
        ids[start:start + len(block)] = np.argmin(distance, axis=1)
    residual = (points - CENTERS[ids]) / SIGMA
    return ids, residual


def moments(x, ids):
    counts = np.bincount(ids, minlength=K)
    denominator = np.maximum(counts, 1)
    means = np.stack([np.bincount(ids, weights=x[:, j], minlength=K) / denominator
                      for j in range(2)], axis=1)
    second = np.empty((K, 2, 2), dtype=np.float64)
    for j in range(2):
        for k in range(2):
            second[:, j, k] = np.bincount(ids, weights=x[:, j] * x[:, k], minlength=K) / denominator
    cov = second - means[:, :, None] * means[:, None, :]
    return counts, means, cov


def radial_ks(radii):
    radii = np.sort(np.asarray(radii))
    positions = np.arange(len(radii))
    cdf = (1. - np.exp(-radii ** 2 / 2.)) / TARGET_PRECISION
    return float(max(np.max((positions + 1) / len(radii) - cdf),
                     np.max(cdf - positions / len(radii))))


def summary(points):
    ids, residual = assign(points)
    radii = np.linalg.norm(residual, axis=1)
    hq = radii <= 3.
    counts, means, cov = moments(residual, ids)
    hc, hm, hv = moments(residual[hq], ids[hq])
    trace = np.trace(hv, axis1=1, axis2=2) / (2. * TARGET_CONDITIONAL_VARIANCE)
    missing = np.flatnonzero(hc < 2).tolist()
    metrics = dict(n=len(points), precision=float(hq.mean()), within_radius_n=int(hq.sum()),
                   mass_tv=float(np.abs(counts / len(points) - 1. / K).sum() / 2.),
                   center_rms_sigma=None if missing else float(np.sqrt(np.mean(np.sum(hm ** 2, axis=1)))),
                   cov_trace_bias=None if missing else float(np.mean(trace) - 1.),
                   abs_cov_trace_bias=None if missing else float(abs(np.mean(trace) - 1.)),
                   radial_ks=None if missing else radial_ks(radii[hq]))
    eig = np.linalg.eigvalsh(hv)
    result = dict(metrics=metrics, counts=counts.tolist(), hq_counts=hc.tolist(),
                  missing_hq_modes=missing, min_mode_mass=float(counts.min() / len(points)),
                  max_mode_mass=float(counts.max() / len(points)),
                  unconditional_center_rms_sigma=float(np.sqrt(np.mean(np.sum(means ** 2, axis=1)))),
                  unconditional_mode_mean_sigma=means.tolist(), conditional_mode_mean_sigma=hm.tolist(),
                  unconditional_cov_trace_per_axis_sigma2=stats(np.trace(cov, axis1=1, axis2=2) / 2.),
                  conditional_cov_trace_ratio=stats(trace), conditional_cov_eig_sigma2=stats(eig),
                  conditional_radial_ks_after_empirical_mode_centering=radial_ks(
                      np.linalg.norm(residual[hq] - hm[ids[hq]], axis=1)),
                  conditional_covariance_sigma2=hv.tolist())
    internals = dict(ids=ids, residual=residual, hq=hq, counts=counts, means=means, cov=cov,
                     hq_counts=hc, hq_means=hm, hq_cov=hv)
    return result, internals


def mean_square(x):
    return float(np.mean(np.sum(x * x, axis=1)))


def centroid_comparison(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    norm = np.linalg.norm(aa) * np.linalg.norm(bb)
    return dict(mode_mean_difference_rms_sigma=math.sqrt(mean_square(aa - bb)),
                mode_mean_vector_cosine=float(np.sum(aa * bb) / norm) if norm else None,
                first_mean_square_sigma2=mean_square(aa), second_mean_square_sigma2=mean_square(bb))


def paired_decomposition(clean, noisy, clean_summary, noisy_summary):
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    noise = (noisy - clean) / SIGMA
    ids, _, hq = noisy_summary['ids'], noisy_summary['residual'], noisy_summary['hq']
    clean_component = (clean - CENTERS[ids]) / SIGMA
    counts, cm, cv = moments(clean_component[hq], ids[hq])
    _, em, ev = moments(noise[hq], ids[hq])
    selected_ids = ids[hq]
    centered_clean = clean_component[hq] - cm[selected_ids]
    centered_noise = noise[hq] - em[selected_ids]
    ce = np.empty((K, 2, 2))
    for j in range(2):
        for k in range(2):
            ce[:, j, k] = np.bincount(selected_ids,
                weights=centered_clean[:, j] * centered_noise[:, k], minlength=K) / np.maximum(counts, 1)
    cross_cov = ce + ce.transpose(0, 2, 1)
    expected = cv + ev + cross_cov
    assert np.max(np.abs(expected - noisy_summary['hq_cov'])) < 1e-10
    assert np.max(np.abs(cm + em - noisy_summary['hq_means'])) < 1e-10
    noise_cov = np.cov(noise, rowvar=False, bias=True)
    return dict(same_saved_pairs=True,
                nearest_mode_assignment_changes=int(np.sum(ids != clean_summary['ids'])),
                noise_global_mean_sigma=noise.mean(axis=0).tolist(),
                noise_global_covariance_sigma2=noise_cov.tolist(),
                noise_global_variance_per_axis_sigma2=float(np.trace(noise_cov) / 2.),
                center_mean_square_sigma2=dict(clean_component=mean_square(cm),
                    selected_output_noise_component=mean_square(em),
                    twice_cross=float(2 * np.mean(np.sum(cm * em, axis=1))),
                    sum=mean_square(cm + em), original_noisy=mean_square(noisy_summary['hq_means'])),
                conditional_cov_trace_per_axis_sigma2=dict(clean_component=float(np.trace(cv,axis1=1,axis2=2).mean()/2),
                    selected_output_noise_component=float(np.trace(ev,axis1=1,axis2=2).mean()/2),
                    cross=float(np.trace(cross_cov,axis1=1,axis2=2).mean()/2),
                    sum=float(np.trace(expected,axis1=1,axis2=2).mean()/2)),
                conditional_clean_component_mode_means_sigma=cm.tolist(),
                conditional_output_noise_component_mode_means_sigma=em.tolist(),
                identity_max_covariance_residual=float(np.max(np.abs(expected - noisy_summary['hq_cov']))))


def main():
    prep = verify()
    rng = torch.get_rng_state().clone()
    assert not torch.cuda.is_initialized()
    state = torch.load(RUN / 'final-state.pt', map_location='cpu', weights_only=False)['trainer']
    assert state['completed_steps'] == 7000 and state['recipe']['num_particles'] == 20000
    models = state['models']
    anchors = {}
    affine = {}
    z = {}
    for model, role, prior in (('live', 'G', 'prior'), ('ema', 'ema_G', 'ema_prior')):
        w = models[role]['weight'].numpy().astype(np.float64)
        b = models[role]['bias'].numpy().astype(np.float64)
        z[model] = models[prior]['z'].numpy().astype(np.float64)
        anchors[model] = z[model] @ w.T + b
        affine[model] = dict(weight=w.tolist(), bias=b.tolist(),
                             identity_matrix_max_abs_error=float(np.max(np.abs(w - np.eye(2)))))
    summaries = {}
    internal = {}
    original_checks = []
    for split, fname in (('terminal', 'final_samples.npz'), ('holdout', 'holdout_samples.npz')):
        for kind in ('clean', 'noisy'):
            path = RUN / f'native-{kind}' / fname
            with np.load(path, allow_pickle=False) as archive:
                expected = 20000 if split == 'terminal' else 100000
                assert set(archive.files) == {'live', 'ema', 'target'}
                for model in ('live', 'ema', 'target'):
                    points = archive[model]
                    assert points.shape == (expected, 2)
                    key = f'{split}/{kind}/{model}'
                    summaries[key], internal[key] = summary(points)
            if split == 'terminal':
                official = {x['model']: x['accuracy'] for x in (
                    json.loads(line) for line in (RUN / f'native-{kind}' / 'events.jsonl').read_text().splitlines())
                    if x['event'] == 'eval' and x['step'] == 7000}
            else:
                official = read(RUN / f'native-{kind}' / 'summary.json')['holdout']
            for model, metrics in official.items():
                observed = summaries[f'{split}/{kind}/{model}']['metrics']
                for field in ('precision', 'mass_tv', 'center_rms_sigma', 'cov_trace_bias', 'abs_cov_trace_bias', 'radial_ks'):
                    assert abs(observed[field] - metrics[field]) < 1e-11, (split, kind, model, field)
                original_checks.append(dict(split=split, kind=kind, model=model, frozen_fidelity_fields_exact=True))
    anchor_summaries = {}
    anchor_internal = {}
    for model in anchors:
        anchor_summaries[model], anchor_internal[model] = summary(anchors[model])
    decompositions = {}
    centroid_checks = {}
    for split, fname in (('terminal', 'final_samples.npz'), ('holdout', 'holdout_samples.npz')):
        with np.load(RUN / 'native-clean' / fname, allow_pickle=False) as cfile, np.load(
                RUN / 'native-noisy' / fname, allow_pickle=False) as nfile:
            assert np.array_equal(cfile['target'], nfile['target'])
            for model in ('live', 'ema'):
                key = f'{split}/{model}'
                decompositions[key] = paired_decomposition(cfile[model], nfile[model],
                    internal[f'{split}/clean/{model}'], internal[f'{split}/noisy/{model}'])
                distance, row = cKDTree(anchors[model]).query(cfile[model], k=1, workers=1)
                clean = internal[f'{split}/clean/{model}']
                centroid_checks[key] = dict(
                    anchor_to_clean_unconditional=centroid_comparison(anchor_internal[model]['means'], clean['means']),
                    anchor_to_clean_HQ=centroid_comparison(anchor_internal[model]['hq_means'], clean['hq_means']),
                    clean_distance_to_nearest_unperturbed_anchor_sigma=stats(distance/SIGMA),
                    clean_distance_to_nearest_anchor_rms_sigma=float(np.sqrt(np.mean(distance**2))/SIGMA),
                    nearest_anchor_and_clean_same_mode_fraction=float(np.mean(anchor_internal[model]['ids'][row]==clean['ids'])),
                    nearest_anchor_is_not_saved_sampled_row_identity=True)
    persistence = {}
    for kind in ('clean', 'noisy'):
        for model in ('live', 'ema', 'target'):
            persistence[f'{kind}/{model}'] = centroid_comparison(
                internal[f'terminal/{kind}/{model}']['hq_means'], internal[f'holdout/{kind}/{model}']['hq_means'])
    fast, ema = anchor_internal['live'], anchor_internal['ema']
    wfast = np.asarray(affine['live']['weight'])
    wema = np.asarray(affine['ema']['weight'])
    bfast = np.asarray(affine['live']['bias'])
    bema = np.asarray(affine['ema']['bias'])
    prior_delta = (z['ema']-z['live']) @ wema.T
    affine_delta = z['live'] @ (wema-wfast).T + (bema-bfast)
    total_delta = anchors['ema']-anchors['live']
    assert np.max(np.abs(prior_delta+affine_delta-total_delta)) < 1e-12
    correspondence = dict(same_row_same_oracle_mode_fraction=float(np.mean(fast['ids']==ema['ids'])),
        both_rows_inside_3sigma_fraction=float(np.mean(fast['hq']&ema['hq'])),
        same_mode_and_both_inside_fraction=float(np.mean((fast['ids']==ema['ids'])&fast['hq']&ema['hq'])),
        paired_output_displacement_sigma=norm_stats(total_delta/SIGMA),
        prior_row_component_sigma=norm_stats(prior_delta/SIGMA), affine_component_sigma=norm_stats(affine_delta/SIGMA),
        anchor_centroids=centroid_comparison(fast['hq_means'],ema['hq_means']))
    diagnostic = [json.loads(line) for line in (RUN/'native100-diagnostics.jsonl').read_text().splitlines()]
    last = next(x for x in diagnostic if x['step']==7000)
    bd = state['birth_death']
    context = dict(completed_steps=state['completed_steps'], backend_schema=bd['backend_schema'],
        latent_bandwidth=state['controller']['latent_bandwidth'], output_sigma_saved=last['output_sigma'],
        learned_output_sigma=float(state['output_noise']['log_sigma'].exp()),
        applied_output_noise_variance_fraction=(last['output_sigma']/SIGMA)**2,
        controller_mobility=state['controller']['mobility'],
        table_clock={k:state['lr_settle'][0][1][k] for k in ('s','b','last_decisive','last_decisive_scale')},
        recipe_noise_mode=state['recipe']['output_noise_mode'],
        recipe_serve_average=state['recipe']['serve_average'], saved_affine=affine,
        G_is_trainable_affine=state['requires_grad']['G'])
    assert torch.equal(torch.get_rng_state(),rng)
    assert not torch.cuda.is_initialized()
    verify()
    output = dict(status='VALID_SAVED_DIAGNOSTIC', utc=datetime.now(timezone.utc).isoformat(),
        source_and_input_sha256=prep['source_and_input_sha256'], helper_sha256=sha(__file__),
        context=context, cloud_summaries=summaries, anchor_summaries=anchor_summaries,
        paired_noise_decomposition=decompositions, anchor_to_clean=centroid_checks,
        terminal_holdout_center_persistence=persistence, paired_anchor_correspondence=correspondence,
        original_scoring_reproduction=original_checks,
        cpu_only=True,cuda_initialized=False, input_bytes_unchanged=True, global_cpu_rng_unchanged=True,
        new_draws=0,new_emissions=0,new_training_steps=0,checkpoint_selection='fixed_final7000_only',
        production_changes=False, quality_verdict_override=False,
        limits=['Nearest-anchor distances are lower bounds; sampled row IDs were not saved.',
                'Cloud versus anchor moment differences include replacement-sampling error and latent perturbation.',
                'Centered-HQ radial KS uses an empirical center and fixed old mask; it is not a new output or quality verdict.',
                'Saved FAST/EMA row correspondence describes the final state, not historical copy actions.',
                'Oracle centers are used only in this offline diagnostic.'])
    path=HERE/'receipt.json'
    assert not path.exists()
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    compact = {key:value['metrics'] for key,value in summaries.items() if not key.endswith('/target')}
    print(json.dumps(dict(status=output['status'], output=str(path), context=context,
        cloud_metrics=compact,anchor_metrics={k:v['metrics'] for k,v in anchor_summaries.items()},
        paired_anchor_correspondence=correspondence),allow_nan=False),flush=True)


if __name__=='__main__':
    main()
