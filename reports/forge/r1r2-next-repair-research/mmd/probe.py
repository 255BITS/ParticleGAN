"""Analytic saved/explicit-fixture MMD check; no models, autograd, draws or steps.

One fixed kernel mixture, not a bandwidth search. Exact Gaussian target
expectations and saved evaluation outputs are a mathematical cohort, not grading.
"""
import hashlib
import importlib.util
import json
import argparse
from pathlib import Path
import random

import numpy as np
import torch
from scipy.special import ive

from experiments.forge.state import state_digest

ROOT = Path(__file__).resolve().parents[4]
ENERGY = ROOT / 'reports/forge/r1r2-energy-distance-repair'
FROZEN = ROOT / 'reports/forge/r1r2-energy-prior-freeze-repair'
SOURCE = '2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12'
EXECUTION = '67920d9090727bd74adbd331187f0e6f22533e48'
PUBLICATION = 'cb3f7b89d1677591a12dd86377d072b55cba1b95e99060b2841cdf42c418b75c'
CHECKPOINT = 'ac391adc90c90cc3dd445b4201765163a20c1187b39c90ba954d450afc8be328'
PANELS = '2db2838d479c63cb7903c27253542dc5814a12a277393b6c4fbb0d526a2b6ebe'
HELPER = ROOT / 'reports/forge/r1r2-objective-memory-repair/research/score-law/probe.py'
HELPER_SHA = '9e39de0c1b045962370383d224c47ad1f60e0e5157f083f78ba794418e59d1d0'
WIDTHS = np.array([.0625, .125, .25, .5, 1., 2., 4.])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_contract(task):
    # Recipe ownership receipts necessarily differ; actual task bytes do not.
    return {key: value for key,value in task.items() if key != 'field_ownership'}


def pairs(a, b):
    return a[:, None] - b[None]


def mmd_u(x, y):
    n, m = len(x), len(y)
    assert n > 1 and m > 1 and x.shape[1:] == y.shape[1:]
    xx, xy, yy = pairs(x, x), pairs(x, y), pairs(y, y)
    dxx, dxy, dyy = (np.square(value).sum(-1) for value in (xx, xy, yy))
    mask_x, mask_y = ~np.eye(n, dtype=bool), ~np.eye(m, dtype=bool)
    value, gradient = 0., np.zeros_like(x)
    for sigma in WIDTHS:
        kxx, kxy, kyy = (np.exp(-distance / (2 * sigma ** 2))
                         for distance in (dxx, dxy, dyy))
        value += kxx[mask_x].mean() + kyy[mask_y].mean() - 2 * kxy.mean()
        gradient += (-2 * (xx * kxx[:, :, None]).sum(1) / (n * (n-1))
                     + 2 * (xy * kxy[:, :, None]).sum(1) / (n*m)) / sigma ** 2
    return float(value / len(WIDTHS)), gradient / len(WIDTHS)


def gaussian_target(x, centers, target_sigma=.1):
    """FF empirical U; cross/RR exact uniform isotropic-Gaussian expectations.

    Return n-scaled gradients so numerical force comparisons do not depend on
    the 4096 evaluation count. Not a substitute for the actual 128 train batch.
    """
    n, dimension = x.shape
    cross_delta = pairs(x, centers)
    cross_distance2 = np.square(cross_delta).sum(-1)
    center_distance2 = np.square(pairs(centers, centers)).sum(-1)
    variance = target_sigma ** 2
    value, gradient = 0., np.zeros_like(x)
    for sigma in WIDTHS:
        s2 = sigma ** 2
        ff_sum, ff_gradient = 0., np.zeros_like(x)
        for start in range(0, n, 128):
            stop = min(start+128, n)
            delta = pairs(x[start:stop], x)
            kernel = np.exp(-np.square(delta).sum(-1)/(2*s2))
            # Exact ordered self terms excluded; duplicate different rows stay.
            kernel[np.arange(stop-start), np.arange(start, stop)] = 0.
            ff_sum += float(kernel.sum())
            ff_gradient[start:stop] = -2 * (delta*kernel[:, :, None]).sum(1)/((n-1)*s2)
        cross = (s2/(s2+variance))**(dimension/2) * np.exp(-cross_distance2/(2*(s2+variance)))
        rr = (s2/(s2+2*variance))**(dimension/2) * np.exp(-center_distance2/(2*(s2+2*variance)))
        value += ff_sum/(n*(n-1)) - 2*cross.mean() + rr.mean()
        gradient += ff_gradient + 2*(cross_delta*cross[:, :, None]).mean(1)/(s2+variance)
    # Exact energy cross-gradient via Rice-mean derivative, d=2 only.
    assert dimension == 2
    radius = np.sqrt(cross_distance2)
    z = cross_distance2/(4*variance)
    mean_distance_derivative = np.sqrt(np.pi/2)*radius/(2*target_sigma)*(ive(0,z)+ive(1,z))
    cross_unit = np.divide(cross_delta, radius[:, :, None], out=np.zeros_like(cross_delta),
                           where=radius[:, :, None] > 0)
    energy = 2*(cross_unit*mean_distance_derivative[:, :, None]).mean(1)
    for start in range(0, n, 128):
        stop = min(start+128, n)
        delta = pairs(x[start:stop], x)
        radius = np.sqrt(np.square(delta).sum(-1))[:, :, None]
        unit = np.divide(delta, radius, out=np.zeros_like(delta), where=radius > 0)
        energy[start:stop] -= 2*unit.sum(1)/(n-1)
    return float(value/len(WIDTHS)), gradient/len(WIDTHS), energy


def drift(x, gradient, centers):
    labels = np.square(pairs(x, centers)).sum(-1).argmin(1)
    radial = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    tangent = np.stack([-radial[:, 1], radial[:, 0]], 1)
    records = []
    for component in range(len(centers)):
        mask = labels == component
        if mask.sum() < 2:
            continue
        centered, force = x[mask] - x[mask].mean(0), -gradient[mask]
        derivative = [float(2 * ((centered @ axis[component]) * (force @ axis[component])).mean())
                      for axis in (radial, tangent)]
        records.append(dict(component=component, samples=int(mask.sum()),
                            radial=derivative[0], tangent=derivative[1]))
    return dict(output_gradient_norm=float(np.linalg.norm(gradient)), components=records,
        radial_expanding=sum(r['radial'] > 0 for r in records),
        tangent_contracting=sum(r['tangent'] < 0 for r in records),
        mean_radial=float(np.mean([r['radial'] for r in records])),
        mean_tangent=float(np.mean([r['tangent'] for r in records])))


def fixtures():
    x = np.array([[-.7, .2], [.3, -.4], [1.1, .8], [-.1, 1.4]])
    y = np.array([[-.3, .7], [1., -.1], [.2, .9]])
    value, gradient = mmd_u(x, y)
    errors = []
    for coordinate in np.ndindex(x.shape):
        offset = np.zeros_like(x)
        offset[coordinate] = 1e-6
        derivative = (mmd_u(x + offset, y)[0] - mmd_u(x - offset, y)[0]) / 2e-6
        errors.append(abs(derivative - gradient[coordinate]))
    assert max(errors) < 2e-9
    matched = np.array([[-1.], [1.]] * 16)
    signed, _ = mmd_u(matched, matched)
    assert signed < 0
    token_x, token_y = np.zeros((8, 2)), np.tile([0., 1.], (8, 1))
    token_value, token_gradient = mmd_u(token_x, token_y)
    assert np.isfinite(token_gradient).all()
    offsets = np.linspace(-.05, .05, 6)
    poles = np.r_[-1 + offsets, 1 + offsets][:, None]
    _, pole_gradient = mmd_u(np.zeros((12, 1)), poles)
    assert abs(pole_gradient).max() < 1e-15
    chars, words = 'abcdefghijklmnopqrstuvwxyz_ ', ('apple', 'grape', 'lemon', 'melon', 'berry')
    bank = np.zeros((5, len(chars), 6))
    for row, word in enumerate(words):
        for position, char in enumerate(word + '_'):
            bank[row, chars.index(char), position] = 1
    word_records = []
    for confidence in (2., 80.):
        logits = bank[:1].repeat(20, axis=0) * confidence
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        loss, g = mmd_u(p.reshape(20, -1), bank.repeat(4, axis=0).reshape(20, -1))
        g = g.reshape(p.shape)
        logit_gradient = p * (g - (g*p).sum(axis=1, keepdims=True))
        word_records.append(dict(logit_gap=confidence, value=loss,
            output_gradient_norm=float(np.linalg.norm(g)), logit_gradient_norm=float(np.linalg.norm(logit_gradient))))
    assert word_records[-1]['logit_gradient_norm'] < word_records[0]['logit_gradient_norm'] * 1e-20
    gaussian_centers = np.array([[-1., 0.], [1., 0.]])
    _, gaussian_gradient, gaussian_energy_gradient = gaussian_target(x, gaussian_centers)
    gaussian_errors, energy_errors = [], []
    def energy_variable_value(points):
        distance2 = np.square(pairs(points, gaussian_centers)).sum(-1)
        z = distance2/(4*.1**2)
        mean_radius = .1*np.sqrt(np.pi/2)*((1+2*z)*ive(0,z)+2*z*ive(1,z))
        fake_distance = np.sqrt(np.square(pairs(points,points)).sum(-1))
        return 2*mean_radius.mean() - fake_distance[~np.eye(len(points),dtype=bool)].mean()
    for coordinate in np.ndindex(x.shape):
        offset = np.zeros_like(x)
        offset[coordinate] = 1e-6
        derivative = (gaussian_target(x+offset, gaussian_centers)[0]
                      - gaussian_target(x-offset, gaussian_centers)[0])/2e-6
        gaussian_errors.append(abs(derivative-gaussian_gradient[coordinate]/len(x)))
        energy_derivative = (energy_variable_value(x+offset)-energy_variable_value(x-offset))/2e-6
        energy_errors.append(abs(energy_derivative-gaussian_energy_gradient[coordinate]/len(x)))
    assert max(gaussian_errors) < 2e-9
    assert max(energy_errors) < 2e-9
    return dict(finite_difference_coordinates=8, max_error=max(errors), error_bound=2e-9,
        gaussian_expectation_gradient_coordinates=8, gaussian_max_error=max(gaussian_errors),
        gaussian_energy_gradient_coordinates=8, gaussian_energy_max_error=max(energy_errors),
        unit_invariance_claimed=False, signed_matched_u=signed,
        token=dict(value=token_value, identical_target_rows=8, target_distance_median=0,
            shared_and_concept_slot_gradient=token_gradient.sum(0).tolist(),
            unused_slot_direct_gradient=[0., 0.], unused_shared_path_still_moves=True),
        pole=dict(gradient_max=float(abs(pole_gradient).max()),
            symmetry_breaking=False), word_softmax=word_records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, default=ROOT,
        help='Verify the original 1216 files here; use the pinned B/E snapshot after later core preparation.')
    source_root = parser.parse_args().source_root.resolve()
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    publication = ENERGY / 'publication/results.json'
    assert sha(publication) == PUBLICATION and sha(HELPER) == HELPER_SHA
    pub = json.loads(publication.read_text())
    assert pub['complete'] and pub['source_digest'] == SOURCE and pub['source_commit'] == EXECUTION
    row = next(r for r in pub['certified_results'] if r['role'] == 'non_saturating'
               and r['task_id'] == 'ring16_acquisition')
    attempt = Path(row['checkpoint']['artifact_root']).parent
    request = json.loads((attempt / 'request.json').read_text())['request']
    assert request['source']['digest'] == SOURCE and len(request['source']['files']) == 1216
    assert all(sha(source_root / path) == digest for path, digest in request['source']['files'].items())
    checkpoint, panels_path = attempt / 'provenance/provenance-state.pt', attempt / 'observed-samples.pt'
    assert sha(checkpoint) == CHECKPOINT == row['checkpoint']['sha256'] and sha(panels_path) == PANELS
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state_digest(saved) == row['checkpoint']['state_sha256']
    recipe = saved['trainer']['recipe']
    assert recipe['loss'] == 'non_saturating' and recipe['reg_arm'] == 'a_r1r2'
    assert recipe['reg_coeff'] == .1 and recipe['reg_every'] == 1 and recipe['amsgrad']
    # Zero additive fields are omitted by the pinned recipe-compat serializer.
    assert recipe.get('energy_distance_weight', 0) == 0 and recipe.get('prior_update', 'learned') == 'learned'
    assert tuple(recipe['betas']) == (.5, .99) and tuple(recipe['prior_betas']) == (0., .99)
    panels = torch.load(panels_path, map_location='cpu', weights_only=True)
    assert len(panels) == 96 and panels[-1]['step'] == saved['trainer']['completed_steps'] == 1600
    x = panels[-1]['samples'].double().numpy()
    assert len(x) == 4096
    definition = request['tasks']['ring16_acquisition']['execution']['host_definition']
    centers = np.array(definition['means'])
    assert np.allclose(definition['covariances'], np.tile(.01*np.eye(2), (16,1,1)), atol=0, rtol=0)
    assert np.allclose(definition['masses'], np.full(16, 1/16), atol=0, rtol=0)
    assert np.allclose(np.linalg.norm(centers,axis=1), 3., atol=1e-14, rtol=0)
    value, gradient, energy = gaussian_target(x, centers)
    spec = importlib.util.spec_from_file_location('frozen_score_math', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    f, critic_gradient = helper.critic(torch.from_numpy(x),
        {k: v.double() for k, v in saved['trainer']['models']['D'].items()})
    ns_gradient = -(torch.sigmoid(-f)[:, None] * critic_gradient).numpy()
    comparison = {'mmd_u_mixture': drift(x, gradient, centers),
                  'energy_u': drift(x, energy, centers),
                  'native_non_saturating': drift(x, ns_gradient, centers)}
    wide_row = next(r for r in pub['certified_results'] if r['role'] == 'energy_u'
                    and r['task_id'] == 'ring16_acquisition')
    wide_attempt = Path(wide_row['checkpoint']['artifact_root']).parent
    wide_state_path = wide_attempt/'provenance/provenance-state.pt'
    wide_panels_path = wide_attempt/'observed-samples.pt'
    assert sha(wide_state_path) == '676740b80e7dfff7862a42b0962bfdea07b292437ec32f60efe9fb2db84c7426'
    assert sha(wide_state_path) == wide_row['checkpoint']['sha256']
    assert sha(wide_panels_path) == '6eaf89d0bddcac23caa0d1dbabc9d9df62b69fb6141184bfad18a64842560872'
    wide_state = torch.load(wide_state_path,map_location='cpu',weights_only=False)
    assert state_digest(wide_state) == wide_row['checkpoint']['state_sha256']
    wide_request = json.loads((wide_attempt/'request.json').read_text())['request']
    assert wide_request['source'] == request['source']
    assert task_contract(wide_request['tasks']['ring16_acquisition']) == task_contract(request['tasks']['ring16_acquisition'])
    wide_panels = torch.load(wide_panels_path,map_location='cpu',weights_only=True)
    assert len(wide_panels) == 96 and wide_panels[-1]['step'] == 1600
    wide_x = wide_panels[-1]['samples'].double().numpy()
    assert len(wide_x) == 4096
    wide_value, wide_gradient, wide_energy = gaussian_target(wide_x,centers)
    wide = dict(role='energy_u',attempt_id=wide_row['attempt_id'],
        candidate_revision=wide_row['candidate_revision'], checkpoint_sha256=sha(wide_state_path),
        checkpoint_state_sha256=wide_row['checkpoint']['state_sha256'], panels_sha256=sha(wide_panels_path),
        value=wide_value, sample_count=4096,
        comparison=dict(mmd_u_mixture=drift(wide_x,wide_gradient,centers),
                        energy_u=drift(wide_x,wide_energy,centers)))
    latest_path = FROZEN / 'publication/results.json'
    assert sha(latest_path) == '013696f61094eb8c06e9d9c41a9422270315678992f9a1a9680430c56bf7f207'
    latest = json.loads(latest_path.read_text())
    assert latest['complete'] and len(latest['aggregate_charge_attempt_ids']) == 238
    frozen_row = next(r for r in latest['certified_results'] if r['role'] == 'frozen_prior'
                      and r['task_id'] == 'ring16_acquisition')
    frozen_attempt = Path(frozen_row['checkpoint']['artifact_root']).parent
    frozen_state_path = frozen_attempt/'provenance/provenance-state.pt'
    frozen_panels_path = frozen_attempt/'observed-samples.pt'
    assert sha(frozen_state_path) == 'cae963187dfca81c7bde6fb691baebc978a5b5a4196404829699a52bca719290'
    assert sha(frozen_state_path) == frozen_row['checkpoint']['sha256']
    assert sha(frozen_panels_path) == 'a71ca74e390eb980597c440b726fdbf184adf0b3311f0576b64ff49db963f862'
    frozen_state = torch.load(frozen_state_path,map_location='cpu',weights_only=False)
    assert state_digest(frozen_state) == frozen_row['checkpoint']['state_sha256']
    frozen_request = json.loads((frozen_attempt/'request.json').read_text())['request']
    assert frozen_request['source']['digest'] == SOURCE
    assert frozen_request['source']['files'] == request['source']['files']
    assert task_contract(frozen_request['tasks']['ring16_acquisition']) == task_contract(request['tasks']['ring16_acquisition'])
    frozen_panels = torch.load(frozen_panels_path,map_location='cpu',weights_only=True)
    assert len(frozen_panels) == 96 and frozen_panels[-1]['step'] == 1600
    frozen_x = frozen_panels[-1]['samples'].double().numpy()
    assert len(frozen_x) == 4096
    frozen_value, frozen_gradient, frozen_energy = gaussian_target(frozen_x,centers)
    frozen = dict(role='frozen_prior',attempt_id=frozen_row['attempt_id'],
        candidate_revision=frozen_row['candidate_revision'],
        execution_origin=frozen_request['source']['origin_commit'],
        checkpoint_sha256=sha(frozen_state_path),
        checkpoint_state_sha256=frozen_row['checkpoint']['state_sha256'],
        panels_sha256=sha(frozen_panels_path), value=frozen_value, sample_count=4096,
        comparison=dict(mmd_u_mixture=drift(frozen_x,frozen_gradient,centers),
                        energy_u=drift(frozen_x,frozen_energy,centers)))
    fixture_results = fixtures()
    after = np.random.get_state()
    assert random.getstate() == py_rng and torch.equal(torch_rng, torch.get_rng_state())
    assert after[0] == np_rng[0] and np.array_equal(after[1], np_rng[1]) and after[2:] == np_rng[2:]
    print(json.dumps(dict(schema_version=1, scope='read_only_saved_and_explicit_mathematical_fixtures',
        source_digest=SOURCE, source_files_verified=1216, source_files_verified_root=str(source_root),
        execution_commit=EXECUTION,
        candidate_revision=row['candidate_revision'], attempt_id=row['attempt_id'],
        publication_sha256=PUBLICATION, checkpoint_sha256=CHECKPOINT,
        checkpoint_state_sha256=row['checkpoint']['state_sha256'], panels_sha256=PANELS,
        helper_sha256=HELPER_SHA, reproduction_source_sha256=sha(Path(__file__)),
        latest_publication_sha256=sha(latest_path), historical_attempts=238,
        aggregate_paid_seconds=latest['aggregate_total_paid_seconds'],
        remaining_paid_seconds=8000-latest['aggregate_total_paid_seconds'],
        coefficient=1, mixture_stddev_native_units=WIDTHS.tolist(), equal_weights=True,
        statistic='signed squared-MMD offdiagonal U-form; no sqrt/clamp',
        fixed_kernel_u_unbiased_for_iid_minibatches=True,
        fixtures=fixture_results, saved_overwide_control=wide, saved_frozen_control=frozen,
        ring=dict(value=value, comparison=comparison,
            target='exact uniform original16-Gaussian expectation; radius3, isotropic sigma.1',
            fake='all4096 saved terminal evaluation outputs; ordered FF self exclusions',
            gradient_normalization='n times derivative of mean loss; input-space force only',
            actual_seen_training_pair=False, target_component_parameters_used_only_in_diagnostic=True,
            prospective_objective_uses_only_existing_real_fake_minibatches=True,
            local_variance_is_only_output_space_first_order_proxy=True),
        qualification_input=False, models_constructed=0, autograd_calls=0,
        sampling_draws_added=0, optimizer_updates_added=0, rng_unchanged=True), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
