"""Saved-only score-law fixture: no models, samplers, autograd or updates."""
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import numpy as np
import torch

from experiments.forge.api import task_formulation_context
from experiments.forge.state import state_digest
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'reports/forge/r1r2-energy-distance-repair'
EXECUTION = '67920d9090727bd74adbd331187f0e6f22533e48'
SOURCE = '2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12'
PUBLICATION_SHA = 'cb3f7b89d1677591a12dd86377d072b55cba1b95e99060b2841cdf42c418b75c'
CHECKPOINT_SHA = 'ac391adc90c90cc3dd445b4201765163a20c1187b39c90ba954d450afc8be328'
PANELS_SHA = '2db2838d479c63cb7903c27253542dc5814a12a277393b6c4fbb0d526a2b6ebe'
HELPER = ROOT / 'reports/forge/r1r2-objective-memory-repair/research/score-law/probe.py'
HELPER_SHA = '9e39de0c1b045962370383d224c47ad1f60e0e5157f083f78ba794418e59d1d0'
TASKS = ('gaussian1d_smoke', 'two_pole', 'unused_token_hold', 'ae_gan_hold',
         'ring16_acquisition', 'five_word_joint_smoke', 'clockfree_audit_measurement_v1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    publication = OUT / 'publication/results.json'
    assert sha(publication) == PUBLICATION_SHA and sha(HELPER) == HELPER_SHA
    pub = json.loads(publication.read_text())
    assert pub['complete'] and pub['source_commit'] == EXECUTION and pub['source_digest'] == SOURCE
    row = next(r for r in pub['certified_results']
               if r['role'] == 'non_saturating' and r['task_id'] == 'ring16_acquisition')
    attempt = Path(row['checkpoint']['artifact_root']).parent
    request = json.loads((attempt / 'request.json').read_text())['request']
    assert request['source']['digest'] == SOURCE and request['source']['origin_commit'] == EXECUTION
    assert all(sha(ROOT / path) == digest for path, digest in request['source']['files'].items())
    checkpoint, panels_path = attempt / 'provenance/provenance-state.pt', attempt / 'observed-samples.pt'
    assert sha(checkpoint) == CHECKPOINT_SHA == row['checkpoint']['sha256']
    assert sha(panels_path) == PANELS_SHA
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state_digest(saved) == row['checkpoint']['state_sha256']
    trainer = saved['trainer']
    panels = torch.load(panels_path, map_location='cpu', weights_only=True)
    assert len(panels) == 96 and panels[-1]['step'] == trainer['completed_steps'] == 1600
    recipe = Recipe(**trainer['recipe'])
    assert recipe.loss == 'non_saturating' and recipe.loss_labels == (0., 1., 1.)
    assert recipe.reg_arm == 'a_r1r2' and recipe.reg_coeff == .1 and recipe.reg_every == 1
    assert recipe.amsgrad and recipe.betas == (.5, .99) and recipe.prior_betas == (0., .99)
    assert recipe.energy_distance_weight == 0 and recipe.prior_update == 'learned'
    host_deltas = {}
    for task_id in TASKS:
        task = request['tasks'][task_id]
        base = task_formulation_context(request['candidate'], task, request['protocol'],
                                        device='cpu', root=ROOT).recipe
        changed = replace(base, loss='least_squares')
        delta = [key for key in asdict(base) if asdict(base)[key] != asdict(changed)[key]]
        assert delta == ['loss'] and changed.energy_distance_weight == 0
        assert changed.reg_arm == 'a_r1r2' and changed.reg_coeff == .1 and changed.reg_every == 1
        host_deltas[task_id] = delta
    spec = importlib.util.spec_from_file_location('frozen_score_math', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    points = panels[-1]['samples'].double()
    critic_state = {key: value.double() for key, value in trainer['models']['D'].items()}
    centers = points.new_tensor(request['tasks']['ring16_acquisition']['execution']['host_definition']['means'])
    fake, gradients = helper.critic(points, critic_state)
    real, _ = helper.critic(centers, critic_state)
    labels = torch.cdist(points, centers).argmin(1)
    radial = centers / centers.norm(dim=1, keepdim=True)
    tangent = torch.stack([-radial[:, 1], radial[:, 0]], 1)
    weights = {'non_saturating': torch.sigmoid(-fake), 'least_squares': 1 - fake}
    shape = {}
    for name, weight in weights.items():
        result = helper.shape_proxy(points, labels, radial, tangent, gradients, weight)
        shape[name] = {key: value for key, value in result.items() if not isinstance(value, list)}
    np_after = np.random.get_state()
    assert random.getstate() == py_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert np_after[0] == np_rng[0] and np.array_equal(np_after[1], np_rng[1]) and np_after[2:] == np_rng[2:]
    print(json.dumps(dict(
        schema_version=1, scope='saved_endpoint_plus_explicit_target_center_fixture',
        execution_commit=EXECUTION, source_digest=SOURCE,
        source_files_verified=len(request['source']['files']),
        publication_sha256=PUBLICATION_SHA, helper_sha256=HELPER_SHA,
        attempt_id=row['attempt_id'], candidate_revision=row['candidate_revision'],
        checkpoint_sha256=CHECKPOINT_SHA, checkpoint_state_sha256=row['checkpoint']['state_sha256'],
        panels_sha256=PANELS_SHA, saved_panel_count=96, saved_endpoint_samples=len(points),
        target_centers_are_historical_training_pairs=False,
        inherited_labels=[0., 1., 1.], host_recipe_deltas=host_deltas,
        fake_scores=helper.summary(fake), target_center_scores=helper.summary(real),
        critic_input_gradient_norm=helper.summary(gradients.norm(dim=1)),
        fake_scores_above_generator_target=int((fake > 1).sum()),
        ascent_weights={name: helper.summary(value) for name, value in weights.items()},
        least_squares_over_non_saturating_weight=helper.summary(weights['least_squares'] / weights['non_saturating']),
        shape_proxy=shape,
        common_shift_fixture=dict(non_saturating_gradient=float(-torch.sigmoid(-real).mean() + torch.sigmoid(fake).mean()),
                                  least_squares_gradient=float((real - 1).mean() + fake.mean()),
                                  non_saturating_curvature=float((torch.sigmoid(real) * torch.sigmoid(-real)).mean()
                                                                + (torch.sigmoid(fake) * torch.sigmoid(-fake)).mean()),
                                  least_squares_curvature=2.),
        qualification_input=False, models_constructed=0, sampling_draws_added=0,
        optimizer_updates_added=0, rng_unchanged=True), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
