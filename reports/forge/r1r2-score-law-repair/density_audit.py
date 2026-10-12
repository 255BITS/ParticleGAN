"""Reproduce the new score-law Ring metrics from exactly 96 frozen panels.

Run only after publication completes. Import pinned historical equations;
construct no models, draw no samples, update no weights and rewrite no grades.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import MISSING, fields
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from experiments.forge.state import state_digest
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'reports/forge/r1r2-objective-memory-repair'
PINS = {'density_audit.py': '1107399d5003887ceb26ef821ba631fe638390bd865d6d696fa11c74b3c3a0e3',
        'audit_collection.py': '596ba6cbefbfcea8dc059450ba0c157c7cc72980e9a8f5e384324d619f3bf384'}
SOURCE = '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'
RUNTIME = '640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc'
RING = 'ring16_acquisition'


def imported(name):
    import hashlib
    path = PREVIOUS / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[name]
    spec = importlib.util.spec_from_file_location('frozen_score_law_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_recipe(compact):
    """Restore static declared defaults; construct no Recipe or modules."""
    declared = fields(Recipe)
    assert set(compact) <= {field.name for field in declared}
    assert all(field.default is not MISSING and field.default_factory is MISSING for field in declared)
    return {field.name: deepcopy(compact.get(field.name, field.default)) for field in declared}


def parent_geometry(published, h, predecessor):
    attempt = Path(published['checkpoint']['artifact_root']).parent
    request = h.read(attempt / 'request.json')['request']
    raw, graded = h.read(attempt / 'raw-result.json'), h.read(attempt / 'graded-result.json')
    checkpoint = raw['evidence']['provenance_checkpoint']
    state_path = attempt / 'provenance/provenance-state.pt'
    assert published['candidate_id'] == request['candidate']['id'] and published['candidate_revision'] == request['candidate_revision']
    assert published['source_digest'] == request['source']['digest'] == graded['source_digest'] == SOURCE
    assert h.canonical_sha(request['runtime']) == RUNTIME and request['protocol']['seed'] == request['rng']['seed'] == 0
    assert graded['raw_hash'] == h.canonical_sha(raw)
    assert all(h.sha(ROOT / path) == digest for path, digest in request['source']['files'].items())
    assert h.sha(state_path) == checkpoint['sha256'] == published['checkpoint']['sha256']
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    assert state_digest(state) == checkpoint['state_sha256'] == published['checkpoint']['state_sha256']
    assert all(state_digest(state['streams']['states'][name]) == digest
               for name, digest in checkpoint['named_stream_state_sha256'].items())
    assert h.canonical_sha(raw['evidence']['comparison_protocol']) == h.canonical_sha(published['comparison_protocol'])
    trainer = state['trainer']
    assert trainer['completed_steps'] == 1600
    recipe = complete_recipe(trainer['recipe'])
    groups = predecessor.optimizer_bindings(trainer, recipe, 'ams_m110')
    panels = torch.load(attempt / 'observed-samples.pt', map_location='cpu', weights_only=True)
    observations = raw['evidence']['observations']
    assert len(panels) == len(observations) == 96 and panels[-1]['step'] == observations[-1]['step'] == 1600
    assert all(observations[-1][name] == graded['grades'][RING]['metrics'][name] == published['metrics'][name] for name in h.METRICS)
    task = request['tasks'][RING]
    geometry = h.geometry(panels[-1]['samples'], trainer['models'], task['execution']['host_definition'], task['execution']['prior'])
    convergence = graded['grades'][RING]['evaluator_result']['convergence']
    finite = [row for row in observations if h.finite(row['component_covariance_error'])]
    receipt = dict(role='ams_m110', candidate_id=published['candidate_id'], candidate_revision=published['candidate_revision'],
        source_digest=SOURCE, source_origin_commit=request['source']['origin_commit'], runtime_sha256=RUNTIME,
        attempt_id=published['attempt_id'], attempt_archive_path=str(attempt),
        artifacts_sha256={name: h.sha(attempt / name) for name in ('request.json', 'raw-result.json', 'graded-result.json',
                         'observed-samples.pt', 'provenance/provenance-state.pt')},
        checkpoint_state_sha256=checkpoint['state_sha256'], actual_optimizer_groups=groups,
        endpoint=h.selected(observations[-1]), best_covariance_observation=h.selected(min(finite,
            key=lambda row: row['component_covariance_error'])) if finite else None,
        original_passing_observations=convergence['passing_observations'], original_terminal_passing_suffix=convergence['passing_suffix'],
        gate_metrics_recomputed=0, additional_saved_endpoint_geometry_evaluations=1, endpoint_geometry=geometry)
    return receipt, recipe


def audit(publication_path, execution, predecessor):
    assert len(execution) == 40 and all(c in '0123456789abcdef' for c in execution)
    h = predecessor.load_helper()
    publication, registration, manifest = h.read(publication_path), h.read(OUT / 'registration.json'), h.read(OUT / 'manifest.json')
    assert publication['complete'] and publication['source_commit'] == execution
    assert publication['source_digest'] == SOURCE and publication['registration_sha256'] == h.sha(OUT / 'registration.json')
    assert publication['optimizer_updates_added_by_collection'] == publication['sampling_draws_added_by_collection'] == 0
    assert not publication['selection_or_defaults_changed']
    assert set(registration['candidates']) == {'baseline', 'ams_m110', 'non_saturating'}
    records = publication['certified_results']
    assert len(records) == len(publication['task_cells']) == 21
    parents = [row for row in records if row['role'] == 'ams_m110' and row['task_id'] == RING]
    new = [row for row in records if row['role'] == 'non_saturating' and row['task_id'] == RING]
    assert len(parents) == len(new) == 1
    parent, published = parents[0], new[0]
    old_publication_path = PREVIOUS / 'publication/results.json'
    old_parent = next(row for row in h.read(old_publication_path)['certified_results']
                      if row['role'] == 'ams_m110' and row['task_id'] == RING)
    assert parent == old_parent
    parent_shape, parent_recipe = parent_geometry(parent, h, predecessor)
    card = next(row for row in manifest['candidates'] if row['role'] == 'non_saturating')
    admitted = registration['candidates']['non_saturating']
    assert published['candidate_id'] == admitted['candidate'] == card['candidate']
    assert published['candidate_revision'] == admitted['candidate_revision']
    assert published['source_digest'] == admitted['source_digest'] == SOURCE and admitted['runtime_sha256'] == RUNTIME
    assert published['compatibility_key'] == admitted['task_keys'][RING]
    attempt = Path(published['checkpoint']['artifact_root']).parent
    assert attempt.name == published['attempt_id']
    request = h.read(attempt / 'request.json')['request']
    raw, graded = h.read(attempt / 'raw-result.json'), h.read(attempt / 'graded-result.json')
    grade = graded['grades'][RING]
    assert graded['raw_hash'] == h.canonical_sha(raw)
    assert request['candidate']['id'] == card['candidate'] and request['candidate_revision'] == admitted['candidate_revision']
    assert request['source']['origin_commit'] == execution and request['source']['digest'] == graded['source_digest'] == SOURCE
    assert all(h.sha(ROOT / path) == digest for path, digest in request['source']['files'].items())
    assert h.canonical_sha(request['runtime']) == RUNTIME and request['protocol']['seed'] == request['rng']['seed'] == 0
    assert h.canonical_sha(request['protocol']) == admitted['protocol_sha256']
    task = request['tasks'][RING]
    spec, prior = task['execution']['host_definition'], task['execution']['prior']
    assert (spec['layers'], spec['hidden'], spec['particles'], spec['z_dim']) == (2, 64, 256, 4)
    assert prior == dict(kind='mog', sigma=.1, standardize=False) and task['execution']['steps'] == 1600
    thresholds = task['evaluation']['thresholds']
    assert thresholds == [['sample_count', '>=', 4096], ['modes', '>=', 16], ['mass_tv', '<=', .15],
                          ['hq', '>=', .85], ['component_covariance_error', '<=', .85], ['component_min_eigen_ratio', '>=', .15]]
    assert all(h.sha(ROOT / path) == digest for path, digest in task['evaluation']['sources'].items())
    checkpoint = raw['evidence']['provenance_checkpoint']
    state_path = attempt / 'provenance/provenance-state.pt'
    assert h.sha(state_path) == checkpoint['sha256'] == published['checkpoint']['sha256']
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    assert state_digest(state) == checkpoint['state_sha256'] == published['checkpoint']['state_sha256']
    trainer = state['trainer']
    assert trainer['completed_steps'] == checkpoint['completed_steps'] == 1600
    actual = complete_recipe(trainer['recipe'])
    declared = request['candidate']['resolved_recipe']
    assert h.canonical_sha(trainer['recipe']) == h.canonical_sha(raw['recipe'])
    controls = tuple(dict.fromkeys(h.CONTROLS + predecessor.EXTRA_CONTROLS + ('loss', 'loss_labels')))
    assert all(h.canonical_sha(actual[name]) == h.canonical_sha(declared[name]) for name in controls)
    changed = [name for name in actual if h.canonical_sha(actual[name]) != h.canonical_sha(parent_recipe[name])]
    assert changed == ['loss'] and actual['loss'] == 'non_saturating' and parent_recipe['loss'] == 'relativistic'
    assert tuple(actual['loss_labels']) == (0., 1., 1.) and actual['reg_arm'] == 'a_r1r2'
    assert actual['reg_coeff'] == .1 and actual['reg_every'] == 1 and actual['reg_coeff_end'] is None
    # Exactly the parent's native optimizer laws; only the score field differs.
    groups = predecessor.optimizer_bindings(trainer, actual, 'ams_m110')
    record = trainer['optimizers'][1]['regularizer']['record']
    assert record['calls'] == record['observed_steps'] == 1600 and not record['anchor_started']
    assert trainer['penalty_options'] == {}
    stream_hashes, streams = checkpoint['named_stream_state_sha256'], state['streams']['states']
    assert set(streams) == set(stream_hashes) == set(checkpoint['named_stream_keys'])
    assert all(state_digest(streams[name]) == digest for name, digest in stream_hashes.items())
    comparison = raw['evidence']['comparison_protocol']
    assert h.canonical_sha(comparison) == h.canonical_sha(published['comparison_protocol'])
    for field in ('initial_tensors_sha256', 'initial_tensor_metadata', 'actual_target_batches'):
        assert h.canonical_sha(comparison[field]) == h.canonical_sha(parent['comparison_protocol'][field])
    assert raw['evidence']['guards']['unintended_rng_deviations'] == 0
    panels_path = attempt / 'observed-samples.pt'
    panels = torch.load(panels_path, map_location='cpu', weights_only=True)
    observed = raw['evidence']['observations']
    assert len(panels) == len(observed) == 96
    descriptor = raw['evidence']['saved_observer_outputs']
    assert descriptor['path'] == panels_path.name and descriptor['sha256'] == h.sha(panels_path)
    recreated, worst = [], {name: 0. for name in h.METRICS}
    for panel, original in zip(panels, observed, strict=True):
        assert panel['step'] == original['step']
        metrics = dict(step=panel['step'], **h.gate_metrics(panel['samples'], spec))
        for name in h.METRICS:
            left, right = metrics[name], original[name]
            if left is None or right is None:
                assert left is right
            else:
                assert math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6), (panel['step'], name)
                worst[name] = max(worst[name], abs(left - right))
        assert h.passing(metrics, thresholds) == h.passing(original, thresholds)
        recreated.append(metrics)
    flags = [h.passing(row, thresholds) for row in recreated]
    suffix = next((index for index, value in enumerate(reversed(flags)) if not value), len(flags))
    convergence = grade['evaluator_result']['convergence']
    assert sum(flags) == convergence['passing_observations'] and suffix == convergence['passing_suffix']
    assert convergence['minimum_stable_checks'] == 5 and recreated[-1]['step'] == grade['metrics']['step'] == 1600
    # PASS is allowed if and only if the unchanged terminal gate actually passes.
    assert grade['gate_status'] == published['gate_status'] == ('PASS' if suffix >= 5 else 'FAIL')
    assert all(recreated[-1][name] == grade['metrics'][name] == published['metrics'][name] for name in h.METRICS)
    geometry = h.geometry(panels[-1]['samples'], trainer['models'], spec, prior)
    if geometry['status'] != 'UNAVAILABLE':
        assert math.isclose(geometry['reconstructed_float64_full_component_covariance_error'],
                            recreated[-1]['component_covariance_error'], rel_tol=1e-5, abs_tol=1e-5)
    finite = [row for row in recreated if h.finite(row['component_covariance_error'])]
    ring = dict(role='non_saturating', study=card['study'], candidate_id=card['candidate'],
        candidate_revision=admitted['candidate_revision'], source_digest=SOURCE, source_origin_commit=execution,
        runtime_canonical_sha256=RUNTIME, attempt_id=published['attempt_id'], attempt_archive_path=str(attempt),
        exact_cached_parent_role='ams_m110', actual_task_bound_recipe_changed_fields=changed,
        actual_task_bound_recipe_canonical_sha256=h.canonical_sha(actual), recipe_controls={name: declared[name] for name in controls},
        actual_optimizer_groups=groups, full_frozen_source_manifest_verified=True,
        score_law_binding=dict(loss='non_saturating', penalty_arm='a_r1r2', gamma=.1, every=1,
            loss_source_sha256=request['source']['files']['particlegan/gan_loss.py'],
            trainer_source_sha256=request['source']['files']['particlegan/training.py'],
            native_reductions_unscaled=True, historical_paired_force_values_available=False),
        task_evaluation_sources_sha256=task['evaluation']['sources'], protocol_seed=0,
        protocol_canonical_sha256=admitted['protocol_sha256'], checkpoint_state_sha256=checkpoint['state_sha256'],
        artifacts_sha256={name: h.sha(attempt / name) for name in ('request.json', 'raw-result.json', 'graded-result.json',
                         'observed-samples.pt', 'provenance/provenance-state.pt')},
        named_stream_state_sha256=stream_hashes, all_named_stream_checkpoints_verified=True,
        captured_initial_tensor_and_target_batch_hashes_match_own_cached_parent=True,
        comparison_protocol_canonical_sha256=h.canonical_sha(comparison), thresholds=thresholds,
        gate_status=grade['gate_status'], recomputed_observations=96, max_absolute_gate_metric_difference=worst,
        full_passing_observations=sum(flags), full_terminal_passing_suffix=suffix, minimum_stable_checks=5,
        full_passing_windows=h.stable_windows(flags, recreated),
        covariance_bound_passing_observations=sum(h.compare(row['component_covariance_error'], '<=', .85) for row in recreated),
        failures_by_bound={name: sum(not h.compare(row[name], op, bound) for row in recreated) for name, op, bound in thresholds},
        endpoint_failed_bounds=[name for name, op, bound in thresholds if not h.compare(recreated[-1][name], op, bound)],
        endpoint=h.selected(recreated[-1]), best_covariance_observation=h.selected(min(finite,
            key=lambda row: row['component_covariance_error'])) if finite else None,
        last_five_observations=[h.selected(row) for row in recreated[-5:]], endpoint_geometry=geometry)
    return dict(schema_version=1, evidence_type='saved_observation_diagnosis_not_new_experiment',
        scope='One new whole native non-saturating recipe; all96 original Ring panels and saved endpoint geometry.',
        execution_commit=execution, source_digest=SOURCE, runtime_sha256=RUNTIME,
        archive_root=registration['archive'], publication_sha256=h.sha(publication_path),
        manifest_sha256=h.sha(OUT / 'manifest.json'), registration_sha256=h.sha(OUT / 'registration.json'),
        imported_helpers_sha256=PINS, imported_frozen_density_source_sha256=predecessor.HELPER_SHA,
        previous_publication_sha256=h.sha(old_publication_path), recipe_count=1, recomputed_saved_observations=96,
        new_optimizer_updates=0, new_sampling_draws=0, recomputed_metrics=list(h.METRICS),
        cached_parents={'ams_m110': parent_shape}, rings=[ring],
        reference_draws_avoided='Gate-only equations exclude reference/global-moment/SW1 branches.',
        initial_tensor_proof_scope='Exact captured actual-tensor and ordered target-batch receipts match the cached parent; no initialization reconstruction.',
        historical_proof_limit='No historical score pairs, force decomposition or sampled-row ownership is reconstructed. '
            'Beta1=.5 G/D moments average history; only beta1=0 prior moments recover terminal gradients. '
            'Functional first-order Jacobian geometry is not an exact additive covariance partition.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, default=OUT / 'publication/results.json')
    parser.add_argument('--execution-commit', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    predecessor, collection = imported('density_audit.py'), imported('audit_collection.py')
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with collection.forbid_live_execution(), torch.no_grad():
        result = audit(args.publication, args.execution_commit, predecessor)
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0] and before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:] and torch.equal(before[2], after[2])
    result['diagnosis_rng_states_unchanged'] = True
    result['reproduction_source_sha256'] = predecessor.load_helper().sha(Path(__file__))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
