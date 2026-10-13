"""Reproduce the both fresh energy-comparison Ring metrics from exactly 192 frozen panels.

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
import sys

import numpy as np
import torch

from experiments.forge.state import state_digest
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'reports/forge/r1r2-weak-penalty-repair'
HELPERS = ROOT / 'reports/forge/r1r2-objective-memory-repair'
PINS = {'density_audit.py': '1107399d5003887ceb26ef821ba631fe638390bd865d6d696fa11c74b3c3a0e3',
        'audit_collection.py': '596ba6cbefbfcea8dc059450ba0c157c7cc72980e9a8f5e384324d619f3bf384'}
RUNTIME = '640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc'
RING = 'ring16_acquisition'
PYTHON = '/home/martyn/dev/ParticleGAN/.venv/bin/python'


def imported(name):
    import hashlib
    path = HELPERS / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PINS[name]
    spec = importlib.util.spec_from_file_location('frozen_weak_penalty_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_recipe(compact):
    """Restore static declared defaults; construct no Recipe or modules."""
    declared = fields(Recipe)
    assert set(compact) <= {field.name for field in declared}
    assert all(field.default is not MISSING and field.default_factory is MISSING for field in declared)
    return {field.name: deepcopy(compact.get(field.name, field.default)) for field in declared}


def load_collection():
    path = OUT / 'audit_collection.py'
    spec = importlib.util.spec_from_file_location('fresh_energy_collection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(publication_path, execution, predecessor, collection):
    assert len(execution) == 40 and all(c in '0123456789abcdef' for c in execution)
    assert sys.executable == PYTHON and torch.get_num_threads() == 1
    h = predecessor.load_helper()
    inputs = {'publication': publication_path, 'registration': OUT / 'registration.json',
              'manifest': OUT / 'manifest.json', 'previous_publication': PREVIOUS / 'publication/results.json',
              'previous_audit': PREVIOUS / 'collection-audit.json', 'collection_source': OUT / 'audit_collection.py'}
    input_hashes = {name: h.sha(path) for name, path in inputs.items()}
    publication, registration, manifest = [h.read(inputs[key]) for key in ('publication','registration','manifest')]
    assert publication['complete'] and publication['source_commit'] == execution
    assert publication['round_id'] == 'r1r2-energy-distance-repair-v1'
    assert publication['full_reservation_seconds'] == registration['reservation_seconds'] == 5040
    assert publication['paid_cap_seconds'] == registration['paid_cap_seconds'] == 8000
    source = manifest['source_digest']
    assert source == publication['source_digest'] == registration['source_digest']
    assert len(source) == 64 and source != h.read(inputs['previous_publication'])['source_digest']
    assert publication['registration_sha256'] == h.sha(inputs['registration'])
    assert registration['manifest'] == manifest and registration['manifest_sha256'] == h.sha(inputs['manifest'])
    assert all(h.sha(ROOT / path) == digest for path, digest in registration['declarations'].items())
    assert {str((OUT / name).relative_to(ROOT)) for name in ('audit_collection.py', 'density_audit.py')} <= set(registration['declarations'])
    collection.committed_sources(ROOT, execution, registration['declarations'])
    assert publication['optimizer_updates_added_by_collection'] == publication['sampling_draws_added_by_collection'] == 0
    assert not publication['selection_or_defaults_changed']
    assert set(registration['candidates']) == {'non_saturating', 'energy_u'}
    records = publication['certified_results']
    assert len(records) == len(publication['task_cells']) == len(publication['media']) == 14
    old_publication, old_audit = h.read(inputs['previous_publication']), h.read(inputs['previous_audit'])
    assert old_publication['complete'] and old_audit['status'] == 'PASS'
    assert old_audit['inputs_sha256']['publication'] == input_hashes['previous_publication']
    old_attempts = set(old_publication['aggregate_charge_attempt_ids'])
    assert all(row['attempt_id'] not in old_attempts for row in records)
    parent = next(row for row in records if row['role']=='non_saturating' and row['task_id']==RING)
    parent_recipe, rings = None, []
    for role in ('non_saturating', 'energy_u'):
        published = next(row for row in records if row['role']==role and row['task_id']==RING)
        card = next(row for row in manifest['candidates'] if row['role'] == role)
        admitted = registration['candidates'][role]
        assert published['candidate_id'] == admitted['candidate'] == card['candidate']
        assert published['candidate_revision'] == admitted['candidate_revision']
        assert published['source_digest'] == admitted['source_digest'] == source and admitted['runtime_sha256'] == RUNTIME
        assert published['compatibility_key'] == admitted['task_keys'][RING]
        attempt = Path(published['checkpoint']['artifact_root']).parent
        assert attempt.name == published['attempt_id']
        request = h.read(attempt / 'request.json')['request']
        raw, graded = h.read(attempt / 'raw-result.json'), h.read(attempt / 'graded-result.json')
        grade = graded['grades'][RING]
        assert graded['raw_hash'] == h.canonical_sha(raw)
        assert request['candidate']['id'] == card['candidate'] and request['candidate_revision'] == admitted['candidate_revision']
        assert request['source']['origin_commit'] == execution and request['source']['digest'] == graded['source_digest'] == source
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
        collection.committed_sources(ROOT, execution, request['source']['files'])
        energy = collection.energy_receipt(state, dict(evidence=raw['evidence']), RING, role, collection.load_helper(), steps=1600)
        declared = request['candidate']['resolved_recipe']
        assert h.canonical_sha(trainer['recipe']) == h.canonical_sha(raw['recipe'])
        controls = tuple(dict.fromkeys(h.CONTROLS + predecessor.EXTRA_CONTROLS + ('loss', 'loss_labels', 'energy_distance_weight')))
        assert all(h.canonical_sha(actual[name]) == h.canonical_sha(declared[name]) for name in controls)
        if role == 'non_saturating':
            parent_recipe = actual
            changed = []
        else:
            changed = [name for name in actual if h.canonical_sha(actual[name]) != h.canonical_sha(parent_recipe[name])]
            assert changed == ['energy_distance_weight']
        assert actual['energy_distance_weight'] == (1. if role == 'energy_u' else 0.)
        assert actual['loss'] == parent_recipe['loss'] == 'non_saturating'
        assert tuple(actual['loss_labels']) == (0., 1., 1.) and actual['reg_arm'] == 'a_r1r2'
        assert actual['reg_coeff'] == .1 and actual['reg_every'] == 1 and actual['reg_coeff_end'] is None
        # Exact new control optimizer/penalty laws; only the raw energy objective differs.
        groups = predecessor.optimizer_bindings(trainer, actual, 'ams_m110')
        record = trainer['optimizers'][1]['regularizer']['record']
        assert record['calls'] == record['observed_steps'] == 1600 and not record['anchor_started']
        assert trainer['penalty_options'] == {}
        stream_hashes, streams = checkpoint['named_stream_state_sha256'], state['streams']['states']
        assert set(streams) == set(stream_hashes) == set(checkpoint['named_stream_keys'])
        assert all(state_digest(streams[name]) == digest for name, digest in stream_hashes.items())
        comparison = raw['evidence']['comparison_protocol']
        assert h.canonical_sha(comparison) == h.canonical_sha(published['comparison_protocol'])
        collection.matched_ordinary_comparisons(parent['comparison_protocol'], comparison)
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
        ring = dict(role=role, study=card['study'], candidate_id=card['candidate'],
            candidate_revision=admitted['candidate_revision'], source_digest=source, source_origin_commit=execution,
            runtime_canonical_sha256=RUNTIME, attempt_id=published['attempt_id'], attempt_archive_path=str(attempt),
            fresh_control_role='non_saturating', actual_task_bound_recipe_changed_fields=changed,
            actual_task_bound_recipe_canonical_sha256=h.canonical_sha(actual), recipe_controls={name: declared[name] for name in controls},
            actual_optimizer_groups=groups, actual_energy_binding=energy, full_frozen_source_manifest_verified=True,
            score_law_binding=dict(loss='non_saturating', penalty_arm='a_r1r2', gamma=.1, every=1, energy_distance_weight=actual['energy_distance_weight'],
                loss_source_sha256=request['source']['files']['particlegan/gan_loss.py'],
                trainer_source_sha256=request['source']['files']['particlegan/training.py'],
                native_reductions_unscaled=True, historical_paired_force_values_available=False),
            task_evaluation_sources_sha256=task['evaluation']['sources'], protocol_seed=0,
            protocol_canonical_sha256=admitted['protocol_sha256'], checkpoint_state_sha256=checkpoint['state_sha256'],
            artifacts_sha256={name: h.sha(attempt / name) for name in ('request.json', 'raw-result.json', 'graded-result.json',
                             'observed-samples.pt', 'provenance/provenance-state.pt')},
            named_stream_state_sha256=stream_hashes, all_named_stream_checkpoints_verified=True,
            captured_initial_tensor_and_target_batch_hashes_match_own_fresh_control=True,
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
        rings.append(ring)
    assert all(h.sha(path) == input_hashes[name] for name, path in inputs.items())
    return dict(schema_version=1, evidence_type='saved_observation_diagnosis_not_new_experiment',
        scope='Both fresh native NS AMS recipes; 96 original saved Ring panels each, no historical grade reuse.',
        execution_commit=execution, source_digest=source, runtime_sha256=RUNTIME,
        archive_root=registration['archive'], publication_sha256=input_hashes['publication'],
        manifest_sha256=input_hashes['manifest'], registration_sha256=input_hashes['registration'],
        imported_helpers_sha256=PINS, imported_frozen_density_source_sha256=predecessor.HELPER_SHA,
        inputs_sha256=input_hashes, previous_publication_sha256=input_hashes['previous_publication'],
        recipe_count=2, recomputed_saved_observations=192, exact_cached_cases=0,
        new_optimizer_updates=0, new_sampling_draws=0, recomputed_metrics=list(h.METRICS), rings=rings,
        reference_draws_avoided='Gate-only equations exclude reference/global-moment/SW1 branches.',
        initial_tensor_proof_scope='Captured initial tensors and ordered target batches match the new-source control; no reconstruction. Recorded device labels retained; CUDA worker ordinals normalized for metadata comparisons.',
        historical_proof_limit='Old-source grades supply no fresh cell. No historical score pairs, force decomposition or sampled-row ownership is reconstructed. '
            'Beta1=.5 G/D moments average history; only beta1=0 prior moments recover terminal gradients. '
            'Functional first-order Jacobian geometry is not an exact additive covariance partition.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, default=OUT / 'publication/results.json')
    parser.add_argument('--execution-commit', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    predecessor, guard = imported('density_audit.py'), imported('audit_collection.py')
    collection = load_collection()
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with collection.forbid_live_execution(guard), torch.no_grad():
        result = audit(args.publication, args.execution_commit, predecessor, collection)
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0] and before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:] and torch.equal(before[2], after[2])
    result['diagnosis_rng_states_unchanged'] = True
    result['reproduction_source_sha256'] = predecessor.load_helper().sha(Path(__file__))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
