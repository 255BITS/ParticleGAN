"""Audit two new spectral-repair Ring16 runs from frozen saved evidence only.

Requires the completed numerical publication. Reuses the hashed five-track gate
and functional geometry equations; constructs no models and calls no RNG draws.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import random

import torch

from experiments.forge.state import state_digest
from particlegan.recipes import Recipe

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/forge/r1r2-spectral-repair'
HELPER = ROOT / 'reports/forge/r1r2-five-track-repair/density_readout.py'
HELPER_SHA = '82db99554cabc14c42ba564af6a0855af68193e650e279935de2473e2d774387'
SOURCE = '25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507'
RUNTIME = '640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc'
NEW_ROLES = ('spectral', 'critic_spectral')
EXTRA_CONTROLS = ('optimizer_family', 'adam_variant', 'eps', 'd_eps', 'prior_eps',
                  'optimizer_momentum', 'optimizer_adam_lr', 'optimizer_smoothing',
                  'optimizer_convolution', 'optimizer_svd_backend')
RING = 'ring16_acquisition'


def load_helper():
    import hashlib
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location('frozen_r1r2_density', HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def optimizer_bindings(trainer, recipe, role):
    """Inspect saved native/normalized state without constructing optimizers."""
    family = {'spectral': 'dualnorm', 'critic_spectral': 'dualnorm_D_only', 'adam': 'adam'}[role]
    assert recipe['optimizer_family'] == family
    assert recipe['eps'] == 1e-8 and recipe['prior_eps'] is None
    assert recipe['d_eps'] is None
    assert tuple(recipe['betas']) == (0., .99)
    assert recipe['prior_betas'] is None and recipe['d_betas'] is None
    assert recipe['lr'] == .0085 and recipe['d_lr_mult'] == recipe['prior_lr_mult'] == 1
    assert recipe['amsgrad'] is False
    assert recipe['optimizer_momentum'] == recipe['optimizer_smoothing'] == 0
    assert recipe['optimizer_adam_lr'] is None
    assert recipe['optimizer_svd_backend'] == 'native' and recipe['optimizer_convolution'] == 'none'
    optimizers = trainer['optimizers']
    assert len(optimizers) == 2
    groups = optimizers[0]['param_groups'] + optimizers[1]['param_groups']
    assert len(optimizers[0]['param_groups']) == 2 and len(optimizers[1]['param_groups']) == 1
    roles, rows = ('G', 'prior', 'D'), []
    algorithms = {'spectral': ('dualnorm', 'rownorm', 'dualnorm'),
                  'critic_spectral': ('adam', 'adam', 'dualnorm'),
                  'adam': ('adam', 'adam', 'adam')}[role]
    for optimizer_index, optimizer in enumerate(optimizers):
        if family != 'adam':
            metadata = optimizer['dualnorm']
            assert metadata == dict(schema=1, family=family, momentum=0.,
                                    sampled_rows=[None] * len(optimizer['param_groups']))
        else:
            assert 'dualnorm' not in optimizer
        if optimizer_index == 1:
            observer = optimizer['regularizer']
            assert observer['ema'] is None and observer['guard'] is None
            assert observer['record']['observed_steps'] == 1600
            if family != 'adam':
                assert observer['optimizer_family'] == family
        owned = {parameter for group in optimizer['param_groups'] for parameter in group['params']}
        assert set(optimizer['state']) == owned
        for group in optimizer['param_groups']:
            algorithm = group.get('algorithm', 'adam')
            for parameter in group['params']:
                parameter_state = optimizer['state'][parameter]
                assert int(parameter_state['step']) == 1600
                required = {'step', 'exp_avg', 'exp_avg_sq'} if algorithm == 'adam' else {'step'}
                assert set(parameter_state) == required
                for key in required - {'step'}:
                    assert bool(torch.isfinite(parameter_state[key]).all())
                if algorithm == 'adam':
                    assert bool((parameter_state['exp_avg_sq'] >= 0).all())
    for name, group, algorithm, normalized_role in zip(
            roles, groups, algorithms, ('generator', 'prior', 'critic'), strict=True):
        assert tuple(group['betas']) == (0., .99) and group['lr'] == .0085
        assert group['eps'] == 1e-8 and group['amsgrad'] is False
        if family != 'adam':
            assert group['algorithm'] == algorithm and group['role'] == normalized_role
        else:
            assert group.get('algorithm', 'adam') == 'adam'
        if algorithm == 'rownorm':
            assert name == 'prior' and len(group['params']) == 1
            assert group['sampled_rows_required'] is True
        rows.append(dict(role=name, optimizer_family=family, algorithm=group.get('algorithm', 'adam'),
                         normalized_role=group.get('role'), betas=list(group['betas']),
                         lr=group['lr'], eps=group['eps'], amsgrad=group['amsgrad'],
                         saved_parameter_count=len(group['params']), all_saved_parameter_steps=1600,
                         first_moment_saved=algorithm == 'adam', second_moment_saved=algorithm == 'adam',
                         betas_consumed=algorithm == 'adam',
                         actual_sampled_rows_required=group.get('sampled_rows_required', False)))
    return rows


def cached_parent_geometry(published, h):
    """Functional endpoint geometry; retain the parent's original gate results."""
    attempt = Path(published['checkpoint']['artifact_root']).parent
    request = h.read(attempt/'request.json')['request']
    raw = h.read(attempt/'raw-result.json')
    graded = h.read(attempt/'graded-result.json')
    checkpoint = raw['evidence']['provenance_checkpoint']
    state_path = attempt/'provenance/provenance-state.pt'
    assert published['candidate_id'] == request['candidate']['id']
    assert published['candidate_revision'] == request['candidate_revision']
    assert published['source_digest'] == request['source']['digest'] == graded['source_digest'] == SOURCE
    assert h.canonical_sha(request['runtime']) == RUNTIME
    assert request['protocol']['seed'] == request['rng']['seed'] == 0
    assert graded['raw_hash'] == h.canonical_sha(raw)
    assert all(h.sha(ROOT/path) == expected for path, expected in request['source']['files'].items())
    assert h.sha(state_path) == checkpoint['sha256'] == published['checkpoint']['sha256']
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    assert state_digest(state) == checkpoint['state_sha256'] == published['checkpoint']['state_sha256']
    assert h.canonical_sha(raw['evidence']['comparison_protocol']) == h.canonical_sha(published['comparison_protocol'])
    assert all(state_digest(state['streams']['states'][key]) == expected
               for key, expected in checkpoint['named_stream_state_sha256'].items())
    trainer = state['trainer']
    assert trainer['completed_steps'] == 1600
    recipe = asdict(Recipe(**trainer['recipe']))
    groups = optimizer_bindings(trainer, recipe, 'adam')
    panels = torch.load(attempt/'observed-samples.pt', map_location='cpu', weights_only=True)
    observations = raw['evidence']['observations']
    assert len(panels) == len(observations) == 96
    assert panels[-1]['step'] == observations[-1]['step'] == 1600
    for key in h.METRICS:
        assert observations[-1][key] == graded['grades'][RING]['metrics'][key] == published['metrics'][key]
    spec = request['tasks'][RING]['execution']['host_definition']
    prior = request['tasks'][RING]['execution']['prior']
    shape = h.geometry(panels[-1]['samples'], trainer['models'], spec, prior)
    convergence = graded['grades'][RING]['evaluator_result']['convergence']
    finite_rows = [row for row in observations if h.finite(row['component_covariance_error'])]
    return dict(role='adam', candidate_id=published['candidate_id'], candidate_revision=published['candidate_revision'],
        source_digest=SOURCE, source_origin_commit=request['source']['origin_commit'], runtime_sha256=RUNTIME,
        attempt_id=published['attempt_id'], attempt_archive_path=str(attempt),
        artifacts_sha256={name: h.sha(attempt/name) for name in ('request.json', 'raw-result.json',
            'graded-result.json', 'observed-samples.pt', 'provenance/provenance-state.pt')},
        checkpoint_state_sha256=checkpoint['state_sha256'], actual_optimizer_groups=groups,
        endpoint=h.selected(observations[-1]),
        best_covariance_observation=h.selected(min(finite_rows, key=lambda row: row['component_covariance_error'])),
        original_passing_observations=convergence['passing_observations'],
        original_terminal_passing_suffix=convergence['passing_suffix'],
        gate_metrics_recomputed=0, additional_saved_endpoint_geometry_evaluations=1,
        endpoint_geometry=shape)


def audit(publication_path, execution):
    h = load_helper()
    publication = h.read(publication_path)
    registration = h.read(OUT/'registration.json')
    manifest = h.read(OUT/'manifest.json')
    assert publication['complete'] and publication['source_commit'] == execution
    assert publication['source_digest'] == SOURCE
    assert publication['registration_sha256'] == h.sha(OUT/'registration.json')
    assert publication['optimizer_updates_added_by_collection'] == publication['sampling_draws_added_by_collection'] == 0
    records = publication['certified_results']
    baseline = next(row for row in records if row['role'] == 'adam' and row['task_id'] == RING)
    parent_geometry = cached_parent_geometry(baseline, h)
    controls = h.CONTROLS + EXTRA_CONTROLS
    rings = []
    for role in NEW_ROLES:
        selected = [row for row in records if row['role'] == role and row['task_id'] == RING]
        assert len(selected) == 1
        published = selected[0]
        card = next(card for card in manifest['candidates'] if card['role'] == role)
        admitted = registration['candidates'][role]
        assert published['candidate_id'] == admitted['candidate'] == card['candidate']
        assert published['candidate_revision'] == admitted['candidate_revision']
        assert published['source_digest'] == admitted['source_digest'] == SOURCE
        assert admitted['runtime_sha256'] == RUNTIME
        assert published['compatibility_key'] == admitted['task_keys'][RING]
        attempt = Path(published['checkpoint']['artifact_root']).parent
        assert attempt.name == published['attempt_id']
        request = h.read(attempt/'request.json')['request']
        raw = h.read(attempt/'raw-result.json')
        graded = h.read(attempt/'graded-result.json')
        grade = graded['grades'][RING]
        assert graded['raw_hash'] == h.canonical_sha(raw)
        assert request['candidate']['id'] == card['candidate']
        assert request['candidate_revision'] == admitted['candidate_revision']
        assert request['source']['origin_commit'] == execution
        assert request['source']['digest'] == graded['source_digest'] == SOURCE
        assert all(h.sha(ROOT/path) == expected for path, expected in request['source']['files'].items())
        assert h.canonical_sha(request['runtime']) == RUNTIME
        assert request['protocol']['seed'] == request['rng']['seed'] == 0
        assert h.canonical_sha(request['protocol']) == admitted['protocol_sha256']
        task = request['tasks'][RING]
        spec, prior = task['execution']['host_definition'], task['execution']['prior']
        assert (spec['layers'], spec['hidden'], spec['particles'], spec['z_dim']) == (2, 64, 256, 4)
        assert prior == dict(kind='mog', sigma=.1, standardize=False)
        assert task['execution']['steps'] == 1600
        thresholds = task['evaluation']['thresholds']
        assert thresholds == [['sample_count', '>=', 4096], ['modes', '>=', 16],
                              ['mass_tv', '<=', .15], ['hq', '>=', .85],
                              ['component_covariance_error', '<=', .85],
                              ['component_min_eigen_ratio', '>=', .15]]
        assert all(h.sha(ROOT/path) == expected for path, expected in task['evaluation']['sources'].items())
        checkpoint = raw['evidence']['provenance_checkpoint']
        state_path = attempt/'provenance/provenance-state.pt'
        assert h.sha(state_path) == checkpoint['sha256'] == published['checkpoint']['sha256']
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        assert state_digest(state) == checkpoint['state_sha256'] == published['checkpoint']['state_sha256']
        trainer = state['trainer']
        assert trainer['completed_steps'] == checkpoint['completed_steps'] == 1600
        # The checkpoint stores a compact public recipe; Recipe fills only its
        # documented defaults. Task-owned resource bindings stay in that recipe.
        actual_recipe = asdict(Recipe(**trainer['recipe']))
        assert h.canonical_sha(trainer['recipe']) == h.canonical_sha(raw['recipe'])
        declared_recipe = request['candidate']['resolved_recipe']
        for key in controls:
            assert h.canonical_sha(actual_recipe[key]) == h.canonical_sha(declared_recipe[key]), (role, key)
        optimizer_groups = optimizer_bindings(trainer, actual_recipe, role)
        stream_receipt = checkpoint['named_stream_state_sha256']
        states = state['streams']['states']
        assert set(states) == set(stream_receipt) == set(checkpoint['named_stream_keys'])
        assert all(state_digest(states[key]) == expected for key, expected in stream_receipt.items())
        comparison = raw['evidence']['comparison_protocol']
        assert h.canonical_sha(comparison) == h.canonical_sha(published['comparison_protocol'])
        for field in ('initial_tensors_sha256', 'initial_tensor_metadata', 'actual_target_batches'):
            assert h.canonical_sha(comparison[field]) == h.canonical_sha(baseline['comparison_protocol'][field]), (role, field)
        assert raw['evidence']['guards']['unintended_rng_deviations'] == 0
        saved = torch.load(attempt/'observed-samples.pt', map_location='cpu', weights_only=True)
        original = raw['evidence']['observations']
        assert len(saved) == len(original) == 96
        recreated, worst = [], {key: 0. for key in h.METRICS}
        for panel, row in zip(saved, original, strict=True):
            assert panel['step'] == row['step']
            metrics = dict(step=panel['step'], **h.gate_metrics(panel['samples'], spec))
            for key in h.METRICS:
                left, right = metrics[key], row[key]
                if left is None or right is None:
                    assert left is right
                else:
                    assert math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6), (role, panel['step'], key)
                    worst[key] = max(worst[key], abs(left-right))
            assert h.passing(metrics, thresholds) == h.passing(row, thresholds)
            recreated.append(metrics)
        flags = [h.passing(row, thresholds) for row in recreated]
        suffix = next((index for index, value in enumerate(reversed(flags)) if not value), len(flags))
        convergence = grade['evaluator_result']['convergence']
        assert sum(flags) == convergence['passing_observations']
        assert suffix == convergence['passing_suffix']
        assert convergence['minimum_stable_checks'] == 5
        assert recreated[-1]['step'] == grade['metrics']['step'] == 1600
        assert grade['gate_status'] == published['gate_status'] == ('PASS' if suffix >= 5 else 'FAIL')
        for key in h.METRICS:
            assert recreated[-1][key] == grade['metrics'][key] == published['metrics'][key]
        shape = h.geometry(saved[-1]['samples'], trainer['models'], spec, prior)
        if shape['status'] != 'UNAVAILABLE':
            assert math.isclose(shape['reconstructed_float64_full_component_covariance_error'],
                                recreated[-1]['component_covariance_error'], rel_tol=1e-5, abs_tol=1e-5)
        finite_rows = [row for row in recreated if h.finite(row['component_covariance_error'])]
        rings.append(dict(role=role, study=card['study'], candidate_id=card['candidate'],
            candidate_revision=admitted['candidate_revision'], source_digest=SOURCE,
            source_origin_commit=execution, runtime_canonical_sha256=RUNTIME,
            resolved_recipe_canonical_sha256=h.canonical_sha(declared_recipe),
            actual_task_bound_recipe_canonical_sha256=h.canonical_sha(actual_recipe),
            recipe_controls={key: declared_recipe[key] for key in controls},
            actual_optimizer_groups=optimizer_groups,
            full_frozen_source_manifest_verified=True,
            optimizer_implementation_sources_sha256={path: request['source']['files'][path] for path in
                ('particlegan/recipes.py', 'particlegan/optim/dualnorm.py', 'particlegan/recipe_schedules.py',
                 'particlegan/training.py', 'particlegan/policy.py')},
            task_evaluation_sources_sha256=task['evaluation']['sources'],
            attempt_archive_path=str(attempt), attempt_id=published['attempt_id'],
            artifacts_sha256={name: h.sha(attempt/name) for name in ('request.json', 'raw-result.json',
                'graded-result.json', 'observed-samples.pt', 'provenance/provenance-state.pt')},
            checkpoint_state_sha256=checkpoint['state_sha256'],
            protocol_seed=0, protocol_canonical_sha256=admitted['protocol_sha256'],
            named_stream_state_sha256=stream_receipt, all_named_stream_checkpoints_verified=True,
            captured_initial_tensor_and_target_batch_hashes_match_cached_adam=True,
            comparison_protocol_canonical_sha256=h.canonical_sha(comparison),
            thresholds=thresholds, gate_status=grade['gate_status'], recomputed_observations=96,
            max_absolute_gate_metric_difference=worst, full_passing_observations=sum(flags),
            full_terminal_passing_suffix=suffix, minimum_stable_checks=5,
            full_passing_windows=h.stable_windows(flags, recreated),
            covariance_bound_passing_observations=sum(h.compare(row['component_covariance_error'], '<=', .85) for row in recreated),
            failures_by_bound={key: sum(not h.compare(row[key], op, bound) for row in recreated) for key, op, bound in thresholds},
            endpoint_failed_bounds=[key for key, op, bound in thresholds if not h.compare(recreated[-1][key], op, bound)],
            endpoint=h.selected(recreated[-1]),
            best_covariance_observation=h.selected(min(finite_rows, key=lambda row: row['component_covariance_error'])) if finite_rows else None,
            milestones=[h.selected(row) for row in recreated if row['step'] in (400, 800, 1200, 1600)],
            last_five_observations=[h.selected(row) for row in recreated[-5:]], endpoint_geometry=shape))
    assert len(rings) == 2
    return dict(schema_version=1, evidence_type='saved_observation_diagnosis_not_new_experiment',
        scope='Two new whole global full/critic-only DualNorm recipes; all 96 saved Ring16 panels and actual endpoint geometry per recipe.',
        execution_commit=execution, source_digest=SOURCE, runtime_sha256=RUNTIME,
        archive_root=registration['archive'], publication_sha256=h.sha(publication_path),
        manifest_sha256=h.sha(OUT/'manifest.json'), registration_sha256=h.sha(OUT/'registration.json'),
        imported_frozen_density_source_sha256=HELPER_SHA, recipe_count=2,
        recomputed_saved_observations=192, new_optimizer_updates=0, new_sampling_draws=0,
        recomputed_metrics=list(h.METRICS),
        reference_draws_avoided='Gate-only equations exclude reference/global-moment/SW1 branches.',
        initial_tensor_proof_scope='Verifies captured actual-tensor and target-batch receipts against publication and cached Adam; does not reconstruct initialization.',
        family_proof_scope='Checks full DualNorm G/D polar and sampled-prior row law versus D-only polar/native-Adam isolation through actual saved family/role metadata, inactive or retained moments and all parameter steps.',
        prior_row_proof_limit='Full DualNorm requires actual drawn rows and uses the unchanged hashed public training/policy ownership hooks. Optimizer step clears sampled IDs; endpoint checkpoints cannot independently reconstruct each historical row set. No gradient-support substitution is made.',
        cached_parent=parent_geometry,
        rings=rings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path, default=OUT/'publication/results.json')
    parser.add_argument('--execution-commit', required=True, help='Independently supplied admitted execution SHA.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    python_before, torch_before = random.getstate(), torch.get_rng_state().clone()
    result = audit(args.publication, args.execution_commit)
    assert python_before == random.getstate() and torch.equal(torch_before, torch.get_rng_state())
    result['diagnosis_rng_states_unchanged'] = True
    result['reproduction_source_sha256'] = load_helper().sha(Path(__file__))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
