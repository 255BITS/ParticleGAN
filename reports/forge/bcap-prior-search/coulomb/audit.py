"""Audit final Coulomb search identity and matched conditions from saved receipts.

No model construction, training, sampling, or archive qualification reuse.
Run only after the immutable ordinary round has drained and been published.
"""
from dataclasses import asdict
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.forge.api import resolve_public_recipe
from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
from experiments.forge.knowledge import _attempts
from experiments.forge.state import state_digest
import torch

DIRECTORY = Path(__file__).resolve().parent
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/coulomb')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(path):
    spec = importlib.util.spec_from_file_location('coulomb_saved_workflow', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def recipe(values):
    return asdict(resolve_public_recipe(dict(recipe_overrides=values)))


def checkpoint_streams(row):
    descriptor = row.get('evidence', {}).get('provenance_checkpoint')
    if not descriptor:
        return None
    path = Path(descriptor['artifact_root']) / descriptor['path']
    require(file_hash(path) == descriptor['sha256'], row['task_id'] + ': checkpoint bytes differ')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(state_digest(saved) == descriptor['state_sha256'], row['task_id'] + ': complete saved state differs')
    streams = saved.get('streams', {})
    bindings, states = streams.get('manifest', {}).get('bindings', {}), streams.get('states', {})
    require(bindings and set(bindings) == set(states),
            row['task_id'] + ': missing named checkpoint stream states')
    non_eval = {key: state_digest(value) for key, value in states.items()
                if bindings.get(key, {}).get('family') != 'eval'}
    return dict(checkpoint_sha256=descriptor['sha256'], state_sha256=descriptor['state_sha256'],
                named_manifest_sha256=stable_hash(streams.get('manifest')),
                non_eval_stream_sha256=non_eval,
                fixture_data_generator_sha256=state_digest(saved['fixture']['data_generator'])
                    if 'fixture' in saved and 'data_generator' in saved['fixture'] else None)


def terminal_witnesses(workflow, row, card):
    """Compact bound witnesses from the final required saved checks; no regrade."""
    evaluation = card['evaluation']
    if evaluation['kind'] != 'transfer_sustained' or row['gate_status'] != 'FAIL':
        return None
    points = row.get('evidence', {}).get('observations', [])
    count = evaluation['minimum_stable_checks']
    window = points[-count:]
    require(len(window) == count, row['task_id'] + ': missing required saved terminal window')
    bounds = {}
    for name, op, target in evaluation['thresholds']:
        bounds.setdefault(name, []).append((op, target))

    def values(point):
        return point.get('metrics', point)

    failed = [point for point in window if any(workflow.measurement(name,
        values(point).get(name), limits)['status'] != 'PASS' for name, limits in bounds.items())]
    metrics = []
    for name, limits in bounds.items():
        def badness(point):
            score = values(point).get(name)
            if score is None or not math.isfinite(float(score)):
                return math.inf
            return max(((target - score) if op in ('>=', '>') else
                        (score - target) if op in ('<=', '<') else abs(score - target))
                       / (abs(target) or 1) for op, target in limits)
        point = max(window, key=badness)
        metric = workflow.measurement(name, values(point).get(name), limits)
        metric.update(witness_step=point['step'], original_observation_sha256=stable_hash(point))
        metrics.append(metric)
    return dict(original_observations_sha256=stable_hash(points),
        terminal_window_sha256=stable_hash(window), required_checks=count,
        terminal_steps=[point['step'] for point in window], failed_checks=len(failed),
        recorded_passing_suffix=row['evaluator_result']['convergence']['passing_suffix'],
        rows=metrics, qualification_credit=False, original_grade_unchanged=True,
        scope='Frozen bounds applied to saved numerical packets for explanation only; original qualification retained')


def main():
    workflow = module(DIRECTORY.parent / 'workflow.py')
    progress, state = read_json(ARCHIVE / 'progress.json'), read_json(ARCHIVE / 'queue/queue/state.json')
    require(progress['phase'] == 'complete', 'Wait for complete registered round')
    require(not any(e['status'] in ('queued', 'running', 'paused') for e in state['submissions'].values())
            and not any(j['status'] == 'running' for j in state['jobs'].values()), 'Queue is active')
    result, spec = read_json(DIRECTORY / 'results.json'), read_json(DIRECTORY / 'spec.json')
    require(result['source_commit'] == progress['source_commit']
            and result['source_digest'] == progress['source_digest'] == spec['source_digest'], 'Executed identity differs')
    originals, _ = _attempts(ROOT)
    attempts = {a['attempt_id']: a for a in originals}
    entries = [state['submissions'][progress['control_request']]] + [state['submissions'][rid] for rid in progress['search_requests']]
    rows = [workflow.validated_submission_rows(state, e, attempts) for e in entries]
    control = {row['task_id']: row for row in rows[0]}
    records = []
    control_streams = {name: checkpoint_streams(row) for name, row in control.items()}
    for index, (entry, actual_rows) in enumerate(zip(entries, rows)):
        request = entry['request']
        weight = request['candidate']['recipe_overrides']['prior_reg']
        role = 'control' if index == 0 else f'prior_reg={weight}'
        require(request['source']['digest'] == progress['source_digest'], role + ': source differs')
        scientific = {name: digest for name, digest in request['source']['files'].items()
                      if name.startswith(('particlegan/', 'experiments/', 'benchmarks/'))}
        require(all(file_hash(ROOT / name) == digest for name, digest in scientific.items()),
                role + ': frozen scientific implementation bytes differ')
        for key in ('protocol', 'runtime', 'compute_profiles', 'execution_policy'):
            require(request[key] == entries[0]['request'][key], role + ': ' + key + ' differs')
        require(request['protocol']['seed'] == 0, 'Protocol seed changed')
        control_cards = entries[0]['request']['tasks']
        require({name: stable_hash(card) for name, card in control_cards.items()} == spec['task_contracts'],
                'Registered control task contracts differ')
        # Resolved task packets also embed the candidate's two declared recipe
        # values. Permit exactly those metadata substitutions, preserving every
        # other byte of architecture/data/prior/budget/evaluation ownership.
        expected_cards = deepcopy(control_cards)
        for card in expected_cards.values():
            fields = card['field_ownership']['recipe_fields']
            for key, value in dict(prior_reg=weight,
                    prior_regularizer='vicreg' if index == 0 else 'coulomb').items():
                for leaf in ('value', 'reference_recipe_value'):
                    if leaf in fields[key] and (leaf != 'value' or fields[key][leaf] is not None):
                        fields[key][leaf] = value
        require(request['tasks'] == expected_cards, role + ': undeclared task condition/ownership delta')
        global_recipe = recipe(request['candidate']['recipe_overrides'])
        baseline_recipe = recipe(entries[0]['request']['candidate']['recipe_overrides'])
        delta = {k: dict(control=baseline_recipe[k], candidate=global_recipe[k])
                 for k in baseline_recipe if baseline_recipe[k] != global_recipe[k]}
        require(delta == ({} if index == 0 else {'prior_reg': dict(control=0., candidate=weight),
            'prior_regularizer': dict(control='vicreg', candidate='coulomb')}), role + ': undeclared global trainer delta')
        cells = []
        for row in actual_rows:
            task_id = row['task_id']
            require(row['receipt_binding']['valid_receipt'], role + '/' + task_id + ': invalid original certificate')
            actual, before = workflow.compact_row(row), workflow.compact_row(control[task_id])
            card = request['tasks'][task_id]
            initial_equal = actual.get('initialization') == before.get('initialization')
            initial_prior_equal = actual.get('initial_prior') == before.get('initial_prior')
            sampling_equal = actual['sampling'] == before['sampling']
            require(initial_equal and initial_prior_equal and sampling_equal, role + '/' + task_id + ': fixed cohort changed')
            guards = row.get('evidence', {}).get('guards', {})
            clock = task_id == 'clockfree_audit_measurement_v1'
            if not clock:
                require('unintended_rng_deviations' in guards,
                        role + '/' + task_id + ': missing consumed RNG guard')
            require(guards.get('unintended_rng_deviations', 0) == 0, role + '/' + task_id + ': unintended RNG deviation')
            evidence = row.get('evidence', {})
            counts = guards.get('optimizer_updates', {})
            # Hold resumes the arm's own certified earliest acquisition state.
            continuity = evidence.get('continuity') if task_id == 'five_word_joint_hold' else None
            budget = card['execution']['steps']
            if continuity:
                require(continuity['restored_exactly'] and continuity['same_recipe_prior_architecture']
                        and not continuity['history_reset'], role + ': word continuation changed')
                require(evidence['executed_updates'] == budget == 4000, role + ': own-prefix hold allowance changed')
                expected_updates = continuity['prefix_steps'] + budget
                require(all(value == expected_updates for value in counts.values()), role + ': word total updates differ')
            elif card['adapter'] == 'ring_endurance':
                # Retained ring tasks can declare adaptive host execution; use
                # the actual completed host counter if their guard owns it.
                expected_updates = row.get('cost', {}).get('completed_steps', budget)
                require(all(value == expected_updates for value in counts.values()), role + '/' + task_id + ': update counters differ')
            elif not clock:
                require(counts and all(value == budget for value in counts.values()),
                        role + '/' + task_id + ': fixed declared update counters differ')
            actual_recipe = recipe(actual['recipe'])
            applicability = (actual.get('prior_policy') or {}).get('applicability')
            if not clock:
                require(actual_recipe['prior_reg'] == weight, role + '/' + task_id + ': actual regularization coefficient differs')
                require(actual_recipe['prior_regularizer'] == ('vicreg' if index == 0 else 'coulomb'), role + '/' + task_id + ': actual regularizer differs')
                for k, value in dict(prior_coulomb_softening=.1, prior_coulomb_confinement=.35, prior_coulomb_neighbors=8).items():
                    require(actual_recipe[k] == value, role + '/' + task_id + ': geometry setting differs')
            stream_proof = checkpoint_streams(row)
            require(clock or stream_proof, role + '/' + task_id + ': missing saved consumed streams')
            stream_comparison = None
            if stream_proof and control_streams[task_id]:
                old_streams = control_streams[task_id]
                stream_comparison = dict(non_eval_stream_states_equal=
                    stream_proof['non_eval_stream_sha256'] == old_streams['non_eval_stream_sha256'],
                    fixture_data_generator_equal=stream_proof['fixture_data_generator_sha256']
                        == old_streams['fixture_data_generator_sha256'],
                    own_prefix_hold=bool(continuity), optional_clock_probe=clock,
                    same_optimizer_update_counts=counts == before.get('guards', {}).get('optimizer_updates', {}),
                    same_word_prefix=(continuity['prefix_steps'] ==
                        control[task_id]['evidence']['continuity']['prefix_steps']) if continuity else None)
                # Different certified word prefixes and adaptive ring execution
                # are retained as protocol exceptions, never common-clock claims.
                stream_comparison['comparison_scope'] = ('own-prefix word continuation'
                    if continuity else 'adaptive ring host' if card['adapter'] == 'ring_endurance'
                    else 'optional smaller clock probe' if clock else 'same declared task budget')
                if not clock and card['adapter'] != 'ring_endurance' and (
                        not continuity or stream_comparison['same_word_prefix']):
                    require(stream_comparison['non_eval_stream_states_equal']
                            and stream_comparison['fixture_data_generator_equal'],
                            role + '/' + task_id + ': fixed-budget consumed data/prior streams differ')
            cells.append(dict(task_id=task_id, status=row['gate_status'], attempt_id=row['receipt_binding']['attempt_id'],
                valid_original_receipt=True, initial_tensor_receipts_equal=initial_equal,
                initial_prior_equal=initial_prior_equal, sampling_equal=sampling_equal,
                initial_tensor_receipts_sha256=stable_hash(actual.get('initialization')),
                actual_recipe_sha256=stable_hash(actual['recipe']), initial_prior_sha256=stable_hash(actual.get('initial_prior')),
                sampling_sha256=stable_hash(actual['sampling']), prior_policy_sha256=stable_hash(actual.get('prior_policy')),
                optimizer_updates=counts, declared_updates=budget,
                own_prefix_continuation=continuity, optional_clock_diagnostic=clock,
                feature_invariance_claim=False if clock else None,
                prior_applicability=applicability, checkpoint_streams=stream_proof,
                matched_consumed_streams=stream_comparison,
                required_terminal_window_witnesses=terminal_witnesses(workflow, row, card)))
        records.append(dict(role=role, weight=weight, request_id=request['request_id'],
            actual_global_recipe_sha256=stable_hash(global_recipe), trainer_delta=delta,
            task_bindings_sha256=stable_hash(spec['task_contracts']),
            resolved_task_bindings_sha256=stable_hash({name: stable_hash(card)
                for name, card in request['tasks'].items()}),
            task_metadata_delta_scope='Only declared prior_reg/prior_regularizer values in field_ownership.recipe_fields; all other task bytes equal',
            verified_scientific_files=len(scientific), scientific_manifest_sha256=stable_hash(scientific), cells=cells))
    archived_path = ROOT / 'reports/forge/recipe-prior-refactor/publication/results.json'
    archive = read_json(archived_path)
    archived = {r['task_id']: r for r in archive['certified_results'] if r['family'] == 'bcap-pure' and r['scope'] == 'ordinary'}
    require(set(control) == set(archived), 'Fresh/archived baseline denominators differ')
    baseline_comparisons = []
    for task_id, old in sorted(archived.items()):
        fresh = control[task_id]
        metrics = fresh.get('metrics', {})
        changed = [name for name, score in old['metrics'].items() if name not in metrics or metrics[name] != score]
        require(old['gate_status'] == fresh['gate_status'] and not changed, task_id + ': archived baseline grade/metric values differ')
        baseline_comparisons.append(dict(task_id=task_id, gate_equal=True, archived_metric_values_equal=True,
            exact_metric_dictionary_equal=old['metrics'] == metrics,
            additional_metrics=sorted(set(metrics) - set(old['metrics'])),
            fresh_attempt_id=fresh['receipt_binding']['attempt_id'], archived_attempt_id=old['attempt_id']))
    receipt = dict(schema_version=1, status='PASS', scope='final_saved_coulomb_recipe_and_cohort_audit',
        source_commit=progress['source_commit'], source_digest=progress['source_digest'],
        result_sha256=file_hash(DIRECTORY / 'results.json'), spec_sha256=file_hash(DIRECTORY / 'spec.json'),
        records=records, baseline_reproduction=dict(archived_path=str(archived_path.relative_to(ROOT)),
            archived_sha256=file_hash(archived_path), archived_source_digest=archive['source_digest'],
            archived_evidence_reused=False, comparisons=baseline_comparisons,
            exact_gate_matches=len(baseline_comparisons),
            exact_archived_metric_value_matches=len(baseline_comparisons),
            exact_metric_dictionary_matches=sum(r['exact_metric_dictionary_equal'] for r in baseline_comparisons)),
        optimizer_updates_added=0, sampling_draws_added=0, qualification_input=False,
        data_sequence_scope='Identical declared task/data laws and initial tensor receipts; verified final consumed non-evaluation stream states at each fixed common budget and each matched word prefix. Original isolated/checkpointed RNG proof and complete budgets retained. Own-prefix word holds can consume different post-acquisition windows (.01 does); adaptive ring host counters remain explicit. Per-stream draw counters are not instrumented; no common-clock claim for different hold prefixes.',
        script_sha256=file_hash(Path(__file__)), workflow_sha256=file_hash(DIRECTORY.parent / 'workflow.py'))
    atomic_json(DIRECTORY / 'audit.json', receipt)
    print(json.dumps(dict(status='PASS', certified_rows=sum(len(r['cells']) for r in records), baseline_gates=len(baseline_comparisons))))


if __name__ == '__main__':
    main()
