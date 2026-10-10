#!/usr/bin/env python3
"""Saved-only publication supplement for the frozen direction-only pair.

Run after root-owned admission/drain reaches diagnostic_complete. No model
construction, restoration, sampling, training, scorer, or worker calls occur.
Initial/data/RNG/own-producer proofs remain the main publication audit's job.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/home/martyn/dev/ParticleGAN-bcap-projection-baseline-review')
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/baseline-repairs/projection')
FROZEN = '442dc7ef52658a727cad6014e2e2d5fbcb1787a5'
REGISTRATION_SHA = '91e303bf4422f80cd07cc94d8bec021bcdea0c3ee6c80d0102f514b830e1ad1b'
DIGEST = 'd944540c70e280d8f981367368792920a70bbec0d919d681b4bdf0770a96f59f'
INACTIVE_SCOPE = {
    'gaussian1d_smoke', 'gaussian1d_stability', 'ring16_acquisition',
    'five_word_joint_smoke', 'five_word_joint_hold', 'vector_unequal_mass',
    'vector_unequal_width', 'vector_anisotropic', 'grid100', 'rotated100', 'staggered100',
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scalar(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else None
    return value if value is None or type(value) in (str, bool, int, float) else None


def optimizer_records(saved):
    """Keep ownership paths/roles, raw counters and clock coverage explicit."""
    records = []

    def visit(value, path='state', supplied_roles=None):
        if isinstance(value, dict):
            if isinstance(value.get('param_groups'), list) and isinstance(value.get('state'), dict):
                groups = []
                for i, group in enumerate(value['param_groups']):
                    role = group.get('role')
                    if role is None and isinstance(supplied_roles, (list, tuple)) and i < len(supplied_roles):
                        role = supplied_roles[i]
                    groups.append(dict(index=i, role=role, parameter_ids=group.get('params', []),
                        algorithm=group.get('algorithm'), sampled_rows_required=group.get('sampled_rows_required')))
                counts = [scalar(row['step']) for row in value['state'].values()
                          if isinstance(row, dict) and 'step' in row]
                counts = sorted(set(c for c in counts if type(c) in (int, float)))
                wrappers = {name: dict(schema=value[name].get('schema'), mode=value[name].get('mode'),
                    stats=value[name].get('stats'), pending_present=value[name].get('pending') is not None)
                    for name in ('projection', 'constraint_geometry', 'direction_blend', 'strict_progress')
                    if isinstance(value.get(name), dict)}
                geometry = wrappers.get('constraint_geometry', {}).get('stats') or {}
                blend = wrappers.get('direction_blend', {}).get('stats') or {}
                denominator = blend.get('conflict_steps', 0)
                records.append(dict(owner_path=path, groups=groups, base_step_counts=counts,
                    geometry_steps_equal_max_base_clock=bool(counts) and geometry.get('steps') == max(counts),
                    dualnorm={k: v for k, v in value.get('dualnorm', {}).items() if k != 'sampled_rows'},
                    wrapper_metadata=wrappers, top_level_keys=sorted(value),
                    retained_norm_ratio_mean_on_conflicts=(blend.get('retained_norm_ratio_sum', 0) / denominator
                                                          if denominator else None)))
                return
            for key, child in value.items():
                if key in {'models', 'averages', 'role_parameters', 'streams', 'initialization'}:
                    continue
                roles = value.get('roles') if key == 'optimizers' else None
                visit(child, f'{path}.{key}', roles)
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                roles = supplied_roles[i] if isinstance(supplied_roles, (list, tuple)) and i < len(supplied_roles) else None
                visit(child, f'{path}[{i}]', roles)

    visit(saved)
    return records


def actual_tensors(saved):
    """Live public model/prior states plus saved averages and external policy tables.

    Optimizer/stream tensors are deliberately excluded. Policy.table is included
    because some learned prior tables are owned outside the prior module.
    """
    import torch
    tensors, containers = {}, []

    def flatten(value, path):
        if isinstance(value, torch.Tensor):
            tensors[path] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                flatten(child, f'{path}.{key}')
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                flatten(child, f'{path}[{i}]')

    def visit(value, path='state'):
        if isinstance(value, dict):
            if isinstance(value.get('models'), dict):
                containers.append(dict(path=f'{path}.models', names=sorted(value['models'])))
                flatten(value['models'], f'{path}.models')
                for key in ('averages', 'table', 'averaged_table', 'output_noise'):
                    if key in value:
                        flatten(value[key], f'{path}.{key}')
            for key, child in value.items():
                if key not in {'models', 'averages', 'optimizers', 'streams', 'initialization', 'role_parameters'}:
                    visit(child, f'{path}.{key}')
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                visit(child, f'{path}[{i}]')

    visit(saved)
    names = {name for container in containers for name in container['names']}
    complete = bool(tensors) and bool(names & {'G', 'generator'}) and bool(names & {'D', 'critic'}) and 'prior' in names
    return tensors, containers, complete


def task_delta_proof(baseline, candidate, stable_hash):
    """Validate the exact declared ownership delta; retain original identities."""
    a, b = deepcopy(baseline['task']), deepcopy(candidate['task'])
    path = ('field_ownership', 'recipe_fields', 'constraint_geometry_mode')
    am, bm = a, b
    for key in path[:-1]:
        am, bm = am[key], bm[key]
    before, after = am.pop(path[-1]), bm.pop(path[-1])
    expected_before = dict(owner='technique', status='effective', source='public Recipe preset bcap', value='none')
    expected_after = dict(owner='technique', status='effective', source='candidate.recipe_overrides.constraint_geometry_mode', value='direction_blend')
    return dict(task_id=baseline['item']['task_id'],
        exact_task_declarations_equal_except_registered_ownership_delta=(a == b
            and before == expected_before and after == expected_after),
        baseline_task_declaration_sha256=stable_hash(baseline['task']),
        candidate_task_declaration_sha256=stable_hash(candidate['task']),
        declared_delta=dict(path='.'.join(path), baseline=before, candidate=after))


def compare_pair(task, baseline, candidate, state_digest):
    import torch
    from experiments.forge.contracts import stable_hash
    result = dict(task_id=task, status='UNVERIFIED', exact_tensor_equality=None)
    if not baseline or not candidate:
        return dict(result, reason='A paired certified final checkpoint is unavailable.')
    for entry in (baseline, candidate):
        if entry['item']['gate_status'] not in {'PASS', 'FAIL'} or entry['saved'] is None:
            return dict(result, reason='A scientific arm is incomplete or lacks a certified checkpoint.')
    left, right = baseline['item'], candidate['item']
    task_proof = task_delta_proof(baseline, candidate, stable_hash)
    result['task_declaration_delta_proof'] = task_proof
    if (not task_proof['exact_task_declarations_equal_except_registered_ownership_delta']
            or left['source_digest'] != right['source_digest']
            or left['provenance_checkpoint']['completed_steps'] != right['provenance_checkpoint']['completed_steps']):
        return dict(result, reason='Task/source/actual completed-step bindings differ; equality is not inferred.')
    opts = optimizer_records(candidate['saved'])
    wrapped = [o for o in opts if 'direction_blend' in o['wrapper_metadata']]
    uncovered = [o['owner_path'] for o in opts
                 if any(g['role'] in {'generator', 'encoder', 'prior'} for g in o['groups'])
                 and 'direction_blend' not in o['wrapper_metadata']]
    if not wrapped or uncovered:
        return dict(result, reason='Active wrapper coverage is missing for an owned generator-side optimizer.',
                    uncovered_optimizer_paths=uncovered)
    for opt in wrapped:
        meta = opt['wrapper_metadata']
        geometry = meta.get('constraint_geometry', {}).get('stats') or {}
        blend = meta['direction_blend'].get('stats') or {}
        if any(blend.get(key, 0) != 0 for key in ('conflict_steps', 'blended_steps', 'pareto_stalls')) or geometry.get('projected_steps', 0):
            return dict(result, status='NOT_APPLICABLE', reason='The actual dot criterion activated; inactive equality is not asserted.')
        if (geometry.get('steps', 0) <= 0 or geometry.get('max_derivative_before') != 0
                or geometry.get('max_derivative_after') != 0 or not opt['geometry_steps_equal_max_base_clock']
                or any(m['pending_present'] for m in meta.values())):
            return dict(result, reason='Zero activation/complete optimizer-clock coverage cannot be established.')
    a, ac, complete_a = actual_tensors(baseline['saved'])
    b, bc, complete_b = actual_tensors(candidate['saved'])
    if not complete_a or not complete_b:
        return dict(result, reason='Actual public generator/critic/prior tensor containers are unavailable.',
                    baseline_containers=ac, candidate_containers=bc)
    missing = sorted(set(a) ^ set(b))
    different = [p for p in sorted(set(a) & set(b)) if a[p].dtype != b[p].dtype or a[p].shape != b[p].shape
        or not torch.equal(a[p].detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                           b[p].detach().cpu().contiguous().reshape(-1).view(torch.uint8))]
    equal = not missing and not different
    return dict(result, status='PASS' if equal else 'FAIL', exact_tensor_equality=equal,
        comparison='All captured model/prior tensor bytes, dtype, shape and paths; optimizer wrappers excluded explicitly.',
        baseline_checkpoint=left['provenance_checkpoint'], candidate_checkpoint=right['provenance_checkpoint'],
        baseline_tensor_state_sha256=state_digest(a), candidate_tensor_state_sha256=state_digest(b),
        tensor_count=len(a), model_containers=ac, missing_tensor_paths=missing, differing_tensor_paths=different,
        optimizer_wrapper_metadata_difference='Candidate constraint_geometry/direction_blend metadata is intentional and retained separately; whole checkpoints are not claimed equal.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT)
    parser.add_argument('--artifacts', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=ARCHIVE / 'analysis/saved-counters.json')
    options = parser.parse_args()
    root, artifacts = options.repository.resolve(), options.artifacts.resolve()
    sys.path.insert(0, str(root))
    import torch
    torch.set_num_threads(1)
    from experiments.forge.state import state_digest
    from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
    workflow = load_module('projection_saved_phase3', root / 'reports/forge/bcap-three-phase/phase3.py')
    registration_path = root / 'reports/forge/bcap-projection-baseline-review/registration.json'
    registration = read_json(registration_path)
    require = workflow.require
    require(workflow.head(root) == FROZEN, 'Keep the exact frozen source worktree for this supplement.')
    require(file_hash(registration_path) == REGISTRATION_SHA and registration['source_digest'] == DIGEST,
            'The frozen registration differs.')
    progress_path = artifacts / 'phase3-progress.json'
    progress = read_json(progress_path)
    require(progress['phase'] == 'diagnostic_complete' and progress['source_commit'] == FROZEN
            and progress['registration_sha256'] == REGISTRATION_SHA, 'Read only the completed frozen campaign.')
    module = workflow.phase2.publisher(root)
    module.ROLES = workflow.ROLES
    module.required_questions = lambda _root: registration['original_requirements']
    module.scopes = lambda _options: workflow.diagnostic_scopes(module, artifacts, progress_path)
    collection = module.collect(argparse.Namespace(repository=root, allow_partial=False))
    for context in collection['scopes']:
        for role, sub in context['submissions'].items():
            require(workflow.request_summary(sub['request'], registration['task_ids']) == registration['arms'][role],
                    f'{role}: admitted request differs from frozen registration.')
            require(sub['request']['source']['origin_commit'] == FROZEN
                    and sub['request']['source']['digest'] == DIGEST, 'Measured source provenance differs.')
    rows, selected = [], {}
    for entry in collection['final']:
        item = entry['item']
        selected[(item['role'], item['task_id'])] = entry
        rows.append(dict(role=item['role'], task_id=item['task_id'], gate_status=item['gate_status'],
            request_id=item['request_id'], attempt_id=item['attempt_id'], source_commit=item['source_commit'],
            source_digest=item['source_digest'], source_files_sha256=stable_hash(entry['request']['source']['files']),
            task_contract_sha256=item['task_contract_sha256'], checkpoint=item['provenance_checkpoint'],
            owned_optimizers=optimizer_records(entry['saved']) if entry['saved'] is not None else [],
            mechanism_stats=item['mechanism_stats']))
    pairs = [compare_pair(task, selected.get(('baseline', task)), selected.get(('candidate', task)), state_digest)
             for task in registration['task_ids'] if task in INACTIVE_SCOPE]
    task_proofs = [task_delta_proof(selected[('baseline', task)], selected[('candidate', task)], stable_hash)
                   for task in registration['task_ids']
                   if ('baseline', task) in selected and ('candidate', task) in selected]
    require(len(task_proofs) == 16 and all(p['exact_task_declarations_equal_except_registered_ownership_delta']
                                         for p in task_proofs), 'Original paired task laws differ beyond the registered trainer ownership delta.')
    result = dict(schema_version=1, qualification_input=False, scope='saved_direction_only_counter_and_tensor_supplement',
        frozen_commit=FROZEN, source_digest=DIGEST, registration_sha256=REGISTRATION_SHA,
        helper_sha256=file_hash(Path(__file__)), optimizer_updates_added=0, sampling_draws_added=0,
        certified_final_rows=rows, inactive_tensor_comparisons=pairs,
        all_sixteen_task_declaration_delta_proofs=task_proofs,
        inactive_tensor_comparison_counts=dict(Counter(p['status'] for p in pairs)),
        audit_boundary='Main audit owns actual initial/data/named-stream/own-producer continuity proofs. No grade transfers or endpoint-only replacement claims.',
        derivative_interpretation='Stored derivative bounds are maxima floored at zero; zero before/after with zero conflict counters establishes no observed positive dot criterion, not a global Lipschitz or finite-loss guarantee.',
        optimizer_metadata_interpretation='Raw wrapper statistics are retained by optimizer ownership path and role. Model/prior equality excludes intentional wrapper metadata differences.',
        accounting=collection['accounting'])
    atomic_json(options.output, result)
    print(json.dumps(dict(event='saved_counter_supplement_complete', output=str(options.output),
        certified_rows=len(rows), inactive_tensor_comparison_counts=result['inactive_tensor_comparison_counts'],
        optimizer_updates_added=0, sampling_draws_added=0)), flush=True)


if __name__ == '__main__':
    main()
