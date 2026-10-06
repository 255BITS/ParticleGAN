"""Pure strict serializer for actual ROOT-paid889 software controls.

This contract performs no scientific imports, measurements or authority actions.
"""
from __future__ import annotations

import hashlib
import json

CANDIDATE_ID = 'atlas-original-two-pole-passive889-v1'
VIEW_ID = 'atlas_original_two_pole_passive889_v1'
STUDY_ID = 'atlas-original-two-pole-passive889-study-v1'
TRACK_ID = 'atlas889_original_two_pole_passive'
TASK_ID = 'two_pole'
TASK_DIGEST = '2f0207310d6bb7b290bdc520d7992eb4e6da411becae69a76d76b1232897db8b'
EFFECTIVE_RECIPE_SHA256 = 'd5f20a8c4a9a7a3e2f0ac6d4562ae0a364677ffe73611ff9f9b7432434024b95'
RESOURCES = dict(backend='cpu', cpu_threads=1, gpus=0, budget_seconds=300)
SCHEMA = 'forge_atlas889_two_pole_passive_software_proof_v1'
CONTROL_SCHEMA = 'forge_atlas889_two_pole_passive_control_result_v1'
CHECKS = ('original_adam_off_on_populated_moments', 'original_policy_hooks_off_on',
    'actual_beta_lr_q_inputs_and_return', 'original_finish_event_identity',
    'closure_exception_passthrough', 'adam_exception_and_restore_passthrough',
    'observer_fault_does_not_replace_q', 'capture_fault_marks_unknown',
    'bounded_records_and_declared_scope_unknown', 'private_source_origin_no884_patch',
    'original_recipe_task_registry_binding')
ZERO_OPERATIONS = ('extra_prior_samples', 'extra_rng_draws', 'extra_model_forwards',
    'extra_backward_calls', 'extra_optimizer_steps', 'extra_decision_evaluations',
    'state_getter_calls', 'state_mutations', 'foreign_device_initializations')
FIELDS = ('schema', 'status', 'candidate_id', 'view_id', 'study_id', 'track_marker',
    'request_id', 'request_sha256', 'source_digest', 'candidate_revision', 'task_id',
    'task_digest', 'source_pins', 'checks', 'added_operations',
    'effective_recipe_sha256', 'resources')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def task_digest(task):
    return digest({k: v for k, v in task.items() if k not in ('preflight_blockers', 'field_ownership')})


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def validate_source_pins(pins):
    if not isinstance(pins, dict) or not pins:
        raise ValueError('actual copied software Source pins are required')
    for path, pin in pins.items():
        if (not isinstance(path, str) or path.startswith('/') or '..' in path.split('/')
                or not isinstance(pin, dict) or set(pin) != {'sha256', 'bytes'}
                or not _sha(pin['sha256']) or type(pin['bytes']) is not int or pin['bytes'] <= 0):
            raise ValueError('malformed exact typed software Source pin')


def build_software_proof(request, measured_controls, source_pins):
    """Bind real passing controls to a complete admitted request; never invent PASS."""
    validate_source_pins(source_pins)
    if (measured_controls.get('schema') != CONTROL_SCHEMA
            or measured_controls.get('status') != 'PASS'
            or measured_controls.get('checks') != dict.fromkeys(CHECKS, 'PASS')
            or set(measured_controls.get('added_operations', {})) != set(ZERO_OPERATIONS)
            or any(type(v) is not int or v != 0 for v in measured_controls['added_operations'].values())):
        raise ValueError('fresh actual passing controls and strict measured integer-zero operations are required')
    if (not isinstance(request.get('request_id'), str) or not request['request_id']
            or request['candidate']['id'] != CANDIDATE_ID or request['view']['id'] != VIEW_ID
            or request['candidate']['claim_contract']['experimental_track'] != TRACK_ID
            or set(request['tasks']) != {TASK_ID}
            or task_digest(request['tasks'][TASK_ID]) != TASK_DIGEST
            or request['protocol']['seed'] != 0 or not _sha(request['source']['digest'])
            or not _sha(request['candidate_revision'])):
        raise ValueError('one original seeded direct889 admitted request is required')
    jobs = request['jobs']
    if (len(jobs) != 1 or jobs[0]['task_id'] != TASK_ID or jobs[0]['budget_seconds'] != 300
            or jobs[0]['resources'].get('backend') != 'cpu'
            or jobs[0]['resources'].get('cpu_threads') != 1 or jobs[0]['resources'].get('gpus') != 0):
        raise ValueError('original CPU1/gpus0/full300 job is required')
    if measured_controls.get('effective_recipe_sha256') != EFFECTIVE_RECIPE_SHA256:
        raise ValueError('actual task-effective full79 d5f20 owner proof is required')
    for path, pin in source_pins.items():
        if request['source']['files'].get(path) != pin['sha256']:
            raise ValueError('software Source does not join the actual frozen request: ' + path)
    proof = dict(schema=SCHEMA, status='PASS', candidate_id=CANDIDATE_ID, view_id=VIEW_ID,
        study_id=STUDY_ID, track_marker=TRACK_ID, request_id=request['request_id'],
        request_sha256=digest(request), source_digest=request['source']['digest'],
        candidate_revision=request['candidate_revision'], task_id=TASK_ID, task_digest=TASK_DIGEST,
        source_pins=source_pins, checks=measured_controls['checks'],
        added_operations=measured_controls['added_operations'],
        effective_recipe_sha256=EFFECTIVE_RECIPE_SHA256, resources=RESOURCES)
    if set(proof) != set(FIELDS):
        raise ValueError('software proof fields differ')
    return proof
