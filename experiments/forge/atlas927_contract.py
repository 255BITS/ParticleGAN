"""Pure full-request proof for the explicit927 table-only optimizer ablation.

Base Recipe provenance and the effective optimizer law are separate facts.
This serializer does not run controls, construct a model or grant admission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

CANDIDATE_ID = 'atlas-two-pole-particle-amsgrad-off927-v1'
VIEW_ID = 'atlas_two_pole_particle_amsgrad_off927_v1'
STUDY_ID = 'atlas-two-pole-particle-amsgrad-off927-study-v1'
TRACK_ID = 'atlas927_two_pole_particle_amsgrad_off'
TASK_ID = 'two_pole'
TASK_DIGEST = '2f0207310d6bb7b290bdc520d7992eb4e6da411becae69a76d76b1232897db8b'
BASE_RECIPE_SHA256 = 'd5f20a8c4a9a7a3e2f0ac6d4562ae0a364677ffe73611ff9f9b7432434024b95'
RESOURCES = dict(backend='cpu', cpu_threads=1, gpus=0, budget_seconds=300)
OPTIMIZER_VARIANT = {
    'schema': 'forge_atlas927_two_pole_particle_amsgrad_variant_v1',
    'base_recipe_sha256': BASE_RECIPE_SHA256,
    'groups': [
        {'optimizer': 'generator', 'group_index': 0, 'role': 'table', 'amsgrad': False},
        {'optimizer': 'generator', 'group_index': 1, 'role': 'noise', 'amsgrad': True},
        {'optimizer': 'critic', 'group_index': 0, 'role': 'critic', 'amsgrad': True}],
    'table_lr_clock': 'completed_direct_adam_lr/base_lr'}
OPTIMIZER_VARIANT_SHA256 = '950edc64094c417d98d43b2eaf59fbaaba0524f7823e2acb001ba7576db1d051'
SCHEMA = 'forge_atlas927_particle_amsgrad_software_proof_v1'
CONTROL_SCHEMA = 'forge_atlas927_particle_amsgrad_control_result_v1'
CHECKS = ('registry_base_recipe_and_task_bound', 'fresh_owned_table_only_group_override',
    'populated_adam_denominator_branch', 'completed_lr_clock_and_q_preserved',
    'checkpoint_recorder_and_roundtrip_flags', 'closure_and_failure_forwarding',
    'wrong_variant_or_role_rejected', 'original_task_gate_and_core_scope')
RETAINED_CONTROL_SCHEMA = 'forge_atlas889_two_pole_passive_control_result_v1'
RETAINED_CHECKS = ('original_adam_off_on_populated_moments', 'original_policy_hooks_off_on',
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
    'task_digest', 'source_pins', 'checks', 'added_operations', 'base_recipe_sha256',
    'effective_optimizer_variant', 'effective_optimizer_variant_sha256', 'resources')
UNCHANGED_SOURCE_PINS = {
    'configs/forge/tasks/two_pole.json': {
        'sha256': '55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5', 'bytes': 3099},
    'experiments/forge/atlas889_contract.py': {
        'sha256': 'e4850919918d913600dce641f1c1e421c4e1fe483a4dedac74a520c03458043d', 'bytes': 5520},
    'experiments/forge/atlas889_two_pole_observer.py': {
        'sha256': '0c2070cbaa5da9f056ded548b629165ace50746472aa45e0dc8abd20a4e2a646', 'bytes': 23311},
    'tests/test_atlas889_two_pole_observer.py': {
        'sha256': '52f82b6b0bd5bc313ee341466463cee415d61e9804658dfc554280931d144145', 'bytes': 29610},
    'tests/test_atlas921_completed_lr_clock.py': {
        'sha256': '3da79006816bcb88488333c200f55077af5d6e3c97f2dafbbedd863fc7279a40', 'bytes': 13934}}
REQUIRED_PROOF_PATHS = (
    'experiments/forge/atlas927_contract.py',
    'experiments/forge/atlas927_two_pole_owner.py',
    'tests/test_atlas927_particle_amsgrad.py',
    'configs/forge/ideas/' + CANDIDATE_ID + '.json',
    'configs/forge/views/' + VIEW_ID + '.json')


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
        raise ValueError('actual copied927 software Source pins are required')
    for path, pin in pins.items():
        if (not isinstance(path, str) or not path or path.startswith('/') or '\\' in path
                or any(part in ('', '.', '..') for part in path.split('/'))
                or not isinstance(pin, dict) or set(pin) != {'sha256', 'bytes'}
                or not _sha(pin['sha256']) or type(pin['bytes']) is not int or pin['bytes'] <= 0):
            raise ValueError('malformed exact typed software Source pin')


def _passed(report, schema, checks):
    if (not isinstance(report, dict) or report.get('schema') != schema
            or report.get('status') != 'PASS'
            or report.get('checks') != dict.fromkeys(checks, 'PASS')
            or set(report.get('added_operations', {})) != set(ZERO_OPERATIONS)
            or any(type(v) is not int or v != 0 for v in report['added_operations'].values())):
        raise ValueError('actual complete passing controls and nine measured integer-zero operations are required')


def build_software_proof(request, measured_controls, source_pins, *, reused_observer_controls):
    """Join fresh927 controls and separate retained recorder controls to one request."""
    validate_source_pins(source_pins)
    _passed(measured_controls, CONTROL_SCHEMA, CHECKS)
    _passed(reused_observer_controls, RETAINED_CONTROL_SCHEMA, RETAINED_CHECKS)
    if reused_observer_controls.get('effective_recipe_sha256') != BASE_RECIPE_SHA256:
        raise ValueError('retained recorder controls must identify their original base Recipe')
    if (measured_controls.get('base_recipe_sha256') != BASE_RECIPE_SHA256
            or canonical(measured_controls.get('effective_optimizer_variant')) != canonical(OPTIMIZER_VARIANT)
            or measured_controls.get('effective_optimizer_variant_sha256') != OPTIMIZER_VARIANT_SHA256
            or measured_controls.get('resources') != RESOURCES):
        raise ValueError('actual927 base/variant/resources do not match the owned ablation')
    controlled = measured_controls.get('source_pins')
    validate_source_pins(controlled)
    required = set(REQUIRED_PROOF_PATHS) | set(UNCHANGED_SOURCE_PINS)
    if not required <= source_pins.keys():
        raise ValueError('complete copied927 owner/contract/task/helper/card proof Source is required')
    if (not {'experiments/forge/atlas927_two_pole_owner.py', 'experiments/forge/atlas927_contract.py',
            'tests/test_atlas927_particle_amsgrad.py'} <= controlled.keys()
            or any(source_pins.get(path) != pin for path, pin in controlled.items())
            or any(source_pins.get(path) != pin for path, pin in UNCHANGED_SOURCE_PINS.items())):
        raise ValueError('actual controlled Source and unchanged recorder/helpers differ from copied proof Source')
    if (not isinstance(request.get('request_id'), str) or not request['request_id']
            or request.get('campaign_id') != STUDY_ID
            or request['candidate']['id'] != CANDIDATE_ID or request['view']['id'] != VIEW_ID
            or request.get('study', {}).get('id') != STUDY_ID
            or request.get('study', {}).get('candidate') != CANDIDATE_ID
            or request.get('study', {}).get('status') != 'ready'
            or request.get('study_review', {}).get('status') != 'READY'
            or request.get('study_admission') != request['study_review'].get('receipt')
            or request['candidate']['claim_contract']['experimental_track'] != TRACK_ID
            or canonical(request['candidate']['claim_contract'].get('optimizer_variant')) != canonical(OPTIMIZER_VARIANT)
            or request['view'].get('evidence_scope') != 'research_diagnostic'
            or request.get('through_tier') != 1 or request.get('execution_backend') != 'cpu'
            or set(request['tasks']) != {TASK_ID}
            or task_digest(request['tasks'][TASK_ID]) != TASK_DIGEST
            or type(request['protocol']['seed']) is not int or request['protocol']['seed'] != 0
            or not _sha(request['source']['digest']) or not _sha(request['candidate_revision'])):
        raise ValueError('one fresh ready927 seeded original direct diagnostic request is required')
    jobs = request['jobs']
    if (len(jobs) != 1 or jobs[0]['task_id'] != TASK_ID
            or jobs[0].get('task_ids', [TASK_ID]) != [TASK_ID]
            or type(jobs[0]['budget_seconds']) is not int or jobs[0]['budget_seconds'] != 300
            or jobs[0]['resources'].get('backend') != 'cpu'
            or type(jobs[0]['resources'].get('cpu_threads')) is not int or jobs[0]['resources']['cpu_threads'] != 1
            or type(jobs[0]['resources'].get('gpus')) is not int or jobs[0]['resources']['gpus'] != 0
            or jobs[0]['science'].get('candidate_revision') != request['candidate_revision']
            or jobs[0]['science'].get('evidence_use') != 'research_diagnostic'
            or digest(jobs[0]['science']) != jobs[0]['compatibility_key']):
        raise ValueError('exact original CPU1/gpus0/full300 diagnostic job is required')
    for path, pin in source_pins.items():
        if request['source']['files'].get(path) != pin['sha256']:
            raise ValueError('software Source does not join actual copied request: ' + path)
    proof = dict(schema=SCHEMA, status='PASS', candidate_id=CANDIDATE_ID, view_id=VIEW_ID,
        study_id=STUDY_ID, track_marker=TRACK_ID, request_id=request['request_id'],
        request_sha256=digest(request), source_digest=request['source']['digest'],
        candidate_revision=request['candidate_revision'], task_id=TASK_ID, task_digest=TASK_DIGEST,
        source_pins=deepcopy(source_pins), checks={
            'retained_observer': deepcopy(reused_observer_controls['checks']),
            'variant': deepcopy(measured_controls['checks'])},
        added_operations=deepcopy(measured_controls['added_operations']),
        base_recipe_sha256=BASE_RECIPE_SHA256, effective_optimizer_variant=deepcopy(OPTIMIZER_VARIANT),
        effective_optimizer_variant_sha256=OPTIMIZER_VARIANT_SHA256, resources=deepcopy(RESOURCES))
    if set(proof) != set(FIELDS) or digest(OPTIMIZER_VARIANT) != OPTIMIZER_VARIANT_SHA256:
        raise ValueError('927 software proof fields or canonical optimizer descriptor differ')
    return proof
