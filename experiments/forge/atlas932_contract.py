"""Pure full-request proof for the explicit932 particle-L2 objective ablation.

Base Recipe, retained927 optimizer law and new objective law are separate facts.
This serializer does not run controls, construct a model or grant admission.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

CANDIDATE_ID = 'atlas-two-pole-l2-off932-v1'
VIEW_ID = 'atlas_two_pole_l2_off932_v1'
STUDY_ID = 'atlas-two-pole-l2-off932-study-v1'
TRACK_ID = 'atlas932_two_pole_l2_off'
TASK_ID = 'two_pole_l2_off932_v1'
TASK_DIGEST = 'e3eb1e6df5cc334cb5f71684d25f57c1c3321877c2ace336207fb233577d2ffb'
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
OBJECTIVE_VARIANT = {'parent_task_id': 'two_pole', 'parent_task_sha256': '55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5', 'particle_l2': 0.0, 'reference_particle_l2': 0.02, 'schema': 'forge_atlas932_two_pole_l2_objective_variant_v1', 'training_term': 'particle_l2 * particles.square().mean()'}
OBJECTIVE_VARIANT_SHA256 = '1514574ea06c8cd602293ef0fa99121c4b9f1df1de6d00f5066aa35c69b6036f'
SCHEMA = 'forge_atlas932_particle_l2_software_proof_v1'
CONTROL_SCHEMA = 'forge_atlas932_particle_l2_control_result_v1'
CHECKS = ('declared_objective_and_actual_owner_coefficient', 'pinned_loss_coefficient_gradient_delta',
    'preserved_optimizer_clock_and_recorder_checkpoint')
OPTIMIZER_CONTROL_SCHEMA = 'forge_atlas927_particle_amsgrad_control_result_v1'
OPTIMIZER_CHECKS = ('registry_base_recipe_and_task_bound', 'fresh_owned_table_only_group_override',
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
    'effective_optimizer_variant', 'effective_optimizer_variant_sha256',
    'effective_objective_variant', 'effective_objective_variant_sha256', 'resources', 'control_producers')
UNCHANGED_SOURCE_PINS = {
    'configs/forge/tasks/two_pole_l2_off932_v1.json': {'sha256': '868a3e2a8da8a92539051fedd1b08e8ce33635f343dfa34d81ea2d50c99a3f08', 'bytes': 3714},
    'configs/forge/tasks/two_pole.json': {
        'sha256': '55ac2d3883ba6c173da304fa7f10648a0b559c202fc35b451b1d0c8870f61cf5', 'bytes': 3099},
    'experiments/forge/atlas927_contract.py': {'sha256': '7a2be1b94d723776a4a938868cefc8cc6962fd37946d5b41e1fe0c49ea15e01d', 'bytes': 10860},
    'experiments/forge/atlas927_two_pole_owner.py': {'sha256': '0fdd73021ab38f660b8c1ee4b292e32ef9fa39077751ba85d7679a64d283a587', 'bytes': 68998},
    'tests/test_atlas927_particle_amsgrad.py': {'sha256': '64e562da2a8fe3a5412f78c8fcf23eaeaa86ddf4b5b7b693ca508bc6d1ded2e5', 'bytes': 19154},
    'experiments/forge/atlas889_contract.py': {
        'sha256': 'e4850919918d913600dce641f1c1e421c4e1fe483a4dedac74a520c03458043d', 'bytes': 5520},
    'experiments/forge/atlas889_two_pole_observer.py': {
        'sha256': '0c2070cbaa5da9f056ded548b629165ace50746472aa45e0dc8abd20a4e2a646', 'bytes': 23311},
    'tests/test_atlas889_two_pole_observer.py': {
        'sha256': '52f82b6b0bd5bc313ee341466463cee415d61e9804658dfc554280931d144145', 'bytes': 29610},
    'tests/test_atlas921_completed_lr_clock.py': {
        'sha256': '3da79006816bcb88488333c200f55077af5d6e3c97f2dafbbedd863fc7279a40', 'bytes': 13934}}
REQUIRED_PROOF_PATHS = (
    'experiments/forge/atlas932_contract.py',
    'experiments/forge/atlas932_two_pole_owner.py',
    'tests/test_atlas932_particle_l2.py',
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
        raise ValueError('actual copied932 software Source pins are required')
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


def build_software_proof(request, measured_controls, source_pins, *, reused_optimizer_controls, reused_observer_controls):
    """Join fresh932 controls and separate retained927/recorder producers to one request."""
    validate_source_pins(source_pins)
    _passed(measured_controls, CONTROL_SCHEMA, CHECKS)
    _passed(reused_observer_controls, RETAINED_CONTROL_SCHEMA, RETAINED_CHECKS)
    _passed(reused_optimizer_controls, OPTIMIZER_CONTROL_SCHEMA, OPTIMIZER_CHECKS)
    if reused_observer_controls.get('effective_recipe_sha256') != BASE_RECIPE_SHA256:
        raise ValueError('retained recorder controls must identify their original base Recipe')
    if (measured_controls.get('base_recipe_sha256') != BASE_RECIPE_SHA256
            or canonical(measured_controls.get('effective_optimizer_variant')) != canonical(OPTIMIZER_VARIANT)
            or measured_controls.get('effective_optimizer_variant_sha256') != OPTIMIZER_VARIANT_SHA256
            or measured_controls.get('resources') != RESOURCES):
        raise ValueError('actual932 base/variant/resources do not match the owned ablation')
    if (canonical(measured_controls.get('effective_objective_variant')) != canonical(OBJECTIVE_VARIANT)
            or measured_controls.get('effective_objective_variant_sha256') != OBJECTIVE_VARIANT_SHA256
            or reused_optimizer_controls.get('base_recipe_sha256') != BASE_RECIPE_SHA256
            or canonical(reused_optimizer_controls.get('effective_optimizer_variant')) != canonical(OPTIMIZER_VARIANT)
            or reused_optimizer_controls.get('effective_optimizer_variant_sha256') != OPTIMIZER_VARIANT_SHA256
            or reused_optimizer_controls.get('resources') != RESOURCES):
        raise ValueError('distinct932 objective and retained927 optimizer producer are required')
    prior_controlled = reused_optimizer_controls.get('source_pins')
    validate_source_pins(prior_controlled)
    if any(source_pins.get(path) != pin for path, pin in prior_controlled.items()):
        raise ValueError('retained927 producer Source must be preserved unchanged in the new snapshot')
    controlled = measured_controls.get('source_pins')
    validate_source_pins(controlled)
    required = set(REQUIRED_PROOF_PATHS) | set(UNCHANGED_SOURCE_PINS)
    if not required <= source_pins.keys():
        raise ValueError('complete copied932 owner/contract/task/helper/card proof Source is required')
    if (not {'experiments/forge/atlas932_two_pole_owner.py', 'experiments/forge/atlas932_contract.py',
            'tests/test_atlas932_particle_l2.py'} <= controlled.keys()
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
            or canonical(request['candidate']['claim_contract'].get('objective_variant')) != canonical(OBJECTIVE_VARIANT)
            or request['view'].get('evidence_scope') != 'research_diagnostic'
            or request.get('through_tier') != 1 or request.get('execution_backend') != 'cpu'
            or set(request['tasks']) != {TASK_ID}
            or task_digest(request['tasks'][TASK_ID]) != TASK_DIGEST
            or type(request['protocol']['seed']) is not int or request['protocol']['seed'] != 0
            or not _sha(request['source']['digest']) or not _sha(request['candidate_revision'])):
        raise ValueError('one fresh ready932 seeded original direct diagnostic request is required')
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
            'retained_optimizer': deepcopy(reused_optimizer_controls['checks']),
            'objective': deepcopy(measured_controls['checks'])},
        added_operations=deepcopy(measured_controls['added_operations']),
        base_recipe_sha256=BASE_RECIPE_SHA256, effective_optimizer_variant=deepcopy(OPTIMIZER_VARIANT),
        effective_optimizer_variant_sha256=OPTIMIZER_VARIANT_SHA256,
        effective_objective_variant=deepcopy(OBJECTIVE_VARIANT),
        effective_objective_variant_sha256=OBJECTIVE_VARIANT_SHA256, resources=deepcopy(RESOURCES),
        control_producers={
            'retained_observer': {'schema': RETAINED_CONTROL_SCHEMA,
                'canonical_record_sha256': digest(reused_observer_controls),
                'source_scope': 'unchanged recorder0c207 and original11 helper52f; actual receipt retained separately by ROOT'},
            'retained_optimizer': {'schema': OPTIMIZER_CONTROL_SCHEMA,
                'canonical_record_sha256': digest(reused_optimizer_controls), 'source_pins': deepcopy(prior_controlled)},
            'objective': {'schema': CONTROL_SCHEMA,
                'canonical_record_sha256': digest(measured_controls), 'source_pins': deepcopy(controlled)}})
    if (set(proof) != set(FIELDS) or digest(OPTIMIZER_VARIANT) != OPTIMIZER_VARIANT_SHA256
            or digest(OBJECTIVE_VARIANT) != OBJECTIVE_VARIANT_SHA256):
        raise ValueError('932 software proof fields or canonical optimizer descriptor differ')
    return proof
