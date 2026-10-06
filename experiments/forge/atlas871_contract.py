"""Definitions-only fresh software-proof contract for the 2000/48 Gaussian diagnostic."""
from copy import deepcopy
import hashlib
import json

CANDIDATE_ID = 'atlas-existing-mog-longer871-v1'
VIEW_ID = 'atlas_existing_mog_longer871_v1'
STUDY_ID = 'atlas-existing-mog-longer871-study-v1'
TRACK_ID = 'atlas871_existing_mog_longer'
TASK_ID = 'gaussian1d_acquisition_longer871_v1'
TASK_DIGEST = '09025c182ec7670f3346238ae2e95c2b92c5312cae8ae19a5f8bb4d52701e256'
PROOF_SCHEMA = 'forge_atlas871_longer_gaussian_software_proof_v1'
SOFTWARE_CHECKS = ('task_duration_delta', 'original_task_pin', 'original_prefix_clocks',
    'appended_clocks', 'unchanged_five_suffix_gates', 'incomplete_curve_rejected',
    'strict_task_mutations', 'full79_recipe_binding', 'fixed_rng_and_initializer',
    'ordinary_admission_resources', 'scoped_registry', 'observer_omitted')
EXTRA_OPERATIONS = ('extra_prior_samples', 'extra_rng_draws', 'extra_model_forwards',
    'extra_backward_calls', 'extra_optimizer_steps', 'extra_decision_evaluations',
    'state_getter_calls', 'state_mutations', 'foreign_device_initializations')


def build_software_proof(request, measured_controls, source_pins):
    """Serialize ROOT's measured new controls after full copied Queue admission.

    This function performs no I/O, test, model operation, or numerical grading.
    The runtime independently compares persisted bytes to its literal proof map.
    """
    from .atlas871_longer import validate, digest
    expected_checks = {name:'PASS' for name in SOFTWARE_CHECKS}
    expected_ops = {name:0 for name in EXTRA_OPERATIONS}
    candidate = request['candidate']
    if (measured_controls.get('schema') != 'forge_atlas871_software_control_result_v1'
            or measured_controls.get('status') != 'PASS'
            or measured_controls.get('checks') != expected_checks
            or measured_controls.get('added_operations') != expected_ops
            or any(type(v) is not int for v in measured_controls['added_operations'].values())
            or candidate.get('id') != CANDIDATE_ID
            or candidate.get('claim_contract', {}).get('experimental_track') != TRACK_ID
            or request.get('view', {}).get('id') != VIEW_ID
            or request.get('study', {}).get('id') != STUDY_ID
            or request['study'].get('status') != 'ready'
            or not isinstance(request.get('request_id'), str) or not request['request_id']
            or set(request['tasks']) != {TASK_ID}):
        raise ValueError('actual new controls and one full ready admitted Gaussian request required')
    if validate(request['tasks'][TASK_ID])['task_payload_sha256'] != TASK_DIGEST:
        raise ValueError('the exact duration-only Gaussian task differs')
    files = request['source']['files']
    if digest(files) != request['source']['digest'] or not source_pins:
        raise ValueError('full copied Source index required')
    for path,pin in source_pins.items():
        if (set(pin) != {'sha256','bytes'} or type(pin['bytes']) is not int or pin['bytes'] <= 0
                or files.get(path) != pin['sha256']):
            raise ValueError('new software-proof Source pin differs: ' + path)
    return dict(schema=PROOF_SCHEMA, status='PASS', candidate_id=CANDIDATE_ID,
        view_id=VIEW_ID, study_id=STUDY_ID, track_marker=TRACK_ID,
        request_id=request['request_id'], request_sha256=digest(request),
        source_digest=request['source']['digest'], candidate_revision=request['candidate_revision'],
        task_id=TASK_ID, task_digest=TASK_DIGEST, source_pins=deepcopy(source_pins),
        checks=expected_checks, added_operations=expected_ops)
