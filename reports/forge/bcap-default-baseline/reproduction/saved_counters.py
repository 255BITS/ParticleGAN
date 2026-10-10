"""Read-only ordinary counter and actual tensor supplement; no science calls."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path('/home/martyn/dev/ParticleGAN-bcap-projection-baseline-review')
A = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-default-adoption-20261010')
SOURCE = 'd378734f40b09ce223a389e8f54a9783ec6a0c75'
DIGEST = '6a225fcdd6922cdad37c9c947e163fb6f164f6ad4390293a3d3091b8f741ce44'
REG = '91b2ce2eee3d452bb11694256331d1122adba789bea9a812a4c2201982ae8db9'
PREVIOUS = Path('/mnt/ml7tb/ParticleGAN-forge/bcap-tier1-repair-next/baseline-repairs/projection/saved_counters.py')

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=A / 'publication/saved-counters.json')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    import torch
    torch.set_num_threads(1)
    from experiments.forge.contracts import atomic_json, file_hash, read_json, stable_hash
    from experiments.forge.state import state_digest
    workflow = load('ordinary_saved_counter_workflow', ROOT / 'reports/forge/bcap-default-baseline/workflow.py')
    require = workflow.require
    require(workflow.head(ROOT) == SOURCE, 'Keep exact scientific HEAD until saved extraction finishes.')
    registration_path = ROOT / 'reports/forge/bcap-default-baseline/registration.json'
    registration = read_json(registration_path)
    progress = read_json(A / 'progress.json')
    require(progress['phase'] == 'ordinary_complete' and progress['source_commit'] == SOURCE
        and progress['source_digest'] == DIGEST and progress['registration_sha256'] == REG
        and file_hash(registration_path) == REG, 'Exact completed ordinary registration required.')
    module = workflow.shared.publisher(ROOT)
    module.ROLES = workflow.ROLES
    module.required_questions = lambda _root: registration['original_requirements']
    collection = module.collect(argparse.Namespace(repository=ROOT, queue=A / 'queue', progress=A / 'progress.json',
        diagnostic_queue=None, diagnostic_progress=None, allow_partial=False))
    previous = load('prior_saved_counter_functions', PREVIOUS)
    previous_sha = file_hash(PREVIOUS)
    require(previous_sha == '14ba9d85b45990ee1ce78b064cf85c204bc4debde28b61bcf3e13d50c4b3d470',
        'Preserved prior helper implementation changed; retain its original identity.')

    def ownership(control, candidate, stable_hash):
        left, right = deepcopy(control['task']), deepcopy(candidate['task'])
        am = left['field_ownership']['recipe_fields'].pop('constraint_geometry_mode')
        bm = right['field_ownership']['recipe_fields'].pop('constraint_geometry_mode')
        before = dict(owner='technique', status='effective',
            source='candidate.recipe_overrides.constraint_geometry_mode', value='none')
        after = dict(before, value='direction_blend')
        return dict(task_id=control['item']['task_id'],
            exact_task_declarations_equal_except_registered_ownership_delta=(left == right and am == before and bm == after),
            control_task_declaration_sha256=stable_hash(control['task']),
            candidate_task_declaration_sha256=stable_hash(candidate['task']),
            declared_delta=dict(path='field_ownership.recipe_fields.constraint_geometry_mode', control=am, candidate=bm))

    # Reuse the preserved byte-comparison implementation, with the new explicit
    # control declaration. Neither previous helper bytes nor receipts change.
    previous.task_delta_proof = ownership
    original_tensors = previous.actual_tensors

    def actual_tensors(saved):
        tensors, containers, complete = original_tensors(saved)
        role_parameters = saved.get('role_parameters')
        if not isinstance(role_parameters, dict):
            return tensors, containers, complete
        applied = saved.get('applied', {})
        roles = applied.get('active_roles')
        valid = isinstance(roles, list) and set(roles) == set(role_parameters) and bool(saved.get('models'))
        for role, parameters in role_parameters.items():
            valid = valid and isinstance(parameters, list) and bool(parameters)
            containers.append(dict(path=f'state.role_parameters.{role}',
                names=[p.get('name') for p in parameters], source='Actual certified standalone/model parameter snapshots'))
            for index, parameter in enumerate(parameters):
                value = parameter.get('value')
                valid = valid and isinstance(value, torch.Tensor) and parameter.get('index') == index
                if isinstance(value, torch.Tensor):
                    tensors[f'state.role_parameters.{role}[{index}].value'] = value
                if parameter.get('representation') == 'direct_sample_coordinates':
                    valid = valid and role == 'prior' and parameter.get('name', '').startswith('direct_particles.')
                else:
                    name = parameter.get('name')
                    model_path = f'state.models.{name}'
                    valid = valid and model_path in tensors and torch.equal(tensors[model_path], value)
        # BehaviorComponents.provenance_state includes every model state and
        # every owned role parameter. Two-pole trains standalone coordinates,
        # while unipolar has G/D networks and no consumed latent-prior module.
        # Require the actual active-role receipt; never invent missing models.
        valid = valid and 'discriminator' in role_parameters
        if 'generator' in role_parameters:
            valid = valid and 'generator' in saved['models']
        if 'encoder' in role_parameters:
            valid = valid and 'encoder' in saved['models']
        if 'prior' not in role_parameters:
            containers.append(dict(path='state.applied.active_roles', names=roles,
                prior_tensor_applicability='No latent-prior parameter role consumed by this original host.'))
        return tensors, containers, bool(valid)

    previous.actual_tensors = actual_tensors
    selected, rows = {}, []
    for context in collection['scopes']:
        for role, sub in context['submissions'].items():
            require(workflow.summary(sub['request']) == registration['arms'][role], 'Executed registration differs.')
    for entry in collection['final']:
        item = entry['item']
        selected[item['role'], item['task_id']] = entry
        rows.append(dict(role=item['role'], task_id=item['task_id'], gate_status=item['gate_status'],
            source_commit=item['source_commit'], source_digest=item['source_digest'],
            task_contract_sha256=item['task_contract_sha256'], checkpoint=item['provenance_checkpoint'],
            owned_optimizers=previous.optimizer_records(entry['saved']) if entry['saved'] is not None else [],
            mechanism_stats=item['mechanism_stats']))
    proofs, pairs = [], []
    for assignment in registration['original_requirements']:
        task = assignment['task']
        control, candidate = selected.get(('control', task)), selected.get(('candidate', task))
        if control and candidate:
            proofs.append(ownership(control, candidate, stable_hash))
            if control['saved'] is not None and candidate['saved'] is not None:
                pairs.append(previous.compare_pair(task, control, candidate, state_digest))
    require(all(p['exact_task_declarations_equal_except_registered_ownership_delta'] for p in proofs),
        'Actual task declarations differ beyond the single registered ownership delta.')
    atomic_json(args.output, dict(schema_version=1, scope='ordinary_saved_direction_counter_tensor_supplement',
        source_commit=SOURCE, source_digest=DIGEST, registration_sha256=REG,
        helper_sha256=file_hash(Path(__file__)), reused_helper=dict(path=str(PREVIOUS), sha256=previous_sha,
            adaptation='Explicit control ownership now candidate.recipe_overrides.constraint_geometry_mode=none; preserved helper bytes unchanged.'),
        saved_layout_adaptation='Capture actual role_parameters.value snapshots, validate their active_roles/names and duplicate model bytes; direct coordinates and absent latent-prior roles remain explicitly distinguished.',
        optimizer_updates_added=0, sampling_draws_added=0, certified_final_rows=rows,
        task_declaration_delta_proofs=proofs, inactive_tensor_comparisons=pairs,
        comparison_counts=dict(Counter(p['status'] for p in pairs)),
        metadata_boundary='Actual model/prior tensor byte equality is separate from intentional optimizer wrapper metadata differences. Activated dot criterion yields NOT_APPLICABLE; missing/unequal states remain UNVERIFIED.',
        derivative_boundary='Bounds are stored maxima floored at zero, not global Lipschitz or finite-step-loss guarantees.',
        qualification_boundary='Independent audit owns source/initialization/data/non-eval streams/own producers. This supplement never changes grades.',
        accounting=collection['accounting']))
    print(dict(output=str(args.output), sha256=file_hash(args.output), comparisons=dict(Counter(p['status'] for p in pairs))), flush=True)

if __name__ == '__main__':
    main()
