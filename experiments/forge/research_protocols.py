"""Explicit paired replications; the seed-zero screening protocol remains unchanged."""
from pathlib import Path

from .contracts import file_hash, identifier, read_json, stable_hash


def protocol_semantics(protocol):
    # Registration names and revisions cannot turn an exact repeat into new work.
    return {k: v for k, v in protocol.items() if k not in ('id', 'revision', 'research')}


def comparison_identity(registry):
    return dict(roles={role: declaration['recipe_overrides_sha256'] for role, declaration in registry['roles'].items()},
                tasks=registry['tasks'])


def registered_protocol(root, protocol_id):
    root = Path(root)
    identifier(protocol_id, 'protocol')
    path = root / 'configs/forge/protocols' / f'{protocol_id}.json'
    protocol = read_json(path)
    if protocol.get('id') != protocol_id:
        raise ValueError('protocol filename and identity differ')
    if protocol_id == 'screening':
        if type(protocol.get('seed')) is not int or protocol['seed'] != 0 or 'research' in protocol:
            raise ValueError('original screening must retain seed 0')
        return protocol, None
    research = protocol.get('research', {})
    cohort_id = research.get('cohort_id', '')
    identifier(cohort_id, 'research cohort')
    registry_path = root / 'configs/forge/research-cohorts' / f'{cohort_id}.json'
    registry = read_json(registry_path)
    seeds = registry.get('seed_set', [])
    if (registry.get('schema_version') != 1 or registry.get('status') != 'preregistered'
            or registry.get('qualification_input') is not False
            or research.get('purpose') != 'paired_replication'
            or not seeds or any(type(seed) is not int or seed <= 0 for seed in seeds)
            or len(seeds) != len(set(seeds)) or protocol.get('seed') not in seeds
            or set(registry.get('roles', {})) != {'control', 'candidate'}):
        raise ValueError('alternate seeds require a complete nonqualifying paired-replication registration')
    if registry.get('protocols', {}).get(protocol_id) != file_hash(path):
        raise ValueError('registered protocol bytes changed')
    original_path = root / 'configs/forge/protocols/screening.json'
    original = read_json(original_path)
    if registry.get('screening_sha256') != file_hash(original_path):
        raise ValueError('registered original screening protocol changed')
    expected = protocol_semantics(original)
    expected['seed'] = protocol['seed']
    if protocol_semantics(protocol) != expected:
        raise ValueError('replication may change only the explicitly registered seed')
    code = registry.get('source_semantics', {})
    if not code:
        raise ValueError('registration requires the scientific implementation identity')
    for relative, sha in code.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or file_hash(target) != sha:
            raise ValueError('registered scientific implementation changed')
    return protocol, registry


def make_unit(root, request, registry):
    root = Path(root)
    protocol = request['protocol']
    matches = [role for role, value in registry['roles'].items() if value['candidate_id'] == request['candidate']['id']]
    if len(matches) != 1:
        raise ValueError('registered paired role is ambiguous or absent')
    role = matches[0]
    registry_path = root / 'configs/forge/research-cohorts' / f"{protocol['research']['cohort_id']}.json"
    return dict(role=role, replication_role='paired_seed', seed=protocol['seed'],
                    protocol_semantics_sha256=stable_hash(protocol_semantics(protocol)),
                    comparison_sha256=stable_hash(comparison_identity(registry)),
                    source_semantics_sha256=stable_hash(registry['source_semantics']),
                    source_digest=request['source']['digest'],
                    candidate_revision=request['candidate_revision'],
                    registration_sha256=file_hash(registry_path), protocol_id=protocol['id'])


def validate_unit(request, unit):
    root = Path(request['source']['snapshot_path'])
    protocol, registry = registered_protocol(root, request['protocol']['id'])
    if registry is None or protocol != request['protocol']:
        raise ValueError('replication request does not bind its registered protocol')
    expected = make_unit(root, request, registry)
    role = unit.get('role')
    declared = registry['roles'].get(role, {})
    if unit != expected or role not in registry['roles']:
        raise ValueError('experimental unit differs from the registered role/protocol/source')
    candidate = request['candidate']
    if (candidate['id'] != declared['candidate_id']
            or stable_hash(candidate.get('recipe_overrides', {})) != declared['recipe_overrides_sha256']):
        raise ValueError('experimental unit changed its frozen candidate')
    if set(registry['tasks']) != set(request['tasks']):
        raise ValueError('registered task cohort changed')
    for name, sections in registry['tasks'].items():
        if any(request['tasks'][name].get(section) != value for section, value in sections.items()):
            raise ValueError('registered original task contract changed')
    if request['view'] != registry['view'] or request['view'].get('evidence_scope') != 'research_diagnostic':
        raise ValueError('replication must remain separate from screening qualification')
    return registry


def validate_submission(request):
    """Separate, fully registered research admission; never weaken screening."""
    if any(k in request for k in ('promotion', 'calibration_lane', 'promotion_campaign', 'calibration_campaign')):
        raise ValueError('paired research cannot carry promotion/calibration metadata')
    validate_unit(request, request.get('research_unit', {}))
    from .api import FormulationContext
    candidate, protocol = request['candidate'], request['protocol']
    context = FormulationContext(recipe_preset=candidate.get('recipe_preset'), recipe_overrides=candidate.get('recipe_overrides', {}),
        prior=candidate.get('prior'), seed=protocol['seed'], requires_capabilities=candidate.get('requires_capabilities', ()),
        extensions=candidate.get('extensions', {}), initializer=candidate.get('initializer', 'deterministic_orthogonal'),
        execution_path=candidate.get('execution_path', 'public_trainer'))
    expected_rng = context.streams.manifest()
    if request.get('rng') != expected_rng:
        raise ValueError('research RNG does not match its frozen named-stream protocol')
    for job in request['jobs']:
        science = job['science']
        if (science.get('seed') != protocol['seed'] or science.get('protocol') != protocol
                or science.get('rng') != expected_rng or science.get('evidence_use') != 'research_diagnostic'):
            raise ValueError('research job changed protocol/RNG or qualification scope')
