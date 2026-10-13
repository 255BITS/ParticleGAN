"""Finite native-direction/critic-floor comparison on unchanged Forge tasks.

This report wrapper preserves the original workflow and queue. Root owns the
five declarations, preparation, committed-source admission and collection.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012')
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-direction-floor-repair-20261012')
PRIOR_ARCHIVES = (ORIGINAL,
    Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-five-track-repair-20261012'),
    Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-role-repair-20261012'))
CAMPAIGN = 'r1r2-direction-floor-repair-v1'
ZERO_CANDIDATE = 'r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9'
ROLES = {'baseline', 'adam', 'adam_eps', 'ada', 'ada_eps'}

spec = importlib.util.spec_from_file_location(
    'r1r2_original_direction_floor_workflow', ROOT / 'reports/forge/r1r2-tier1-search/workflow.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
original_requests = shared.requests
original_storage = shared.storage
original_declarations = shared.declaration_paths


def validate_manifest(value):
    shared.require(value['schema_version'] == 1 and value['round'] == 1
                   and value['id'] == CAMPAIGN, 'Use the separately registered finite direction/floor campaign')
    shared.require(value['max_candidates'] == 5 and len(value['candidates']) == 5
                   and not value.get('searches'), 'Three new whole recipes and two exact cached controls')
    shared.require({card['role'] for card in value['candidates']} == ROLES,
                   'Use the explicit native optimizer by critic epsilon comparison')
    shared.require(value['baseline_role'] == 'baseline', 'Preserve the scheduled reference')
    shared.require(value['paid_cap_seconds'] == 8000, 'Preserve the shared aggregate paid ceiling')
    shared.require(value['campaign']['id'] == CAMPAIGN
                   and value['campaign']['candidate_budget_seconds'] == shared.ALLOWANCE
                   and value['campaign']['budget_seconds'] == 5 * shared.ALLOWANCE,
                   'Reserve each unchanged full task allowance')
    return value


def requests(root, queue_root, manifest):
    resolved, searches = original_requests(root, queue_root, manifest)
    first = shared.read_json(root / 'reports/forge/r1r2-tier1-search/registration.json')
    shared.require(resolved['adam']['projection']['candidate'] == ZERO_CANDIDATE,
                   'Adam control must retain the exact measured zero-momentum card')
    measured = [p for p in first['candidates'].values() if p['candidate'] == ZERO_CANDIDATE]
    shared.require(len(measured) == 1, 'Missing unique original Adam control binding')
    controls = {'baseline': first['candidates']['baseline'], 'adam': measured[0]}
    state = shared.read_json(queue_root / 'queue/state.json')
    for role, previous in controls.items():
        expected = {key: value for key, value in previous.items() if key != 'card'}
        shared.require(resolved[role]['projection'] == expected,
                       f'{role}: retain exact measured source/runtime/protocol/task/job identity')
        shared.require(set(expected['task_keys']) == shared.TASKS,
                       f'{role}: preserve all six ordinary tasks and own clock proof')
        shared.require(all(state['jobs'].get(key, {}).get('result') is not None
                           and len(state['jobs'][key]['attempts']) == 1
                           for key in expected['task_keys'].values()),
                       f'{role}: all seven original one-attempt results must exist; no control rerun')
    return resolved, searches


def declaration_paths(root, manifest, resolved):
    # Reusing a measured configuration never moves it into the idea registry.
    paths = original_declarations(root, manifest, resolved)
    for item in resolved.values():
        candidate = item['card']['candidate']
        idea = f'configs/forge/ideas/{candidate}.json'
        configuration = f'configs/forge/configurations/{candidate}.json'
        if not (root / idea).exists():
            shared.require((root / configuration).exists(), 'Missing original candidate declaration')
            paths.remove(idea)
            paths.add(configuration)
    return paths


def storage(archive):
    shared.require(archive == ARCHIVE and (archive / 'queue').is_symlink()
                   and (archive / 'queue').resolve() == ORIGINAL / 'queue',
                   'Retain the exact original queue, artifacts and every historical charge')
    locations = (archive, *PRIOR_ARCHIVES)
    readings = [original_storage(location) for location in locations]
    shared.require(sum(reading['bytes'] for reading in readings) < shared.ARCHIVE_CAP,
                   'All four research archives exceed the combined storage cap')
    return dict(archives={str(path): reading for path, reading in zip(locations, readings, strict=True)})


shared.validate_manifest = validate_manifest
shared.requests = requests
shared.declaration_paths = declaration_paths
shared.storage = storage
shared.OUT = OUT

if __name__ == '__main__':
    shared.main()
