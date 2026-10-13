"""Finite role-momentum comparison on the unchanged public Forge tasks."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ORIGINAL = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012')
ARCHIVE = Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-role-repair-20261012')
CAMPAIGN = 'r1r2-role-repair-v1'

spec = importlib.util.spec_from_file_location(
    'r1r2_original_workflow', ROOT / 'reports/forge/r1r2-tier1-search/workflow.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
original_requests = shared.requests
original_storage = shared.storage
original_declarations = shared.declaration_paths


def validate_manifest(value):
    shared.require(value['schema_version'] == 1 and value['round'] == 1
                   and value['id'] == CAMPAIGN, 'Use the independently registered finite campaign')
    shared.require(value['max_candidates'] == 10 and len(value['candidates']) == 10
                   and not value.get('searches'), 'Seven new whole recipes and three cached controls')
    shared.require(value['paid_cap_seconds'] == 8000, 'Preserve the shared aggregate paid ceiling')
    shared.require(value['campaign']['candidate_budget_seconds'] == shared.ALLOWANCE
                   and value['campaign']['budget_seconds'] == 10 * shared.ALLOWANCE,
                   'Reserve each unchanged full task budget')
    shared.require(value['baseline_role'] == 'baseline', 'Preserve the scheduled reference')
    shared.require({c['role'] for c in value['candidates']} ==
                   {'baseline', 'm000', 'm111', 'm100', 'm010', 'm001',
                    'm110', 'm101', 'm011', 'ams_m111'}, 'Use the declared role factorial')
    return value


def requests(root, queue_root, manifest):
    resolved, searches = original_requests(root, queue_root, manifest)
    first = shared.read_json(root / 'reports/forge/r1r2-tier1-search/registration.json')
    last = shared.read_json(root / 'reports/forge/r1r2-five-track-repair/registration.json')
    zeros = next(v for v in first['candidates'].values()
                 if v['candidate'] == resolved['m000']['projection']['candidate'])
    controls = {'baseline': last['candidates']['baseline'],
                'm000': zeros, 'm111': last['candidates']['momentum']}
    state = shared.read_json(queue_root / 'queue/state.json')
    for role, previous in controls.items():
        expected = {k: v for k, v in previous.items() if k != 'card'}
        shared.require(resolved[role]['projection'] == expected,
                       f'{role}: retain exact measured source/runtime/task/job identity')
        shared.require(all(state['jobs'].get(key, {}).get('result') is not None
                           and len(state['jobs'][key]['attempts']) == 1
                           for key in expected['task_keys'].values()),
                       f'{role}: all seven original results must already exist; no control rerun')
    return resolved, searches


def declaration_paths(root, manifest, resolved):
    # A measured configuration remains in its original registry category.
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
                   'Retain original artifacts and every original paid charge')
    locations = [archive, ORIGINAL,
                 Path('/mnt/ml7tb/ParticleGAN-forge/r1r2-five-track-repair-20261012')]
    readings = [original_storage(location) for location in locations]
    shared.require(sum(r['bytes'] for r in readings) < shared.ARCHIVE_CAP,
                   'Combined research archive exceeds its storage cap')
    return dict(archives={str(p): r for p, r in zip(locations, readings)})


shared.validate_manifest = validate_manifest
shared.requests = requests
shared.declaration_paths = declaration_paths
shared.storage = storage
shared.OUT = OUT

if __name__ == '__main__':
    shared.main()
