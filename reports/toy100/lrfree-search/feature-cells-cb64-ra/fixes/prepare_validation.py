"""Create a fresh, immutable GPU acceptance lane for an integrated correction.

Only the candidate, output locations and execution plan differ from the completed
CUDA baseline. Data, model initialization, scorers, quality gates and seeds stay
unchanged. Run after integration; refuses a second preparation.
"""
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'feature-cells-cuda-retest-20260929'
OUT = ROOT / 'validation'
PACKAGE = ROOT / 'pkg-CB64-RA2'
CONFIG = ROOT / 'configs/overrides-CB64-RA2.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def package_identity():
    sources = {}
    digest = hashlib.sha256()
    for path in sorted((PACKAGE / 'particlegan').rglob('*.py')):
        name = str(path.relative_to(PACKAGE / 'particlegan'))
        sources[name] = sha(path)
        digest.update(name.encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest(), sources


def metric_nodes(source):
    tree = ast.parse(source)
    wanted = {'frechet', 'embedding_score', 'diagnostics', 'ImageEvaluation'}
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in wanted}


def main():
    if OUT.exists():
        raise SystemExit('Validation directory exists; refusing to overwrite frozen evidence.')
    package_sha, source_map = package_identity()
    config = json.loads(CONFIG.read_text())
    learned = OUT / 'learned'
    screens = OUT / 'screens'
    learned.mkdir(parents=True)
    screens.mkdir()
    common = (BASE / 'learned/common.py').read_text().replace(
        'VARIANTS = ("E22", "CB64-RA")', 'VARIANTS = ("CB64-RA2",)')
    (learned / 'common.py').write_text(common)
    for name in ('run_training.py', 'replay.py'):
        (learned / name).write_bytes((BASE / 'learned' / name).read_bytes())
    assert metric_nodes((learned / 'run_training.py').read_text()) == metric_nodes(
        (BASE / 'learned/run_training.py').read_text())
    protocol = '''# Corrected CB64-RA2 CUDA acceptance

The candidate is frozen before any quality execution. Use physical GPU 0 only,
one numerical process at a time, deterministic algorithms and no TF32. Keep the
completed baseline's data, evaluator, seeds, G/D/prior initialization and gates.
Learned toy and MNIST: N=1024, z=128, batch=128, seed=314159, 2000 updates and all
ten saved checkpoints. Replay checkpoint 1000 twice for ten updates; only
birth_death.last.eval_seconds may differ. Reference measurements come from the
completed matched-input E22/CB64-RA CUDA runs; do not rerun unchanged controls.
Canonical screen.py and its scorers are unmodified, live noisy output is primary.
All 16 tasks receive their original budgets. Each native task has 7000 updates,
34 observations, five terminal 20k evaluations and a separate 100k holdout.
No source, config, quality gate, fixture or seed may change during this lane.
'''
    (OUT / 'PROTOCOL.md').write_text(protocol)
    (learned / 'PROTOCOL.md').write_text(protocol)
    inputs = json.loads((BASE / 'learned/INPUTS.json').read_text())
    inputs['variants'] = {'CB64-RA2': dict(package_root=str(PACKAGE),
        package_sha256=package_sha, source_sha256=source_map,
        config_path=str(CONFIG), config_sha256=sha(CONFIG), config=config)}
    inputs['origin_completed_cuda_baseline'] = dict(path=str(BASE),
        source_freeze_sha256=sha(BASE / 'source-freeze.json'),
        execution_results_sha256=sha(BASE / 'execution-results.json'))
    write(learned / 'INPUTS.json', inputs)
    learned_local = {name: sha(learned / name) for name in
        ('common.py', 'run_training.py', 'replay.py', 'PROTOCOL.md', 'INPUTS.json')}
    write(learned / 'SOURCE-FREEZE.json', dict(status='FROZEN_PRE_EXECUTION',
        local_source_sha256=learned_local, candidate_package_sha256=package_sha,
        candidate_config_sha256=sha(CONFIG)))
    write(learned / 'preparation-receipt.json', dict(status='PREPARED_NO_NUMERICAL_EXECUTION',
        unchanged_metric_helpers=True, baseline=str(BASE),
        package_sha256=package_sha, config_sha256=sha(CONFIG),
        source_freeze_sha256=sha(learned / 'SOURCE-FREEZE.json')))
    write(learned / 'READY.json', dict(status='READY', local_source_sha256=learned_local,
        source_freeze_sha256=sha(learned / 'SOURCE-FREEZE.json')))

    lane = (BASE / 'screens/lane.py').read_text().replace(
        "STUDY = Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929')",
        f'STUDY = Path({str(ROOT)!r})').replace('CB64-RA', 'CB64-RA2')
    lane = lane.replace("PACKAGE_SHA = '13f5bbe4de824e6899cb28ee4dff8f35d74bbaa6243bfe9c18b8df44d173a1ce'",
        f'PACKAGE_SHA = {package_sha!r}')
    lane = lane.replace("CONFIG_SHA = 'd2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7'",
        f'CONFIG_SHA = {sha(CONFIG)!r}')
    (screens / 'lane.py').write_text(lane)
    for name in ('run_screen.py', 'collect.py'):
        (screens / name).write_text((BASE / 'screens' / name).read_text().replace('CB64-RA', 'CB64-RA2'))
    (screens / 'candidate-options.json').write_bytes((BASE / 'screens/candidate-options.json').read_bytes())
    # Preserve all original fixture/scorer input hashes, replacing candidate and
    # wrapper maps by complete maps of this newly frozen candidate and lane.
    old_screen_freeze = json.loads((BASE / 'screens/source-freeze.json').read_text())
    files = {key: value for key, value in old_screen_freeze['files'].items()
        if not Path(value['path']).is_relative_to(BASE / 'screens')
        and not Path(value['path']).is_relative_to(ROOT.parent / 'feature-cells-config-20260929')}
    for path in sorted((PACKAGE / 'particlegan').rglob('*.py')) + [CONFIG]:
        files[str(path)] = dict(path=str(path), sha256=sha(path))
    for path in sorted(screens.glob('*')):
        files[str(path)] = dict(path=str(path), sha256=sha(path))
    write(screens / 'source-freeze.json', dict(status='FROZEN_PRE_EXECUTION', files=files,
        candidate_package_sha256=package_sha, candidate_config_sha256=sha(CONFIG)))
    screen_local = {p.name: sha(p) for p in sorted(screens.glob('*'))}
    write(screens / 'READY.json', dict(status='READY', lane_source_sha256=screen_local,
        source_freeze_sha256=sha(screens / 'source-freeze.json')))

    launcher = (BASE / 'launch.py').read_text().replace(
        "for variant in ('E22', 'CB64-RA'):", "for variant in ('CB64-RA2',):")
    # Diagnose former failures early; nevertheless execute every original task.
    old_order = "TASKS = ('mode_hold', 'img_intensity2', 'img_blobs4', 'img_stripes2', 'img_bars4',\n         'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width',\n         'vector_anisotropic', 'vector_overlap', 'vector_spiral',\n         'ring_shift', 'stationary', 'grid100', 'rotated100', 'staggered100')"
    new_order = "TASKS = ('mode_hold', 'img_blobs4', 'vector_unequal_mass', 'ring_shift',\n         'grid100', 'rotated100', 'staggered100', 'stationary',\n         'img_intensity2', 'img_stripes2', 'img_bars4', 'vector_two_broad',\n         'vector_unequal_width', 'vector_anisotropic', 'vector_overlap', 'vector_spiral')"
    assert old_order in launcher
    launcher = launcher.replace(old_order, new_order)
    launcher = launcher.replace('        event(\'job_complete\', **receipt)\n',
        '        event(\'job_complete\', **receipt)\n'
        '        if process.returncode != 0 or status == \'ERROR\':\n'
        '            event(\'queue_aborted\', name=job[\'name\'], reason=\'runtime or evidence error\')\n'
        '            raise SystemExit(1)\n')
    (OUT / 'launch.py').write_text(launcher)
    local = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    external = dict(inputs['read_only_file_sha256'])
    external.update({v['path']: v['sha256'] for v in files.values()
        if not Path(v['path']).is_relative_to(OUT)})
    external.update({str(PACKAGE / 'particlegan' / p): h for p, h in source_map.items()})
    external[str(CONFIG)] = sha(CONFIG)
    external[str(ROOT / 'READY.json')] = sha(ROOT / 'READY.json')
    external[str(Path(__file__))] = sha(Path(__file__))
    write(OUT / 'source-freeze.json', dict(status='FROZEN_PRE_EXECUTION',
        frozen_at=datetime.now(timezone.utc).isoformat(), local_sources=local,
        external_sources=external, baseline=str(BASE), package_sha256=package_sha,
        config_sha256=sha(CONFIG), seed_policy='unchanged, no seed experiments'))
    print(json.dumps(dict(status='READY', validation=str(OUT),
        package_sha256=package_sha, config_sha256=sha(CONFIG),
        frozen_local_files=len(local), frozen_external_files=len(external))), flush=True)


if __name__ == '__main__':
    main()
