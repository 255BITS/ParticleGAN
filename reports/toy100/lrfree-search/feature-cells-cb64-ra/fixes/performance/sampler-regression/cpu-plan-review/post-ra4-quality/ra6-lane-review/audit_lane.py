"""Read-only prepared-lane audit using stdlib hashes and ASTs."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
BASE = ROOT.parent / 'feature-cells-cuda-retest-20260929'
LANE = ROOT / 'validation-cb64-ra6'
HERE = Path(__file__).resolve().parent
VARIANT = 'CB64-RA6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(node):
    return ast.dump(node, include_attributes=False)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def assignment(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def verify_map(entries, prefix=None):
    for name, expected in entries.items():
        path = Path(name) if prefix is None else prefix / name
        assert sha(path) == expected, str(path)


def jobs(tree):
    fn = function(tree, 'jobs')
    tasks = next(n for n in tree.body if assignment(n, 'TASKS'))
    ns = dict(ROOT=LANE, PYTHON='/tmp/pr38-default-env/bin/python')
    exec(compile(ast.Module(body=[tasks, fn], type_ignores=[]), '<pure jobs>', 'exec'), ns)
    return list(ns['jobs']())


def main():
    ready_path = ROOT / 'quality/ra6/READY.json'
    assert sha(ready_path) == '873c4394d6cce5967dba4298bc626c96953bde744ba65960e777e89200bdec0e'
    ready = read(ready_path)
    verify_map(ready['numerical_source_sha256'])
    frozen_path = LANE / 'source-freeze.json'
    frozen = read(frozen_path)
    verify_map(frozen['local_sources'], LANE)
    verify_map(frozen['external_sources'])
    assert frozen['package_sha256'] == ready['package_sha256'] == '6bb967c405c486d2a5cbd5dcc3bce7d1bd2cb93b3f9c7e916229f1327a8e5fc5'
    assert frozen['config_sha256'] == ready['config_sha256'] == 'd2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7'
    screen_freeze = read(LANE / 'screens/source-freeze.json')
    for item in screen_freeze['files'].values():
        assert sha(item['path']) == item['sha256'], item['path']
    screen_ready = read(LANE / 'screens/READY.json')
    verify_map(screen_ready['lane_source_sha256'], LANE / 'screens')
    assert screen_ready['source_freeze_sha256'] == sha(LANE / 'screens/source-freeze.json')
    learned_freeze = read(LANE / 'learned/SOURCE-FREEZE.json')
    verify_map(learned_freeze['local_source_sha256'], LANE / 'learned')
    baseline_inputs = read(BASE / 'learned/INPUTS.json')
    inputs = read(LANE / 'learned/INPUTS.json')
    assert {k: v for k, v in inputs.items() if k not in ('variants', 'origin_completed_cuda_baseline')} == {
        k: v for k, v in baseline_inputs.items() if k not in ('variants', 'origin_completed_cuda_baseline')}
    assert set(inputs['variants']) == {VARIANT}
    candidate = inputs['variants'][VARIANT]
    assert candidate['package_sha256'] == ready['package_sha256']
    assert candidate['source_sha256'] == ready['package_source_sha256']
    assert candidate['config_sha256'] == ready['config_sha256']
    assert (LANE / 'learned/common.py').read_text() == (BASE / 'learned/common.py').read_text().replace(
        'VARIANTS = ("E22", "CB64-RA")', 'VARIANTS = ("CB64-RA6",)')
    for name in ('run_training.py', 'replay.py'):
        assert (LANE / 'learned' / name).read_bytes() == (BASE / 'learned' / name).read_bytes()
    assert (LANE / 'screens/candidate-options.json').read_bytes() == (BASE / 'screens/candidate-options.json').read_bytes()
    old_screen = read(BASE / 'screens/source-freeze.json')
    retained = {key: item for key, item in old_screen['files'].items()
        if not Path(item['path']).is_relative_to(BASE / 'screens')
        and not Path(item['path']).is_relative_to(ROOT.parent / 'feature-cells-config-20260929')}
    for key, item in retained.items():
        assert screen_freeze['files'][key] == item

    original_collector = (BASE / 'screens/collect.py').read_text().replace('CB64-RA', VARIANT)
    collector_tree = ast.parse((LANE / 'screens/collect.py').read_text())
    declaration = read(LANE / 'screens/DECLARED-API-EXPECTATION.json')
    assert declaration['adapted_ast_sha256'] == hashlib.sha256(dump(collector_tree).encode()).hexdigest()
    assert declaration['original_ast_sha256'] == hashlib.sha256(dump(ast.parse(original_collector)).encode()).hexdigest()
    collect_fn = function(collector_tree, 'collect')
    options = [n for n in ast.walk(collect_fn) if assignment(n, 'expected_options')]
    assert len(options) == 1
    value = next(k.value for k in options[0].value.keywords if k.arg == 'evaluation_generate')
    assert value.value == 'indexed'
    value.value = 'plain'
    assert dump(collector_tree) == dump(ast.parse(original_collector))
    assert declaration['declared_before_numerical_execution'] is True
    training_tree = ast.parse((ROOT / 'pkg-CB64-RA6/particlegan/training.py').read_text())
    trainer = next(n for n in training_tree.body if isinstance(n, ast.ClassDef) and n.name == 'GANTrainer')
    generate = function(trainer, '_generate')
    assert len(generate.args.args) >= 6 and generate.args.args[5].arg == 'indices'

    launch_tree = ast.parse((LANE / 'launch.py').read_text())
    original_launch = ast.parse((ROOT / 'validation-ra4/launch.py').read_text().replace('CB64-RA4', VARIANT))
    plan, original_plan = jobs(launch_tree), jobs(original_launch)
    assert len(plan) == len(original_plan) == 19
    assert {j['name']: j for j in plan} == {j['name']: j for j in original_plan}
    assert len([j for j in plan if j['name'].startswith('screen-')]) == 16
    assert [j['name'] for j in plan[:4]] == ['learned-toy-CB64-RA6', 'screen-grid100', 'learned-mnist-CB64-RA6', 'replay-CB64-RA6']
    fn = function(launch_tree, 'jobs')
    launch_tree.body[launch_tree.body.index(fn)] = function(original_launch, 'jobs')
    assert dump(launch_tree) == dump(original_launch)
    assert (LANE / 'QUALITY-TARGETS.md').read_bytes() == (ROOT / 'quality/PROTOCOL.md').read_bytes()
    screen = Path('/ml2/hypergan/lrfree-20260926/harness/screen.py')
    assert sha(screen) == 'ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c'
    assert str(ready_path) in frozen['external_sources']
    for name in ('prepare_validation4.py', 'quality/prepare_lane.py'):
        assert str(ROOT / name) in frozen['external_sources']
    for p in LANE.rglob('*.py'):
        compile(p.read_text(), str(p), 'exec')
    monitor = ROOT / 'integration/review/monitor_validation.py'
    assert sha(monitor) == '4ee4cae810342b675a39b269112ca907be3ffe88ae9cadab157e4933c487328b'
    receipt = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(), validation=str(LANE),
        ready_sha256=sha(ready_path), source_freeze_sha256=sha(frozen_path),
        package_sha256=frozen['package_sha256'], config_sha256=frozen['config_sha256'],
        checked_maps=dict(root_numerical=len(ready['numerical_source_sha256']), local=len(frozen['local_sources']),
            external=len(frozen['external_sources']), screen=len(screen_freeze['files']), retained_original_screen_inputs=len(retained)),
        checks=dict(all_frozen_maps_exact=True, full_original_learned_input_maps_equal=True,
            learned_scorers_replay_bytes_exact=True, screen_fixture_scorer_maps_preserved=True,
            collector_only_declared_plain_to_indexed_AST_literal=True, positional_indices_API_compatible=True,
            launcher_only_jobs_AST_changes=True, all_19_job_specs_and_16_screens_exact=True,
            toy_first_grid_second=True, original_candidate_options_noise_and_stream_checks=True,
            helper_template_ready_and_declaration_frozen=True, all_generated_sources_compile=True,
            canonical_harness_and_monitor_sources_unchanged=True),
        job_order=[j['name'] for j in plan], declaration=declaration,
        monitor_source_sha256=sha(monitor), numerical_execution=False, defects=[])
    assert not (HERE / 'receipt.json').exists()
    (HERE / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status='PASS', jobs=19, screens=16, source_freeze_sha256=receipt['source_freeze_sha256'])))


if __name__ == '__main__':
    main()
