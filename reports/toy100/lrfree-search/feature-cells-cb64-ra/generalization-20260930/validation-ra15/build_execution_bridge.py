"""Pin unchanged no-fire paths to actual closed RA13/RA14 execution receipts.

This source and JSON audit uses stdlib only: no tensors, models, scorers or GPU.
Inherited records retain their original source and fresh-training labels.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
BASE = STUDY / 'pkg-RA14-replay'
PACKAGE = STUDY / 'pkg-RA15-partial-recovery'
CONFIG = STUDY / 'configs/RA15-partial-recovery.json'
OLD = STUDY / 'validation-ra14'
CORRECTED = STUDY / 'validation-ra14-r2'
LEARNED = STUDY / 'mnist/ra13-settled'
REPLAY = STUDY / 'mnist/ra14-replay-r2'
pins = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path)
    pins[str(path)] = sha(path)
    return path


def read(path):
    return json.loads(pin(path).read_text())


def package_digest(package):
    h = hashlib.sha256()
    for path in sorted((package / 'particlegan').rglob('*.py')):
        pin(path)
        h.update(str(path.relative_to(package / 'particlegan')).encode() + b'\0' + path.read_bytes() + b'\0')
    return h.hexdigest()


def validate_closed(lane, closed):
    for name, expected in closed['file_sha256'].items():
        assert sha(pin(lane / name)) == expected, name


def verify_fire_monotonicity():
    continuous = BASE / 'particlegan/continuous.py'
    assert continuous.read_bytes() == (PACKAGE / 'particlegan/continuous.py').read_bytes()
    tree = ast.parse(continuous.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'OptimizerSurprise')
    writes = []
    for method in cls.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        for node in ast.walk(method):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self' and target.attr == 'fires':
                        writes.append((method.name, type(node).__name__, ast.dump(node.value, include_attributes=False), node.lineno))
    assert [(method, kind, value) for method, kind, value, line in writes] == [
        ('__init__', 'Assign', 'Constant(value=0)'), ('decide', 'AugAssign', 'Constant(value=1)')]
    decision = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'decide')
    increments = [node for node in ast.walk(decision) if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Attribute) and node.target.attr == 'fires']
    assert len(increments) == 1 and isinstance(increments[0].op, ast.Add)
    # Restore is an explicit checkpoint operation, never an ordinary decide reset.
    host = pin(CORRECTED / 'screen_current.py').read_text()
    assert 'frozen.load_state_dict(saved)' in host
    assert 'trainer.load_state_dict(' not in host
    return dict(status='PASS', source=str(continuous), sha256=sha(continuous), writes=writes,
        fresh_initial_fires=0, only_ordinary_update_write='fires += 1', no_ordinary_reset=True,
        host_main_trainer_restored_during_fresh_gate=False,
        implication='A fresh-run endpoint fires0 proves the trigger remainedFalse throughout that run.',
        restore_scope='Explicit load_state_dict may restore a saved count; inherited gates are fresh runs and disable ring frozencontrol.')


def gate(task, lane, kind):
    folder = lane / 'runs' / task
    receipt = read(folder / 'acceptance-receipt.json')
    execution = read(folder / 'execution-receipt.json')
    result = read(folder / 'result.json')
    assert receipt['acceptance_status'] == receipt['quality_status'] == 'PASS'
    assert receipt['evidence_validity'] == 'VALID' and not receipt['reasons']
    assert execution['status'] == 'COMPLETE' and execution['process_exit_code'] == 0
    assert execution['source_integrity_before']['status'] == execution['source_integrity_after']['status'] == 'VALID'
    assert execution['source_integrity_before']['package_sha256'] == package_digest(BASE)
    assert execution['source_integrity_after']['package_sha256'] == package_digest(BASE)
    assert result['header']['package_sha256'] == package_digest(BASE)
    assert result['header']['options']['ring_frozen_control'] is False
    assert result['completed_steps'] == receipt['original_plan']['steps']
    assert result['stream_deviations'] == 0
    assert sha(folder / 'result.json') == receipt['result_sha256']
    rows = [json.loads(line) for line in pin(folder / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == receipt['original_plan']['observation_steps']
    state_path = pin(folder / 'final-state.pt')
    mechanism = receipt['mechanisms']
    fires = mechanism['surprise']['fires']
    assert type(fires) is int and fires >= 0
    backend = mechanism['selection']['actual_backend']
    assert backend in ('knn', 'feature_cells')
    if task != 'ring_shift':
        assert fires == 0
    native = receipt.get('native')
    summary = None
    if kind == 'native':
        assert result['completed_steps'] == 7000 and len(rows) == 34
        accuracy = native['accuracy']
        assert accuracy['status'] == 'PASS' and accuracy['passed']
        assert all(row['passed'] for row in accuracy['terminal_checks']) and len(accuracy['terminal_checks']) == 5
        assert accuracy['holdout_metrics']['passed'] and accuracy['holdout_metrics']['n'] == 100000
        summary = dict(observations=34, terminal_steps=[r['step'] for r in accuracy['terminal_checks']],
            terminal_cloud_size=20000, holdout_size=100000, final_hq=result['final']['hq'],
            final_modes=result['final']['modes'], accuracy_status='PASS', coverage_status='PASS')
    return dict(task=task, kind=kind, execution_source=str(lane), original_candidate='RA14-replay',
        original_fresh_execution=True, fresh_RA15_execution=False, original_quality_status='PASS',
        original_evidence_validity='VALID', completed_steps=result['completed_steps'], observations=len(rows),
        backend=backend, endpoint_R1_fires=fires, final_state_path=str(state_path),
        final_state_sha256=sha(state_path), source_equivalence_eligible=task != 'ring_shift',
        bridge_reason='KNN law unchanged' if backend == 'knn' else 'fresh endpoint fires0 and monotonic detector prove defaultFalse path',
        native=summary, final=result['final'], receipt=str(folder / 'acceptance-receipt.json'))


def build():
    assert not (ROOT / 'NO-FIRE-EXECUTION-BRIDGE.json').exists()
    base_sha = package_digest(BASE)
    candidate_sha = package_digest(PACKAGE)
    assert base_sha == '68f5706590683a44348cb04b3798917411bfaa54e5d45b17d57c345a4da33c15'
    assert candidate_sha == '741fc3933654b985a74314b730b93be814d408ad46f6f7f6547b561f64619228'
    changed = [path.name for path in sorted((BASE / 'particlegan').glob('*.py'))
        if path.read_bytes() != (PACKAGE / 'particlegan' / path.name).read_bytes()]
    assert changed == ['feature_cells.py', 'feature_policy.py', 'mean_transport.py', 'output_moments.py']
    assert pin(CONFIG).read_bytes() == pin(STUDY / 'configs/RA14-replay.json').read_bytes()
    proof = read(STUDY / 'portability/partial-recovery/SOURCE-BRIDGE.json')
    independent = read(STUDY / 'diagnostics/moving-rotated-controller/PATCH-REVIEW.json')
    assert proof['status'] == 'CPU_PASS' and independent['status'] == 'PASS_INDEPENDENT_SOURCE_REVIEW'
    assert proof['static_no_fire_fixture_exact_checkpoint_and_RNG_parity']
    assert proof['package_sha256'] == candidate_sha and proof['base_package_sha256'] == base_sha
    for lane in (OLD, CORRECTED):
        frozen = read(lane / 'SOURCE-FREEZE.json')
        assert frozen['package_sha256'] == base_sha
        for path, expected in frozen['hashes'].items():
            assert sha(pin(path)) == expected, path
    monotonic = verify_fire_monotonicity()
    ports = ('mode_hold', 'img_intensity2', 'img_blobs4', 'img_bars4', 'img_stripes2',
        'vector_two_broad', 'vector_unequal_mass', 'vector_unequal_width', 'vector_anisotropic',
        'vector_overlap', 'vector_spiral', 'stationary')
    gates = [gate(task, OLD, 'port') for task in ports]
    gates += [gate(task, CORRECTED, 'native') for task in ('grid100', 'rotated100', 'staggered100')]
    affected_ring = gate('ring_shift', OLD, 'port')
    assert affected_ring['backend'] == 'feature_cells' and affected_ring['endpoint_R1_fires'] == 1
    old_learned = read(LEARNED / 'CLOSED.json')
    old_replay = read(REPLAY / 'CLOSED.json')
    validate_closed(LEARNED, old_learned)
    validate_closed(REPLAY, old_replay)
    alias = read(STUDY / 'portability/replay-alias/SOURCE-BRIDGE.json')
    restoration = read(REPLAY / 'RESTORATION-BRIDGE.json')
    assert alias['status'] == 'PASS' and alias['fresh_training_projection_byte_identical']
    assert alias['only_validation_or_restoration_callsites'] and alias['checkpoint_schema_unchanged']
    assert restoration['proof']['fresh_training_arithmetic_identical']
    assert restoration['proof']['candidate_package_sha256'] == base_sha
    assert old_replay['status'] == 'PASS_RESTORATION_ONLY_CHECKPOINT_BRIDGE_AND_ORIGINAL_CUDA_REPLAY'
    assert old_replay['fresh_training_updates'] == 0 and old_replay['fresh_replay_updates_total'] == 40
    assert old_replay['replay_status'] == dict(toy='PASS', mnist='PASS')
    learned = []
    for task in ('toy', 'mnist'):
        folder = LEARNED / 'training' / task / 'RA13-settled'
        result = read(folder / 'result.json')
        rows = [json.loads(line) for line in pin(folder / 'metrics.jsonl').read_text().splitlines()]
        assert result['status'] == 'COMPLETE' and result['steps'] == 2000
        assert len(rows) == 10 and rows[-1]['step'] == 2000
        assert all(type(row['diagnostics']['surprise']['fires']) is int and row['diagnostics']['surprise']['fires'] == 0 for row in rows)
        diag = result['final']['diagnostics']
        assert diag['surprise']['fires'] == 0
        for step in (1000, 2000):
            path = pin(folder / f'checkpoint-{step:04d}.pt')
            assert sha(path) == result['checkpoint_sha256'][path.name]
        learned.append(dict(problem=task, original_training_source='RA13-settled', original_fresh_updates=2000,
            fresh_RA14_training_updates=0, fresh_RA15_training_updates=0, observed_R1_fires=[0] * 10,
            backend=diag['backend_selection']['actual_backend'], endpoint_R1_fires=0,
            inherited_metrics=result['final']['metrics'], original_quality_gate=result['original_quality_gate'],
            checkpoint1000=str(folder / 'checkpoint-1000.pt'), checkpoint2000=str(folder / 'checkpoint-2000.pt'),
            bridge='RA13 fresh execution → RA14 restoration-only source equivalence → RA15 unchanged no-fire/KNN law',
            fresh_RA15_original_native_CPU_map_CUDA_replay='PENDING_REQUIRED_2x10_UPDATES'))
    affected = []
    for task in ('grid100', 'rotated100', 'staggered100'):
        receipt = read(CORRECTED / 'moving' / task / 'COMPLETION.json')
        assert receipt['status'] == 'COMPLETE' and receipt['source_integrity_after']['status'] == 'VALID'
        assert receipt['quality_status'] in ('PASS', 'FAIL')
        affected.append(dict(task=task, kind='moving', previous_status=receipt['quality_status'],
            fresh_required_steps=1500, turns=2, degrees=30, turn_every=500, seed=1234,
            status='PENDING_FRESH_RA15_ORIGINAL_FULL_GATE', prior_receipt=str(CORRECTED / 'moving' / task / 'COMPLETION.json')))
    affected.insert(0, dict(task='ring_shift', kind='port', previous_status='PASS', previous_R1_fires=1,
        fresh_required_steps=4600, status='PENDING_FRESH_RA15_ORIGINAL_FULL_GATE', prior_receipt=affected_ring['receipt']))
    receipt = dict(status='PASS_SOURCE_EQUIVALENCE_ONLY_FRESH_AFFECTED_GATES_PENDING',
        candidate='RA15-partial-recovery', package_sha256=candidate_sha, base_package_sha256=base_sha,
        config_sha256=sha(CONFIG), changed_modules=changed, detector_monotonicity=monotonic,
        no_fire_math_proof=independent['no_fire_source_proof'], CPU_no_fire_exact_fixture=proof['static_no_fire_fixture_exact_checkpoint_and_RNG_parity'],
        bridged_quality_tasks=gates, bridged_quality_task_count=15, original_quality_task_count=19,
        required_fresh_quality_tasks=affected, learned_source_bridges=learned,
        required_latest_full_suite='PENDING_ACTUAL_RA15_INTEGRATED_SOURCE_TESTS_AND17_RECOVERY_CONTRACTS',
        required_latest_learned_CUDA_replay='PENDING_ACTUAL_RA15_40_UPDATES_TWO_FIXTURES_TWO_BRANCHES_TEN_EACH',
        candidate_GPU_diagnostic_outcome='UNKNOWN_NOT_USED_FOR_FULL_GATE_ACCEPTANCE',
        original_RA14_quality_totals=dict(PASS=18, FAIL=1),
        prior_RA13_full_suite=dict(passed=1404, skipped=12, applies_to='RA13 only; not the latest-source test result'),
        no_new_quality_claims=True, fresh_RA15_training_updates_this_preparation=0,
        model_calls=0, scorer_calls=0, tensor_loads=0, gpu_operations=0,
        original_fresh_training_labels_preserved=True,
        limitations=['Bridged tasks are original RA14 executions with source equivalence, never relabelled freshRA15 executions.',
            'The causal1000→1500diagnostic is not a substitute for three fresh full1500moving gates.',
            'MNIST retains checkpoint metric/LR parity and ten classes without an invented numerical quality gate.',
            'Final acceptance requires every affected original gate, actual latest-source fullsuite and original CUDA replay.'],
        read_only_file_sha256=pins)
    path = ROOT / 'NO-FIRE-EXECUTION-BRIDGE.json'
    with path.open('x') as out:
        out.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(status=receipt['status'], bridged_tasks=15, fresh_quality_tasks=4,
        read_only_pins=len(pins), bridge_sha256=sha(path)), sort_keys=True))


if __name__ == '__main__':
    build()
