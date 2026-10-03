"""Freeze exact current PR155 E22 on the original learned fixtures; no GPU."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERAL = ROOT.parents[1]
PARENT = ROOT.parent / 'ra13-settled'
BASELINE = GENERAL / 'visualization-pr223-convergence'
NAME = 'PR155-E22-cabe2084'
PACKAGE = BASELINE / 'pkg-pr155-e22-cabe2084'
CONFIG = BASELINE / 'configs/pr155-e22.json'
BASE = 'cabe2084284db923d525918cbf3e18de6f20faac'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def digest(root):
    out = hashlib.sha256()
    package = root / 'particlegan'
    for path in sorted(package.rglob('*.py')):
        out.update(str(path.relative_to(package)).encode() + b'\0' + path.read_bytes() + b'\0')
    return out.hexdigest()


def main():
    assert not (ROOT / 'SOURCE-FREEZE.json').exists(), 'preserve every frozen preparation'
    (ROOT / 'logs').mkdir(exist_ok=True)
    source_freeze = BASELINE / 'BASELINE-SOURCE-FREEZE.json'
    assert sha(source_freeze) == '2c34fe4f5f8950c16f1f551a9ede4e3082e64af1d19e69655b254b25677225e4'
    source = json.loads(source_freeze.read_text())
    assert source['base_commit'] == BASE
    assert source['package_sha256'] == digest(PACKAGE) == 'c174ba0b805cc8e49ca40ebbea11785bb908f3ea32615afdf2ecb43e77221316'
    assert sha(CONFIG) == '03a8c7a8d35512a4e06735c37c375895f6ecc24011c4411a606eddd92f4c202c'
    for path, expected in source['hashes'].items():
        assert sha(path) == expected, path
    changes = []
    def replace(text, old, new, label):
        assert text.count(old) == 1, label
        changes.append(dict(label=label, before=old, after=new))
        return text.replace(old, new)
    common = (PARENT / 'common.py').read_text()
    common = replace(common, "VARIANTS = ('RA13-settled',)", f"VARIANTS = ('{NAME}',)", 'current baseline report identity only')
    (ROOT / 'common.py').write_text(common)
    adapter = (PARENT / 'adapter.py').read_text()
    adapter = adapter[:adapter.index('\ndef selection_diagnostics(')] + '\n'
    (ROOT / 'adapter.py').write_text(adapter)
    runner = (PARENT / 'run_training.py').read_text()
    runner = replace(runner, 'initialize_fixture_models, selection_diagnostics', 'initialize_fixture_models', 'drop Atlas-only diagnostic import')
    runner = replace(runner, "    out['backend_selection']=selection_diagnostics(trainer)\n", '', 'no feature selection on E22')
    runner = replace(runner, "    if trainer.policy.reopen_guard is not None:out['reopen_guard']=trainer.policy.reopen_guard.state_dict()\n", '', 'no optional settled guard in PR155')
    runner = replace(runner,
        "                record['backend_selection']=validate_checkpoint_state(state,trainer.policy.roles,args.problem,recipe)\n"
        "                require(record['backend_selection']==record['diagnostics']['backend_selection'],'record/checkpoint selection disagrees')\n"
        "                require(record['diagnostics'].get('reopen_guard')==state.get('reopen_guard'),'record/checkpoint reopen guard disagrees')\n",
        "                record['baseline_recipe_contract']=validate_checkpoint_state(state,trainer.policy.roles,args.problem,recipe)\n",
        'validate current named E22 instead of Atlas selection')
    (ROOT / 'run_training.py').write_text(runner)
    write(ROOT / 'source-transform-receipt.json', dict(parent_runner=str(PARENT / 'run_training.py'),
        parent_runner_sha256=sha(PARENT / 'run_training.py'), adapted_runner_sha256=sha(ROOT / 'run_training.py'),
        parent_common_sha256=sha(PARENT / 'common.py'), parent_adapter_sha256=sha(PARENT / 'adapter.py'),
        transformations=changes, training_loop_edits=0, scorer_edits=0, model_edits=0, sampling_edits=0))
    from contracts import verify_fixture_sources
    contracts = verify_fixture_sources()
    parent = json.loads((PARENT / 'INPUTS.json').read_text())
    readonly = dict(parent['read_only_file_sha256'])
    readonly.update(source['hashes'])
    extra = [source_freeze, PARENT / 'run_training.py', PARENT / 'common.py', PARENT / 'adapter.py',
        PARENT / 'contracts.py', PARENT / 'SOURCE-FREEZE.json', PARENT / 'INPUTS.json', PARENT / 'CLOSED.json',
        GENERAL / 'release-prep/final-v4-attempt1/QUALIFICATION.json',
        GENERAL / 'release-prep/final-v4-attempt1/FROZEN.json',
        GENERAL / 'portability/ra17-current-pr155/SOURCE-BRIDGE.json',
        GENERAL / 'portability/ra17-current-pr155/SOURCE-FREEZE.json',
        GENERAL / 'mnist/ra17-replay/CLOSED.json']
    for problem in ('toy', 'mnist'):
        extra += [PARENT / 'training' / problem / 'RA13-settled' / filename
                  for filename in ('config.json', 'result.json', 'metrics.jsonl')]
        if problem == 'mnist':
            extra.append(PARENT / 'training' / problem / 'RA13-settled/evaluator.json')
    for path in extra:
        readonly[str(path)] = sha(path)
    for path, expected in readonly.items():
        assert sha(path) == expected, path
    selected = dict(package_root=str(PACKAGE), package_sha256=digest(PACKAGE),
        source_sha256={str(path.relative_to(PACKAGE / 'particlegan')): sha(path)
                      for path in sorted((PACKAGE / 'particlegan').rglob('*.py'))},
        config_path=str(CONFIG), config_sha256=sha(CONFIG), config=json.loads(CONFIG.read_text()),
        upstream_base_commit=BASE, source_variant=NAME,
        intervention='Fresh current PR155 E22 baseline, optimizer reopening/release retained; no Atlas controls.')
    inputs = dict(prepared_at=datetime.now(timezone.utc).isoformat(), variants={NAME: selected},
        upstream_base_commit=BASE, read_only_file_sha256=readonly,
        expected_initial_hashes=parent['expected_initial_hashes'], gpu_policy=parent['gpu_policy'],
        training=parent['training'], evaluator_hash_scopes=parent['evaluator_hash_scopes'],
        original_toy_quality_gate=parent['original_toy_quality_gate'], mnist_quality_gate=None,
        mnist_quality_scope='Actual comparative original metrics; no invented threshold.',
        atlas_source_training=str(PARENT / 'training'), atlas_execution_label='RA13-settled, fresh original2000 updates',
        atlas_current_source_bridge=str(GENERAL / 'portability/ra17-current-pr155/SOURCE-BRIDGE.json'),
        historical_e22_none_preserved=True, original_failed_candidates_preserved=True,
        execution_policy='Only root may launch; exact two original learned fixtures under shared GPU0 serial mutex.')
    assert selected['config']['reopen_signal'] == 'optimizer' and selected['config']['reopen_anchor'] == 'release'
    assert 'reopen_guard' not in selected['config'] and 'birth_death_backend' not in selected['config']
    write(ROOT / 'INPUTS.json', inputs)
    write(ROOT / 'preparation-receipt.json', dict(status='SOURCE_FROZEN_WAITING_FOR_CPU_PREFLIGHT_AND_ROOT_LAUNCH',
        model_initialization='Original public deterministic orthogonal G=0/D=1, no training RNG consumed',
        source_contract=contracts, planned_fresh_training_updates=4000, planned_replay_updates=0,
        steps_per_fixture=2000, fixtures=2, cuda_launches=0,
        baseline_label=NAME, current_e22_optimizer_reopening=True, atlas_controls_added=False,
        archived_e22_none_not_relabelled=True, original_learning_rates_retained=True,
        comparator_scope='Named currentPR155 E22 vscurrent Atlas recipe onoriginal learned fixtures; no ablation attribution.'))
    files = [path for path in sorted(ROOT.iterdir()) if path.suffix in ('.py', '.md')]
    files.append(ROOT / 'source-transform-receipt.json')
    write(ROOT / 'SOURCE-FREEZE.json', dict(local_source_sha256={p.name: sha(p) for p in files},
        upstream_base_commit=BASE, gpu_execution_started=False))
    print(json.dumps(dict(status='SOURCE_FROZEN_NO_GPU_EXECUTION', source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'),
        inputs_sha256=sha(ROOT / 'INPUTS.json'), read_only_guards=len(readonly), package_sha256=selected['package_sha256'])))


if __name__ == '__main__':
    main()
