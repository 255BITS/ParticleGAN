"""Freeze root-reviewed candidate and original learned inputs; never launch."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERAL = ROOT.parents[1]
ORIGINAL = Path('/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929')
LANE = ORIGINAL / 'validation-cb64-ra11/learned'
NAME = 'RA12-auto'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2) + '\n')


def package_sources(root):
    package = root / 'particlegan'
    return {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*.py'))}


def package_digest(root):
    digest = hashlib.sha256()
    package = root / 'particlegan'
    for path in sorted(package.rglob('*.py')):
        digest.update(str(path.relative_to(package)).encode() + b'\0' + path.read_bytes() + b'\0')
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-receipt', type=Path)
    parser.add_argument('--seal-cpu', action='store_true')
    args = parser.parse_args()
    if args.seal_cpu:
        import common
        common.verify_inputs()
        receipt = json.loads((ROOT / 'preflight-cpu.json').read_text())
        assert receipt['status'] == 'PASS' and receipt['training_updates'] == 0
        assert receipt['begin_step_calls'] == 0 and not receipt['cuda_context_initialized']
        files = ['SOURCE-FREEZE.json', 'INPUTS.json', 'preparation-receipt.json', 'preflight-cpu.json',
                 'logs/preflight-cpu.log', 'ADAPTER-PREPARATION.json', 'source-transform-receipt.json']
        write(ROOT / 'CPU-CLOSED.json', dict(status='PASS_ORIGINAL_FIXTURE_CURRENT_API_PREFLIGHT',
              training_updates=0, model_forwards=0, sampling_calls=0, begin_step_calls=0,
              cuda_context_initialized=False, file_sha256={name: sha(ROOT / name) for name in files}))
        print(json.dumps(dict(status='PASS_CPU_CLOSED', sha256=sha(ROOT / 'CPU-CLOSED.json'))))
        return
    assert args.review_receipt is not None and args.review_receipt.is_file(), 'root-reviewed candidate receipt is required'
    assert not (ROOT / 'SOURCE-FREEZE.json').exists(), 'learned adapter already frozen'
    from contracts import verify_fixture_sources
    verify_fixture_sources()
    package = GENERAL / 'pkg-RA12-auto'
    config_path = GENERAL / 'configs/RA12-auto.json'
    config = json.loads(config_path.read_text())
    reference = json.loads((GENERAL / 'configs/RA11-R1-historical-rates.json').read_text())
    reference.pop('initialization')
    reference['birth_death_backend'] = 'auto'
    assert config == reference, 'shared auto config differs beyond removed initialization/backend auto'
    assert 'initialization' not in config
    selected = dict(package_root=str(package), package_sha256=package_digest(package),
                    source_sha256=package_sources(package), config_path=str(config_path),
                    config_sha256=sha(config_path), config=config,
                    intervention='current PR155 R1 plus declared feature/reference auto capability and role rates',
                    source_variant='RA12-auto')
    original = json.loads((LANE / 'INPUTS.json').read_text())
    read_only = dict(original['read_only_file_sha256'])
    for path, expected in read_only.items():
        assert sha(path) == expected, path
    extra = [LANE / name for name in ('INPUTS.json', 'run_training.py', 'common.py', 'replay.py')]
    extra += [ORIGINAL / 'quality/leaderboard.py', GENERAL / 'configs/RA11-R1-historical-rates.json',
              config_path, args.review_receipt.resolve()]
    controls = {
        'CB64-RA11': LANE / 'training',
        'E22': ORIGINAL.parent / 'feature-cells-cuda-retest-20260929/learned/training',
    }
    for problem in ('toy', 'mnist'):
        for variant, directory in controls.items():
            for name in ('config.json', 'result.json', 'metrics.jsonl'):
                extra.append(directory / problem / variant / name)
            if problem == 'mnist':
                extra.append(directory / problem / variant / 'evaluator.json')
    for path in extra:
        read_only[str(path)] = sha(path)
    inputs = dict(original, prepared_at=datetime.now(timezone.utc).isoformat(), variants={NAME: selected},
                  read_only_file_sha256=read_only, origin_lane=str(LANE),
                  candidate_review_receipt=dict(path=str(args.review_receipt.resolve()), sha256=sha(args.review_receipt)),
                  comparative_control_training_roots={k: str(v) for k, v in controls.items()},
                  execution_policy='Root authorizes exact actual-candidate runs inside existing owned outer GPU0 slot and shared serial lock.',
                  original_toy_quality_gate={'precision_min': .9, 'coverage': 25, 'mass_tv_max': .1, 'mode_mass_min': .01},
                  mnist_quality_gate=None, mnist_quality_scope='Original comparative regression; no threshold added.')
    write(ROOT / 'INPUTS.json', inputs)
    commands = {problem: ['/tmp/pr38-default-env/bin/python', '-u', '-B', str(ROOT / 'launch_first.py'), '--problem', problem]
                for problem in ('toy', 'mnist', 'replay')}
    write(ROOT / 'preparation-receipt.json', dict(status='SOURCE_FROZEN_WAITING_FOR_CPU_AND_ROOT_GPU_AUTHORIZATION',
          shared_config_source=str(config_path), candidate_review_receipt=inputs['candidate_review_receipt'],
          current_public_init=True, model_init_hashes_required_exact=True,
          no_synthetic_updates_or_begin_step=True, scorer_ast_and_original_gate_unchanged=True,
          estimated_train_wall_seconds={'toy': 160, 'mnist': 110, 'replay': 10},
          estimate_scope='prior original diagnostic costs; actual current API candidate remains unmeasured',
          proposed_commands=commands, cuda_launches=0))
    files = [path for path in sorted(ROOT.iterdir()) if path.suffix in ('.py', '.md')]
    files += [ROOT / 'source-transform-receipt.json']
    write(ROOT / 'SOURCE-FREEZE.json', dict(local_source_sha256={path.name: sha(path) for path in files}))
    print(json.dumps(dict(status='SOURCE_FROZEN_NO_NUMERICAL_EXECUTION',
                         source_freeze_sha256=sha(ROOT / 'SOURCE-FREEZE.json'), inputs_sha256=sha(ROOT / 'INPUTS.json'),
                         proposed_commands=commands)))


if __name__ == '__main__':
    main()
