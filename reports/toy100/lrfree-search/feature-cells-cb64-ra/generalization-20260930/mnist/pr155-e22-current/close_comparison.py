"""Close actual current PR155 learned comparisons from receipts only; no Torch."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GENERAL = ROOT.parents[1]
ATLAS = ROOT.parent / 'ra13-settled'
NAME = 'PR155-E22-cabe2084'
CHECKPOINTS = [0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + '\n')


def curves(directory):
    rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == CHECKPOINTS, 'original checkpoints changed'
    return rows


def toy_gate(metrics):
    return (metrics['precision'] >= .90 and metrics['coverage'] == 25
        and metrics['mass_tv'] <= .10 and len(metrics['supported_mass']) == 25
        and min(metrics['supported_mass']) >= .01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'comparison-closure')
    args = parser.parse_args()
    assert not args.output.exists(), 'preserve every closure attempt'
    import common
    common.verify_inputs()
    from launch_training import verify
    integrity = verify()
    qualification = read(GENERAL / 'release-prep/final-v4-attempt1/QUALIFICATION.json')
    assert qualification['status'] == 'COMPLETE' and qualification['quality_qualification'] == 'PASS'
    assert qualification['evidence_validity'] == 'VALID'
    inputs = read(ROOT / 'INPUTS.json')
    baseline = inputs['variants'][NAME]
    pins = dict(inputs['read_only_file_sha256'])
    pins.update({str(ROOT / key): value for key, value in read(ROOT / 'SOURCE-FREEZE.json')['local_source_sha256'].items()})
    pins.update({str(path): sha(path) for path in (ROOT / 'SOURCE-FREEZE.json', ROOT / 'INPUTS.json', ROOT / 'CPU-CLOSED.json', Path(__file__).resolve())})
    rows = {}
    for problem in ('toy', 'mnist'):
        actual_dir = ROOT / 'training' / problem / NAME
        atlas_dir = ATLAS / 'training' / problem / 'RA13-settled'
        completion_path = ROOT / f'COMPLETION-{problem}.json'
        completion = read(completion_path)
        assert completion['status'] == 'COMPLETE' and completion['returncode'] == 0
        assert completion['fresh_training_updates'] == 2000
        result = read(actual_dir / 'result.json')
        candidate = read(atlas_dir / 'result.json')
        assert result['status'] == candidate['status'] == 'COMPLETE'
        assert result['steps'] == candidate['steps'] == 2000 and result['device'] == candidate['device'] == 'cuda:0'
        assert completion['result_sha256'] == sha(actual_dir / 'result.json')
        assert completion['log_sha256'] == sha(completion['log'])
        assert result['receipt']['package'] == baseline
        assert result['receipt']['seed'] == candidate['receipt']['seed'] == 314159
        assert result['receipt']['serial_backward'] and candidate['receipt']['serial_backward']
        assert result['receipt']['primary_sampling']['output_noise'] and candidate['receipt']['primary_sampling']['output_noise']
        for key, value in inputs['expected_initial_hashes'][problem].items():
            assert result['receipt'][key] == candidate['receipt'][key] == value
        bcurves, acurves = curves(actual_dir), curves(atlas_dir)
        assert result['final']['metrics'] == bcurves[-1]['metrics']
        assert candidate['final']['metrics'] == acurves[-1]['metrics']
        for step, row in zip(CHECKPOINTS, bcurves):
            assert result['checkpoint_sha256'][f'checkpoint-{step:04d}.pt'] == sha(actual_dir / f'checkpoint-{step:04d}.pt')
            assert math.isfinite(row['training_seconds']) and row['training_seconds'] >= 0
        if problem == 'mnist':
            assert read(actual_dir / 'evaluator.json') == read(atlas_dir / 'evaluator.json'), 'original evaluator scope differs'
        bm, am = result['final']['metrics'], candidate['final']['metrics']
        if problem == 'toy':
            assert result['original_quality_gate'] == ('PASS' if toy_gate(bm) else 'FAIL')
            assert candidate['original_quality_gate'] == ('PASS' if toy_gate(am) else 'FAIL')
            compact = dict(precision={'current_e22': bm['precision'], 'atlas': am['precision'], 'higher_is_better': True},
                coverage={'current_e22': bm['coverage'], 'atlas': am['coverage'], 'maximum': 25},
                mass_tv={'current_e22': bm['mass_tv'], 'atlas': am['mass_tv'], 'lower_is_better': True})
        else:
            assert result.get('original_quality_gate') is None and candidate.get('original_quality_gate') is None
            compact = {key: dict(current_e22=bm['active_embedding'][key], atlas=am['active_embedding'][key],
                       higher_is_better=(key != 'embedding_frechet'))
                       for key in ('embedding_frechet', 'embedding_precision', 'embedding_recall')}
            compact['confident_class_coverage'] = dict(current_e22=bm['confident_class_coverage'],
                atlas=am['confident_class_coverage'], maximum=10)
        rows[problem] = dict(current_e22_execution='fresh2000, PR155-E22-cabe2084',
            atlas_execution='fresh2000, RA13-settled; current-source bridge/replay retained',
            current_e22_result=str(actual_dir / 'result.json'), atlas_result=str(atlas_dir / 'result.json'),
            original_steps=2000, initial_hashes_exact=True, compact_metrics=compact,
            current_e22_metrics=bm, atlas_metrics=am,
            current_e22_gate=result.get('original_quality_gate'), atlas_gate=candidate.get('original_quality_gate'),
            current_e22_actual_training_seconds=result['training_seconds'],
            atlas_actual_training_seconds=candidate['training_seconds'],
            current_e22_started_at=result['receipt']['started_at'],
            current_e22_completed_at_epoch=completion['completed'],
            atlas_started_at=candidate['receipt']['started_at'],
            current_e22_fires=result['final']['diagnostics']['surprise']['fires'],
            atlas_fires=candidate['final']['diagnostics']['surprise']['fires'],
            current_e22_reopen_events=result['final']['diagnostics']['surprise']['log'],
            atlas_reopen_events=candidate['final']['diagnostics']['surprise']['log'],
            checkpoint_comparison=[dict(step=b['step'], current_e22_metrics=b['metrics'], atlas_metrics=a['metrics'],
                current_e22_lrs=b['diagnostics']['lr'], atlas_lrs=a['diagnostics']['lr'],
                current_e22_surprise=b['diagnostics']['surprise'], atlas_surprise=a['diagnostics']['surprise'],
                current_e22_output_sigma=b['diagnostics']['output_sigma'], atlas_output_sigma=a['diagnostics']['output_sigma'],
                metrics_identical=b['metrics'] == a['metrics'],
                lrs_identical=b['diagnostics']['lr'] == a['diagnostics']['lr']) for b, a in zip(bcurves, acurves)])
        for filename in ('config.json', 'metrics.jsonl', 'result.json'):
            for directory in (actual_dir, atlas_dir):
                pins[str(directory / filename)] = sha(directory / filename)
        for path in (completion_path, Path(completion['log'])):
            pins[str(path)] = sha(path)
        if problem == 'mnist':
            for directory in (actual_dir, atlas_dir):
                pins[str(directory / 'evaluator.json')] = sha(directory / 'evaluator.json')
    for path, expected in pins.items():
        assert sha(path) == expected, path
    args.output.mkdir()
    report = dict(status='COMPLETE_CURRENT_PR155_E22_VS_ATLAS_COMPARISON', validity='VALID',
        created_utc=datetime.now(timezone.utc).isoformat(),
        upstream_base_commit='cabe2084284db923d525918cbf3e18de6f20faac',
        current_e22_package_sha256=baseline['package_sha256'],
        current_e22_config_sha256=baseline['config_sha256'], current_e22_optimizer_reopening=True,
        latest_atlas_package_sha256='500ff0e966beb649dd7cafa0b91d7bb30cb451e5d62ece2883411a0507c8df61',
        fresh_current_e22_updates_total=4000, new_atlas_training_updates=0, source_integrity=integrity,
        fixtures=rows, scope='Two original fixed-seed learned fixtures; recipe comparison, not isolated mechanism attribution.',
        caveats=['MNIST uses existing kNN controls in both recipes.',
            'No numerical MNIST acceptance threshold added.',
            'Recorded timings are actual observations, not a controlled speed claim.',
            'Historical E22-without-reopening remains distinct.',
            'Atlas original fresh execution labels and source/replay bridges remain unchanged.'])
    write(args.output / 'COMPARISON.json', report)
    write(args.output / 'INPUTS.json', pins)
    write(args.output / 'FROZEN.json', dict(status=report['status'], validity='VALID',
        artifact_sha256={name: sha(args.output / name) for name in ('COMPARISON.json', 'INPUTS.json')},
        input_count=len(pins), gpu_calls_by_closer=0, model_calls_by_closer=0))
    print(json.dumps(dict(status=report['status'], validity='VALID', output=str(args.output),
        comparison_sha256=sha(args.output / 'COMPARISON.json'), input_count=len(pins))), flush=True)


if __name__ == '__main__':
    main()
