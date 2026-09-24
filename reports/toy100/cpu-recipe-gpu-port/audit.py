"""Regrade all backend controls and verify initialization/random-stream parity."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from prepare import prepare

ROOT = Path(__file__).resolve().parent
TASKS = ['trajectory', 'img_intensity2', 'img_bars4', 'img_blobs4', 'mode_hold', 'vector_unequal_mass']
PROFILES = ['cpu', 'cuda', 'cuda_cpu_random', 'cuda_cpu_init']


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def main():
    retention = json.loads((ROOT / 'retention.json').read_text())
    for name, expected in retention.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with tempfile.TemporaryDirectory(prefix='recipe-port-audit-') as temporary:
        prepared = prepare(Path(temporary) / 'sources')
        assert json.loads((prepared / 'prepared-sources.json').read_text()) == json.loads(
            (ROOT / 'prepared-sources.json').read_text())
        sys.path.insert(0, str(prepared / 'repos/cpu'))
        from benchmarks.transfer_suite.protocol import test_verdict
        original_config = json.loads((ROOT.parents[2] / 'configs/toy100/constraints_simple_regularization.json').read_text())
        worker_hashes = {hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                         for name in ('probe.py', 'probe-v1.py')}
        rows = {}
        for profile in PROFILES:
            for task in TASKS:
                record = read(ROOT / 'runs' / profile / (task + '.json.gz'))
                assert record['worker_sha256'] in worker_hashes
                expected_config = original_config | {'device': 'cpu' if profile == 'cpu' else 'cuda:0'}
                assert record['config'] == expected_config
                assert record['backend'] == ('cpu' if profile == 'cpu' else 'cuda')
                assert record['cpu_random'] == (profile == 'cuda_cpu_random')
                assert record['environment']['LD_PRELOAD'] is None
                assert record['torch'] == '2.13.0+cu126'
                assert record['proof']['adam_calls'] == 2 * record['spec']['steps']
                assert all(v['device'] == expected_config['device'] for v in record['proof']['optimizers'].values())
                assert sum(v['calls'] for v in record['proof']['optimizers'].values()) == record['proof']['adam_calls']
                verdict = test_verdict(record['spec'], record['result'])
                assert verdict == record['verdict'] and verdict['status'] == record['status']
                rows[profile, task] = record
        comparisons = []
        for task in TASKS:
            cpu, cuda, bridge, init = [rows[name, task] for name in PROFILES]
            assert cpu['proof']['initial_optimizers'] == bridge['proof']['initial_optimizers']
            assert cpu['proof']['initial_optimizers'] == init['proof']['initial_optimizers']
            assert cpu['proof']['initial_optimizers'] != cuda['proof']['initial_optimizers']
            assert cpu['randomness'] == bridge['randomness']
            assert cuda['randomness'] == init['randomness']
            fixture = ROOT / 'initialization-fixtures' / task / 'initial-values.pt'
            assert hashlib.sha256(fixture.read_bytes()).hexdigest() == init['initialization_fixture_sha256']
            fixture_record = read(fixture.with_name('result.json.gz'))
            assert fixture_record['status'] == 'INITIALIZATION_CAPTURED'
            assert fixture_record['proof']['adam_calls'] == 0
            assert fixture_record['proof']['initial_optimizers'] == cpu['proof']['initial_optimizers']
            # The instrumented native-CUDA control must preserve the earlier run.
            earlier = json.loads((ROOT.parent / 'gpu-known-winner-control/runs/simpler22_reference' / task / 'result.json').read_text())
            assert earlier['result']['live'] == cuda['result']['live']
            comparisons.append(dict(task=task, verdicts={p: rows[p, task]['status'] for p in PROFILES},
                                    initial_parameters_match=True, all_cpu_random_draws_match=True,
                                    initialization_only_preserves_cuda_draws=True,
                                    cpu_random_calls=cpu['randomness']['calls']))
        errors = read(ROOT / 'setup-errors.json.gz')
        assert all(row['record']['status'] == 'ERROR' and row['record']['proof']['adam_calls'] == 0 for row in errors)
        result = dict(status='PASS', scored_runs=len(rows), fixed_seed=0,
                      initializations_captured_without_updates=len(TASKS),
                      preserved_setup_errors=len(errors),
                      counts={p: sum(rows[p, task]['status'] == 'PASS' for task in TASKS) for p in PROFILES},
                      comparisons=comparisons,
                      scope='Six failing transfer hosts only; no combined 22-toy score for either diagnostic variant')
        (ROOT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
