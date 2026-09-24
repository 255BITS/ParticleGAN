"""Recompute GPU leaderboard verdicts from the saved evidence, without training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rows = json.loads((root / 'candidates.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    latest = {}
    for line in (root / 'ledger.jsonl').read_text().splitlines():
        item = json.loads(line)
        latest[item['candidate'], item['task']] = item
    expected = {(r['name'], task) for r in rows for task in protocol['toys'] + ['convergence']}
    assert set(latest) <= expected
    if not args.partial:
        assert set(latest) == expected, 'incomplete matrix'
    archives = json.loads((root / 'source-archives.json').read_text())
    checks = []
    reference_env = None
    specs = {}
    with tempfile.TemporaryDirectory(prefix='gpu-score-audit-') as temporary:
        code = Path(temporary)
        for name, info in archives.items():
            archive = root / 'archives' / info['archive']
            assert hashlib.sha256(archive.read_bytes()).hexdigest() == info['sha256']
            with tarfile.open(archive, 'r:gz') as stream:
                members = stream.getmembers()
                assert set(m.name for m in members) == set(info['source_hashes'])
                for member in members:
                    assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                    data = stream.extractfile(member).read()
                    assert hashlib.sha256(data).hexdigest() == info['source_hashes'][member.name]
                    if name == 'eps_net_1m':
                        dest = code / member.name
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
        sys.path[:0] = [str(code), str(code / 'reports/toy100/h_stability')]
        from benchmarks.transfer_suite.protocol import test_verdict
        from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
        from benchmarks.transfer_suite.public_default_verification import load_declaration, declared_spec
        from benchmarks.toy100.gate import evaluate_suite
        from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
        from convergence_gate import ConvergenceGate
        for row in rows:
            for task in protocol['toys'] + ['convergence']:
                event = latest.get((row['name'], task))
                if event is None:
                    continue
                if event['status'] == 'UNSUPPORTED':
                    assert event['executed'] is False
                    continue
                assert event['status'] not in ('ERROR', 'PENDING', 'SETTLING', 'HOLDING'), event
                suffix = event['artifact'].split('/runs/', 1)[1]
                result_path = root / 'runs' / suffix
                folder = result_path.parent
                result = json.loads(result_path.read_text())
                declaration = json.loads((folder / 'declaration.json').read_text())
                proof = json.loads((folder / 'device-proof.json').read_text())
                assert declaration['config'] == row['config'] and declaration['options'] == row['options']
                assert declaration['worker_sha256'] == hashlib.sha256((root / 'worker.py').read_bytes()).hexdigest()
                environment = declaration['environment']
                if reference_env is None:
                    reference_env = environment
                assert environment == reference_env == event['environment'], 'mixed hardware or numerical profile'
                assert proof == event['device_proof']
                assert proof['adam_calls'] > 0 and proof['cpu_parameter_calls'] == 0
                assert proof['gradient_devices'] == ['cuda:0']
                assert sum(x['calls'] for x in proof['optimizers'].values()) == proof['adam_calls']
                assert all(x['device'] == 'cuda:0' and x['calls'] > 0 for x in proof['optimizers'].values())
                dynamics = json.loads(gzip.decompress((folder / 'dynamics.json.gz').read_bytes()))
                if row['kind'] == 'signal':
                    assert dynamics['updates'], 'candidate optimizer policy was not applied'
                    if 'adam_response' in row['options']:
                        assert dynamics['adam_response']['name'] == row['options']['adam_response']
                    if 'network_geometry' in row['options']:
                        assert dynamics['network_geometry'], 'missing network geometry receipt'
                    if 'gradient_averaging' in row['options']:
                        assert dynamics['gradient_average'], 'missing averaging receipt'
                else:
                    assert dynamics['rng_replay_verified'] and dynamics['records']
                if task == 'convergence':
                    cold_event = latest[row['name'], 'mode_hold']
                    cold = json.loads((root / 'runs' / cold_event['artifact'].split('/runs/', 1)[1]).read_text())
                    prefix = next(point for point in result['diagnostic'] if point['step'] == 1200)
                    assert all(prefix[key] == cold['result']['live'][key] for key in ('modes', 'hq')), 'cold prefix differs from toy run'
                    gate = ConvergenceGate()
                    for point in result['diagnostic']:
                        if point['step'] > 1200:
                            gate.observe(point)
                    assert gate.summary() == result['convergence'] == event['convergence']
                    assert gate.status == event['status']
                    assert proof['adam_calls'] == 2 * result['diagnostic'][-1]['step']
                elif task in ('grid100', 'rotated100', 'staggered100'):
                    coverage = evaluate_suite(folder / 'native', problem=task, write=False)
                    accuracy = accuracy_suite(folder / 'native', problem=task, write=False)
                    assert coverage['status'] == result['coverage']['status']
                    assert accuracy['status'] == result['accuracy']['status']
                    actual = dict(coverage['problems'][task])
                    recorded = dict(result['coverage']['problems'][task])
                    actual.pop('run_dir', None)
                    recorded.pop('run_dir', None)
                    assert actual['status'] in ('PASS', 'FAIL'), actual
                    assert actual == recorded, 'native coverage evidence changed'
                    assert accuracy['problems'][task] == result['accuracy']['problems'][task], 'native accuracy evidence changed'
                    verdict = 'PASS' if coverage['status'] == accuracy['status'] == 'PASS' else 'FAIL'
                    assert verdict == event['status']
                    assert proof['adam_calls'] == 14000
                else:
                    spec = result['spec']
                    jobs, profile = load_declaration()
                    job = next(job for job in jobs if job['spec']['name'] == task)
                    recipe, _, _ = declared_recipe(row['config'])
                    declared, _, _ = declared_spec(job, profile, recipe)
                    assert declared == spec, 'saved spec differs from declared candidate recipe'
                    scoring = {k: v for k, v in spec.items() if k not in ('lr', 'lr_g', 'lr_d', 'd_lr_mult')}
                    assert specs.setdefault(task, scoring) == scoring, 'candidate-specific scoring rules'
                    verdict = test_verdict(spec, result['result'])
                    assert verdict == result['verdict'] == event['verdict']
                    assert verdict['status'] == event['status']
                    assert proof['adam_calls'] == 2 * spec['steps']
                checks.append(dict(candidate=row['name'], task=task, status=event['status'], verified=True))
    report = dict(status='PASS', complete=set(latest) == expected, expected_cells=len(expected),
                  scored_runs=len(checks), unsupported=sum(x['status'] == 'UNSUPPORTED' for x in latest.values()),
                  environment=reference_env, checks=checks)
    (root / ('audit-partial.json' if args.partial else 'audit.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'environment')}))


if __name__ == '__main__':
    main()
