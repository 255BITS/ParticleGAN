"""Read source and closed JSON only; no third-party imports or PT loads."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PKG = ROOT / 'pkg-RA12-auto' / 'particlegan'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def node_sha(path, name):
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def main():
    inputs = [PKG / name for name in (
        'continuous.py', 'policy.py', 'ka2.py', 'k3p.py', 'training.py',
        'recipes.py', 'feature_policy.py')]
    inputs += [ROOT / 'configs/RA12-auto.json', ROOT / 'test_r1_composition.py',
               ROOT / 'pkg-RA11-R1/particlegan/continuous.py',
               ROOT / 'moving-quarter/grid100-r1-on.log',
               ROOT / 'moving-quarter/grid100-r1-on/frames.npz.verdict.json',
               ROOT / 'moving-quarter/SOURCE-FREEZE.json',
               ROOT / 'mnist/ra12-auto/SOURCE-FREEZE.json',
               ROOT / 'mnist/ra12-auto/CLOSED.json']
    summaries = {}
    for task in ('toy', 'mnist'):
        folder = ROOT / 'mnist/ra12-auto/training' / task / 'RA12-auto'
        paths = [folder / name for name in ('metrics.jsonl', 'config.json', 'result.json')]
        inputs.extend(paths)
        result = json.loads(paths[2].read_text())
        assert result['status'] == 'COMPLETE' and result['steps'] == 2000
        rows = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
        assert [r['step'] for r in rows] == [0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
        compact = []
        for row in rows:
            d = row['diagnostics']
            counters = d['birth_death']['counters']
            compact.append(dict(
                step=row['step'], lr=d['lr'], output_sigma=d['output_sigma'],
                surprise=d['surprise'],
                scales={k: v['s'] for k, v in d['lr_settle'].items()},
                population_moves=counters['moves'], isolation_moves=counters['iso_moves']))
        summaries[task] = compact
    assert all(r['population_moves'] == r['isolation_moves'] == 0
               for r in summaries['mnist'] if r['step'] <= 1250)
    assert summaries['toy'][5]['surprise']['log'][0] == [820, 2.096]
    assert summaries['mnist'][2]['surprise']['log'][0] == [202, 3.984]
    current_ast = node_sha(PKG / 'continuous.py', 'OptimizerSurprise')
    prior_ast = node_sha(ROOT / 'pkg-RA11-R1/particlegan/continuous.py', 'OptimizerSurprise')
    assert current_ast == prior_ast
    mechanisms = []
    for line in (ROOT / 'moving-quarter/grid100-r1-on.log').read_text().splitlines():
        if line.startswith('MECHANISM '):
            item = json.loads(line[len('MECHANISM '):])
            mechanisms.append(dict(step=item['step'], surprise=item['surprise']))
    verdict = json.loads((ROOT / 'moving-quarter/grid100-r1-on/frames.npz.verdict.json').read_text())
    assert verdict['status'] == 'PASS' and verdict['passed_periods'] == 2
    receipt = dict(
        status='PASS_SOURCE_AND_CLOSED_JSON_REVIEW',
        scope='read-only source/JSON; no PT/model/forward/update/draw/scoring/CUDA',
        detector_ast_sha256=current_ast,
        same_detector_ast_in_earlier_moving_package=True,
        observations=summaries,
        earlier_moving_mechanisms=mechanisms,
        earlier_moving_verdict=verdict,
        attribution=dict(
            toy_internal_KA2_phase='source-consistent hypothesis; exact trace pending',
            mnist_KA2_warmup_switch=False,
            mnist_population_transport_before_first_fire=False,
            auxiliary_group_or_network_cause='exact fire ratios pending'),
        proposed_scope='known internal optimizer/loss epoch plus explicit role scope; not qualified',
        protected_sha256={str(path): sha(path) for path in inputs},
    )
    target = HERE / 'receipt.json'
    with target.open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], receipt_sha256=sha(target), inputs=len(inputs))))


if __name__ == '__main__':
    main()
