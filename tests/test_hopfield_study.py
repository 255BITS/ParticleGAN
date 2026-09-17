"""Protect the one-seed plan and its predeclared gate."""
import json
from pathlib import Path

from experiments.run_hopfield import gate, generate
from experiments.run_grid import code_provenance


def test_single_seed_grid_and_kill_gate(tmp_path):
    kill, manifests = generate(tmp_path, seed=1234, device='cpu')
    assert len(kill) == 2
    assert {cfg['read'] for cfg in kill} == {'uniform', 'hopfield'}
    configs = [json.loads(path.read_text()) for path in manifests.values()]
    assert [len(paths) for paths in configs] == [2, 28]
    assert len(set(sum(configs, []))) == 30
    for cfg in kill:
        assert cfg['num_particles'] == 100 and cfg['seed'] == 1234
        out = Path(cfg['out_dir'])
        out.mkdir(parents=True)
        (out / 'summary.json').write_text(json.dumps({'final': {'tv': 0.5 if cfg['read'] == 'uniform' else 0.3}}))
    assert gate(kill)['passed']
    hopfield = next(cfg for cfg in kill if cfg['read'] == 'hopfield')
    (Path(hopfield['out_dir']) / 'summary.json').write_text(json.dumps({'final': {'tv': 0.301}}))
    assert not gate(kill)['passed']


def test_study_provenance_includes_actual_training_loop():
    import sys
    sources = code_provenance('experiments/train_hopfield.py', sys.executable)['sources']
    assert 'examples/100gaussians.py' in sources
    assert 'lib/hopfield_metrics.py' in sources
