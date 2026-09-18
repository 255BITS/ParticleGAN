import pytest
import hashlib
import json
import sys
import yaml

from experiments.analyze_cifar_ae_lazy import runtime_estimate, analyze, ROOT
from experiments.run_grid import code_provenance, MANIFEST_VERSION
from experiments.train_cifar_particle_ae import DEFAULTS


def test_runtime_projection_separates_training_and_sample_counts():
    summary = {'config': {'steps': 5000, 'eval_samples': 5000, 'recon_samples': 1000},
               'train_seconds': 250., 'total_seconds': 377.}
    evaluations = [
        {'generation': {'samples': 5000}, 'generation_seconds': 10., 'reconstruction_seconds': 1.},
        {'generation': {'samples': 50000}, 'generation_seconds': 100., 'reconstruction_seconds': 1.},
    ]
    result = runtime_estimate(summary, evaluations, 30000)
    assert result['train_minutes'] == 25.
    # 1500 training + 5*10 small FID + 100 final FID + 6*10 reconstruction + 15 residual.
    assert result['estimated_total_minutes'] == pytest.approx(1725 / 60)


def test_certified_grid_ranks_quality_and_blocks_partial_promotion(tmp_path):
    provenance = code_provenance(str(ROOT / 'experiments/train_cifar_particle_ae.py'), sys.executable)
    paths = []
    for every, fid in [(4, 25.), (8, 24.), (16, 28.)]:
        run = tmp_path / f'n{every:02d}'
        run.mkdir()
        cfg = {**DEFAULTS, 'arm': 'bounded', 'steps': 5000, 'reg_every': every,
               'recon_samples': 1000, 'out_dir': str(run)}
        path = tmp_path / f'n{every:02d}.yaml'
        path.write_text(yaml.safe_dump(cfg))
        paths.append(str(path))
        s = {'config': cfg, 'final': {'step': 5000, 'samples': 50000, 'fid': fid,
             'reconstruction': {'recon_mse': .1}}, 'train_seconds': 250. - every,
             'total_seconds': 400. - every, 'sigma_unchanged': True, 'frozen_features_unchanged': True,
             'metadata': {'initialization_sha256': 'matched'}, 'rng_sha256': {'data': 'matched'}}
        summary = run / 'summary.json'
        summary.write_text(json.dumps(s))
        (run / 'run_grid_complete.json').write_text(json.dumps({
            'version': MANIFEST_VERSION, 'config': cfg, 'provenance': provenance,
            'summary_sha256': hashlib.sha256(summary.read_bytes()).hexdigest()}))
        (run / 'metrics.jsonl').write_text('\n'.join(json.dumps({
            'step': step, 'generation': {'samples': count}, 'generation_seconds': seconds,
            'reconstruction_seconds': 1.}) for step, count, seconds in [(2500, 5000, 10.), (5000, 50000, 100.)]))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(paths))
    report = tmp_path / 'report'
    assert analyze(manifest, report) == 0
    assert yaml.safe_load((report / 'winner_long.yaml').read_text())['reg_every'] == 8
    assert json.loads((report / 'leaderboard.json').read_text())['rows'][0]['name'] == 'n08'
    (tmp_path / 'n16' / 'run_grid_complete.json').unlink()
    assert analyze(manifest, report) == 1
    assert not (report / 'winner_long.yaml').exists()
