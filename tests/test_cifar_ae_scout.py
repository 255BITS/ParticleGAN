"""Scout ranking must use certified final results and wait for the full grid."""
import hashlib
import json
import sys

import pytest
import yaml

from experiments.analyze_cifar_ae_scout import analyze, ROOT
from experiments.run_grid import code_provenance, MANIFEST_VERSION
from experiments.train_cifar_particle_ae import DEFAULTS


def make_grid(tmp_path):
    paths, summaries = [], []
    provenance = code_provenance(str(ROOT / 'experiments/train_cifar_particle_ae.py'), sys.executable)
    for name, fid, early in [('01_scratch_control', 45., 20.), ('02_pretrained', 40., 50.)]:
        run = tmp_path / name
        run.mkdir()
        cfg = {**DEFAULTS, 'arm': 'bounded', 'steps': 5000, 'final_samples': 5000,
               'recon_samples': 1000, 'out_dir': str(run),
               'encoder_backbone': 'scratch' if name.startswith('01') else 'pretrained_resnet18'}
        path = tmp_path / f'{name}.yaml'
        path.write_text(yaml.safe_dump(cfg))
        paths.append(str(path))
        summary = {'config': cfg, 'final': {'step': 5000, 'samples': 5000, 'fid': fid,
                   'reconstruction': {'recon_mse': .1, 'effective_particles': 100., 'offset_saturation': .02}},
                   'train_seconds': 300., 'sigma_unchanged': True, 'frozen_features_unchanged': True,
                   'frozen_encoder_features_unchanged': True}
        summary_path = run / 'summary.json'
        summary_path.write_text(json.dumps(summary))
        summaries.append(summary_path)
        (run / 'run_grid_complete.json').write_text(json.dumps({
            'version': MANIFEST_VERSION, 'config': cfg, 'provenance': provenance,
            'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest()}))
        (run / 'metrics.jsonl').write_text('\n'.join(json.dumps({
            'step': step, 'generation': {'samples': 5000, 'fid': value}})
            for step, value in [(2500, early), (5000, fid)]) + '\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(paths))
    return manifest, summaries


def test_rank_final_fid_and_prepare_longer_config(tmp_path):
    manifest, _ = make_grid(tmp_path)
    report = tmp_path / 'report'
    assert analyze(manifest, report) == 0
    result = json.loads((report / 'leaderboard.json').read_text())
    assert result['rows'][0]['name'] == '02_pretrained'
    winner = yaml.safe_load((report / 'winner_long.yaml').read_text())
    assert winner['steps'] == 30000 and winner['final_samples'] == 50000
    assert winner['encoder_backbone'] == 'pretrained_resnet18'
    assert winner['seed'] == DEFAULTS['seed']


def test_tampered_result_blocks_promotion_and_removes_stale_winner(tmp_path):
    manifest, summaries = make_grid(tmp_path)
    report = tmp_path / 'report'
    assert analyze(manifest, report) == 0
    summaries[0].write_text(summaries[0].read_text() + '\n')
    assert analyze(manifest, report) == 1
    assert not (report / 'winner_long.yaml').exists()
    assert json.loads((report / 'leaderboard.json').read_text())['missing'] == ['01_scratch_control']


def test_mismatched_evaluation_counts_are_rejected(tmp_path):
    manifest, _ = make_grid(tmp_path)
    path = tmp_path / '02_pretrained.yaml'
    cfg = yaml.safe_load(path.read_text())
    cfg['final_samples'] = 10000
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='unmatched scout protocol: final_samples'):
        analyze(manifest, tmp_path / 'report')
