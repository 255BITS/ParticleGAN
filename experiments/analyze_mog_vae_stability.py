#!/usr/bin/env python
"""Verify the lower-LR particle-VAE round, including constant-KL hard inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary, load_config, trainer_defaults


def analyze(root, report):
    trainer = str(ROOT / 'experiments/train_mog_vae_stability.py')
    provenance = code_provenance(trainer, sys.executable)
    configs = sorted((ROOT / 'configs/mog_vae/stability').glob('*.yaml'))
    assert len(configs) == 5
    report.mkdir(parents=True, exist_ok=True)
    rows = {}
    for path in configs:
        name = path.stem
        cfg = load_config(str(path), trainer_defaults(trainer))
        run = root / name
        assert Path(cfg['out_dir']).resolve() == run.resolve()
        assert has_valid_summary(str(run), cfg, provenance), name
        s = json.loads((run / 'summary.json').read_text())
        f = s['final']
        assert f['step'] == cfg['steps'] == 6000
        assert cfg['eval_samples'] == cfg['final_samples'] == 100000 and s['sigma_unchanged']
        assert np.isfinite(f['sample_sw1']) and np.isfinite(f['hq'])
        history = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        assert [v['step'] for v in history] == [2000, 4000, 6000]
        assert history[-1] == f
        assert len(s['checkpoints']) == 3
        for filename, digest in s['checkpoints'].items():
            assert hashlib.sha256((run / filename).read_bytes()).hexdigest() == digest
        r = f['reconstruction']
        if r:
            with np.load(run / 'reconstruction_006000.npz') as a:
                for key, metric in [('errors', 'recon_mse'), ('map_errors', 'map_mse'),
                                    ('shuffle_errors', 'shuffled_mse'), ('center_errors', 'zero_offset_mse'),
                                    ('categorical_kl', 'categorical_kl'), ('local_kl', 'local_kl'), ('same_mode', 'same_mode')]:
                    assert len(a[key]) == cfg['recon_samples'] == 8192
                    assert np.isfinite(a[key]).all()
                    assert np.isclose(a[key].mean(dtype=np.float64), r[metric], atol=1e-7, rtol=1e-5), (name, metric)
                assert np.isclose(np.sqrt(a['pair_squared'].mean(dtype=np.float64)), r['pair_rms'], atol=1e-7)
                assert np.isclose(a['aggregate_q'].sum(), 1.)
                assert a['counts'].sum() == cfg['recon_samples'] * (1 if cfg['posterior'] == 'ae' else 8)
                if cfg['posterior'] in ('categorical', 'hard'):
                    assert r['local_kl'] == 0 and r['posterior_std_ratio'] == 1.
                if cfg['posterior'] == 'hard':
                    assert np.isclose(r['categorical_kl'], np.log(cfg['num_particles']))
                    assert r['posterior_effective_particles'] == 1.
                    assert r['offset_rms'] == 0.
                    assert np.allclose(a['aggregate_q'], a['counts']/a['counts'].sum(), atol=1e-7)
        dest = report / name
        dest.mkdir(exist_ok=True)
        for filename in ['summary.json', 'config.yaml', 'metadata.json', 'provenance.json', 'metrics.jsonl', 'run_grid_complete.json']:
            shutil.copy2(run / filename, dest / filename)
        if r:
            shutil.copy2(run / 'reconstruction_006000.npz', dest / 'reconstruction_006000.npz')
        s['history'] = history
        rows[name] = s
    for key in ['initialization_sha256', 'sigma']:
        assert len({s['metadata'][key] for s in rows.values()}) == 1, key
    for stream in ['data', 'prior']:
        assert len({s['rng_sha256'][stream] for s in rows.values()}) == 1, stream
    allowed = {'posterior', 'temperature', 'obs_sigma', 'gan_weight', 'kl_weight', 'out_dir'}
    common = [{k: v for k, v in s['config'].items() if k not in allowed} for s in rows.values()]
    assert all(v == common[0] for v in common)
    ordered = sorted(rows, key=lambda k: (-rows[k]['final']['modes'], -rows[k]['final']['hq'], rows[k]['final']['sample_sw1']))
    (report / 'leaderboard.json').write_text(json.dumps({k: rows[k] for k in ordered}, indent=2, allow_nan=False) + '\n')
    def fmt(v, digits=4):
        return '—' if v is None else f'{v:.{digits}g}'
    lines = ['# Particle VAE lower-LR leaderboard', '',
        'Five prespecified configurations; one shared seed; final online weights at 6,000 updates.',
        'Generation uses 100k decoder means G(z). Rank: coverage, then HQ, then SW1. Width should approach 1.',
        'This rank is generation-focused; reconstruction and posterior usefulness have separate tradeoffs.', '',
        '| Rank | Run | Modes /100 | HQ % ↑ | Width /real ≈1 | Balance KL ↓ | SW1 ↓ | Sampled recon MSE ↓ | MAP MSE ↓ | Train s |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rank, name in enumerate(ordered, 1):
        s = rows[name]; f = s['final']; r = f['reconstruction'] or {}
        values = [rank, name, f['modes'], fmt(100*f['hq']), fmt(f['width_ratio']), fmt(f['kl_balance']), fmt(f['sample_sw1']), fmt(r.get('recon_mse')), fmt(r.get('map_mse')), fmt(s['train_seconds'])]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '## Posterior and variation', '',
        'Eight draws/input on 8,192 held-out inputs. Pair RMS is Euclidean output distance; same-mode checks input-mode retention.',
        'Conditional effective K is exp(mean categorical entropy). MI is a held-out categorical mutual-information estimate.',
        'AE has no posterior: zero pair RMS is expected; its soft routing probabilities are only gradient diagnostics.', '',
        '| Run | KL categorical | KL local | Conditional effective K | Aggregate effective K (sampled) | MI nats | Pair RMS | Same-mode % | Shuffled MSE | Local std /sigma |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ordered:
        r = rows[name]['final']['reconstruction']
        if r:
            values = [name] + [fmt(r[k]) for k in ['categorical_kl', 'local_kl', 'posterior_effective_particles', 'effective_particles', 'categorical_mutual_information', 'pair_rms']] + [fmt(100*r['same_mode']), fmt(r['shuffled_mse']), fmt(r['posterior_std_ratio'])]
            lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## Likelihood check', '',
        'These samples include the decoder likelihood noise: G(z)+tau*noise. They are distinct from decoder means above.',
        'No-KL control has an evaluable bound but does not optimize it. ELBO column is negative ELBO in nats, lower is better.', '',
        '| Run | Tau | Negative ELBO ↓ | Predictive modes | Predictive HQ % ↑ |',
        '|---|---:|---:|---:|---:|']
    for name in ordered:
        s = rows[name]; r = s['final']['reconstruction']; pred = s['final'].get('likelihood_predictive')
        if pred:
            lines.append('| ' + ' | '.join([name, fmt(s['config']['obs_sigma']), fmt(r['negative_elbo_nats']), str(pred['modes']), fmt(100*pred['hq'])]) + ' |')
    lines += ['', '## Learning curves and cost', '',
        'Every evaluation uses 100k prior samples. SW1 always uses 8192.', '',
        '| Run | 2k modes / HQ% / SW1 | 4k | 6k | Total process s | Peak GiB |',
        '|---|---|---|---|---:|---:|']
    for name in ordered:
        s = rows[name]
        values = [name] + [f"{v['modes']} / {100*v['hq']:.2f} / {v['sample_sw1']:.4f}" for v in s['history']] + [fmt(s['total_seconds']), fmt(s['peak_memory_gb'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', f"Total training: {sum(s['train_seconds'] for s in rows.values())/60:.2f} GPU minutes; total child-process time: {sum(s['total_seconds'] for s in rows.values())/60:.2f} minutes (includes evaluation/I/O).", '',
        'All 5 completion certificates, configurations, matched initializations/data/prior RNGs, fixed sigma, 15 checkpoint hashes, and per-input reconstruction metrics verified.',
        'No seed-only runs, statistical superiority claim, or automatic promotion. See PROTOCOL.md for objective and caveats.', '']
    lines += ['', '## Comparison with previous full learning rates', '',
        'Same seed/initialization and final 6k update budget. New learning rates are half; the new round has 100k at every evaluation.',
        'Old no-KL likelihood tau was .1; new .03 does not affect its training or decoder-mean metrics.', '',
        '| Run | Old modes / HQ % / MSE | New modes / HQ % / MSE |', '|---|---|---|']
    for name in ordered:
        if name == 'hard_constant_kl_gan':
            continue
        old_path = ROOT / 'runs/mog_vae/scout' / name
        old = json.loads((old_path / 'summary.json').read_text())
        old_provenance = code_provenance(str(ROOT / 'experiments/train_mog_vae.py'), sys.executable)
        assert has_valid_summary(str(old_path), old['config'], old_provenance)
        new = rows[name]
        assert old['metadata']['initialization_sha256'] == new['metadata']['initialization_sha256']
        for key in ['data', 'prior']:
            assert old['rng_sha256'][key] == new['rng_sha256'][key]
        for key in ['lr', 'd_lr', 'prior_lr']:
            assert old['config'][key] == 2*new['config'][key]
        cells = [name]
        for item in [old, new]:
            f = item['final']; r = f['reconstruction'] or {}
            cells.append(f"{f['modes']} / {100*f['hq']:.2f} / {fmt(r.get('recon_mse'))}")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Hard posterior has KL=log(400) constant, omitted only from optimization and retained in ELBO diagnostics.',
        'Its categorical effective K is one; only within-particle noise is stochastic. Encoder routing uses a biased straight-through surrogate.', '']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in ordered:
        f = rows[name]['final']; r = f['reconstruction']
        if r:
            axes[0].scatter(r['recon_mse'], 100*f['hq'], label=name)
            axes[1].scatter(r['pair_rms'], 100*r['same_mode'], label=name)
    axes[0].set(xscale='log', xlabel='Sampled reconstruction MSE (lower better)', ylabel='Prior decoder HQ % (higher better)')
    axes[1].set(xlabel='Posterior pair RMS', ylabel='Posterior input-mode retention %')
    axes[1].legend(fontsize=6, loc='best')
    fig.tight_layout(); fig.savefig(report / 'tradeoffs.png', dpi=150); plt.close(fig)
    print((report / 'LEADERBOARD.md').read_text())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/mog_vae/stability'))
    p.add_argument('--report', type=Path, default=Path('reports/mog-vae/stability'))
    args = p.parse_args()
    analyze(args.root, args.report)
