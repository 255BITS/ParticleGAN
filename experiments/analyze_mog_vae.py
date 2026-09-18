#!/usr/bin/env python
"""Verify the queued particle-VAE scout and publish its numerical leaderboard."""
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
    trainer = str(ROOT / 'experiments/train_mog_vae.py')
    provenance = code_provenance(trainer, sys.executable)
    configs = sorted((ROOT / 'configs/mog_vae/scout').glob('*.yaml'))
    assert len(configs) == 12
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
        assert cfg['final_samples'] == 100000 and s['sigma_unchanged']
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
                if cfg['posterior'] == 'categorical':
                    assert r['local_kl'] == 0 and r['posterior_std_ratio'] == 1.
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
    lines = ['# Particle VAE toy leaderboard', '',
        'Twelve prespecified configurations; one shared seed; final online weights at 6,000 updates.',
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
        'At 2k/4k generation uses 20k samples; final uses 100k. SW1 always uses 8192.', '',
        '| Run | 2k modes / HQ% / SW1 | 4k | 6k | Total process s | Peak GiB |',
        '|---|---|---|---|---:|---:|']
    for name in ordered:
        s = rows[name]
        values = [name] + [f"{v['modes']} / {100*v['hq']:.2f} / {v['sample_sw1']:.4f}" for v in s['history']] + [fmt(s['total_seconds']), fmt(s['peak_memory_gb'])]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', f"Total training: {sum(s['train_seconds'] for s in rows.values())/60:.2f} GPU minutes; total child-process time: {sum(s['total_seconds'] for s in rows.values())/60:.2f} minutes (includes evaluation/I/O).", '',
        'All 12 completion certificates, configurations, matched initializations/data/prior RNGs, fixed sigma, 36 checkpoint hashes, and per-input reconstruction metrics verified.',
        'No seed-only runs, statistical superiority claim, or automatic promotion. See PROTOCOL.md for objective and caveats.', '']
    audit_path = report / 'late_audit.json'
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        lines += ['', '## Matched-count late-regression audit', '',
            'Read-only 4k checkpoints evaluated with the same 100k sample count and RNG as final 6k; no additional training.', '',
            '| Run | 4k modes | 4k HQ % | 6k modes | 6k HQ % |', '|---|---:|---:|---:|---:|']
        for name, row in audit['rows'].items():
            assert row['samples'] == 100000 and row['checkpoint_unchanged']
            assert row['checkpoint_sha256'] == rows[name]['checkpoints']['checkpoint_004000.pt']
            assert row['step6000']['hq'] == rows[name]['final']['hq']
            a, b = row['step4000'], row['step6000']
            lines.append(f"| {name} | {a['modes']} | {100*a['hq']:.2f} | {b['modes']} | {100*b['hq']:.2f} |")
        lines += ['', 'AE-GAN was substantially stronger at 4k; endpoint ranking is not a stability or general superiority claim.', '']
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
    p.add_argument('--root', type=Path, default=Path('runs/mog_vae/scout'))
    p.add_argument('--report', type=Path, default=Path('reports/mog-vae'))
    args = p.parse_args()
    analyze(args.root, args.report)
