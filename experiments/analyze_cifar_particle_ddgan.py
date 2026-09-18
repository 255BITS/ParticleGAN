#!/usr/bin/env python
"""Verify complete four-arm results and publish a portable leaderboard."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_grid import code_provenance, has_valid_summary

NAMES = {'direct_gan': 'Direct GAN', 'direct_bounded': 'Particle AE-GAN',
         'ddgan_gan': 'DDGAN', 'ddgan_bounded': 'DDGAN + particle AE'}


def analyze(root, report):
    provenance = code_provenance(str(ROOT / 'experiments/train_cifar_particle_ddgan.py'), sys.executable)
    rows, histories = {}, {}
    report.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        run = root / name
        s = json.loads((run / 'summary.json').read_text())
        cfg = s['config']
        assert Path(cfg['out_dir']).resolve() == run.resolve()
        assert has_valid_summary(str(run), cfg, provenance), name
        assert s['final']['step'] == cfg['steps'] == 10000
        assert s['final']['samples'] == cfg['final_samples'] == 50000
        assert np.isfinite(s['final']['fid']) and np.isfinite(s['final']['fid5k'])
        assert s['sigma_unchanged'] and s['frozen_features_unchanged']
        histories[name] = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        ev = [v for v in histories[name] if 'generation' in v]
        assert [v['step'] for v in ev] == [2500, 5000, 7500, 10000]
        assert [v['generation']['samples'] for v in ev] == [5000, 5000, 5000, 50000]
        s['checkpoint_hashes'] = {str(v['step']): hashlib.sha256((run / f"checkpoint_{v['step']:06d}.pt").read_bytes()).hexdigest() for v in ev}
        r = s['final']['reconstruction']
        if r:
            with np.load(run / 'recon_010000.npz') as a:
                if cfg['model'] == 'ddgan':
                    for t, values in r['per_t'].items():
                        for key, value in values.items():
                            saved = a[f't{t}_{key.removesuffix("_mse")}']
                            assert len(saved) == cfg['recon_samples']
                            assert np.isclose(saved.mean(dtype=np.float64), value, atol=1e-7)
                else:
                    for k in ['recon','zero_offset','random_offset','shuffled_particle']:
                        assert np.isclose(a[k].mean(dtype=np.float64), r[f'{k}_mse'], atol=1e-7)
                if cfg['arm'] == 'bounded':
                    assert np.array_equal(np.bincount(a['ids'], minlength=cfg['num_particles']), a['counts'])
                    assert a['counts'].sum() == cfg['recon_samples']
        dest = report / name
        dest.mkdir(exist_ok=True)
        for filename in ['summary.json','config.yaml','metadata.json','provenance.json','metrics.jsonl','run_grid_complete.json']:
            shutil.copy2(run / filename, dest / filename)
        if r:
            shutil.copy2(run / 'recon_010000.npz', dest / 'recon_010000.npz')
        variation = root.parent / 'variation' / name
        if (variation / 'summary.json').exists():
            v = json.loads((variation / 'summary.json').read_text())
            assert v['protocol']['checkpoint_sha256'] == s['checkpoint_hashes']['10000']
            assert v['checkpoint_unchanged'] and v['model_state_unchanged']
            with np.load(variation / 'per_input_metrics.npz') as arrays:
                for row in v['rows']:
                    prefix = f"t{row['t']}_{row['kind']}_{row['noise_scale_sigma']:g}__"
                    for key in arrays.files:
                        if key.startswith(prefix):
                            metric = key[len(prefix):]
                            assert len(arrays[key]) == 512
                            assert np.isclose(arrays[key].mean(dtype=np.float64), row[metric], rtol=1e-6, atol=1e-7), key

            shutil.copytree(variation, dest / 'variation', dirs_exist_ok=True)
            s['variation'] = v
        rows[name] = s
    for model in ['direct','ddgan']:
        a,b = rows[f'{model}_gan'],rows[f'{model}_bounded']
        common = lambda s: {k:v for k,v in s['config'].items() if k not in ['arm','out_dir']}
        assert common(a) == common(b)
        assert a['metadata']['initialization_sha256'] == b['metadata']['initialization_sha256']
        assert a['rng_sha256'] == b['rng_sha256']
    assert len({s['metadata']['sigma'] for s in rows.values()}) == 1
    for stream in ['data','prior']:
        assert len({s['rng_sha256'][stream] for s in rows.values()}) == 1
    if all('variation' in rows[k] for k in ['ddgan_gan','ddgan_bounded']):
        assert rows['ddgan_gan']['variation']['context_hashes'] == rows['ddgan_bounded']['variation']['context_hashes']
    order = sorted(rows, key=lambda k: rows[k]['final']['fid'])
    (report / 'leaderboard.json').write_text(json.dumps({k:rows[k] for k in order}, indent=2)+'\n')
    lines = ['# CIFAR particle DDGAN comparison', '',
             'Final EMA at exactly 10k training updates. FID uses 50k generated images; lower is better.',
             'Unconditional, same fixed-sigma MoG prior and learning rates. One shared seed, no best-checkpoint selection.', '',
             '| Model | FID50k ↓ | FID5k ↓ | Feature variance /real | Train min | Total min | Peak GiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for k in order:
        s=rows[k];f=s['final']
        lines.append(f"| {NAMES[k]} | {f['fid']:.3f} | {f['fid5k']:.3f} | {f['feature_variance_ratio']:.3f} | {s['train_seconds']/60:.2f} | {s['total_seconds']/60:.2f} | {s['peak_memory_gb']:.2f} |")
    lines += ['', 'Particle AE arms use deterministic bounded encoders plus reconstruction; no KL or variational objective.',
              'Feature variance is a coarse spread metric, not semantic mode coverage. Cross-architecture compute and parameter counts differ.', '',
              '## Same-count learning curves', '', '| Model | FID5k at 2500 | 5000 | 7500 | 10000 |', '|---|---:|---:|---:|---:|']
    for k in NAMES:
        values=[v['generation']['fid5k'] for v in histories[k] if 'generation' in v]
        lines.append('| '+NAMES[k]+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', '## DDGAN clean prediction at fixed noisy input', '',
              'MSE[-1,1] on all 10k test images; each t has a different corruption level. These are not latent-only reconstructions.', '',
              '| t | DDGAN prior | DDGAN + particle AE encoded | Shuffled code | Prior code | Zero offset | Shuffled particle |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for t in ['1','2','3','4']:
        b=rows['ddgan_gan']['final']['reconstruction']['per_t'][t]
        r=rows['ddgan_bounded']['final']['reconstruction']['per_t'][t]
        values=[b['prior_mse']]+[r[k+'_mse'] for k in ['recon','shuffled_code','prior','zero_offset','shuffled_particle']]
        lines.append('| '+t+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    r=rows['direct_bounded']['final']['reconstruction']
    lines += ['', f"Direct Particle AE-GAN test MSE {r['recon_mse']:.6f}, PSNR {r['recon_psnr']:.2f}dB; zero-offset {r['zero_offset_mse']:.6f}, random-offset {r['random_offset_mse']:.6f}, shuffled-particle {r['shuffled_particle_mse']:.6f}.", '',
              '| Encoder | Used /1024 | Effective /1024 | Offset RMS | Saturated coordinates |', '|---|---:|---:|---:|---:|']
    for k in ['direct_bounded','ddgan_bounded']:
        r=rows[k]['final']['reconstruction']
        lines.append(f"| {NAMES[k]} | {r['used_particles']} | {r['effective_particles']:.1f} | {r['offset_rms']:.3f} | {100*r['offset_saturation']:.2f}% |")
    lines += ['', '## Numerical variation', '',
              '512 test inputs, eight draws each; DDGAN holds X_t and t fixed and measures predicted clean images before transition noise.',
              'Own-anchor retrieval is relative to deterministic reconstructions, not semantic identity accuracy.', '',
              '| Model | t | Code | Pair pixel RMSE (0–255 levels) | Feature cosine distance | MSE | Own-anchor retrieval |',
              '|---|---:|---|---:|---:|---:|---:|']
    for k,s in rows.items():
        for v in s.get('variation',{}).get('rows',[]):
            ret=f"{100*v['own_anchor_nearest']:.2f}%" if 'own_anchor_nearest' in v else '—'
            code=v['kind']+(f" + {v['noise_scale_sigma']:g}sigma" if v['kind']=='encoded' else '')
            lines.append(f"| {NAMES[k]} | {v['t']} | {code} | {v['pair_pixel_rmse']:.2f} | {v['pair_feature_cosine']:.4f} | {v['mse']:.6f} | {ret} |")
    lines += ['', 'Verified: current-source completion certificates, full budgets, pair initializations and RNG states, cross-architecture data/prior RNG states, fixed sigma/frozen critic, per-image reconstruction errors, and four checkpoint hashes per arm.', '']
    (report / 'LEADERBOARD.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k in NAMES:
        ev=[v for v in histories[k] if 'generation' in v]
        axes[0].plot([v['step'] for v in ev],[v['generation']['fid5k'] for v in ev],'o-',label=NAMES[k])
    axes[0].set(xlabel='Training updates',ylabel='FID5k (lower better)',title='Same-count generation diagnostics')
    for key in ['recon','prior','shuffled_code','zero_offset']:
        r=rows['ddgan_bounded']['final']['reconstruction']['per_t']
        axes[1].plot([1,2,3,4],[r[str(t)][key+'_mse'] for t in [1,2,3,4]],'o-',label=key)
    axes[1].set(xlabel='Noise timestep',ylabel='Test clean prediction MSE',title='DDGAN + particle AE latent ablations')
    for ax in axes:
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(report / 'learning_curves.png',dpi=140)
    print((report / 'LEADERBOARD.md').read_text())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'runs/cifar_particle_ddgan/scout')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/cifar-particle-ddgan')
    args=parser.parse_args();analyze(args.root,args.report)
