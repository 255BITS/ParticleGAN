#!/usr/bin/env python
"""Verify observational diagnostics, then run two sequential GPU1 LR forks."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import cifar_ae_particle_next as pipeline
from experiments.run_grid import code_provenance, has_valid_summary

TRACK = 'particle_lr_80k_scout'
REPORT = ROOT/'reports/cifar-particle-ae'/TRACK
PARENT = 'runs/cifar_particle_ae/particle_16k_80k/16k_80k/checkpoint_080000.pt'
SHA = '9ddcef1fb0bf82c47a581fa1b2d58872315d5d67b302eadc533c0abc41b7d5c7'
TRAINER = 'experiments/train_cifar_ae_lr_diagnostics.py'
LEGACY = 'experiments/train_cifar_ae_scaling.py'
ARMS = {'half_g': {'g_lr_scale': .5, 'lr_scale': 1.},
        'half_all': {'g_lr_scale': 1., 'lr_scale': .5}}
CONTROL_FID = 16.4609082272
PARENT_FID = 15.752682149


def status(stage, **kwargs):
    pipeline.write(REPORT/'STATUS.json', dict(stage=stage, **kwargs))
    print('STAGE', stage, kwargs, flush=True)


def config(arm, smoke=False):
    parent = json.loads((ROOT/'reports/cifar-particle-ae/particle_16k_80k/results.json').read_text())[0]
    assert hashlib.sha256((ROOT/PARENT).read_bytes()).hexdigest() == SHA
    assert has_valid_summary(str(ROOT/parent['config']['out_dir']), parent['config'],
                             code_provenance(str(ROOT/LEGACY), sys.executable))
    track = TRACK + ('_smoke' if smoke else '')
    cfg = {**parent['config'], **ARMS[arm], 'generator_diagnostics': True,
           'steps': 80016 if smoke else 100000, 'eval_interval': 80016 if smoke else 5000,
           'initial_eval_samples': 0, 'eval_samples': 0 if smoke else 50000,
           'final_samples': 0 if smoke else 50000, 'recon_samples': 64 if smoke else 10000,
           'keep_checkpoints': True, 'resume_checkpoint': PARENT, 'resume_sha256': SHA,
           'max_train_seconds': 7200., 'out_dir': f'runs/cifar_particle_ae/{track}/{arm}'}
    return track, cfg


def verify_run(r, cfg):
    assert r['start_step'] == 80000 and r['final']['step'] == cfg['steps']
    assert r['sigma_unchanged'] and r['frozen_features_unchanged']
    rows = [json.loads(s) for s in (ROOT/cfg['out_dir']/'metrics.jsonl').read_text().splitlines()]
    expected = {'G': cfg['lr']*cfg['g_lr_scale']*cfg['lr_scale'],
                'E': cfg['lr']*cfg['lr_scale'], 'prior': cfg['prior_lr']*cfg['lr_scale'],
                'D': cfg['d_lr']*cfg['lr_scale']}
    assert all(m['learning_rates'] == expected for m in rows)
    assert all('generator_updates' in m for m in rows)
    curve = [m for m in rows if 'generation' in m]
    assert [m['step'] for m in curve] == ([80016] if cfg['steps'] == 80016 else [85000, 90000, 95000, 100000])
    assert all('generator_diagnostics' in m for m in curve)
    initial = json.loads((ROOT/cfg['out_dir']/'generator_diagnostics_initial.json').read_text())
    assert all(initial[k]['parent_drift_rms'] == 0 for k in ('live', 'ema'))
    return rows, curve


def assert_same(a, b, path=''):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), f'diagnostics changed tensor {path}; max delta {float((a-b).abs().max())}'
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a: assert_same(a[k], b[k], path+'/'+str(k))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): assert_same(x, y, path+'/'+str(i))
    else:
        assert a == b, path


def diagnostics_audit():
    """Exact state/gradient audit, independent of nondeterministic D backward."""
    from experiments import train_cifar_ae_lr_diagnostics as trainer
    ck = torch.load(ROOT/PARENT, map_location='cpu', weights_only=False)
    cfg = ck['config']
    g = trainer.DirectGenerator(cfg['z_dim'], cfg['width']).cuda()
    g.load_state_dict(ck['G'])
    eg = copy.deepcopy(g).eval().requires_grad_(False)
    eg.load_state_dict(ck['ema_G'])
    prior = trainer.MoGParticlePrior(cfg['num_particles'], cfg['z_dim'], sigma_rel=cfg['sigma_rel']).cuda()
    trainer.activate_expansion(prior, cfg['expansion_factor'], cfg['seed']+90000, False)
    prior.load_state_dict(ck['ema_prior'])
    # Real checkpoint weights and Adam moments; common fixed gradients isolate the observer.
    for p in g.parameters():
        p.grad = torch.full_like(p, .01)
    reference = copy.deepcopy(g)
    for a, b in zip(reference.parameters(), g.parameters()):
        a.grad = b.grad.clone()
    group = copy.deepcopy(ck['optimizer_g']['param_groups'][0])
    group['lr'] = .00015
    state = {'param_groups': [group], 'state': {k: v for k, v in ck['optimizer_g']['state'].items() if k in group['params']}}
    opt = torch.optim.Adam(g.parameters(), lr=.00015, betas=(0., .999), fused=True)
    refopt = torch.optim.Adam(reference.parameters(), lr=.00015, betas=(0., .999), fused=True)
    opt.load_state_dict(copy.deepcopy(state)); refopt.load_state_dict(copy.deepcopy(state))
    models_before = copy.deepcopy([m.state_dict() for m in (g, eg, prior)])
    grads_before = [p.grad.clone() for p in g.parameters()]
    optimizer_before = copy.deepcopy(opt.state_dict())
    modes_before = [(m.training, [p.requires_grad for p in m.parameters()]) for m in (g, eg, prior)]
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    observer = trainer.GeneratorDiagnostics(g, eg, prior, cfg['seed'])
    observed = observer.observe(g, eg)
    assert_same(models_before, [m.state_dict() for m in (g, eg, prior)], 'model/prior/EMA state')
    assert_same(grads_before, [p.grad for p in g.parameters()], 'gradients')
    assert_same(optimizer_before, opt.state_dict(), 'Adam state')
    assert_same(cpu_rng, torch.get_rng_state(), 'CPU RNG')
    assert_same(cuda_rng, torch.cuda.get_rng_state_all(), 'CUDA RNG')
    assert modes_before == [(m.training, [p.requires_grad for p in m.parameters()]) for m in (g, eg, prior)]
    assert all(observed[k]['parent_drift_rms'] == 0 for k in ('live', 'ema'))
    before = {n: p.detach().clone() for n, p in g.named_parameters()}
    opt.step(); refopt.step()
    trainer.generator_update_metrics(g, before)
    assert_same(reference.state_dict(), g.state_dict(), 'Adam update')
    assert_same(refopt.state_dict(), opt.state_dict(), 'updated Adam state')
    pipeline.write(REPORT/'DIAGNOSTICS_AUDIT.json', {
        'state_gradient_optimizer_rng_preserved_bitwise': True,
        'identical_Adam_update_on_common_gradients': True,
        'parent_sha256': SHA,
        'trainer_sha256': hashlib.sha256((ROOT/TRAINER).read_bytes()).hexdigest(),
        'limitation': 'Cross-run training is not bitwise deterministic: CUDA adaptive_avg_pool2d backward has no deterministic implementation. This audit isolates observer side effects and a common-gradient Adam update; it does not claim bitwise-identical production trajectories.'})
    print('DIAGNOSTICS_AUDIT_PASS', flush=True)


def verify_smokes():
    status('verifying_diagnostics_and_rates')
    subprocess.run([sys.executable, '-u', __file__, '--diagnostics-audit'], cwd=ROOT,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'}, check=True)
    audit = {'observer': json.loads((REPORT/'DIAGNOSTICS_AUDIT.json').read_text())}
    for arm in ARMS:
        track, cfg = config(arm, smoke=True)
        r = pipeline.grid(track, TRAINER, '1', {arm: cfg})[0]
        rows, curve = verify_run(r, cfg)
        saved = torch.load(ROOT/cfg['out_dir']/'checkpoint_080016.pt', map_location='cpu', weights_only=False)
        parent = torch.load(ROOT/PARENT, map_location='cpu', weights_only=False)
        for key in ('G', 'D', 'E', 'prior'):
            assert any(not torch.equal(v, parent[key][k]) for k, v in saved[key].items() if torch.is_tensor(v)), key
        assert_same(parent['prior']['sigma'], saved['prior']['sigma'], 'sigma')
        assert rows[0]['generator_updates']['input.weight']['update_rms'] > 0
        audit[arm] = {'rates': rows[0]['learning_rates'], 'resume': r['metadata']['resume'],
                      'G_update_rms': rows[0]['generator_updates']['input.weight']['update_rms']}
        print('SMOKE_VERIFIED', arm, 'rates, full-state resume, finite diagnostics', flush=True)
    pipeline.write(REPORT/'VALIDATION.json', audit)


def control_probe():
    """Read existing checkpoints only; matched fixed-z drift for the unchanged arm."""
    from experiments import train_cifar_ae_lr_diagnostics as trainer
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    ck = torch.load(ROOT/PARENT, map_location='cpu', weights_only=False)
    cfg = ck['config']
    g = trainer.DirectGenerator(cfg['z_dim'], cfg['width']).cuda().eval()
    eg = trainer.DirectGenerator(cfg['z_dim'], cfg['width']).cuda().eval()
    p = trainer.MoGParticlePrior(cfg['num_particles'], cfg['z_dim'], sigma_rel=cfg['sigma_rel']).cuda()
    trainer.activate_expansion(p, cfg['expansion_factor'], cfg['seed']+90000, False)
    g.load_state_dict(ck['G']); eg.load_state_dict(ck['ema_G']); p.load_state_dict(ck['ema_prior'])
    diag = trainer.GeneratorDiagnostics(g, eg, p, cfg['seed'])
    results = [{'step': 80000, 'checkpoint_sha256': SHA, 'diagnostics': diag.initial}]
    for step in (90000, 100000):
        path = ROOT/f'runs/cifar_particle_ae/particle_16k_160k/16k_160k/checkpoint_{step:06d}.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        for name, sha in saved['sources'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
        g.load_state_dict(saved['G']); eg.load_state_dict(saved['ema_G'])
        results.append({'step': step, 'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'diagnostics': diag.observe(g, eg)})
    pipeline.write(REPORT/'CONTROL_DIAGNOSTICS.json', results)
    print('CONTROL_DIAGNOSTICS_COMPLETE', flush=True)


def review(results):
    lines = ['# Learning-rate forks from the 80k residual CNN', '',
             '| Arm | FID85k | FID90k | FID95k | FID100k | Coverage100k |', '|---|---:|---:|---:|---:|---:|',
             '| unchanged (existing) | — | 15.7901 | — | 16.4609 | 63.36% |']
    for r in results:
        fids = ' | '.join(f"{m['generation']['fid']:.4f}" for m in r['curve'])
        quality = r.get('endpoint_quality')
        coverage = f"{quality['results'][0]['quality']['coverage']:.2%}" if quality else 'pending'
        lines.append(f"| {r['name']} | {fids} | {coverage} |")
    lines += ['', 'Starting parent FID15.7527. Full-state resume, original sigma and moving16k prior. All scores use FID50k.',
              'Fixed-input drift uses256 latent draws from the parent EMA prior; it measures G movement independently of center movement. Pixel drift is not a quality metric.',
              'Per-tensor update/weight RMS is one actual Adam step at each log, not a cumulative update. Zero/small weights can make ratios large; inspect absolute RMS too.',
              'Control checkpoints have snapshot diagnostics only, not historical per-step updates. Previous-probe drift spans10k for controls and5k for interventions; compare parent drift at matching90k/100k.',
              'This tests late learning rates, not whether residual connections themselves cause the plateau. Older1k lower-rate trials failed; no seed-only reruns or automatic extension.']
    (REPORT/'LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
    notes = ['# Interpretation and next-step recommendation', '']
    for r in results:
        best = min(r['curve'], key=lambda m: m['generation']['fid'])
        delta = r['final']['fid'] - CONTROL_FID
        notes.append(f"- {r['name']}: finalFID{r['final']['fid']:.4f}, change versus unchanged100k {delta:+.4f}; best resumed sampledFID{best['generation']['fid']:.4f} at{best['step']}. Parent80kFID{PARENT_FID:.4f} remains a separate reference.")
    if len(results) == 2 and all('endpoint_quality' in r for r in results):
        winner = min(results, key=lambda r: r['final']['fid'])
        if winner['final']['fid'] < PARENT_FID - .2:
            advice = f"Review {winner['name']} for extension: endpoint beats both the parent and unchanged continuation. Check coverage and trajectory before promoting."
        elif winner['final']['fid'] < CONTROL_FID - .2:
            advice = f"{winner['name']} mitigates deterioration versus unchanged continuation, but does not establish a substantial improvement on the parent. Review layer diagnostics before extending."
        else:
            advice = 'Neither arm shows a substantial endpoint improvement versus unchanged continuation. Do not blindly extend; inspect conditioning and per-layer movement before choosing an architecture or layer-specific intervention.'
        notes += ['', advice, 'The0.2 margin is a descriptive review threshold, not a statistical significance test. Lower pixel drift alone is not a success: lower rates mechanically reduce motion.']
    else:
        notes += ['', 'Incomplete experiment set; defer a winner recommendation until both endpoints and quality probes finish.']
    notes += ['', 'Half-G versus half-all distinguishes G-only slowing from slowing the coupled system; neither establishes that residual blocks are the cause. Layer/conditioning diagnostics are observational. No further jobs automatically queued.']
    (REPORT/'FINDINGS.md').write_text('\n'.join(notes)+'\n')
    print('\n'.join(lines), flush=True)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    pipeline.COMMON = ROOT/f'runs/cifar_particle_ae/{TRACK}/PIPELINE.log'
    verify_smokes()
    status('reading_existing_control_checkpoints')
    subprocess.run([sys.executable, '-u', __file__, '--control-probe'], cwd=ROOT,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'}, check=True)
    results, checkpoints = [], {}
    for arm in ARMS:
        status('training', arm=arm, gpu=1, start=80000, stop=100000)
        track, cfg = config(arm)
        r = pipeline.grid(track, TRAINER, '1', {arm: cfg})[0]
        rows, curve = verify_run(r, cfg)
        assert all(m['generation']['samples'] == 50000 for m in curve)
        result = {'name': arm, 'curve': curve, **r}
        results.append(result)
        best = min(curve, key=lambda m: m['generation']['fid'])
        checkpoints[arm] = {}
        for label, step, fid in [('final', 100000, r['final']['fid']),
                                  ('best_sampled', best['step'], best['generation']['fid'])]:
            path = ROOT/cfg['out_dir']/f'checkpoint_{step:06d}.pt'
            checkpoints[arm][label] = {'path': str(path.relative_to(ROOT)),
                                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'fid50k': fid}
        pipeline.write(REPORT/'results.json', results)
        pipeline.write(REPORT/'CHECKPOINTS.json', checkpoints)
        review(results)
        status('endpoint_quality', arm=arm, gpu=1)
        ck = checkpoints[arm]['final']
        probe_track = TRACK+'_quality'
        probe_cfg = {'checkpoint': ck['path'], 'checkpoint_sha256': ck['sha256'],
                     'expected_fid': r['final']['fid'], 'noise_scales': [1.],
                     'out_dir': f'runs/cifar_particle_ae/{probe_track}/{arm}'}
        q = pipeline.grid(probe_track, 'experiments/probe_cifar_ae_overlap.py', '1', {arm: probe_cfg})[0]
        result['endpoint_quality'] = q
        pipeline.write(REPORT/'results.json', results)
        review(results)
    status('complete', gpu=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-probe', action='store_true')
    parser.add_argument('--diagnostics-audit', action='store_true')
    args = parser.parse_args()
    try:
        if args.diagnostics_audit:
            diagnostics_audit()
        elif args.control_probe:
            control_probe()
        else:
            main()
    except Exception as exc:
        REPORT.mkdir(parents=True, exist_ok=True)
        status('failed', error=repr(exc))
        raise
