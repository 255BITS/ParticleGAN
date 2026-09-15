#!/usr/bin/env python
"""CIFAR-10 U-Net DDGAN + UCD. No arguments loads configs/cifar_ddgan/default.yaml."""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import importlib.metadata
from pathlib import Path
import sys
import time
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lib.denoising_toy import DiffusionSchedule, DrawSource, FixedConditionCritic
from lib.image_ddgan import sample_images, update_ema
from lib.image_moonshots import build_models
from lib.gan_loss import GANLoss
from lib.grad_regularizers import GradRegularizer
from lib.vicreg_loss import VICRegLikeLoss
from lib.cifar_speed import SpeedProfiler, cifar_penalty

DEFAULTS = {
    'model': 'ddgan', 'architecture': 'unet', 'd_mode': 'ucd', 'prior': 'learned', 'noise': 'gaussian',
    'ucd_target': 'time_class',
    'd_backbone': 'pretrained_resnet18', 'g_depth': 6, 'g_heads': 4, 'spatial_channels': 16,
    'ncsnpp_ch_mult': [1, 2, 2, 2], 'ncsnpp_res_blocks': 2,
    'ncsnpp_attn_resolutions': [16], 'ncsnpp_z_emb_dim': 256, 'ncsnpp_n_mlp': 4,
    'cache_condition': True, 'channels_last': False, 'fused_adam': True,
    'reg_method': 'autograd', 'reg_every': 4, 'reg_fd_eps': 0.05, 'reg_sync_stats': False,
    'profile_start': 100, 'profile_steps': 0,
    'seed': 24002, 'classes': 10, 'alpha_bar': [1.0, 0.9, 0.5, 0.05, 0.0001],
    'g_width': 32, 'd_width': 32, 'd_norm': 'group', 'z_dim': 128, 'num_particles': 20000,
    'steps': 10000, 'batch_size': 64, 'lr': 0.0006, 'd_lr_mult': 1.5,
    'prior_lr_mult': 10.0, 'beta1': 0.0, 'prior_reg': 1.0,
    'reg_arm': 'b_cap', 'reg_coeff': 1.0, 'reg_kappa': 1.0,
    'gan_mode': 'rp', 'loss_type': 'logistic', 'ucd_lambda': 0.02,
    'ema': 0.995, 'lr_anneal_start': 0.6, 'lr_floor': 1.0,
    'horizontal_flip': True, 'tf32': True, 'log_interval': 100,
    'eval_interval': 10000, 'eval_samples': 5000, 'final_samples': 50000,
    'eval_batch_size': 128, 'save_checkpoint': True,
    'data_dir': 'data', 'fid_cache': 'results/cifar_ddgan/fid_cache',
    'out_dir': 'results/cifar_ddgan/default',
}
DEFAULT_CONFIG = ROOT / 'configs/cifar_ddgan/default.yaml'


def validate(cfg):
    if cfg.get('reg_method', 'autograd') not in ('autograd', 'finite_difference'):
        raise ValueError('invalid regularizer method')
    if type(cfg.get('reg_every', 1)) is not int or cfg.get('reg_every', 1) < 1 or cfg.get('reg_fd_eps', .05) <= 0:
        raise ValueError('invalid regularizer interval/epsilon')
    if (cfg.get('reg_method', 'autograd') != 'autograd' or not cfg.get('reg_sync_stats', True)) and cfg['reg_arm'] != 'b_cap':
        raise ValueError('speed regularizer implementation requires b_cap')
    if cfg.get('profile_steps', 0) < 0 or cfg.get('profile_start', 100) < 0:
        raise ValueError('invalid profiling window')
    if cfg.get('cache_condition', False) and cfg.get('d_backbone') != 'pretrained_resnet18':
        raise ValueError('condition cache requires frozen pretrained D')
    target = cfg.get('ucd_target', 'class')
    if target not in ('class', 'time_class') or (target == 'time_class' and cfg['d_mode'] != 'ucd'):
        raise ValueError('time_class requires a UCD discriminator')
    if cfg['model'] != 'ddgan' or cfg['architecture'] not in ('unet', 'flat_hybrid', 'ncsnpp') or cfg['noise'] != 'gaussian':
        raise ValueError('Image trainer requires DDGAN with Gaussian step noise and a supported architecture')
    if cfg.get('d_backbone', 'pixel') not in ('pixel', 'pretrained_resnet18'):
        raise ValueError('invalid D backbone')
    if cfg.get('d_backbone') == 'pretrained_resnet18' and (target != 'time_class' or cfg['d_mode'] != 'ucd'):
        raise ValueError('pretrained D requires joint UCD')
    if cfg['architecture'] == 'flat_hybrid':
        if cfg['g_width'] < 16 or cfg['g_heads'] < 1 or cfg['g_width'] % cfg['g_heads'] or cfg['g_depth'] < 1 or cfg['spatial_channels'] < 1:
            raise ValueError('invalid flat generator dimensions')
    if cfg['architecture'] == 'ncsnpp':
        if (len(cfg['ncsnpp_ch_mult']) != 4 or
                any(type(x) is not int or x < 1 for x in cfg['ncsnpp_ch_mult']) or
                cfg['ncsnpp_res_blocks'] < 1 or cfg['ncsnpp_z_emb_dim'] < 1 or
                cfg['ncsnpp_n_mlp'] < 0 or
                any(x not in (4, 8, 16, 32) for x in cfg['ncsnpp_attn_resolutions'])):
            raise ValueError('invalid NCSN++ dimensions')
    if cfg['d_norm'] not in ('none', 'group'):
        raise ValueError('d_norm must be none or group')
    if cfg['d_mode'] not in ('ucd', 'concat') or cfg['prior'] not in ('learned', 'fixed', 'gaussian'):
        raise ValueError('invalid discriminator/prior mode')
    for k in ('steps', 'batch_size', 'z_dim', 'num_particles', 'log_interval', 'eval_interval', 'eval_batch_size'):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError(f'{k} must be a positive integer')
    if cfg['batch_size'] < 2 or cfg['num_particles'] < 2 or cfg['classes'] != 10:
        raise ValueError('CIFAR requires ten classes, batch and particle count >=2')
    for k in ('g_width', 'd_width'):
        if cfg[k] < 8 or cfg[k] % 8:
            raise ValueError('widths must be multiples of eight')
    for k in ('eval_samples', 'final_samples'):
        if cfg[k] < 10 or cfg[k] % 10:
            raise ValueError('evaluation counts must be positive multiples of ten')
    if not 0 <= cfg['ema'] < 1 or not 0 <= cfg['lr_anneal_start'] < 1 or not 0 <= cfg['lr_floor'] <= 1:
        raise ValueError('invalid EMA/LR schedule')
    DiffusionSchedule(cfg['alpha_bar'])


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_cifar(cfg, download=False):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(str(ROOT / cfg['data_dir']), train=True, download=download)
    return torch.from_numpy(dataset.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(dataset.targets)


def train(cfg, resume=None):
    validate(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    from lib.cifar_metrics import FIDEvaluator, PROTOCOL, uint8_images, save_grid
    from experiments.run_grid import code_provenance
    import torchvision
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = cfg['tf32']
    torch.backends.cudnn.allow_tf32 = cfg['tf32']
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    out = ROOT / cfg['out_dir']
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.jsonl').exists() and resume is None:
        raise FileExistsError(f'{out} already contains training; use a new out_dir or --resume')
    provenance = code_provenance(__file__, sys.executable)
    ck = None
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        if ck['config'] != cfg or ck['sources'] != provenance['sources']:
            raise ValueError('Resume requires identical config and source; use planned full-budget configs')
    write_json(out / 'provenance.json', provenance)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    with zipfile.ZipFile(out / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, digest in provenance['sources'].items():
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError('source changed during capture')
            archive.writestr(name, data)
    env = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'cuda': torch.version.cuda,
           'python': platform.python_version(),
           'packages': {k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'torch-fidelity', 'PyYAML')},
           'gpu': torch.cuda.get_device_name(), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'tf32': cfg['tf32'], 'precision': 'float32', 'data_sampling': 'iid with replacement; horizontal flip as configured'}
    write_json(out / 'environment.json', env)
    write_json(out / 'fid_protocol.json', PROTOCOL)
    images, labels = load_cifar(cfg)
    print('Preparing FID evaluator/cache', flush=True)
    evaluator = FIDEvaluator(images, ROOT / cfg['fid_cache'], cfg['eval_batch_size'])
    images, labels = images.to(device), labels.to(device)
    rngs = {k: torch.Generator(device=device).manual_seed(cfg['seed'] + i) for i, k in enumerate(('data', 'time', 'corruption', 'latent', 'noise', 'penalty'), 11)}
    schedule = DiffusionSchedule(cfg['alpha_bar']).to(device)
    g, d = build_models(cfg)
    g, d = g.to(device), d.to(device)
    if cfg.get('channels_last', False):
        g, d = g.to(memory_format=torch.channels_last), d.to(memory_format=torch.channels_last)
    if hasattr(d, 'pretrained_metadata'):
        env['pretrained_D'] = d.pretrained_metadata
        write_json(out / 'environment.json', env)
    prior = DrawSource(cfg['prior'], cfg['num_particles'], cfg['z_dim'], cfg['seed'] + 101, device)
    initial_prior = prior.table.detach().clone()
    eg, ep = copy.deepcopy(g).eval().requires_grad_(False), copy.deepcopy(prior).requires_grad_(False)
    groups = [{'params': list(g.parameters()), 'lr': cfg['lr']}]
    if cfg['prior'] == 'learned':
        groups.append({'params': list(prior.parameters()), 'lr': cfg['lr'] * cfg['prior_lr_mult']})
    og = torch.optim.Adam(groups, betas=(cfg['beta1'], .999), fused=cfg.get('fused_adam', False))
    od = torch.optim.Adam((p for p in d.parameters() if p.requires_grad), lr=cfg['lr'] * cfg['d_lr_mult'], betas=(cfg['beta1'], .999), fused=cfg.get('fused_adam', False))
    bases = [[v['lr'] for v in o.param_groups] for o in (og, od)]
    gan, vic = GANLoss(cfg['loss_type'], cfg['gan_mode']), VICRegLikeLoss()
    reg = GradRegularizer(cfg['reg_arm'], cfg['reg_coeff'], kappa=cfg['reg_kappa'], lazy_k=cfg.get('reg_every', 1),
                          method=cfg.get('reg_method', 'autograd'), fd_eps=cfg.get('reg_fd_eps', .05))
    start_step, train_seconds = 0, 0.
    if resume:
        for name, obj in [('G', g), ('D', d), ('prior', prior), ('ema_G', eg), ('ema_prior', ep), ('opt_G', og), ('opt_D', od)]:
            obj.load_state_dict(ck[name])
        for k, r in rngs.items():
            r.set_state(ck['rngs'][k].cpu())
        torch.set_rng_state(ck['cpu_rng'].cpu())
        torch.cuda.set_rng_state(ck['cuda_rng'].cpu())
        start_step, train_seconds = ck['step'], ck['train_seconds']
        # Remove metrics beyond the restored checkpoint before replaying them.
        metric_path = out / 'metrics.jsonl'
        if metric_path.exists():
            lines = [line for line in metric_path.read_text().splitlines()
                     if json.loads(line)['step'] <= start_step]
            metric_path.write_text('\n'.join(lines) + ('\n' if lines else ''))
    parameters = {k: sum(p.numel() for p in obj.parameters()) for k, obj in [('G', g), ('D', d), ('prior', prior)]}
    trainable_parameters = {k: sum(p.numel() for p in obj.parameters() if p.requires_grad) for k, obj in [('G', g), ('D', d), ('prior', prior)]}
    print(f"ARCH G={cfg['architecture']} D={cfg.get('d_backbone', 'pixel')} UCD={cfg['ucd_target']} trainable={trainable_parameters}", flush=True)
    print(f"START seed={cfg['seed']} G_width={cfg['g_width']} D_width={cfg['d_width']} T={schedule.steps} steps={cfg['steps']} params={parameters}", flush=True)

    def batch():
        ids = torch.randint(len(images), (cfg['batch_size'],), device=device, generator=rngs['data'])
        x0, c = images[ids].float() / 127.5 - 1, labels[ids]
        if cfg['horizontal_flip']:
            flip = torch.rand((len(c), 1, 1, 1), device=device, generator=rngs['data']) < .5
            x0 = torch.where(flip, x0.flip(-1), x0)
        t = torch.randint(1, schedule.steps + 1, (len(c),), device=device, generator=rngs['time'])
        real, xt = schedule.forward_pair(x0, t, rngs['corruption'])
        if cfg.get('channels_last', False):
            real, xt = real.contiguous(memory_format=torch.channels_last), xt.contiguous(memory_format=torch.channels_last)
        return c, real, xt, t

    def fake(c, xt, t):
        z, ids = prior.sample(len(c), rngs['latent'])
        clean = g(z, c, xt, t)
        eta = torch.randn(xt.shape, device=device, generator=rngs['noise'])
        return schedule.reverse(clean, xt, t, eta), ids

    def conditioned(c, xt, t):
        if cfg.get('cache_condition', False):
            features = d.condition_features(xt)
            return lambda x: d(x, c, xt, t, condition_features=features)
        return lambda x: d(x, c, xt, t)

    @torch.no_grad()
    def evaluate(n, step):
        # Dedicated generator: evaluation cadence cannot change training draws.
        r = torch.Generator(device=device).manual_seed(99000)
        chunks = []
        ts = time.perf_counter()
        for lo in range(0, n, cfg['eval_batch_size']):
            c = torch.arange(lo, min(n, lo + cfg['eval_batch_size']), device=device) % 10
            x = sample_images(eg, ep, schedule, c, r)
            if not torch.isfinite(x).all():
                raise FloatingPointError('nonfinite samples')
            chunks.append(uint8_images(x).cpu())
        sampling_s = time.perf_counter() - ts
        fid = evaluator(torch.cat(chunks))
        r.manual_seed(88000)
        grid = sample_images(eg, ep, schedule, torch.arange(10, device=device).repeat_interleave(10), r)
        save_grid(grid, out / f'samples_{step:06d}.png')
        save_grid(grid, out / 'samples.png')
        return {'fid': fid, 'samples': n, 'sampling_seconds': sampling_s}

    def checkpoint(step):
        if not cfg['save_checkpoint']:
            return
        ck = {name: obj.state_dict() for name, obj in [('G', g), ('D', d), ('prior', prior), ('ema_G', eg), ('ema_prior', ep), ('opt_G', og), ('opt_D', od)]}
        ck.update(config=cfg, sources=provenance['sources'], step=step, train_seconds=train_seconds,
                  rngs={k: r.get_state() for k, r in rngs.items()}, cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        torch.save(ck, out / 'checkpoint.tmp')
        (out / 'checkpoint.tmp').replace(out / 'checkpoint.pt')

    profiler = SpeedProfiler(out, cfg.get('profile_start', 100), cfg.get('profile_steps', 0))
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    torch.cuda.synchronize()
    block_start = time.perf_counter()
    with (out / 'metrics.jsonl').open('a' if resume else 'w') as log:
        for step in range(start_step + 1, cfg['steps'] + 1):
            profiler.begin(step)
            frac = max(0, (step - 1 - cfg['lr_anneal_start'] * cfg['steps']) / ((1 - cfg['lr_anneal_start']) * cfg['steps']))
            scale = cfg['lr_floor'] + (1 - cfg['lr_floor']) * .5 * (1 + math.cos(math.pi * frac))
            for opt, base in zip((og, od), bases):
                for group, lr in zip(opt.param_groups, base):
                    group['lr'] = lr * scale
            with profiler.region('D_data_fake_condition'):
                d.requires_grad_(True)
                c, real, xt, t = batch()
                with torch.no_grad():
                    xf, _ = fake(c, xt, t)
                critic = conditioned(c, xt, t)
            with profiler.region('D_adversarial_forward'):
                dr, cr = critic(real)
                df, cf = critic(xf)
                ld = gan.d_loss(dr, df)
                if cfg['d_mode'] == 'ucd':
                    targets = d.ucd_labels(c, t)
                    ld = ld + cfg['ucd_lambda'] * (F.cross_entropy(cr, targets) + F.cross_entropy(cf, targets))
            with profiler.region('D_penalty_forward_input_grad'):
                penalty = cifar_penalty(reg, lambda x: critic(x)[0], real, xf, step, rngs['penalty'], cfg)
                ld = ld + penalty
            with profiler.region('D_backward_optimizer'):
                od.zero_grad(set_to_none=True)
                ld.backward()
                od.step()
            with profiler.region('G_data_forward_condition'):
                d.requires_grad_(False)
                c, real, xt, t = batch()
                xf, ids = fake(c, xt, t)
                critic = conditioned(c, xt, t)
            with profiler.region('G_critic_loss'):
                df = critic(xf)[0]
                with torch.no_grad():
                    dr = critic(real)[0]
                lg = gan.g_loss(df, dr)
                if cfg['prior'] == 'learned' and cfg['prior_reg']:
                    selected = prior.table[ids.unique()]
                    if len(selected) > 1:
                        lg = lg + cfg['prior_reg'] * vic(selected)
            with profiler.region('G_backward_optimizer'):
                og.zero_grad(set_to_none=True)
                lg.backward()
                og.step()
            with profiler.region('EMA'):
                update_ema(eg, g, cfg['ema'])
                update_ema(ep, prior, cfg['ema'])
            profiler.end(step)
            report = step % cfg['log_interval'] == 0 or step == cfg['steps']
            evaluate_now = step % cfg['eval_interval'] == 0 or step == cfg['steps']
            if report or evaluate_now:
                torch.cuda.synchronize()
                train_seconds += time.perf_counter() - block_start
                metrics = {'step': step, 'd_loss': float(ld.detach()), 'g_loss': float(lg.detach()), 'penalty': float(penalty.detach()), 'train_seconds': train_seconds,
                           'steps_per_second': step / train_seconds, 'samples_per_second': step * cfg['batch_size'] / train_seconds,
                           'real_draws': 2 * step * cfg['batch_size'], 'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30, 'prior_rms_movement': float((prior.table.detach() - initial_prior).square().mean().sqrt())}
                if not all(math.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError(str(metrics))
                print(f"step={step}/{cfg['steps']} D={metrics['d_loss']:.3f} G={metrics['g_loss']:.3f} cap={metrics['penalty']:.3f} particle_move={metrics['prior_rms_movement']:.4f} train_s={train_seconds:.1f} steps/s={metrics['steps_per_second']:.2f} samples/s={metrics['samples_per_second']:.1f} peak_GB={metrics['peak_memory_gb']:.2f}", flush=True)
                if evaluate_now:
                    print(f"EVAL step={step} samples={cfg['eval_samples']} (diagnostic FID)", flush=True)
                    metrics.update(evaluate(cfg['eval_samples'], step))
                    print(f"FID step={step} n={metrics['samples']} fid={metrics['fid']:.3f}", flush=True)
                    checkpoint(step)
                log.write(json.dumps(metrics, allow_nan=False) + '\n')
                log.flush()
                torch.cuda.synchronize()
                block_start = time.perf_counter()
    print(f"FINAL EVAL samples={cfg['final_samples']}", flush=True)
    final = ({k: metrics[k] for k in ('fid', 'samples', 'sampling_seconds')}
             if cfg['final_samples'] == cfg['eval_samples'] else evaluate(cfg['final_samples'], cfg['steps']))
    summary = {'config': cfg, 'final': final, 'train_seconds': train_seconds,
               'total_seconds': time.perf_counter() - total_start,
               'samples_per_second': cfg['steps'] * cfg['batch_size'] / train_seconds,
               'real_draws': 2 * cfg['steps'] * cfg['batch_size'],
               'peak_memory_gb': torch.cuda.max_memory_allocated() / 2**30, 'environment': env,
               'provenance': provenance, 'parameters': parameters, 'trainable_parameters': trainable_parameters, 'fid_protocol': PROTOCOL}
    write_json(out / 'summary.json', summary)
    print(f"COMPLETE fid={final['fid']:.3f} n={final['samples']} train_s={train_seconds:.1f}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--resume', help='Restore complete training state; config and source must match')
    parser.add_argument('--prepare-data', action='store_true', help='Download verified CIFAR and build FID cache, then exit')
    args = parser.parse_args()
    user = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(user, dict) or set(user) - set(DEFAULTS):
        raise ValueError('config must be a mapping with known keys')
    cfg = {**DEFAULTS, **user}
    validate(cfg)
    if args.prepare_data:
        torch.set_num_threads(4)
        from lib.cifar_metrics import FIDEvaluator
        images, _ = load_cifar(cfg, download=True)
        FIDEvaluator(images, ROOT / cfg['fid_cache'], cfg['eval_batch_size'])
        print('CIFAR and real FID cache ready', flush=True)
    else:
        train(cfg, args.resume)


if __name__ == '__main__':
    main()
