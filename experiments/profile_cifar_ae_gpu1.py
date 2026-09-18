#!/usr/bin/env python
"""Isolated GPU-1 microbenchmarks; never writes production training outputs."""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import time

if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
    raise SystemExit('Run with CUDA_VISIBLE_DEVICES=1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from torchvision.datasets import CIFAR10
from experiments.train_cifar_particle_ae import DEFAULTS, rng
from lib.image_ddgan import update_ema
from lib.image_particle_autoencoder import DirectGenerator, DirectDiscriminator, build_encoder
from particlegan import GANLoss, GradientPenalty, MoGParticlePrior, ParticleRegularizer


class Harness:
    def __init__(self, cfg, images, variant):
        self.cfg, self.images, self.variant = cfg, images, variant
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        self.g = DirectGenerator(cfg['z_dim'], cfg['width']).cuda()
        self.d = DirectDiscriminator(cfg['width']).cuda()
        self.e = build_encoder(cfg).cuda()
        self.prior = MoGParticlePrior(num_particles=cfg['num_particles'], z_dim=cfg['z_dim'],
            sigma_rel=cfg['sigma_rel'], generator=rng(cfg['seed'] + 1, 'cpu')).cuda()
        self.ema = [copy.deepcopy(m).eval().requires_grad_(False) for m in (self.g, self.e, self.prior)]
        self.og = torch.optim.Adam([
            {'params': self.g.parameters(), 'lr': cfg['lr']},
            {'params': self.e.parameters(), 'lr': cfg['lr']},
            {'params': self.prior.parameters(), 'lr': cfg['prior_lr'], 'betas': (.5, .999)},
        ], betas=(0., .999), fused=True)
        self.od = torch.optim.Adam([p for p in self.d.parameters() if p.requires_grad],
            lr=cfg['d_lr'], betas=(0., .999), fused=True)
        self.adv, self.pen, self.spread = GANLoss(), GradientPenalty(lazy_k=cfg['reg_every']), ParticleRegularizer()
        self.data_rng, self.prior_rng = rng(cfg['seed'] + 2), rng(cfg['seed'] + 3)
        self.events = []
        self.instrument = False
        self.annotate = False
        self.ema_targets = [p for m in self.ema for p in m.parameters()]
        self.ema_sources = [p for m in (self.g, self.e, self.prior) for p in m.parameters()]

    @contextlib.contextmanager
    def stage(self, name):
        annotation = torch.profiler.record_function(name) if self.annotate else contextlib.nullcontext()
        with annotation:
            if self.instrument:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                start = time.perf_counter()
            yield
            if self.instrument:
                elapsed = time.perf_counter() - start
                b.record()
                self.events.append((self.regularized, name, a, b, elapsed))

    def batch(self):
        ids = torch.randint(len(self.images), (self.cfg['batch_size'],), device='cuda', generator=self.data_rng)
        x = self.images[ids].float() / 127.5 - 1
        flip = torch.rand((len(x), 1, 1, 1), device='cuda', generator=self.data_rng) < .5
        return torch.where(flip, x.flip(-1), x)

    def step(self, step):
        cfg, g, d, e, p = self.cfg, self.g, self.d, self.e, self.prior
        self.regularized = step % cfg['reg_every'] == 0
        reuse = self.variant in ('reuse_bcap', 'combined', 'batched_d', 'reuse_pruned_d')
        with self.stage('D_prepare_data'):
            d.requires_grad_(True)
            real = self.batch()
        with self.stage('D_prior_and_G_forward'), torch.no_grad():
            z, _ = p.sample(cfg['batch_size'], self.prior_rng)
            fake = g(z)
        self.od.zero_grad(set_to_none=True)
        with self.stage('D_bcap_forward_input_grad'):
            if reuse and self.regularized:
                real_grad = real.detach().clone().requires_grad_(True)
                fake_grad = fake.detach().clone().requires_grad_(True)
                if self.variant == 'batched_d':
                    joined = torch.cat([real_grad, fake_grad])
                    real_logits, fake_logits = d(joined).chunk(2)
                    grads = torch.autograd.grad(real_logits.sum() + fake_logits.sum(), joined, create_graph=True)[0].chunk(2)
                else:
                    real_logits, fake_logits = d(real_grad), d(fake_grad)
                    grads = [torch.autograd.grad(logits.sum(), x, create_graph=True)[0]
                             for logits, x in ((real_logits, real_grad), (fake_logits, fake_grad))]
                norms = [(grad.square().flatten(1).sum(1) + 1e-12).sqrt() for grad in grads]
                dp = self.pen.coeff * cfg['reg_every'] / 2 * sum((norm - self.pen.kappa).relu().square().mean() for norm in norms)
            else:
                dp = self.pen(d, real, fake, step)
        with self.stage('D_adversarial_forward'):
            if not (reuse and self.regularized):
                if self.variant == 'batched_d':
                    real_logits, fake_logits = d(torch.cat([real, fake])).chunk(2)
                else:
                    real_logits, fake_logits = d(real), d(fake)
            dl = self.adv.d_loss(real_logits, fake_logits) + dp
        with self.stage('D_backward_including_bcap'):
            if self.variant in ('pruned_d', 'reuse_pruned_d'):
                dl.backward(inputs=[q for q in d.parameters() if q.requires_grad])
            else:
                dl.backward()
        with self.stage('D_optimizer'):
            self.od.step()
        with self.stage('G_prepare_data_prior'):
            d.requires_grad_(False)
            real = self.batch()
            self.og.zero_grad(set_to_none=True)
            z, _ = p.sample(cfg['batch_size'], self.prior_rng)
        with self.stage('G_real_D_forward'), torch.no_grad():
            dr = d(real)
        with self.stage('G_sample_forward'):
            sample = g(z)
        with self.stage('G_fake_D_forward'):
            gl = self.adv.g_loss(d(sample), dr)
        with self.stage('E_routing_forward'):
            encoded, _, _, _ = e(real, p.means(), p.sigma, cfg['temperature'])
        with self.stage('G_reconstruction_forward'):
            rec = (g(encoded) - real).square().mean()
        with self.stage('prior_regularizer'):
            loss = gl + cfg['recon_weight'] * rec + self.spread(p.z)
        with self.stage('G_E_prior_backward'):
            loss.backward()
        with self.stage('G_E_prior_optimizer'):
            self.og.step()
        with self.stage('EMA'), torch.no_grad():
            if self.variant in ('foreach_ema', 'combined', 'batched_d'):
                torch._foreach_lerp_(self.ema_targets, self.ema_sources, 1 - cfg['ema'])
                for target, source in zip(self.ema, (g, e, p)):
                    for a, b in zip(target.buffers(), source.buffers()):
                        a.copy_(b)
            else:
                for target, source in zip(self.ema, (g, e, p)):
                    update_ema(target, source, cfg['ema'])
        return torch.stack([dl.detach(), gl.detach(), rec.detach(), dp.detach()])

    def snapshot(self):
        modules = (self.g, self.d, self.e, self.prior, *self.ema)
        return {f'{i}:{name}': p.detach().cpu().clone() for i, m in enumerate(modules) for name, p in m.named_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='runs/cifar_particle_ae/performance_gpu1')
    parser.add_argument('--steps', type=int, default=160)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULTS, **yaml.safe_load((ROOT / 'configs/cifar_particle_ae/lazy_long/n08.yaml').read_text()))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    images = torch.from_numpy(CIFAR10(cfg['data_dir'], train=True, download=False).data).permute(0, 3, 1, 2).contiguous().cuda()
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'config': cfg,
              'threads': 2, 'warmup_steps': 40, 'measured_steps': args.steps, 'benchmarks': [], 'correctness': {}}
    print('START GPU1 baseline/candidates, same seed and exact training config; 2 CPU threads', flush=True)
    # Correctness checks include an active bcap step and an ordinary step.
    reference = None
    for variant in ('baseline', 'reuse_bcap', 'foreach_ema', 'combined', 'batched_d'):
        h = Harness(cfg, images, variant)
        losses = torch.stack([h.step(s) for s in (8, 9)]).cpu()
        snap = h.snapshot()
        if reference is None:
            reference = (losses, snap)
        diffs = {k: float((snap[k] - reference[1][k]).abs().max()) for k in snap}
        grad_finite = all(bool(torch.isfinite(p.grad).all()) for m in (h.g, h.d, h.e, h.prior) for p in m.parameters() if p.grad is not None)
        result['correctness'][variant] = {'losses': losses.tolist(), 'loss_max_abs': float((losses - reference[0]).abs().max()),
            'parameter_max_abs': max(diffs.values()), 'worst_parameter': max(diffs, key=diffs.get), 'gradients_finite': grad_finite}
        print('CHECK', variant, json.dumps(result['correctness'][variant]), flush=True)
        del h, snap
    del reference
    for variant in ('baseline', 'reuse_bcap', 'foreach_ema', 'combined', 'batched_d', 'baseline'):
        h = Harness(cfg, images, variant)
        for s in range(1, 41):
            h.step(s)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for s in range(41, 41 + args.steps):
            last = h.step(s)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = {'variant': variant, 'seconds': elapsed, 'steps_per_second': args.steps / elapsed,
               'ms_per_step': elapsed * 1000 / args.steps, 'last_losses': last.cpu().tolist()}
        result['benchmarks'].append(row)
        print('BENCH', json.dumps(row), flush=True)
        if variant == 'baseline' and 'stages' not in result:
            h.instrument = True
            for s in range(1, 33):
                h.step(s)
            torch.cuda.synchronize()
            groups = {}
            for reg, name, a, b, host in h.events:
                groups.setdefault(('regularized' if reg else 'ordinary') + '/' + name, []).append((a.elapsed_time(b), host * 1000))
            result['stages'] = {key: {'cuda_span_ms': sum(x[0] for x in vals) / len(vals),
                                      'host_enqueue_ms': sum(x[1] for x in vals) / len(vals)} for key, vals in groups.items()}
            print('STAGES', json.dumps(result['stages']), flush=True)
            h.instrument = False
            h.annotate = True
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=False) as prof:
                for s in range(1, 9):
                    h.step(s)
            table = prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=35)
            (out / 'profiler.txt').write_text(table)
            prof.export_chrome_trace(str(out / 'trace.json'))
            result['sync_ops'] = [{'name': v.key, 'count': v.count, 'cpu_total_us': v.cpu_time_total} for v in prof.key_averages()
                                  if any(x in v.key for x in ('Synchronize', '_local_scalar_dense', 'aten::item'))]
            print('PROFILER', table, flush=True)
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        del h
    print('COMPLETE', str(out / 'results.json'), flush=True)


if __name__ == '__main__':
    main()
