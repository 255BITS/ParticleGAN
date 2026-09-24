"""Matched diagnostic of one absolute learning-rate horizon cap.

Apply the existing cosine to min(total_steps, horizon), then hold its floor.
The same rule would leave every older host with budget <= horizon unchanged.
This extra rule is not part of the production shared recipe; these results are
NOT common-gate evidence. No target samples, labels, or centers initialize G.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.toy100 import train as runner
from benchmarks.toy100.accuracy_gate import evaluate_suite as accuracy_suite
from benchmarks.toy100.gate import evaluate_suite
from benchmarks.toy100.models import InputNoise, OutputNoise
from lib.toy_models import SimpleMLPDiscriminator
from particlegan import GANTrainer
from particlegan import training as training_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior', choices=('normal', 'disk', 'square'), default='normal')
    parser.add_argument('--scale', type=float, default=3.0)
    parser.add_argument('--problem', default='grid100')
    parser.add_argument('--horizon', type=int, default=1600)
    args = parser.parse_args()
    if args.horizon <= 0:
        parser.error('--horizon must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output / 'probe_source.py').write_bytes(source)
    config = json.loads(args.config.read_text())
    config.pop('problem', None)
    config.update(z_dim=2, fourier=3, batch_size=2048)
    options = dict(generator='trainable_affine', prior=args.prior,
                   scale=args.scale, initialization='identity',
                   lr_horizon_cap=args.horizon, shared_gate_eligible=False)
    (args.output / 'model_options.json').write_text(json.dumps(options, indent=2)+'\n')
    (args.output / 'declared_config.json').write_text(json.dumps(config, indent=2)+'\n')
    provenance_before = runner._source_provenance

    def provenance():
        record = provenance_before()
        record['source_sha256']['reports/toy100/accuracy_horizon_probe.py'] = hashlib.sha256(source).hexdigest()
        record['trainer_factory'] = 'affine LR-horizon diagnostic; not common-gate evidence'
        record['model_options'] = options
        return record

    def make_trainer(resolved, recipe):
        device = torch.device(resolved['device'])
        if device.type != 'cpu':
            raise ValueError('This bounded probe is CPU-only')
        with torch.random.fork_rng():
            torch.manual_seed(resolved['seed'])
            prior = recipe.make_prior(learnable=True).to(device)
            with torch.no_grad():
                if args.prior == 'normal':
                    prior.z.mul_(args.scale)
                elif args.prior == 'square':
                    prior.z.uniform_(-args.scale, args.scale)
                else:
                    angle = 2 * torch.pi * torch.rand(recipe.num_particles)
                    radius = args.scale * torch.rand(recipe.num_particles).sqrt()
                    prior.z.copy_(torch.stack((angle.cos(), angle.sin()), 1) * radius[:, None])
            generator = nn.Linear(2, 2)
            with torch.no_grad():
                generator.weight.copy_(torch.eye(2))
                generator.bias.zero_()
            discriminator = SimpleMLPDiscriminator(
                in_dim=2, hidden_dim=resolved['d_hidden'], n_hidden=resolved['n_hidden'],
                fourier=resolved['fourier'],
            )
            runner._init_linear(discriminator)
            if resolved['output_noise_std']:
                generator = OutputNoise(generator, resolved['output_noise_std'])
            if resolved['input_noise_std']:
                discriminator = InputNoise(discriminator, seed=resolved['seed']+901, device=device)
            trainer = GANTrainer(recipe, generator, discriminator, prior=prior,
                                 seed=resolved['seed'], optimizer_options={'fused': False})
            ordinary_step = trainer.step
            def step(*args_, **kwargs):
                result = ordinary_step(*args_, **kwargs)
                trace.write(json.dumps(dict(
                    completed_step=trainer.completed_steps,
                    lr_g=trainer.opt_g.param_groups[0]['lr'],
                    lr_prior=trainer.opt_g.param_groups[1]['lr'],
                    lr_d=trainer.opt_d.param_groups[0]['lr'],
                ))+'\n')
                return result
            trainer.step = step
            return trainer

    runner.make_trainer = make_trainer
    runner._source_provenance = provenance
    ordinary_scale = training_module.learning_rate_scale
    training_module.learning_rate_scale = lambda step, total, start, floor: ordinary_scale(
        step, min(total, args.horizon), start, floor,
    )
    with (args.output / 'optimizer_actions.jsonl').open('w', buffering=1) as trace:
        runner.train({**config, 'problem': args.problem}, args.output / args.problem)
    coverage = evaluate_suite(args.output, problem=args.problem)
    fidelity = accuracy_suite(args.output, problem=args.problem)
    print(json.dumps(dict(coverage=coverage['status'], accuracy=fidelity['status'])), flush=True)


if __name__ == '__main__':
    main()
