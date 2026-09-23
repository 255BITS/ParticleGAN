"""Exploratory data-space particle GAN; no target geometry enters training."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.toy100 import train as runner
from benchmarks.toy100.gate import evaluate_suite
from lib.toy_models import SimpleMLPDiscriminator, SimpleMLPGenerator
from particlegan import GANTrainer


class AffineGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, z):
        return z @ self.weight.T + self.bias


MODEL_OPTIONS = {'generator_kind': 'affine', 'initial_prior': 'disk', 'radius': 7., 'output_noise': 0.}


class NoisyGenerator(nn.Module):
    def __init__(self, model, std):
        super().__init__()
        self.model, self.std = model, std

    def forward(self, z):
        value = self.model(z)
        return value + torch.randn_like(value) * self.std


def factory(config, recipe):
    device = torch.device(config['device'])
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config['seed'])
        prior = recipe.make_prior().to(device)
        # Broad isotropic disk; radius is a declared model hyperparameter.
        # No target samples, component count, centers, or assignments are read.
        with torch.no_grad():
            angles = torch.rand(recipe.num_particles, device=device) * (2 * torch.pi)
            radius = torch.rand(recipe.num_particles, device=device).sqrt() * MODEL_OPTIONS['radius']
            if MODEL_OPTIONS['initial_prior'] == 'disk':
                prior.z.copy_(torch.stack((angles.cos(), angles.sin()), 1) * radius[:, None])
            elif MODEL_OPTIONS['initial_prior'] == 'uniform':
                prior.z.uniform_(-MODEL_OPTIONS['radius'], MODEL_OPTIONS['radius'])
        generator = (AffineGenerator() if MODEL_OPTIONS['generator_kind'] == 'affine' else
                     SimpleMLPGenerator(z_dim=recipe.z_dim, hidden_dim=config['g_hidden'], n_hidden=config['n_hidden'])).to(device)
        discriminator = SimpleMLPDiscriminator(
            hidden_dim=config['d_hidden'], n_hidden=config['n_hidden'], fourier=config['fourier']).to(device)
        if MODEL_OPTIONS['generator_kind'] != 'affine':
            runner._init_linear(generator)
        runner._init_linear(discriminator)
        if MODEL_OPTIONS['output_noise']:
            generator = NoisyGenerator(generator, MODEL_OPTIONS['output_noise'])
        return GANTrainer(recipe, generator, discriminator, prior=prior, seed=config['seed'],
                          optimizer_options={'fused': config['fused_adam']})


original_provenance = runner._source_provenance
def provenance():
    value = original_provenance()
    source = str(Path(__file__).resolve().relative_to(ROOT))
    value['source_sha256'][source] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    value['trainer_factory'] = 'noisy_mlp_allocation_probe.factory'
    value['model_options'] = MODEL_OPTIONS
    return value


if __name__ == '__main__':
    runner.make_trainer = factory
    runner._source_provenance = provenance
    config = json.loads((ROOT/'configs/toy100/baseline.json').read_text())
    config.update(name='direct_disk7_affine', device='cpu', steps=4000, z_dim=2,
                  lr=.00003, prior_lr_mult=1000., d_lr_mult=20., prior_reg=0.,
                  betas=[0., .999], reg_coeff=1., reg_kappa=1., batch_size=512,
                  fourier=4, n_hidden=2, lr_anneal_start=.4)
    if len(sys.argv)>2:
        argument = Path(sys.argv[2])
        overrides = json.loads(argument.read_text() if argument.is_file() else sys.argv[2])
        MODEL_OPTIONS.update(overrides.pop('model_options', {}))
        config.update(overrides)
    out=ROOT/sys.argv[1]
    out.mkdir(parents=True, exist_ok=True)
    (out/'probe_source.py').write_text(Path(__file__).read_text())
    (out/'model_options.json').write_text(json.dumps(MODEL_OPTIONS, indent=2))
    torch.manual_seed(config['seed'])
    runner.train(config, out/config.get('problem','grid100'))
    print(json.dumps(evaluate_suite(out, problem=config.get('problem','grid100'))),flush=True)
