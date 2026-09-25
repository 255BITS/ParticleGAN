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
from lib.toy_models import SimpleMLPDiscriminator
from particlegan import GANTrainer


class AffineGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(2))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, z):
        return z @ self.weight.T + self.bias


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
            radius = torch.rand(recipe.num_particles, device=device).sqrt() * 7.0
            prior.z.copy_(torch.stack((angles.cos(), angles.sin()), 1) * radius[:, None])
        generator = AffineGenerator().to(device)
        discriminator = SimpleMLPDiscriminator(
            hidden_dim=config['d_hidden'], n_hidden=config['n_hidden'], fourier=config['fourier']).to(device)
        runner._init_linear(discriminator)
        return GANTrainer(recipe, generator, discriminator, prior=prior, seed=config['seed'],
                          optimizer_options={'fused': config['fused_adam']})


original_provenance = runner._source_provenance
def provenance():
    value = original_provenance()
    value['source_sha256']['artifacts/toy100/direct_probe.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    value['trainer_factory'] = 'direct_probe.factory: affine G, disk radius 7 initial prior'
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
        config.update(json.loads(sys.argv[2]))
    out=ROOT/sys.argv[1]
    runner.train(config, out/config.get('problem','grid100'))
    print(json.dumps(evaluate_suite(out, problem=config.get('problem','grid100'))),flush=True)
