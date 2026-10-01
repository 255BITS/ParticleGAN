"""Focused geometry, RNG ownership, row-copy and continuation contracts."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
                  OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
sys.dont_write_bytecode = True
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parent
OLD = Path('/ml2/hypergan/gan-attempts/feature-cells-config-20260929')
sys.path.insert(0, str(ROOT.parents[1]/'pkg-CB64-RA2'))
from particlegan import Recipe, ParticlePrior, GANTrainer
from particlegan.feature_cells import BoundedLatentGeometry

SEED = 90229  # Existing integration fixture seed.


def stream():
    return torch.Generator().manual_seed(SEED)


def make_trainer():
    torch.manual_seed(SEED)
    options = json.loads((OLD/'configs'/'overrides-CB64-RA.json').read_text())
    options.update(num_particles=1024, z_dim=2, initialization=None,
                   output_noise_mode='fixed', output_noise_std=0., serve_average=0.)
    G = torch.nn.Linear(2, 2)
    with torch.no_grad():
        G.weight.copy_(torch.eye(2)); G.bias.zero_()
    D = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
    with torch.no_grad():
        D[0].weight[:, 0] = torch.where(D[0].weight[:, 0] >= 0, 100., -100.)
        D[0].weight[:, 1] = 0.; D[0].bias.zero_()
    prior = ParticlePrior(1024, 2, init_std=0., generator=stream())
    with torch.no_grad():
        prior.z[:, 0] = 3.; prior.z[:102, 0] = -3.
    return GANTrainer(Recipe(**options), G, D, prior=prior, seed=SEED, serial_backward=True)


def real_batch():
    real = torch.zeros(1024, 2)
    real[:, 0] = -3.; real[:102, 0] = 3.
    return real[torch.randperm(1024, generator=stream())]


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype \
            and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a if k != 'eval_seconds')
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a,float) and a != a:
        return isinstance(b,float) and b != b
    return a == b


class GeometryTests(unittest.TestCase):
    def test_local_radius_scaling_duplicates_and_cache_invalidation(self):
        points = torch.nn.Parameter(torch.tensor([[0., 0.], [2., 0.], [0., 2.], [2., 2.]]))
        prior = SimpleNamespace(z=points)
        kernel = BoundedLatentGeometry()
        radius = kernel.radius(points, prior)
        self.assertTrue(torch.equal(radius, torch.ones(4)))
        with torch.no_grad():
            points.mul_(3.)
        self.assertTrue(torch.equal(kernel.radius(points, prior), torch.full((4,), 3.)))
        with torch.no_grad():
            points.zero_()
        self.assertTrue(torch.equal(kernel.radius(points, prior), torch.zeros(4)))
        # A duplicate center must still see the nearest nonidentical center.
        with torch.no_grad():
            points[-1, 0] = .2
        self.assertTrue(torch.equal(kernel.radius(points, prior), torch.full((4,), .1)))

    def test_training_fake_pool_and_copy_use_same_draw_law(self):
        trainer = make_trainer(); bd = trainer.birth_death
        latent = trainer.prior.z[:40]
        noise = torch.randn(latent.shape, generator=stream())
        delta = bd.latent_geometry.displacement(latent, trainer.prior, trainer.controller.latent_bandwidth, noise)
        generated = trainer._generate(trainer.G, latent, 0., stream())
        self.assertTrue(torch.equal(generated, trainer.G(latent+delta)))
        bd.stream.set_state(stream().get_state()); bd.sample_shape = (2,)
        captured = bd._capture_generated(trainer, latent, jitter=True)
        expected_features = bd._features(trainer, trainer.G(latent+delta), chunk=256)
        self.assertTrue(torch.equal(captured, expected_features))
        bd.stream.set_state(stream().get_state())
        child = torch.arange(40, 80); parent = torch.arange(40)
        saved = trainer.noise_generator.get_state().clone()
        bd._move(trainer, child, parent)
        self.assertTrue(torch.equal(trainer.prior.z[child], latent.detach()+delta))
        self.assertTrue(torch.equal(saved, trainer.noise_generator.get_state()))
        # Jitter radius is detached while the ordinary latent gradient remains.
        trainer.opt_g.zero_grad()
        trainer._generate(trainer.G, trainer.prior.z[:40], 0., stream()).sum().backward()
        self.assertTrue(torch.equal(trainer.prior.z.grad[:40], torch.ones(40, 2)))

    def test_copy_preserves_optimizer_ema_and_history_rows(self):
        trainer = make_trainer(); bd = trainer.birth_death
        child = torch.tensor([0, 1]); parent = torch.tensor([300, 301])
        with torch.no_grad():
            trainer.ema_prior.z.copy_(trainer.prior.z+10.)
        state = trainer.opt_g.state[trainer.prior.z]
        rows = torch.arange(2048, dtype=torch.float32).reshape(1024, 2)
        for name, offset in (('exp_avg', 1.), ('exp_avg_sq', 2.), ('max_exp_avg_sq', 3.)):
            state[name] = rows+offset
        state['step'] = torch.tensor(7.)
        trainer.opt_g.latent_history.copy_(rows+4.)
        ema = trainer.ema_prior.z.detach().clone()
        moment = deepcopy(state); history = trainer.opt_g.latent_history.clone()
        prior_before = trainer.prior.z.detach().clone()
        draw = torch.Generator().set_state(bd.stream.get_state())
        delta = bd.latent_geometry.displacement(prior_before[parent], trainer.prior,
                trainer.controller.latent_bandwidth, torch.randn((2, 2), generator=draw))
        bd._move(trainer, child, parent)
        self.assertTrue(torch.equal(trainer.ema_prior.z[child], ema[parent]+delta))
        for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            self.assertTrue(torch.equal(state[name][child], moment[name][parent]))
        self.assertTrue(torch.equal(state['step'], moment['step']))
        self.assertTrue(torch.equal(trainer.opt_g.latent_history[child], history[parent]))

    def test_ema_sampling_uses_corresponding_prior_and_preserves_streams(self):
        trainer = make_trainer()
        with torch.no_grad():
            trainer.ema_prior.z.zero_()
        streams = {name: getattr(trainer, name).get_state().clone() for name in trainer._STREAMS}
        private = trainer.birth_death.stream.get_state().clone()
        rng = torch.get_rng_state().clone()
        sample = trainer.sample(40, ema=True, generator=stream())
        self.assertTrue(torch.equal(sample, trainer.ema_G(torch.zeros(40, 2))))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(private, trainer.birth_death.stream.get_state()))
        for name, before in streams.items():
            self.assertTrue(torch.equal(before, getattr(trainer, name).get_state()))

    def test_checkpoint_after_ordinary_moves_replays_exactly(self):
        trainer = make_trainer(); real = real_batch()
        trainer.step(real)
        self.assertGreater(trainer.birth_death.last['ordinary_moves'], 0)
        saved = trainer.state_dict()
        self.assertEqual(saved['birth_death']['backend_schema'], 3)
        self.assertNotIn('latent_geometry', saved['birth_death'])
        first = trainer.step(real); endpoint = trainer.state_dict()
        restored = make_trainer(); restored.load_state_dict(saved)
        self.assertIsNone(restored.birth_death.snapshot)
        self.assertFalse(restored.birth_death.latent_geometry._entries)
        second = restored.step(real)
        for name in ('loss_d', 'loss_g', 'loss_gan', 'prior_regularization', 'penalty'):
            self.assertTrue(same(first[name], second[name]), name)
        self.assertTrue(same(endpoint, restored.state_dict()))
        bad = deepcopy(saved['birth_death']); bad['backend_schema'] = 1
        with self.assertRaises(ValueError):
            restored.birth_death.check_state(bad)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryTests))
    receipt = dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                   seconds=time.perf_counter()-start, seed=SEED, cuda_initialized=torch.cuda.is_initialized(),
                   passed=result.wasSuccessful(), source_sha256={str(p.relative_to(ROOT.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted((ROOT.parents[1]/'pkg-CB64-RA2'/'particlegan').glob('*.py'))})
    (ROOT/'test-results.json').write_text(json.dumps(receipt, indent=2)+'\n')
    assert not receipt['cuda_initialized']
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
