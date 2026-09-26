"""Default sample stream stays on the torch RNG; sobol and r2 repeat."""

import hashlib

import pytest
import torch

import particlegan.sample_stream as sample_stream
from benchmarks.locked_shared.mlp import SimpleMLPDiscriminator, SimpleMLPGenerator
from benchmarks.locked_shared.mode_hold import ModeHoldRecipe, ring_means, sample_ring, train_mode_hold
from benchmarks.toy100.models import OutputNoise
from benchmarks.toy100.problems import sample_real
from particlegan.particle_prior import ParticlePrior
from torch import nn

# Captured on unmodified develop (0ff9a7af) before this patch, CPU, torch RNG.
RING_DRAWS = (
    "0617eed0119672e8f599b28b0539a939a8ee60236ab49466f14568e733f0497a",
    "5f6c55ae129df2c2798d72fc26e88f149c9550f0895ce12d69152599f7d7f4e6",
    "79635bc4ef85a078060e6a015720cbd527887d28b671474749d265c68f47f7de",
)
RING_STATE = "bdface713608d359d379436ad4d70a66bc6e14facdc869479a81f7c772df3285"
PRIOR = (
    "2712779087de0ba1181f9cca548c3229c635b08792e3ea3f95355de1fcf9321e",
    "367753d4fe548ff46b5d2ae5ce92713ec9786e09e1d254d4b8d51ff77ad1b2a4",
    "55020fa4406a36e28337616c123692c559855230ea2f2abcde8a30ac0cf7746d",
    "38cf6537392b3d8019cc5729bab1a8c3c8e37d844d9b51111ed1575b9f479a53",
    "25e51ba266bef03ab0102a61ed9ea4567f1bb9b89e43bd77e94809b8958e0135",
)
OUTPUT_NOISE = "32eb541aee9cd38f1dc6067b16cda0193ee8ffeb713de250d2604465755aab96"
GLOBAL_AFTER_OUTPUT = "7c64e68bca73b21fb0ca6905052830f84c1063dffe7804e35cea456e33489a9b"
SAMPLE_REAL = "92c9d3a9383d68bb2db9556ac87366cda2e3a27edeba42cdd3666e13f26ba8b5"
SAMPLE_REAL_STATE = "dc890b701e9a0f92f6948f880546b5d999960fdd6182dcd86e5562c9b5001023"
HOLD_PARAMS = "7ef69d6b0f0e23d3c76682d58ab75906ad87b61d59f1b94479532ad81a1c2cdb"


def _dig(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _hold_hash(kind):
    sample_stream.configure(kind)
    gens, discs, priors = [], [], []
    og, od, op = SimpleMLPGenerator.__init__, SimpleMLPDiscriminator.__init__, ParticlePrior.__init__

    def ginit(self, *args, **kwargs):
        og(self, *args, **kwargs)
        gens.append(self)

    def dinit(self, *args, **kwargs):
        od(self, *args, **kwargs)
        discs.append(self)

    def pinit(self, *args, **kwargs):
        op(self, *args, **kwargs)
        priors.append(self)

    SimpleMLPGenerator.__init__ = ginit
    SimpleMLPDiscriminator.__init__ = dinit
    ParticlePrior.__init__ = pinit
    try:
        train_mode_hold(ModeHoldRecipe(steps=2), seed=0)
    finally:
        SimpleMLPGenerator.__init__ = og
        SimpleMLPDiscriminator.__init__ = od
        ParticlePrior.__init__ = op
    parts = [_dig(p) for module in gens + discs + priors for _, p in module.named_parameters()]
    return hashlib.sha256("".join(parts).encode()).hexdigest()


@pytest.fixture(autouse=True)
def _reset_stream():
    sample_stream.configure("rng")
    yield
    sample_stream.configure("rng")


def test_default_rng_matches_develop_hashes():
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    means = ring_means()
    draws = [_dig(sample_ring(means, 16, 0.07, generator)) for _ in range(3)]
    assert tuple(draws) == RING_DRAWS
    assert _dig(generator.get_state()) == RING_STATE

    prior = ParticlePrior(32, 4, init_std=0.5, generator=torch.Generator().manual_seed(1))
    stream = torch.Generator().manual_seed(2)
    first, first_idx = prior.sample(8, generator=stream)
    second, second_idx = prior.sample(8, generator=stream)
    assert (_dig(first), _dig(first_idx), _dig(second), _dig(second_idx), _dig(stream.get_state())) == PRIOR

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(torch.eye(2))
        linear.bias.zero_()
    torch.manual_seed(4)
    assert _dig(OutputNoise(linear, 0.1)(torch.zeros(5, 2))) == OUTPUT_NOISE
    assert _dig(torch.get_rng_state()) == GLOBAL_AFTER_OUTPUT

    data = torch.Generator().manual_seed(5)
    assert _dig(sample_real("grid100", 20, generator=data)) == SAMPLE_REAL
    assert _dig(data.get_state()) == SAMPLE_REAL_STATE
    assert _hold_hash("rng") == HOLD_PARAMS


def test_sobol_and_r2_repeat_and_leave_the_torch_rng():
    rng_hash = _hold_hash("rng")
    assert rng_hash == HOLD_PARAMS
    for kind in ("sobol", "r2"):
        first = _hold_hash(kind)
        second = _hold_hash(kind)
        assert first == second
        assert first != rng_hash
    sample_stream.configure("sobol")
    state = torch.get_rng_state()
    generator = torch.Generator().manual_seed(0)
    generator_state = generator.get_state().clone()
    with sample_stream.update():
        sample_ring(ring_means(), 8, 0.07, generator)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(generator.get_state(), generator_state)


def test_indices_are_floor_and_gaussians_are_inverse_cdf():
    eps = torch.finfo(torch.float64).eps
    sample_stream.configure("sobol")
    reference = torch.quasirandom.SobolEngine(1, scramble=False).draw(8, dtype=torch.float64).reshape(-1)
    with sample_stream.update():
        idx = sample_stream.indices("floor", 8, 5)
    assert torch.equal(idx.cpu(), torch.floor(reference * 5).to(torch.long))

    sample_stream.configure("r2")
    phi = 2.0
    for _ in range(64):
        phi = (1.0 + phi) ** (1.0 / 4)
    alpha = torch.tensor([(1.0 / phi) ** (j + 1) for j in range(3)], dtype=torch.float64)
    rows = torch.arange(1, 5, dtype=torch.float64).unsqueeze(1)
    unit = torch.remainder(0.5 + rows * alpha, 1.0)
    expect = torch.special.ndtri(unit.clamp(eps, 1.0 - eps))
    with sample_stream.update():
        got = sample_stream.normal("gauss", (4, 3), dtype=torch.float64)
    assert torch.allclose(got.cpu(), expect)


def test_categorical_is_inverse_cdf_and_allows_duplicate_indices():
    masses = torch.tensor([0.05, 0.95], dtype=torch.float64)
    cdf = torch.cumsum(masses, 0)
    sample_stream.configure("sobol")
    unit = torch.quasirandom.SobolEngine(1, scramble=False).draw(64, dtype=torch.float64).reshape(-1)
    unit = torch.minimum(unit, cdf[-1])
    expect = torch.searchsorted(cdf, unit, right=False)
    with sample_stream.update():
        idx = sample_stream.categorical("mass", 64, masses)
        dup = sample_stream.indices("dup", 32, 4)
    assert torch.equal(idx.cpu(), expect)
    assert int((idx == 1).sum()) > int((idx == 0).sum())
    assert int(torch.bincount(dup).max()) > 1


def test_eval_outside_update_does_not_advance_the_training_stream():
    sample_stream.configure("sobol")
    means = ring_means()
    with sample_stream.update():
        sample_ring(means, 4, 0.07, torch.Generator().manual_seed(0))
    drawn = sample_stream.points_drawn("data", 3)
    assert drawn == 4
    state = torch.get_rng_state()
    generator = torch.Generator().manual_seed(11)
    outside = sample_ring(means, 4, 0.07, generator)
    direct_g = torch.Generator().manual_seed(11)
    direct = means[torch.randint(0, means.shape[0], (4,), generator=direct_g)]
    direct = direct + 0.07 * torch.randn(4, means.shape[1], generator=direct_g)
    assert sample_stream.points_drawn("data", 3) == drawn
    assert torch.equal(outside, direct)
    assert torch.equal(generator.get_state(), direct_g.get_state())
    assert torch.equal(torch.get_rng_state(), state)
