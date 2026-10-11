"""Generator-side critic read: default identity, stall-reach law, RNG and checkpoint contracts."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe
from particlegan.critic_read import BASE, REACH, WINDOW, StallReachRead
from particlegan.init import deterministic_orthogonal_


@pytest.fixture(autouse=True)
def serial_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.autograd.set_multithreading_enabled(False):
        yield
    torch.set_num_threads(threads)


def recipe(read="none", **options):
    return get_recipe("bcap", generator_critic_read=read, total_steps=40, batch_size=16, z_dim=2,
                      num_particles=12, constraint_geometry_mode="direction_blend", **options)


def nets():
    g = deterministic_orthogonal_(nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 2)), seed=0)
    d = deterministic_orthogonal_(nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(.2), nn.Linear(16, 1)), seed=1)
    return g, d


def trainer(read="none", **options):
    torch.manual_seed(0)  # global draws (e.g. prior construction) start identically
    return GANTrainer(recipe(read, **options), *nets(), seed=0)


def run(read="none", steps=20, configure=None):
    model = trainer(read)
    if configure is not None:
        configure(model)
    results = [model.step(real) for real in real_batches(steps)]
    return model, results


def real_batches(n, seed=5):
    stream = torch.Generator().manual_seed(seed)
    return [torch.randn(16, 2, generator=stream) * .1 + 1 for _ in range(n)]


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


class Slope(nn.Module):
    """Linear critic with a chosen input slope."""
    def __init__(self, slope):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[float(slope)], [0.]]))

    def forward(self, x, *args):
        return (x @ self.weight).squeeze(-1)


def test_default_read_is_the_critic_and_trainer_is_bitwise_unchanged():
    critic = nn.Linear(2, 1)
    assert recipe().make_generator_critic_read(critic) is critic
    a, results_a = run()
    b, results_b = run()
    assert a.critic_read is a._noisy_D
    equal(results_a, results_b)
    equal(a.state_dict(), b.state_dict())
    assert "critic_read" not in a.state_dict()
    assert recipe().to_dict() == get_recipe("bcap", total_steps=40, batch_size=16, z_dim=2, num_particles=12,
                                            constraint_geometry_mode="direction_blend").to_dict()
    assert "generator_critic_read" not in recipe().to_dict()


def test_default_trainer_matches_explicit_critic_reads():
    """The off path equals a trainer whose read is forced to the plain noisy critic."""
    a, results_a = run()
    def plain(model):  # explicit pre-change wiring: G reads the noisy critic itself
        model.critic_read = model._noisy_D
    b, results_b = run(configure=plain)
    equal(results_a, results_b)
    equal(a.state_dict(), b.state_dict())


def test_two_dimensional_stencil_is_the_five_point_mean():
    critic = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    read = StallReachRead(critic)
    x = torch.randn(5, 2)
    assert torch.equal(read(x), critic(x))  # plain critic before the first observe
    read.width = .3
    w = .3
    points = [x, x + torch.tensor([w, 0.]), x + torch.tensor([0., w]), x - torch.tensor([w, 0.]), x - torch.tensor([0., w])]
    expected = torch.stack([critic(p) for p in points]).mean(0)
    torch.testing.assert_close(read(x), expected, rtol=0, atol=1e-6)


def test_width_law_counter_and_reset():
    x = torch.zeros(4, 2)
    for slope in (0., .1, .3):
        read = StallReachRead(Slope(slope))
        assert read.observe(x)["width"] == pytest.approx(BASE)
    read = StallReachRead(Slope(1.))
    assert read.observe(x)["width"] == pytest.approx(REACH)  # peak at u = 1
    assert StallReachRead(Slope(2.)).observe(x)["width"] == pytest.approx(.25)
    read = StallReachRead(Slope(.8))
    for index in range(1, WINDOW):
        stats = read.observe(x)
        assert stats["counter"] == index and not stats["stalled"]
        assert stats["width"] == pytest.approx(.4)
    stats = read.observe(x)
    assert stats["stalled"] and stats["width"] == pytest.approx(REACH)
    with torch.no_grad():
        read.probe.weight[0, 0] = .2
    stats = read.observe(x)
    assert stats["counter"] == 0 and not stats["stalled"] and stats["width"] == pytest.approx(BASE)
    # Widths are in data units divided by kappa.
    read = StallReachRead(Slope(.1), kappa=2.)
    assert read.observe(x)["width"] == pytest.approx(BASE / 2)
    with pytest.raises(ValueError):
        recipe("stall_reach", reg_kappa=0.)
    with pytest.raises(ValueError):
        recipe("wide")


def test_observe_leaves_parameter_gradients_and_rng_alone():
    critic = Slope(.5)
    read = StallReachRead(critic)
    state = torch.get_rng_state()
    read.observe(torch.randn(3, 2, generator=torch.Generator().manual_seed(0)))
    assert critic.weight.grad is None
    assert torch.equal(state, torch.get_rng_state())


def test_stall_reach_consumes_no_trainer_rng():
    off, _ = run(steps=3)
    on, _ = run("stall_reach", steps=3)
    for name in off._STREAMS:
        assert torch.equal(getattr(off, name).get_state(), getattr(on, name).get_state()), name
    assert torch.equal(off.state_dict()["cpu_rng"], on.state_dict()["cpu_rng"])
    assert on.critic_read.observes == 3 and on.critic_read.width is not None


def test_state_dict_round_trip_resumes_identical_width_and_counter():
    batches = real_batches(12)
    full, _ = run("stall_reach", steps=12)
    first = trainer("stall_reach")
    for real in batches[:6]:
        first.step(real)
    saved = first.state_dict()
    assert saved["critic_read"]["counter"] == first.critic_read.counter
    resumed = trainer("stall_reach")
    resumed.load_state_dict(saved)
    assert (resumed.critic_read.width, resumed.critic_read.counter) == (first.critic_read.width, first.critic_read.counter)
    for real in batches[6:]:
        resumed.step(real)
    equal(resumed.state_dict(), full.state_dict())
    bad = deepcopy(saved)
    bad["critic_read"]["counter"] = -1
    with pytest.raises(ValueError):
        trainer("stall_reach").load_state_dict(bad)
    with pytest.raises(ValueError):
        trainer().load_state_dict(saved)


def test_collect_stats_surfaces_read_stats():
    run = trainer("stall_reach")
    stats = run.step(real_batches(1)[0], collect_stats=True)["critic_read_stats"]
    assert set(stats) == {"slope", "u", "width", "stalled", "counter"}


def test_conditional_critic_arguments_are_repeated():
    class Conditional(nn.Module):
        def forward(self, x, context, scale):
            return (x.sum(1) * context.squeeze(1)) * scale

    critic = Conditional()
    read = StallReachRead(critic)
    read.width = .2
    x, context = torch.randn(4, 3), torch.randn(4, 1)
    # A linear critic's symmetric stencil mean is the centre value.
    torch.testing.assert_close(read(x, context, 2.), critic(x, context, 2.), rtol=0, atol=1e-6)
    read.observe(x, context, 2.)
    assert read.last_stats["slope"] == pytest.approx(float((2 * context.abs().squeeze() * 3 ** .5).square().mean().sqrt()))


def test_ring8_public_api_toy_runs_with_declared_gate():
    """Short-budget execution check; the full-budget gate result is recorded in the toy report."""
    from benchmarks.toy_audit.stall_reach_ring8 import GATE, train
    result = train(steps=100)
    assert result["gate"] == {"modes": [">=", 7], "hq": [">=", .9]} and GATE
    assert result["critic_read"]["observes"] == 100
    assert type(result["passed"]) is bool and len(result["curve"]) == 2
