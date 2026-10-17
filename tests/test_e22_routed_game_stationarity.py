"""Local frozen-critic stationarity, distinct from true joint GAN equilibrium."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from examples import e22_routed_game_stationarity as law
from examples import e22_routed_convergence_rotated_teacher as parent

held = law.held


@pytest.fixture(autouse=True)
def private_cpu_scope():
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


class FixedNonlinearCritic(nn.Module):
    """D(x)=.7x+.2x²; deterministic algebra, with no learned/trained fixture."""
    def forward(self, error, condition):
        return (.7 * error + .2 * error.square()).mean((1, 2)).unsqueeze(-1)


class ReflectedScore(nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic
    def forward(self, error, condition):
        return .5 * (self.critic(error, condition) + self.critic(-error, condition))


@pytest.mark.parametrize("case,reflection,antithetic,expected", [
    ("current_single", False, False, -.41), ("current_antithetic", False, True, -.35),
    ("even_single", True, False, -.06), ("even_antithetic", True, True, 0.)])
def test_native_rpgan_exact_solution_can_have_force_and_coupled_symmetry_cancels(case, reflection, antithetic, expected):
    # An arbitrary frozen D need not be at its joint GAN equilibrium. This test
    # asserts the local derivative and the explicit ingredient law only.
    theta = torch.zeros((1, 1, 1), dtype=torch.float64, requires_grad=True)
    critic = FixedNonlinearCritic()
    critic = ReflectedScore(critic) if reflection else critic
    g, d = law.paired_games(critic, theta, torch.zeros((1, 1)),
        torch.full_like(theta, .3), antithetic=antithetic)
    gradient = torch.autograd.grad(g.sum(), theta)[0]
    assert float(g) == pytest.approx(math.log(2), abs=1e-14)
    assert float(d) == pytest.approx(math.log(2), abs=1e-14)
    assert float(gradient) == pytest.approx(expected, abs=1e-14)
    assert -float(gradient.square().sum()) == pytest.approx(-expected**2, abs=1e-14)
    assert theta.grad is None


def test_reflection_evenizes_same_learned_features_and_score_without_new_tensors():
    with torch.random.fork_rng(devices=[]):
        current = held.ConditionalCritic(torch.ones(held.WIDTH))
        held.initialize(current, "critic")
        even = law.ReflectionEvenCritic(torch.ones(held.WIDTH))
    even.load_state_dict(current.state_dict(), strict=True)
    current.eval().requires_grad_(False)
    even.eval().requires_grad_(False)
    error = torch.linspace(-.6, .9, 4*held.TOKENS*held.WIDTH).reshape(4,held.TOKENS,held.WIDTH)
    condition = torch.linspace(-.1,.2,4*769).reshape(4,769)
    before = held.digest(even.state_dict())
    assert set(dict(current.named_parameters())) == set(dict(even.named_parameters()))
    assert held.digest(current.state_dict()) == before
    assert torch.equal(even.features(error,condition), even.features(-error,condition))
    assert torch.equal(even(error,condition), even(-error,condition))
    torch.testing.assert_close(even.features(error,condition),
        .5*(current.features(error,condition)+current.features(-error,condition)), rtol=0, atol=0)
    torch.testing.assert_close(even(error,condition), .5*(current(error,condition)+current(-error,condition)),
        rtol=2e-6, atol=2e-7)
    assert held.digest(even.state_dict()) == before


def test_contexts_and_both_role_panels_use_private_fixed_streams():
    data = parent.make_rotated_data()
    before = torch.get_rng_state().clone()
    indices, stream = law.fit_contexts(data)
    raw = law.private_panels()
    assert raw.shape == (2,12,held.TOKENS,held.WIDTH)
    assert not torch.equal(raw[0],raw[1])
    assert held.digest(raw) == held.digest(law.private_panels())
    assert held.digest(stream) == held.digest(law.fit_contexts(data)[1])
    assert [int(data["fit"]["subjects"][index]) for index in indices] == [s for s in range(6) for _ in range(2)]
    assert torch.equal(torch.get_rng_state(),before)


def test_actual_fresh_particle_graph_is_exact_immutable_and_only_Up_can_acquire():
    data = parent.make_rotated_data()
    with torch.random.fork_rng(devices=[]):
        loop = parent.make_rotated_loop("particle_native_game", data)
        current = held.ConditionalCritic(data["scale"])
        held.initialize(current,"critic")
        even = law.ReflectionEvenCritic(data["scale"])
    even.load_state_dict(current.state_dict(),strict=True)
    current.eval().requires_grad_(False); even.eval().requires_grad_(False)
    judges = {"fixed_software_weights": (current,even)}
    owners = {"current":current,"even":even}
    law.fresh_witness(loop)
    before = law.owner_identity(loop,owners)
    indices,_ = law.fit_contexts(data)
    calls = []
    hook = loop.G.first.register_forward_hook(lambda module,args,value: calls.append(1))
    try:
        results = law.observe_batch(loop,judges,data["fit"]["context"][indices[:4]],law.private_panels()[:,:4])
    finally:
        hook.remove()
    assert calls == [1] and len(results) == 4
    assert law.owner_identity(loop,owners) == before
    for result in results:
        assert result["native_context_count"] == 4 and result["tokens_per_context"] == 16
        assert result["known_solution_residual_nonzero_coordinates"] == 0
        for item in result["per_context"]:
            assert item["generator_game"] == pytest.approx(math.log(2),abs=2e-7)
            for owner in ("generator_down","generator_H_b","generator_C","bank","router","code/first","code/second"):
                assert item["norms"][owner] == 0
            assert item["negative_residual_gradient_directional_derivative"] <= 0
            if result["case"] == "even_antithetic":
                assert item["local_stationarity_verified"]
    original = deepcopy(loop.G.first.up.weight)
    with torch.no_grad():loop.G.first.up.weight[0,0] = .01
    with pytest.raises(AssertionError,match="zero-Up"):
        law.fresh_witness(loop)
    with torch.no_grad():loop.G.first.up.weight.copy_(original)
