"""Prior-policy validation must precede caller-network initialization."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from experiments.forge.api import CapabilityError
from experiments.forge.state import state_digest
from test_forge_native_profiles import tiny_context


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def models(context):
    return (context.construct(lambda: nn.Linear(2, 2), component="generator"),
            context.construct(lambda: nn.Linear(2, 1), component="discriminator"))


def test_late_prior_dtype_failure_preserves_caller_models_and_all_live_streams():
    policy = deepcopy(tiny_context().host_initialization)
    policy["components"]["prior"]["parameters"]["z"] = {"kind": "uniform", "low": -1e100, "high": 1e100}
    context = tiny_context(host_initialization=policy)  # finite/ordered, but cannot fit float32
    g, d = models(context)
    before = state_digest({"g": g.state_dict(), "d": d.state_dict(), "streams": context.streams.state_dict()})
    global_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="dtype range"):
        context.build_trainer(g, d)
    after = state_digest({"g": g.state_dict(), "d": d.state_dict(), "streams": context.streams.state_dict()})
    assert after == before
    assert torch.equal(global_rng, torch.get_rng_state())
    assert context.initialization == {} and context._host_initialized_models == {}
    assert context._host_prior is None and context._trainer is None


def test_success_retains_original_named_draw_sequence_and_initialization_receipts():
    automatic, explicit = tiny_context(), tiny_context()
    g, d = models(automatic)
    rg, rd = models(explicit)
    constructor = automatic.streams.generator("init", component="prior", purpose="locations")
    expected = torch.Generator(device=constructor.device).set_state(constructor.get_state())
    # One public factory draw is the existing protocol, regardless of validation.
    automatic.recipe.make_prior(sigma=automatic.prior_config["sigma"], generator=expected,
        device=automatic.device, dtype=g.weight.dtype, learnable=True, init_std=1.)
    initial_global = torch.get_rng_state().clone()
    trainer = automatic.build_trainer(g, d)
    assert torch.equal(constructor.get_state(), expected.get_state())
    # Preserve the former successful order explicitly: G, D, prior, trainer.
    explicit.initialize(rg, component="generator")
    explicit.initialize(rd, component="discriminator")
    explicit.build_prior(dtype=rg.weight.dtype)
    explicit.build_trainer(rg, rd)
    assert torch.equal(initial_global, torch.get_rng_state())
    assert state_digest(automatic.state_dict()) == state_digest(explicit.state_dict())
    assert automatic.receipt() == explicit.receipt()
    assert torch.equal(trainer.prior.sigma, torch.tensor(.025))


def test_cached_prior_dtype_conflict_is_rejected_before_network_mutation():
    context = tiny_context()
    context.build_prior(dtype=torch.float64)
    g, d = models(context)
    before = state_digest({"g": g.state_dict(), "d": d.state_dict(), "streams": context.streams.state_dict(),
                           "initialization": context.initialization})
    with pytest.raises(CapabilityError, match="different dtype"):
        context.build_trainer(g, d)
    assert state_digest({"g": g.state_dict(), "d": d.state_dict(), "streams": context.streams.state_dict(),
                         "initialization": context.initialization}) == before
