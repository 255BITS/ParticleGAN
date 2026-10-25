"""particlegan_forge RNG parity with experiments/forge (review #19)."""
from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from experiments.forge import rng as forge_rng
from particlegan_forge import rng as pgf_rng


def _streams(module, seed=0):
    streams = module.NamedStreams(seed)
    torch.rand(3, generator=streams.generator("data"))
    torch.rand(2, generator=streams.generator("eval", component="probe"))
    return streams


def test_derivation_and_draws_match_experiments_forge():
    assert pgf_rng.RNG_VERSION == forge_rng.RNG_VERSION and pgf_rng.STREAM_NAMES == forge_rng.STREAM_NAMES
    a, b = _streams(pgf_rng), _streams(forge_rng)
    assert a.manifest() == b.manifest() and a.audit() == b.audit()
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["manifest"] == sb["manifest"]
    assert all(torch.equal(sa["states"][k], sb["states"][k]) for k in sa["states"])


def _mutations(state):
    yield "seed", lambda s: s["manifest"].__setitem__("seed", 1)
    yield "version", lambda s: s["manifest"].__setitem__("version", "forge-rng-v0")
    yield "runtime", lambda s: s["manifest"].__setitem__("runtime", {"torch": "0.0", "rng": "torch.Generator"})
    yield "extra_key", lambda s: s.__setitem__("extra", 1)
    yield "binding_seed", lambda s: next(iter(s["manifest"]["bindings"].values())).__setitem__("seed", 5)
    yield "missing_state", lambda s: s["states"].pop(next(iter(s["states"])))
    yield "schema", lambda s: s.__setitem__("schema", 2)


@pytest.mark.parametrize("name", [name for name, _ in _mutations(None)])
def test_validate_state_dict_rejects_like_experiments_forge(name):
    mutate = dict(_mutations(None))[name]
    outcomes = []
    for module in (pgf_rng, forge_rng):
        state = deepcopy(_streams(module).state_dict())
        mutate(state)
        target = module.NamedStreams(0)
        torch.rand(1, generator=target.generator("noise"))
        before = target.audit()
        with pytest.raises(ValueError) as info:
            target.load_state_dict(state)
        assert target.audit() == before  # validation precedes mutation
        outcomes.append(str(info.value))
    assert outcomes[0] == outcomes[1]


def test_round_trip_restores_and_drops_unbound_streams():
    source = _streams(pgf_rng)
    state = source.state_dict()
    for module in (pgf_rng, forge_rng):
        target = module.NamedStreams(0)
        torch.rand(1, generator=target.generator("noise"))
        target.load_state_dict(state)
        assert target.audit() == source.audit()
        assert target.manifest() == source.manifest()


def test_deterministic_scope_is_strict_and_restores_flags():
    cudnn = torch.backends.cudnn
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
              torch.get_num_threads())
    rng_before = torch.get_rng_state()
    with pgf_rng.deterministic_scope(seed=0):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
        assert cudnn.deterministic and not cudnn.benchmark
        torch.rand(4)
    after = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             cudnn.benchmark, cudnn.deterministic, cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             torch.get_num_threads())
    assert after == before and torch.equal(torch.get_rng_state(), rng_before)
