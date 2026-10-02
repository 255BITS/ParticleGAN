"""Public routed-policy variance/transport ownership, not quadratic identities."""
import pytest
import torch
from examples import e22_routed_antithetic_variance as probe


@pytest.mark.parametrize("bf16", [False, True])
def test_public_policy_dv12_variance_transport_and_state_ownership(bf16):
    torch.set_num_threads(1)
    policy, batches, caller = probe.software_fixture(bf16=bf16)
    initial = policy.state_dict()
    restored, _, _ = probe.software_fixture(bf16=bf16)
    restored.load_state_dict(initial)  # public fresh initialization -> public restore
    for owner in (restored.G, restored.D, restored.router):
        for parameter in owner.parameters(): parameter.grad = torch.full_like(parameter, .25)
    before = probe.digest(restored.state_dict()); caller_before = caller.get_state().clone()
    result = probe.probe(restored, batches, caller)
    assert result["pass"] and result["native_updates"] == 0
    assert result["private_dv12_advanced"] and result["private_gaussian_advanced"]
    assert probe.digest(restored.state_dict()) == before
    assert torch.equal(caller.get_state(), caller_before)
    for owner in (restored.G, restored.D, restored.router):
        assert all(torch.equal(p.grad, torch.full_like(p, .25)) for p in owner.parameters())
    total = result["statistics"]["total_dv12_gaussian"]
    assert set(total) == {"generator", "router", "table", "residual_upstream"}
    assert total["generator"]["ratio"] <= .75
    upstream = total["residual_upstream"]["batch_statistics"]
    assert all(x["transport_discrepancy_max_abs"] <= 1e-8 for x in upstream)
    if bf16:
        assert any(x["transport_discrepancy_max_abs"] > 0 for x in total["generator"]["batch_statistics"])


@pytest.mark.parametrize("single,paired", [(0.,0.), (1.,.751), (1.,float("nan")), (float("inf"),0.)])
def test_predeclared_gate_rejects_unsupported_or_failed_variance(single, paired):
    assert not probe.variance_gate(single, paired)


def test_native_monitor_sentinels_preserved_but_nonfinite_learned_state_rejected():
    policy, _, _ = probe.software_fixture()
    state = policy.state_dict(); state["lr_settle"][0][0]["r_b"] = [torch.tensor(float("nan"))]
    policy.load_state_dict(state)  # native public validator accepts dropped-pair sentinels
    before = probe.digest(state); probe.assert_learned_finite(state)
    assert probe.digest(state) == before
    with torch.no_grad(): state["table"][0,0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite learned"):
        probe.assert_learned_finite(state)


def test_deadline_exception_preserves_policy_gradients_and_global_rng():
    policy, batches, caller = probe.software_fixture()
    before = probe.digest(policy.state_dict()); cpu = torch.get_rng_state().clone()
    def exhausted(): raise TimeoutError("bounded software failure")
    with pytest.raises(TimeoutError): probe.probe(policy, batches, caller, deadline=exhausted)
    assert before == probe.digest(policy.state_dict())
    assert torch.equal(cpu, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in policy.G.parameters())


def test_fixed_routed_dv12_fixture_fails_variance_gate_without_state_changes():
    torch.set_num_threads(1)
    policy, batches, caller = probe.software_fixture(bf16=True, fixture="routed-dv12")
    before = probe.digest(policy.state_dict())
    result = probe.probe(policy, batches, caller)
    assert not result["pass"]
    assert result["statistics"]["total_dv12_gaussian"]["generator"]["ratio"] > .75
    assert result["statistics"]["clean_fixed_gaussian"]["generator"]["ratio"] <= .75
    assert result["private_dv12_advanced"] and result["native_updates"] == 0
    assert before == probe.digest(policy.state_dict())
