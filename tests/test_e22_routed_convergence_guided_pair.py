"""Cheap CFG transfer checks; no quality experiment or optimizer substitute."""

from copy import deepcopy
from dataclasses import replace
import io

import pytest
import torch

from examples import e22_routed_convergence as base
from examples import e22_routed_convergence_rotated_teacher as parent
from examples import e22_routed_convergence_guided_pair as law


@pytest.fixture(autouse=True)
def isolated_cpu():
    threads, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        torch.set_rng_state(rng)


@pytest.fixture(scope="module")
def data():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old = parent.make_rotated_data()
        return old, law.make_guided_data(old)
    finally:
        torch.set_num_threads(threads)


def serialization_roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def test_guidance_preserves_existing_tensors_inputs_and_exact_teacher_reachability(data):
    old, guided = data
    before, rng = base.digest(old), torch.get_rng_state().clone()
    witness = law.feasibility_witness(old)
    assert witness["optimizer_updates"] == 0
    assert witness["data_digest"] == guided["digest"]
    assert all(flag for pools in witness["reachability"].values() for flag in pools.values())
    assert witness["neutral_fast_ema_initial_change_only_H_b"]
    assert base.digest(old) == before and torch.equal(torch.get_rng_state(), rng)
    assert guided["digest"] != old["digest"]
    for role in ("initial_ordinary", "initial_particle", "teacher"):
        assert set(guided[role]) == set(old[role]) | set(law.CFG_BUFFERS)
        assert all(torch.equal(value, guided[role][name]) for name, value in old[role].items())
    for pool in base.SPLITS:
        assert torch.equal(old[pool]["context"], guided[pool]["context"])
        assert guided[pool]["targets"].dtype == torch.float32
        assert guided[pool]["base"].dtype == torch.float32
    assert not torch.equal(old["fit"]["targets"], guided["fit"]["targets"])
    assert not torch.equal(old["scale"], guided["scale"])


@pytest.mark.parametrize("arm", law.ARMS)
def test_factory_keeps_parameter_names_seeds_and_isolates_held_globals(arm, data):
    old, guided = data
    before = (base.Host, base.model_forward, base.make_loop.__globals__["Host"])
    loop = law.make_guided_loop(arm, guided, bindings={"source": "software-only"})
    comparison = parent.make_rotated_loop(arm, old)
    assert (base.Host, base.model_forward, base.make_loop.__globals__["Host"]) == before
    assert set(dict(loop.G.named_parameters())) == set(dict(comparison.G.named_parameters()))
    for name, parameter in loop.G.named_parameters():
        assert torch.equal(parameter, dict(comparison.G.named_parameters())[name])
    assert loop.law["init_map"] == comparison.law["init_map"]
    assert loop.law["factory_adapter"] == law.factory_binding_manifest()
    assert loop.law["task"] == law.TASK and loop.law["arm"] == arm
    for pool in base.SPLITS:
        with torch.no_grad():
            assert torch.equal(base.forward(loop, guided[pool]["context"][:4]), guided[pool]["base"][:4])
    for role in ("generator", "average_generator"):
        model = base.modules(loop)[role]
        assert torch.equal(model.cfg_unconditional_source, old["sources"].mean(0))
        assert torch.equal(model.cfg_source_projection, old["source_projection"])


def test_distinct_context_half_order_and_native_usage_do_not_double_ess(data):
    _, guided = data
    loop = law.make_guided_loop(law.ARMS[1], guided)
    p, host = loop.policy, loop.G
    context = guided["fit"]["context"][[0, 16, 32, 48]]
    values = host.half_inputs(context)
    assert torch.equal(values[:4], context[..., :base.WIDTH])
    source = context[:, 0, base.WIDTH:base.WIDTH + 768]
    assert torch.equal(values[4:], values[:4] + ((guided["sources"].mean(0)[None] - source)
                                              @ guided["source_projection"].T)[:, None])
    candidate = p.routed_control.candidate()
    logits, decoded_codes = {}, {}
    callback = p.routed_control.spec.model_forward

    class Capture:
        def __init__(self, routing):
            self.routing = routing

        def mix(self, site, value):
            logits[site] = value.detach().clone()
            return self.routing.mix(site, value)

    def record(models, contexts, proposed, routing):
        return callback(models, contexts, proposed, Capture(routing))

    hooks = [getattr(host, site).register_forward_pre_hook(
        lambda module, args, site=site: decoded_codes.update({site: args[1].detach().clone()}))
        for site in law.SITES]
    p.routed_control.spec.model_forward = record
    try:
        with torch.no_grad():
            output, usage = p.routed_control.spec.forward_with_usage(p._training_modules(), context, candidate)
    finally:
        p.routed_control.spec.model_forward = callback
        for hook in hooks:
            hook.remove()
    assert output.shape == (4, base.TOKENS, base.WIDTH) and output.dtype == torch.float32
    assert usage.shape == (4, base.PARTICLES)
    manual_usage = []
    for site in law.SITES:
        assert logits[site].shape == (4, 2, base.TOKENS, base.PARTICLES)
        weights = (logits[site] + candidate.log_mass).softmax(-1)
        stacked_codes = weights @ candidate.table
        assert torch.equal(decoded_codes[site], torch.cat((stacked_codes[:, 0], stacked_codes[:, 1])))
        manual_usage.append(weights.mean((1, 2)))
    assert torch.allclose(usage, torch.stack(manual_usage).mean(0), rtol=1e-6, atol=1e-8)
    query = p.router.first_query(values) @ candidate.table.T / base.Z_DIM**.5
    assert torch.equal(logits["first"], torch.stack(query.chunk(2), dim=1))
    # Deliberately distinct contexts prevent naive [2B,T,N] -> [B,2,T,N]
    # reshaping from silently passing the ordering check.
    assert not torch.equal(logits["first"], query.reshape(4, 2, base.TOKENS, base.PARTICLES))
    p.routed_control.evidence.observe_effect(0, usage, torch.ones(4), torch.full((4,), 1.1), 1)
    assert p.routed_control.evidence.effect_contexts[0] == 4
    assert p.routed_control.evidence.effective_contexts[0] <= 4 + 1e-12


def test_public_dv12_draws_independent_half_and_site_perturbations(data):
    _, guided = data
    loop = law.make_guided_loop(law.ARMS[1], guided)
    p = loop.policy
    context = guided["fit"]["context"][:4]
    candidate = p.routed_control.candidate()
    prior = p.controller.routed_prior(candidate.table, candidate.log_mass)
    stream = torch.Generator().set_state(p.noise_generator.get_state())
    deltas = []

    def actual_native_perturb(codes):
        perturbed = p.controller.perturb_latent(codes, stream, prior, record=False)
        deltas.append((perturbed - codes).detach().reshape(4, 2, base.TOKENS, base.Z_DIM))
        return perturbed

    before = base.digest(base.checkpoint(loop))
    with torch.no_grad():
        p.routed_control.spec.forward_with_usage(p._training_modules(), context, candidate,
                                               perturb_fn=actual_native_perturb)
    assert len(deltas) == 2
    for displacement in deltas:
        assert displacement.count_nonzero() > 0
        assert not torch.equal(displacement[:, 0], displacement[:, 1])
    assert not torch.equal(deltas[0], deltas[1])
    assert base.digest(base.checkpoint(loop)) == before


def test_candidate_replays_both_sequential_guided_sites(data):
    _, guided = data
    loop = law.make_guided_loop(law.ARMS[1], guided)
    p = loop.policy
    for site in law.SITES:
        with torch.no_grad():
            getattr(loop.G, site).up.weight.copy_(guided["teacher"][site + ".up.weight"])
    context = guided["fit"]["context"][:4]
    candidate = p.routed_control.candidate()
    changed_table = candidate.table.detach().clone()
    changed_table[0] += 2
    changed = replace(candidate, table=changed_table)

    class Capture:
        def __init__(self, proposed):
            self.proposed, self.logits = proposed, {}

        def mix(self, site, logits):
            self.logits[site] = logits.detach().clone()
            return (logits + self.proposed.log_mass).softmax(-1) @ self.proposed.table

    first, second = Capture(candidate), Capture(changed)
    with torch.no_grad():
        old_output = loop.G.forward_routed(context, p.router, candidate, first)
        new_output = loop.G.forward_routed(context, p.router, changed, second)
    assert tuple(first.logits) == tuple(second.logits) == law.SITES
    assert not torch.equal(first.logits["first"], second.logits["first"])
    assert not torch.equal(first.logits["second"], second.logits["second"])
    assert not torch.equal(old_output, new_output)


@pytest.mark.parametrize("arm", law.ARMS)
def test_actual_shared_native_updates_recover_exactly_and_reject_parent_law(arm, data):
    old, guided = data
    loop = law.make_guided_loop(arm, guided, bindings={"source": "software-only"})
    for _ in range(3):
        base.update(loop)
    saved = serialization_roundtrip(base.checkpoint(loop))
    expected_rows = [base.update(loop), base.update(loop)]
    expected = base.checkpoint(loop)
    recovered = law.make_guided_loop(arm, guided, bindings={"source": "software-only"})
    base.restore(recovered, saved)
    assert base.digest([base.update(recovered), base.update(recovered)]) == base.digest(expected_rows)
    assert base.digest(base.checkpoint(recovered)) == base.digest(expected)
    if arm in law.ARMS[1:]:
        assert expected_rows[-1]["bank_gradient_rows"] == base.PARTICLES
        assert expected_rows[-1]["query_gradient_norm"] > 0
        assert recovered.policy.routed_control.spec.max_context_harm == 0
        assert not recovered.policy.routed_control.spec.output_error_guard
    parent_state = base.checkpoint(parent.make_rotated_loop(arm, old))
    before = base.digest(base.checkpoint(recovered))
    with pytest.raises(ValueError, match="must match"):
        base.restore(recovered, parent_state)
    assert base.digest(base.checkpoint(recovered)) == before


def test_changed_guidance_source_or_parent_data_cannot_masquerade_as_fixed_law(data):
    old, guided = data
    changed = deepcopy(guided)
    changed["guided_pair"]["guidance"] = 2
    with pytest.raises(ValueError, match="fixed guidance"):
        law.make_guided_loop(law.ARMS[0], changed)
    changed = deepcopy(guided)
    changed["initial_ordinary"]["cfg_unconditional_source"][0] += .01
    with pytest.raises(ValueError, match="exact fixed parent-source mean"):
        law.make_guided_loop(law.ARMS[0], changed)
    changed = deepcopy(old)
    changed["sources"][0, 0] += .01
    with pytest.raises(ValueError, match="digest"):
        law.make_guided_data(changed)


@pytest.mark.parametrize("role", ("initial_ordinary", "initial_particle", "teacher"))
@pytest.mark.parametrize("buffer", law.CFG_BUFFERS)
def test_redigested_frozen_half_buffer_corruption_is_rejected(role, buffer, data):
    _, guided = data
    changed = deepcopy(guided)
    changed[role][buffer].flatten()[0] += .01
    changed["digest"] = base.digest({key: value for key, value in changed.items() if key != "digest"})
    with pytest.raises(ValueError, match="source projection|parent-source mean"):
        law.make_guided_loop(law.ARMS[1], changed)


def test_redigested_parent_audit_drift_is_rejected(data):
    old, guided = data
    for candidate in (deepcopy(old), deepcopy(guided)):
        candidate["teacher_span_rotation"]["audit_sha256"] = "0" * 64
        candidate["digest"] = base.digest({key: value for key, value in candidate.items() if key != "digest"})
        with pytest.raises(ValueError, match="rotated reachable teacher"):
            if "guided_pair" in candidate:
                law.make_guided_loop(law.ARMS[0], candidate)
            else:
                law.make_guided_data(candidate)
