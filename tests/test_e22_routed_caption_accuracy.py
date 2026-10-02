"""New benchmark software checks; exactly four tiny native updates total."""
from copy import deepcopy
import json
import math
import sys

import pytest
import torch

from examples import e22_routed_caption_accuracy as api


@pytest.fixture(autouse=True)
def isolated_cpu():
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]): yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def tiny_data():
    with torch.random.fork_rng(devices=[]):
        return api.make_data(api.SMALL, torch.device("cpu"))


def metric(v):
    return {"rmse": v, "by_source": {str(i): v for i in range(6)}}


def gate(ordinary=None, particle=None, zero=None, **changes):
    return api.scientific_gate(metric(1) if ordinary is None else ordinary,
        metric(.99) if particle is None else particle, metric(1) if zero is None else zero,
        **{"bank_updates": 511, "query_updates": 511, "C_norms": {s: .1 for s in api.SITES}, **changes})


def test_oracle_and_destructive_controls_run_without_native_updates():
    assert all(api.scorer_controls().values())


def test_physical_float64_rmse_and_unequal_source_weighting():
    labels = torch.repeat_interleave(torch.arange(6), torch.arange(1, 7))
    values = torch.arange(1, 22, dtype=torch.float64) / 37
    residual = values[:, None, None].expand(21, 2, 3)
    result = api.accuracy(residual, labels)
    oracle = math.sqrt(math.fsum(float(v)**2 for v in values) / 21)
    assert result["rmse"] == pytest.approx(oracle, rel=1e-14)
    assert result["contexts"] == 21 and result["coordinates_per_context"] == 6
    for source in range(6):
        power = [float(v)**2 for v in values[labels == source]]
        assert result["by_source"][str(source)] == pytest.approx(math.sqrt(math.fsum(power) / len(power)), rel=1e-14)
        assert result["source_counts"][str(source)] == source + 1
    assert abs(result["rmse"] - math.sqrt(sum(v*v for v in result["by_source"].values()) / 6)) > .01


def test_future_particle_win_passes_without_historical_failure_requirement():
    assert gate()["pass"]
    # Exact inclusive relative thresholds requested by the new protocol.
    candidate = metric(1 - api.RELATIVE_MARGIN)
    assert gate(particle=candidate, zero=metric(candidate["rmse"] * (1 + api.RELATIVE_MARGIN)))["pass"]
    assert not gate(particle=metric(1))["aggregate_accuracy_improved"]


def test_source_harm_tolerance_and_strict_code_benefit_boundaries():
    candidate = metric(.99); candidate["by_source"]["5"] = 1 + api.SOURCE_HARM
    assert gate(particle=candidate, zero=metric(1.01))["no_source_harmed"]
    candidate["by_source"]["5"] += 1e-12
    assert not gate(particle=candidate, zero=metric(1.01))["pass"]
    zero = metric(1); zero["by_source"]["5"] = .99
    assert not gate(zero=zero)["code_beneficial_each_source"]


@pytest.mark.parametrize("role", ["bank", "query"])
def test_90percent_live_denominator_excludes_first_zero_up_update(role):
    assert gate(**{role + "_updates": 460})["pass"]
    assert not gate(**{role + "_updates": 459})["pass"]
    with pytest.raises(ValueError, match="updates2..512"):
        gate(**{role + "_updates": 512})


@pytest.mark.parametrize("bad", [0., float("nan"), float("inf")])
def test_all_six_particle_code_owners_must_be_finite_and_nonzero(bad):
    norms = {s: .1 for s in api.SITES}; norms[api.SITES[-1]] = bad
    assert not gate(C_norms=norms)["pass"]


def test_invalid_source_or_nonfinite_endpoint_fails_explicitly():
    with pytest.raises(ValueError): api.accuracy(torch.zeros(6, 2, 3), [0, 1, 2, 3, 4, 4])
    with pytest.raises(ValueError): api.accuracy(torch.full((6, 2, 3), float("nan")), list(range(6)))
    missing = metric(.99); missing["by_source"].pop("5")
    with pytest.raises(ValueError): gate(particle=missing)
    invalid = metric(.99); invalid["rmse"] = float("inf")
    with pytest.raises(ValueError): gate(particle=invalid)


def test_reduced_public_initialization_checkpoint_and_teacher_pairing(tiny_data):
    before = api.digest(torch.get_rng_state())
    assert api.preflight(tiny_data)["native_updates"] == 0
    assert api.digest(torch.get_rng_state()) == before
    loop = api.make_loop(api.ARMS[1], tiny_data)
    assert isinstance(loop.policy.table, torch.nn.Parameter)
    assert len(loop.policy.table) == 128 and loop.policy.table.shape[1] == 4
    # Same latent/time inputs, six different positive-caption targets.
    source = tiny_data["test"]["context"][0:1].repeat(6, 1, 1)
    source[:, :, api.SMALL.output] = torch.arange(6)[:, None]
    inputs = loop.policy.G.inputs(source, teacher=True)
    assert torch.equal(inputs[2][:6], loop.policy.G.captions[7:13])
    assert torch.equal(inputs[2][6:], loop.policy.G.captions[0:1].expand(6, -1, -1))
    with torch.no_grad(): teacher = loop.policy.G.teacher(source)
    assert len(torch.unique(teacher.flatten(1), dim=0)) == 6
    assert all(b.route is None for b in loop.policy.G.branches())


def test_both_arms_bf16_down_up_match_zero_code_first_gradient(tiny_data):
    ordinary = api.make_loop(api.ARMS[0], tiny_data)
    zero = api.make_loop(api.ARMS[1], tiny_data, software_C_zero=True)
    x = tiny_data["test"]["context"][:4]
    condition = ordinary.policy.encoder.condition(x)
    panel = .125 * torch.randn(4, api.SMALL.tokens, api.SMALL.output, generator=torch.Generator().manual_seed(72))
    gradients = []
    for loop in (ordinary, zero):
        p = loop.policy
        residual = p.G(x) if loop.arm == api.ARMS[0] else p.routed_generate(x, sigma=0, perturb=False)
        with torch.no_grad(): real = p.D(panel, condition)
        loss = p.recipe.make_loss().g_loss(p.D(panel + residual / p.D.scale, condition), real)
        gradients.append(torch.autograd.grad(loss, [b.up.weight for b in p.G.branches()]))
    assert all(torch.equal(a, b) for a, b in zip(*gradients))


def test_four_native_software_updates_replay_through_public_api(tiny_data, monkeypatch):
    """Two original updates plus exactly two replay updates; no quality claim."""
    loop = api.make_loop(api.ARMS[1], tiny_data); initial = api.checkpoint(loop)
    rows = [api.update(loop) for _ in range(2)]; final = api.checkpoint(loop)
    assert rows[1]["bank_live"] and rows[1]["query_live"]
    replay = api.make_loop(api.ARMS[1], tiny_data)
    api.restore(replay, initial)
    repeated = [api.update(replay) for _ in range(2)]
    assert api.digest(rows) == api.digest(repeated)
    assert api.digest(api.checkpoint(replay)) == api.digest(final)
    monkeypatch.setattr(api.init, "initialize_", lambda *a, **k: pytest.fail("trained restore reinitialized owners"))
    api.restore(replay, final)
    assert api.digest(api.checkpoint(replay)) == api.digest(final)
    api.learned_finite(replay.policy)
    before = api.digest(api.checkpoint(replay))
    clean = api.capture(replay.policy, tiny_data["test"]["context"][:4])
    code_zero = api.capture(replay.policy, tiny_data["test"]["context"][:4], zero_code=True)
    assert torch.isfinite(clean).all() and torch.isfinite(code_zero).all()
    assert api.digest(api.checkpoint(replay)) == before


def test_existing_output_refusal_preserves_receipt_and_does_not_initialize_cuda(tmp_path, monkeypatch):
    out = tmp_path / "retained"; out.mkdir(); receipt = out / "completion.json"; receipt.write_text("retained bytes\n")
    before = receipt.read_bytes()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(api, "global_rng", lambda *a: pytest.fail("refusal touched CUDA state"))
    monkeypatch.setattr(sys, "argv", ["accuracy", "--run", "--out", str(out)])
    assert api.main() == 2
    assert receipt.read_bytes() == before


def test_fixed_protocol_is_reusable_and_has_no_native_version_allowlist():
    card = json.loads(api.CARD.read_text())
    assert card["arms"] == list(api.ARMS) and card["steps"] == 512 and card["seconds"] == 300
    assert card["media_steps"] == list(api.MEDIA_STEPS) and card["media_indices"] == list(api.MEDIA_INDICES)
    assert card["prior"]["kind"] == "particle_cloud" and card["prior"]["sigma"] == 0
    assert "native_python_sha256" not in card and "execution_authorized" not in card
    assert api.accuracy(torch.zeros(6, 2, 3), torch.arange(6))["rmse"] == 0


def test_goal_maps_reduce_all_physical_coordinates_and_preserve_source_order():
    from examples import render_e22_routed_caption_accuracy as media
    residual = torch.tensor([3., 4.], dtype=torch.float64).expand(6, 4, 2).clone()
    residual *= torch.arange(1, 7, dtype=torch.float64)[:, None, None]
    maps = media.token_maps(residual)
    assert maps.shape == (6, 2, 2)
    for source in range(6):
        assert maps[source] == pytest.approx(math.sqrt(12.5) * (source + 1), rel=1e-14)
    with pytest.raises(ValueError): media.token_maps(torch.full((6, 4, 2), float("inf")))


def test_goal_frame_uses_same_initial_color_scale_for_target_and_both_arms():
    from examples import render_e22_routed_caption_accuracy as media
    zero = torch.zeros(6, 4, 2)
    frame = media.goal_frame(zero, torch.ones_like(zero), torch.full_like(zero, .5),
        step=512, vmax=1., terminal={"scientific_status": "FAIL", "ordinary": 1., "particle": .5})
    # Centers of source0 tiles; a single color scale covers all rows/steps.
    assert frame.getpixel((242, 171)) == (255, 255, 255)
    assert frame.getpixel((242, 345)) == (255, 0, 0)
    assert frame.getpixel((242, 519)) == (255, 127, 127)
    assert frame.size == (1180, 680)


@pytest.mark.parametrize("existing", ["goal.gif", "goal-final.png", "media-completion.json"])
def test_media_refusal_preserves_prior_outputs_without_touching_models(tmp_path, monkeypatch, existing):
    from examples import render_e22_routed_caption_accuracy as media
    run = tmp_path / "retained-media"; run.mkdir()
    path = run / existing; path.write_bytes(b"previous observation bytes\n")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    monkeypatch.setattr(sys, "argv", ["render", "--run-directory", str(run)])
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("refusal read model/observation tensors"))
    assert media.main() == 2
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
