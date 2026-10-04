"""Four public-only tiny plumbing contracts; no production fixture/campaign."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
SPEC = importlib.util.spec_from_file_location(
    "remote_conditioning_public_contract", EXAMPLES / "routed_remote_conditioning.py"
)
toy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = toy
SPEC.loader.exec_module(toy)


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_initialized()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def tiny_data():
    """Handcrafted unit labels, not production teacher seed4 qualification."""
    data = {}
    content = torch.linspace(-0.2, 0.2, 256).view(1, 16, 16)
    for name, offset in (("fit", 0.0), ("guard", 0.03), ("population", -0.03)):
        context = toy.paired_contexts(content + offset, torch.tensor([0.4]))
        target = 0.1 * context[:, :2]
        data[name + "_context"], data[name + "_mean"] = context, target
        if name != "population":
            count = 128 if name == "fit" else 4
            data[name + "_rows"] = context.repeat(count, 1, 1, 1)
            data[name + "_target"] = target.repeat(count, 1, 1, 1)
    mean, std = toy.fit_channel_calibration(data["fit_target"])
    data["target_mean"], data["target_std"] = mean, std
    for name in ("fit", "guard", "population"):
        data[name + "_mean"] = (data[name + "_mean"] - mean) / std
        if name != "population":
            data[name + "_target"] = (data[name + "_target"] - mean) / std
    data.update(
        teacher_nonlocal_variance=1e-5, teacher_hash="unit_handcrafted_not_production"
    )
    data["fixture_hash"] = toy.reference.digest_tensors(
        {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
    )
    return data


def test_paired_local_views_and_prescribed_public_teacher_readout(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "unit contract cannot run production fixture/teacher qualification"
        )

    monkeypatch.setattr(toy, "fixture", forbidden)
    base = torch.linspace(-0.2, 0.2, 256).view(1, 16, 16)
    context = toy.paired_contexts(base, torch.tensor([0.4]))
    assert torch.equal(context[0, 0], context[1, 0])
    assert torch.equal(context[0, 2], context[1, 2])
    assert torch.equal(context[0, :, 8:, 8:], context[1, :, 8:, 8:])
    assert context[0, 1, :2, :2].eq(0.2).all() and context[1, 1, :2, :2].eq(-0.2).all()
    E = toy.TeacherEncoder()
    declared = deepcopy(E.state_dict())
    toy.init.deterministic_orthogonal_(E, seed=5)
    assert all(
        torch.equal(value, E.state_dict()[name]) for name, value in declared.items()
    )
    with torch.no_grad():
        query = E(context)
    assert (
        query[0, 0] > 0
        and query[1, 0] < 0
        and query[:, 2].eq(1).all()
        and query[:, 3].eq(-1).all()
    )
    fit = torch.tensor([[[[1.0, 3.0]], [[5.0, 5.0]]], [[[5.0, 7.0]], [[5.0, 5.0]]]])
    mean, std = toy.fit_channel_calibration(fit)
    assert torch.equal(mean.flatten(), torch.tensor([4.0, 5.0]))
    assert torch.allclose(std.flatten(), torch.tensor([5.0**0.5, 1e-4]))
    # Frozen pointwise host plus four3x3 layers has radius4. Lower-rightmask
    # starts12, so its local input support starts8, away from marker0:2.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        G = toy.reference.Generator()
    toy.init.deterministic_orthogonal_(G, seed=0)
    layers = [G.input, G.blocks[0].first, G.blocks[0].second, G.output]
    assert sum((layer.kernel_size[0] - 1) // 2 for layer in layers) == 4
    with torch.no_grad():
        codeblind = G(context, torch.zeros(2, 4))
    assert torch.equal(codeblind[0, :, 12:16, 12:16], codeblind[1, :, 12:16, 12:16])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        normalized = toy.NormalizedGenerator(mean, std)
    toy.init.deterministic_orthogonal_(normalized, seed=0)
    assert all(
        torch.equal(value, normalized.state_dict()[name])
        for name, value in G.state_dict().items()
    )
    with torch.no_grad():
        assert torch.equal(
            normalized(context, torch.zeros(2, 4)), (codeblind - mean) / std
        )


def test_matched_callers_and_one_full_public_native_update_are_finite(monkeypatch):
    data = tiny_data()
    a, b = toy.make_loop(16, data), toy.make_loop(64, data)
    assert a.initial_hashes == b.initial_hashes
    for module in (a.policy.G, a.policy.ema_G):
        assert torch.equal(module.target_mean, data["target_mean"])
        assert torch.equal(module.target_std, data["target_std"])
        assert module.target_mean.data_ptr() != data["target_mean"].data_ptr()
    assert a.policy.G.target_std.data_ptr() != a.policy.ema_G.target_std.data_ptr()
    first, second = toy.draw_panel(a), toy.draw_panel(b)
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert torch.equal(first["g_indices"][:16], first["d_indices"])
    assert first["g_gaussian"].shape == (64, 2, 16, 16)
    seen = []
    original = a.policy.begin_step

    def begin(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(float(a.policy.opt_g.param_groups[0]["lr"]))
        return result

    monkeypatch.setattr(a.policy, "begin_step", begin)
    row = toy.update(a)
    assert a.policy.completed_steps == row["step"] == a.health["finite_steps"] == 1
    assert row["dense_rows"] == 128 and a.health["ka2_applied_calls"] == 1
    assert a.policy.opt_g.param_groups[0]["lr"] == 0.25 * seen[0]
    assert a.policy.opt_d.ema_critic is not a.policy.D
    assert a.policy.opt_g.latent_damping.table is a.policy.table
    assert a.policy.table is a.policy.opt_g.param_groups[2]["params"][0]
    toy.native_health(a.policy)
    parameter = a.policy.opt_g.param_groups[0]["params"][0]
    a.policy.opt_g.state[parameter]["exp_avg_sq"].view(-1)[0] = float("inf")
    with pytest.raises(FloatingPointError, match="optimizer"):
        toy.native_health(a.policy)


def test_complete_public_checkpoint_resume_and_missing_caller_rejection():
    data = tiny_data()
    source = toy.make_loop(64, data)
    toy.update(source)
    saved = deepcopy(toy.checkpoint(source))
    clone = toy.make_loop(64, data)
    toy.restore(clone, saved)
    assert clone.policy.completed_steps == 1
    assert clone.policy.table.data_ptr() != source.policy.table.data_ptr()
    assert clone.policy.opt_g.latent_damping.table is clone.policy.table
    assert toy.reference.digest_tensors(
        toy.draw_panel(source)
    ) == toy.reference.digest_tensors(toy.draw_panel(clone))
    assert toy.reference.caller_stream_hashes(
        source
    ) == toy.reference.caller_stream_hashes(clone)
    for kind in ("caller", "fixture", "clock", "calibration"):
        invalid = deepcopy(saved)
        if kind == "caller":
            del invalid["base"]["caller_streams"]["g_gaussian"]
        elif kind == "fixture":
            invalid["fixture_hash"] = "different-fixture"
        elif kind == "clock":
            invalid["base"]["health"]["finite_steps"] = 2
        else:
            invalid["calibration"]["std"].mul_(2)
        with pytest.raises(ValueError):
            toy.restore(clone, invalid)


def test_fixed_two_endpoint_and_codeblind_gates_refuse_partial_or_invalid_quality():
    arms = {
        name: {
            "initial": {"live_mask_error": 1.0},
            "evaluations": {
                "100": {"live_mask_error": 0.1 if name == "G16" else 0.08},
                "200": {"live_mask_error": 0.03 if name == "G16" else 0.02},
            },
        }
        for name in ("G16", "G64")
    }
    checks = toy.quality_gates(arms, 0.1)
    assert all(checks.values())
    wrong = deepcopy(arms)
    wrong["G64"]["evaluations"]["100"]["live_mask_error"] = 0.1
    assert not toy.quality_gates(wrong, 0.1)["G64_better_at100"]
    for value in (float("nan"), float("inf"), -0.01):
        invalid = deepcopy(arms)
        invalid["G64"]["evaluations"]["200"]["live_mask_error"] = value
        with pytest.raises(ValueError):
            toy.quality_gates(invalid, 0.1)
    assert not toy.campaign_passes(
        {
            "checks": checks,
            "committed_updates": {"G16": 200, "G64": 200},
            "failure": None,
        }
    )
    assert not toy.campaign_passes(
        {"checks": {"teacher_qualified": True}, "failure": "source mismatch"}
    )
