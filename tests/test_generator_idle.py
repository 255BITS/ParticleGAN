"""CPU checks for the opt-in generator idle rule and constant-LR overlay."""
import hashlib
import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from particlegan import GANTrainer, get_recipe, learning_rate_scale, learning_rate_scales
from particlegan.generator_idle import generator_is_idle, release_generator_step
from benchmarks.legacy.recipe import LegacyRecipe
from benchmarks.toy100.schedule import policy_multipliers


# Retain the old software receipt. Later develop added checkpoint metadata;
# compare all current tensors, gradients, optimizer/RNG state and metadata to
# exact pinned develop bytes instead of replacing this historical digest.
_HISTORICAL_DEVELOP_SHA256 = "75da077247f3f7e69e7ea32dbc16e16a53aabedc45937a20b3aa53ecfbcd7717"
_DEVELOP_SOURCE = "6cdc209a3956aee44a2f42fac4a92bcaddab46a4"


def _walk(value, prefix, chunks):
    if isinstance(value, torch.Tensor):
        chunks.append(prefix.encode() + value.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            _walk(value[key], prefix + "/" + str(key), chunks)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, prefix + f"[{index}]", chunks)
    else:
        chunks.append(f"{prefix}={value!r}".encode())


def _fingerprint(trainer):
    chunks = []
    _walk(trainer.state_dict(), "state", chunks)
    for name, module in ("G", trainer.G), ("D", trainer.D), ("prior", trainer.prior):
        for pname, parameter in module.named_parameters():
            grad = parameter.grad
            payload = b"" if grad is None else grad.detach().cpu().contiguous().numpy().tobytes()
            chunks.append(f"grad/{name}/{pname}".encode() + payload)
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def _short_run():
    torch.manual_seed(0)
    recipe = get_recipe(num_particles=8, z_dim=2, batch_size=4, total_steps=4)
    generator = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    trainer = GANTrainer(recipe, generator, critic)
    real = torch.randn(4, 2)
    for _ in range(3):
        trainer.step(real)
    return trainer


def _develop_fingerprint(tmp_path):
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", _DEVELOP_SOURCE, "particlegan"],
        cwd=root, text=True).splitlines()
    for name in paths:
        if name.endswith(".py"):
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(subprocess.check_output(
                ["git", "show", f"{_DEVELOP_SOURCE}:{name}"], cwd=root))
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    fixtures = "\n\n".join(ast.get_source_segment(source, node) for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_walk", "_fingerprint", "_short_run"})
    script = ("import hashlib, torch\nfrom torch import nn\n"
              "from particlegan import GANTrainer, get_recipe\n" + fixtures
              + "\nprint(_fingerprint(_short_run()))\n")
    program = tmp_path / "control.py"
    program.write_text(script)
    environment = {**os.environ, "PYTHONPATH": str(tmp_path), "CUDA_VISIBLE_DEVICES": "",
                   "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.check_output([sys.executable, str(program)], cwd=tmp_path,
                                   env=environment, text=True).strip()


def test_disabled_path_matches_develop_and_does_not_score_the_gap(monkeypatch, tmp_path):
    def boom(*_args, **_kwargs):
        raise AssertionError("idle statistic evaluated while the rule is off")

    monkeypatch.setattr("particlegan.generator_idle.generator_is_idle", boom)
    first = _short_run()
    second = _short_run()
    assert _fingerprint(first) == _develop_fingerprint(tmp_path)
    assert _fingerprint(second) == _fingerprint(first)
    assert "generator_idle_se" not in first.state_dict()["recipe"]
    assert "constant_lr" not in first.state_dict()["recipe"]


def test_gap_rule():
    matched = torch.zeros(4, 1)
    assert generator_is_idle(matched, matched.clone(), 1.0)
    assert generator_is_idle(torch.zeros(4), torch.zeros(4), 0.0)
    separated = torch.full((4, 1), 3.0)
    assert not generator_is_idle(separated, torch.zeros(4, 1), 1.0)
    assert not generator_is_idle(torch.zeros(1), torch.zeros(1), 1.0)
    assert not generator_is_idle(torch.tensor([float("nan"), 0.0]), torch.zeros(2), 1.0)
    noisy = torch.tensor([0.1, -0.2, 0.05, 0.0])
    assert generator_is_idle(noisy, torch.zeros(4), 1.0)
    assert not generator_is_idle(noisy + 10, torch.zeros(4), 1.0)


class _Axis(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x[:, :1] + self.bias


def _axis_trainer(real, idle, mode="batch"):
    torch.manual_seed(0)
    generator = nn.Linear(2, 2)
    critic = _Axis()
    recipe = get_recipe(num_particles=4, z_dim=2, batch_size=4, total_steps=2,
                        generator_idle_se=idle, generator_idle_mode=mode)
    trainer = GANTrainer(recipe, generator, critic)
    with torch.no_grad():
        generator.weight.zero_()
        generator.bias.zero_()
    before = [parameter.detach().clone() for parameter in generator.parameters()]
    trainer.step(real)
    return trainer, before


def _moved(trainer, before):
    return any(not torch.equal(parameter.detach(), saved)
               for parameter, saved in zip(trainer.G.parameters(), before))


def test_idle_skips_generator_step_and_resolved_gap_still_steps():
    zeros = torch.zeros(4, 2)
    idle_trainer, idle_before = _axis_trainer(zeros, 1.0)
    live_trainer, live_before = _axis_trainer(zeros, None)
    assert not _moved(idle_trainer, idle_before)
    assert _moved(live_trainer, live_before)
    assert len(idle_trainer.opt_d.state) == 1
    assert idle_trainer.opt_g.state == {}

    separated = torch.zeros(4, 2)
    separated[:, 0] = 10
    stepping, before = _axis_trainer(separated, 1.0)
    assert _moved(stepping, before)


def test_release_is_a_noop_until_the_scope_is_set():
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.ones(2)
    optimizer = torch.optim.Adam([parameter], lr=0.1, betas=(0.0, 0.999))
    assert release_generator_step(optimizer, torch.zeros(4), torch.zeros(4)) is False
    assert torch.equal(parameter.grad, torch.ones(2))
    from particlegan.generator_idle import generator_idle_scope
    with generator_idle_scope(1.0):
        assert release_generator_step(optimizer, torch.zeros(4), torch.zeros(4)) is True
    assert parameter.grad is None
    parameter.grad = torch.ones(2)
    with generator_idle_scope(1.0):
        assert release_generator_step(optimizer, torch.full((4,), 5.0), torch.zeros(4)) is False
    assert torch.equal(parameter.grad, torch.ones(2))


def test_constant_lr_overlay_is_default_off_and_forces_unit_multipliers():
    base = get_recipe(total_steps=100, network_lr_horizon_cap=40, network_lr_floor=0.01, lr_floor=0.05)
    network, prior = learning_rate_scales(90, base)
    assert network == learning_rate_scale(90, 40, base.lr_anneal_start, 0.01)
    assert prior < 1.0
    assert learning_rate_scales(90, base.replace(constant_lr=True)) == (1.0, 1.0)
    assert policy_multipliers(90, 100, base.lr_anneal_start, base.lr_floor, 40,
                              network_lr_floor=0.01) == (network, prior)
    assert policy_multipliers(90, 100, base.lr_anneal_start, base.lr_floor, 40,
                              network_lr_floor=0.01, constant_lr=True) == (1.0, 1.0)
    assert "constant_lr" not in get_recipe().to_dict()
    assert "generator_idle_se" not in get_recipe().to_dict()
    assert "generator_idle_mode" not in get_recipe().to_dict()
    assert get_recipe().replace(constant_lr=True).to_dict()["constant_lr"] is True
    assert get_recipe().replace(generator_idle_se=1).to_dict()["generator_idle_se"] == 1.0
    packet = LegacyRecipe().to_dict()
    assert "constant_lr" not in packet and "generator_idle_se" not in packet
    assert LegacyRecipe(constant_lr=True, generator_idle_se=1.0).to_dict()["generator_idle_se"] == 1.0
    import json
    from pathlib import Path
    from benchmarks.transfer_suite.toy100_compatibility import declared_recipe
    winner = json.loads(Path("configs/toy100/constraints_simple_regularization.json").read_text())
    plain, _, _ = declared_recipe(winner)
    assert plain.constant_lr is False and plain.generator_idle_se is None
    assert plain.generator_idle_mode == "batch"
    overlay = {**winner, "constant_lr": True, "generator_idle_se": 1,
               "generator_idle_mode": "ema"}
    armed, _, _ = declared_recipe(overlay)
    assert armed.constant_lr is True and armed.generator_idle_se == 1.0
    assert armed.generator_idle_mode == "ema"


def _marked_run(standard_errors, mode="batch"):
    torch.manual_seed(0)
    recipe = get_recipe(num_particles=8, z_dim=2, batch_size=4, total_steps=4,
                        generator_idle_se=standard_errors, generator_idle_mode=mode)
    generator = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    critic = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    trainer = GANTrainer(recipe, generator, critic)
    real = torch.randn(4, 2)
    for _ in range(3):
        trainer.step(real)
    return trainer


def test_batch_mode_is_byte_identical_when_ema_is_off():
    assert _fingerprint(_marked_run(1.0)) == _fingerprint(_marked_run(1.0, "batch"))
    assert "generator_idle_mode" not in get_recipe().to_dict()
    assert get_recipe().replace(generator_idle_mode="ema").to_dict()["generator_idle_mode"] == "ema"
    report = _marked_run(1.0)._idle_accounting.public()
    assert report["mode"] == "batch"
    assert report["decisions"] == 3
    assert report["idle_count"] <= report["decisions"]


def test_ema_idles_on_a_matched_constant_gap_and_steps_on_a_resolved_gap():
    zeros = torch.zeros(4, 2)
    idle_trainer, idle_before = _axis_trainer(zeros, 1.0, "ema")
    assert not _moved(idle_trainer, idle_before)
    idle = idle_trainer._idle_accounting.public()
    assert idle["mode"] == "ema"
    assert idle["idle_count"] == 1 and idle["decisions"] == 1
    assert idle["idle_fraction"] == 1.0

    separated = torch.zeros(4, 2)
    separated[:, 0] = 10
    stepping, before = _axis_trainer(separated, 1.0, "ema")
    assert _moved(stepping, before)
    resolved = stepping._idle_accounting.public()
    assert resolved["idle_count"] == 0 and resolved["decisions"] == 1
    assert resolved["idle_fraction"] == 0.0


def test_ema_first_step_matches_the_batch_rule_and_ignores_a_nonfinite_gap():
    from particlegan.generator_idle import IdleAccounting
    noisy = torch.tensor([0.1, -0.2, 0.05, 0.0])
    zeros = torch.zeros(4)
    batch = generator_is_idle(noisy, zeros, 1.0)
    accounting = IdleAccounting("ema", 1.0)
    assert accounting.decide(noisy, zeros, 1.0, "ema") is batch
    poisoned = IdleAccounting("ema", 1.0)
    assert poisoned.decide(torch.tensor([float("nan"), 0.0]), zeros[:2], 1.0, "ema") is False
    assert poisoned._mean is None
    assert poisoned.decide(zeros, zeros, 1.0, "ema") is True


def test_suite_summary_records_idle_decisions(tmp_path):
    from benchmarks.toy100.train import train
    summary = train(dict(
        model="gan", problem="grid100", device="cpu", seed=31, steps=2,
        z_dim=2, num_particles=8, batch_size=4, d_hidden=4, n_hidden=1,
        fourier=0, eval_samples=16, snapshot_samples=4, eval_interval=2,
        snapshot_interval=2, log_interval=2, threads=1,
        output_noise_std=0.0, input_noise_std=0.0,
        generator_idle_se=1.0, generator_idle_mode="ema",
    ), tmp_path)
    idle = summary["generator_idle"]
    assert idle["mode"] == "ema" and idle["standard_errors"] == 1.0
    assert idle["decisions"] == 2
    assert idle["idle_count"] <= idle["decisions"]
    text = (tmp_path / "progress.log").read_text()
    assert f"idle={idle['idle_count']}/{idle['decisions']}" in text


def test_constant_lr_and_idle_threshold_reject_bad_values():
    with pytest.raises(ValueError):
        get_recipe(constant_lr=1)
    with pytest.raises(ValueError):
        get_recipe(generator_idle_se=-1)
    with pytest.raises(ValueError):
        get_recipe(generator_idle_mode="smooth")


@pytest.mark.parametrize("mode", ["batch", "ema"])
def test_active_checkpoint_repeats_the_next_update_and_complete_state(mode):
    first = _marked_run(1.0, mode)
    state = first.state_dict()
    assert state["generator_idle"]["decisions"] == state["completed_steps"] == 3
    resumed = _marked_run(1.0, mode)
    resumed.load_state_dict(state)
    real = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    first.step(real)
    expected = _fingerprint(first)
    resumed.load_state_dict(state)
    resumed.step(real)
    assert _fingerprint(resumed) == expected
    assert resumed._idle_accounting.state_dict() == first._idle_accounting.state_dict()


@pytest.mark.parametrize("field,value", [
    ("decisions", 2), ("decisions", True), ("idle_count", 4),
    ("standard_errors", True), ("standard_errors", 2.0), ("ema_alpha", .5),
    ("mean", float("nan")), ("mean", True), ("variance", -1.),
    ("variance", None), ("weight_sq", 0.), ("weight_sq", True),
])
def test_corrupt_idle_checkpoint_rejects_before_models_optimizers_or_rng_change(field, value):
    trainer = _marked_run(1.0, "ema")
    state = trainer.state_dict()
    state["generator_idle"][field] = value
    before = _fingerprint(trainer)
    with pytest.raises(ValueError, match="generator idle"):
        trainer.load_state_dict(state)
    assert _fingerprint(trainer) == before


def test_active_historical_checkpoint_cannot_silently_restart_the_ema():
    trainer = _marked_run(1.0, "ema")
    state = trainer.state_dict()
    del state["generator_idle"]
    before = _fingerprint(trainer)
    with pytest.raises(ValueError, match="checkpoint schema"):
        trainer.load_state_dict(state)
    assert _fingerprint(trainer) == before


def test_inactive_checkpoint_keeps_its_old_schema_and_rejects_active_filter_state():
    trainer = _short_run()
    state = trainer.state_dict()
    assert "generator_idle" not in state
    trainer.load_state_dict(state)
    state["generator_idle"] = _marked_run(1.0, "ema")._idle_accounting.state_dict()
    with pytest.raises(ValueError, match="checkpoint schema"):
        trainer.load_state_dict(state)


def test_each_legacy_run_gets_an_independent_filter_and_restores_its_outer_scope():
    from particlegan.generator_idle import generator_idle_scope, generator_idle_accounting
    zero = torch.zeros(4)
    with generator_idle_scope(1., "ema") as outer:
        release_generator_step(torch.optim.Adam([nn.Parameter(torch.ones(()))]), zero, zero)
        with generator_idle_scope(1., "ema") as inner:
            assert inner.decisions == 0 and inner._mean is None
            release_generator_step(torch.optim.Adam([nn.Parameter(torch.ones(()))]), zero, zero)
        assert generator_idle_accounting() is outer
        assert outer.decisions == inner.decisions == 1
    assert generator_idle_accounting() is None


@pytest.mark.parametrize("host", ["two_pole", "trajectory", "residual_student", "unipolar",
                                 "ae_gan_hold", "cover_leftover", "unused_token_hold", "mid_scale_identity"])
def test_forge_behavior_hosts_consume_and_record_the_active_recipe(tmp_path, host):
    from copy import deepcopy
    from experiments.forge.behavior_adapters import run_behavior
    from experiments.forge.views import load_tasks, grade_result
    root = Path(__file__).resolve().parents[1]
    original = load_tasks(root)[host]
    fixture = deepcopy(original)
    fixture["execution"]["steps"] = 2  # Software fixture, never a shortened scientific result.
    candidate = {"recipe_overrides": {"generator_idle_se": 1., "generator_idle_mode": "ema",
                                      "input_noise_std": 0., "output_noise_std": 0.}}
    result = run_behavior({"protocol": {"seed": 0}, "candidate": candidate}, fixture, tmp_path)
    state = result["applied"]["generator_idle"]
    assert state["mode"] == "ema" and state["standard_errors"] == 1.
    assert state["decisions"] == 2 and state["mean"] is not None
    assert 0 <= state["idle_count"] <= 2
    assert result["applied"]["recipe"]["generator_idle_mode"] == "ema"
    # A shortened software fixture can also fail an ordinary quality gate.
    # It must never count as a passing full-protocol scientific result.
    assert grade_result(original, result)["status"] in ("INCOMPLETE", "FAIL")


def test_joint_word_admission_has_an_explicit_idle_consumer():
    from experiments.forge.views import load_tasks
    from experiments.forge.word_adapter import word_preflight
    task = load_tasks(Path(__file__).resolve().parents[1])["five_word_joint_smoke"]
    blockers = word_preflight(task, {"recipe_preset": "bcap", "recipe_overrides": {
        "generator_idle_se": 1., "generator_idle_mode": "ema"}})
    assert blockers == []


def _word_fixture(standard_errors, *, noise=False, fixture_class=None):
    from benchmarks.toy_audit.api_images import WordFixture
    from experiments.forge.views import load_tasks
    from experiments.forge.word_adapter import word_context
    root = Path(__file__).resolve().parents[1]
    task = load_tasks(root)["five_word_joint_smoke"]
    overrides = {} if noise else {"input_noise_std": 0., "output_noise_std": 0.}
    if standard_errors is not None:
        overrides.update(generator_idle_se=standard_errors, generator_idle_mode="ema")
    context = word_context({"candidate": {"recipe_preset": "bcap", "recipe_overrides": overrides},
                            "protocol": {"seed": 0}}, task, "cpu")
    return (fixture_class or WordFixture)(device="cpu", seed=0, recipe_name=None,
                                          max_steps=4, components=context)


def test_joint_word_idle_checkpoint_repeats_the_next_update_without_new_draws():
    from experiments.forge.state import state_digest
    first = _word_fixture(1e6)
    first.step()
    first.step()
    saved = first.state_dict()
    assert saved["generator_idle"]["decisions"] == 2
    assert saved["generator_idle"]["idle_count"] == 2
    first.step()
    expected = state_digest(first.state_dict())
    resumed = _word_fixture(1e6)
    resumed.restore_generator_idle(saved["generator_idle"], completed_steps=2, validate_only=True)
    resumed.policy.load_state_dict(saved["api_state"])
    resumed.recipe.restore_prior_regularizer(resumed.spread, saved.get("prior_regularizer"))
    resumed.restore_generator_idle(saved["generator_idle"])
    resumed.data_generator.set_state(saved["data_generator"])
    resumed.step()
    assert state_digest(resumed.state_dict()) == expected


def test_joint_word_rejects_bad_or_missing_idle_state_before_mutation():
    from experiments.forge.state import state_digest
    fixture = _word_fixture(1e6)
    fixture.step()
    state = fixture.state_dict()["generator_idle"]
    before = state_digest(fixture.state_dict())
    with pytest.raises(ValueError, match="generator idle"):
        fixture.restore_generator_idle(None, validate_only=True)
    state["decisions"] = 2
    with pytest.raises(ValueError, match="decisions differ"):
        fixture.restore_generator_idle(state, validate_only=True)
    assert state_digest(fixture.state_dict()) == before


@pytest.mark.parametrize("noise", [False, True])
def test_disabled_joint_word_updates_match_exact_develop_fixture_and_rng(tmp_path, noise):
    import importlib.util
    from experiments.forge.state import state_digest
    root = Path(__file__).resolve().parents[1]
    path = tmp_path / "original_word.py"
    path.write_bytes(subprocess.check_output(["git", "show",
        _DEVELOP_SOURCE + ":benchmarks/toy_audit/api_images.py"], cwd=root))
    name = "benchmarks.toy_audit._pr399_original_word"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        torch.manual_seed(0)
        current = _word_fixture(None, noise=noise)
        for _ in range(2):
            current.step()
        expected = state_digest(current.state_dict())
        torch.manual_seed(0)
        original = _word_fixture(None, noise=noise, fixture_class=module.WordFixture)
        for _ in range(2):
            original.step()
        assert state_digest(original.state_dict()) == expected
    finally:
        sys.modules.pop(name, None)
    with pytest.raises(ValueError):
        get_recipe(constant_lr=True, lr_schedule="constant", network_lr_horizon_cap=None)
    with pytest.raises(ValueError):
        get_recipe(constant_lr=True, continuous_policy="dv7")
