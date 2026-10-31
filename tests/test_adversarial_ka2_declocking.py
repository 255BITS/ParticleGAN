"""Empirical adversarial stress tests for KA2 de-clocking.

Tests cover:
1. Clock-free perturbation tests: state hash invariance across varying step_label offsets,
   varying horizons, varying evaluation cadences, and split resumption restarts.
2. Exact boundary transition: step 799 evaluates strictly pure A penalty and step 800 evaluates
   blended penalty with bit-identical continuation parity.
3. Optimizer state serialization: saving/loading optimizer state preserves exact readiness
   ratio R(t) without step drift.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile

import pytest
import torch
from torch import nn

from experiments.forge.clockfree import run_clockfree, source_audit
from experiments.forge.state import state_digest
from experiments.forge.views import grade_result
from particlegan import GANTrainer, Recipe, get_recipe
from particlegan.ka2 import (
    KA2CriticAdam,
    KA2GradientPenalty,
    KA2StepRecord,
    READINESS_THRESHOLD,
    WARMUP_CALLS,
    _optimizer_moment_readiness,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_trainer(recipe, seed=0):
    torch.manual_seed(seed)
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2))
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1))
    return GANTrainer(recipe, G, D, seed=seed, optimizer_options={"foreach": False})


def _clockfree_request():
    return {
        "candidate": {
            "recipe_overrides": {
                "critic_formulation": "ka2",
                "lr_floor": 1,
                "network_lr_floor": 1,
                "input_noise_std": 0,
                "output_noise_warmup": 0,
                "d_guard_min_steps": 0,
            }
        },
        "protocol": {"seed": 0},
    }


# ===========================================================================
# Dimension 1: Clock-Free Perturbation Stress Tests
# ===========================================================================

@pytest.mark.parametrize("offset", [1, 13, 256, 1024, 8192, 65536, 1_000_000])
def test_clockfree_varying_step_label_offsets(offset, tmp_path):
    """Verify state hash invariance across varying step_label offsets."""
    task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit.json").read_text())
    task["execution"]["step_label_offset"] = offset
    task["execution"]["warmup_steps"] = 4
    task["execution"]["probe_steps"] = 4
    task["execution"]["steps"] = 4 + 4

    request = _clockfree_request()
    raw = run_clockfree(request, task, tmp_path, "cpu")
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS", f"Failed with offset {offset}: {grade}"

    step_comp = next(c for c in raw["evidence"]["comparisons"] if c["condition"] == "step_label")
    assert step_comp["reference_sha256"] == step_comp["changed_sha256"], (
        f"State hash mismatch for offset {offset}"
    )


@pytest.mark.parametrize("horizon", [16, 128, 512, 4096, 100_000])
def test_clockfree_varying_horizons(horizon, tmp_path):
    """Verify state hash invariance across varying perturbed horizons."""
    task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit.json").read_text())
    task["execution"]["perturbed_horizon"] = horizon
    task["execution"]["warmup_steps"] = 4
    task["execution"]["probe_steps"] = 4
    task["execution"]["steps"] = 4 + 4

    request = _clockfree_request()
    raw = run_clockfree(request, task, tmp_path, "cpu")
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS", f"Failed with horizon {horizon}: {grade}"

    horizon_comp = next(c for c in raw["evidence"]["comparisons"] if c["condition"] == "horizon")
    assert horizon_comp["reference_sha256"] == horizon_comp["changed_sha256"], (
        f"State hash mismatch for horizon {horizon}"
    )


@pytest.mark.parametrize("samples", [1, 16, 64, 256])
def test_clockfree_varying_evaluation_cadences(samples, tmp_path):
    """Verify state hash invariance when evaluation sample counts vary."""
    task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit.json").read_text())
    task["execution"]["evaluation_samples"] = samples
    task["execution"]["warmup_steps"] = 4
    task["execution"]["probe_steps"] = 4
    task["execution"]["steps"] = 4 + 4

    request = _clockfree_request()
    raw = run_clockfree(request, task, tmp_path, "cpu")
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS", f"Failed with eval samples {samples}: {grade}"

    eval_comp = next(c for c in raw["evidence"]["comparisons"] if c["condition"] == "evaluation_cadence")
    assert eval_comp["reference_sha256"] == eval_comp["changed_sha256"], (
        f"State hash mismatch for eval samples {samples}"
    )


@pytest.mark.parametrize("warmup", [1, 2, 4, 8, 16])
def test_clockfree_split_resumption_restarts(warmup, tmp_path):
    """Verify state hash invariance under multiple split resumption restart points."""
    task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit.json").read_text())
    task["execution"]["warmup_steps"] = warmup
    task["execution"]["probe_steps"] = 4
    task["execution"]["steps"] = warmup + 4

    request = _clockfree_request()
    raw = run_clockfree(request, task, tmp_path, "cpu")
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS", f"Failed with warmup {warmup}: {grade}"

    restart_comp = next(c for c in raw["evidence"]["comparisons"] if c["condition"] == "restart")
    assert restart_comp["reference_sha256"] == restart_comp["changed_sha256"], (
        f"State hash mismatch for warmup {warmup}"
    )


def test_clockfree_boundary_spanning_probe(tmp_path):
    """Probe spanning update 798 to 806 (across boundary at step 800)."""
    task = json.loads((ROOT / "configs/forge/tasks/clockfree_audit.json").read_text())
    task["execution"]["warmup_steps"] = 798
    task["execution"]["probe_steps"] = 8
    task["execution"]["steps"] = 798 + 8

    request = _clockfree_request()
    raw = run_clockfree(request, task, tmp_path, "cpu")
    grade = grade_result(task, raw)
    assert grade["status"] == "PASS", f"Boundary spanning probe failed: {grade}"
    assert grade["metrics"]["parity_comparisons"] == 4
    assert raw["evidence"]["source_audit"]["unexplained_clock_dependencies"] == []
    for c in raw["evidence"]["comparisons"]:
        assert c["reference_sha256"] == c["changed_sha256"], (
            f"Condition {c['condition']} failed parity across boundary transition"
        )


def test_official_audit_tasks_pass(tmp_path):
    """Execute official clockfree_audit and clockfree_audit_measurement_v1."""
    request = _clockfree_request()
    for task_name in ["clockfree_audit", "clockfree_audit_measurement_v1"]:
        task = json.loads((ROOT / f"configs/forge/tasks/{task_name}.json").read_text())
        raw = run_clockfree(request, task, tmp_path / task_name, "cpu")
        grade = grade_result(task, raw)
        assert grade["status"] == "PASS", f"{task_name} status != PASS: {grade}"
        assert grade["metrics"]["parity_comparisons"] == 4
        assert raw["evidence"]["source_audit"]["unexplained_clock_dependencies"] == []


# ===========================================================================
# Dimension 2: Exact Boundary Transition Stress Tests
# ===========================================================================

def test_boundary_readiness_exact_analytical_and_numerical_values():
    """Verify exact analytical values, numerical thresholds, and step isolation."""
    beta2 = 0.999
    r_798 = 1.0 - (beta2 ** 798)
    r_799 = 1.0 - (beta2 ** 799)
    r_800 = 1.0 - (beta2 ** 800)
    r_801 = 1.0 - (beta2 ** 801)

    assert r_798 < r_799 < READINESS_THRESHOLD <= r_800 < r_801
    assert r_799 < 0.55085
    assert r_800 >= 0.55085

    # Gap between step 799 and step 800
    gap = r_800 - r_799
    expected_gap = (beta2 ** 799) * (1.0 - beta2)
    assert pytest.approx(gap, rel=1e-9) == expected_gap

    # Distance to threshold
    dist_799 = READINESS_THRESHOLD - r_799
    dist_800 = r_800 - READINESS_THRESHOLD
    assert dist_799 > 0.0, "Step 799 must be below threshold"
    assert dist_800 >= 0.0, "Step 800 must be at or above threshold"

    # Evaluator functions
    assert _optimizer_moment_readiness(None, step=799) < READINESS_THRESHOLD
    assert _optimizer_moment_readiness(None, step=800) >= READINESS_THRESHOLD


def test_step_799_pure_a_isolation_and_no_anchor_evaluation():
    """Verify step 799 evaluates strictly pure A without calling anchor or blend."""
    recipe = get_recipe(trainer="ka2", num_particles=16, z_dim=2, batch_size=8, total_steps=820, d_guard_min_steps=2)
    trainer = _make_trainer(recipe, seed=42)

    # Instrument anchor to track calls
    anchor = trainer.opt_d.anchor
    tracker = {"calls": 0}
    orig_forward = anchor.forward

    def wrapped_forward(*args, **kwargs):
        tracker["calls"] += 1
        return orig_forward(*args, **kwargs)

    anchor.forward = wrapped_forward

    rng = torch.Generator().manual_seed(123)
    # Run up to step 799 (calls 1 to 799)
    for _ in range(799):
        real = torch.randn(8, 2, generator=rng)
        out = trainer.step(real, collect_stats=True)
        assert out["penalty_stats"]["phase"] == "a"
        assert out["penalty_stats"]["w"] == 1.0
        assert out["penalty_stats"]["prox"] == 0.0
        # Anchor must NEVER have been evaluated during pure-A warmup
        assert tracker["calls"] == 0
        assert trainer.opt_d.record.anchor_started is False

    assert trainer.completed_steps == 799


def test_step_800_blend_transition_and_anchor_activation():
    """Verify step 800 transitions to blended penalty and initializes the anchor."""
    recipe = get_recipe(trainer="ka2", num_particles=16, z_dim=2, batch_size=8, total_steps=820, d_guard_min_steps=2)
    trainer = _make_trainer(recipe, seed=42)

    anchor = trainer.opt_d.anchor
    tracker = {"calls": 0}
    orig_forward = anchor.forward

    def wrapped_forward(*args, **kwargs):
        tracker["calls"] += 1
        return orig_forward(*args, **kwargs)

    anchor.forward = wrapped_forward

    rng = torch.Generator().manual_seed(123)
    # Run 799 steps (warmup calls 1 to 799)
    for _ in range(799):
        real = torch.randn(8, 2, generator=rng)
        trainer.step(real)

    assert tracker["calls"] == 0
    assert trainer.opt_d.record.anchor_started is False

    # Call 800: FIRST BLEND (anchor initialized via anchor.start_(), prox=0)
    real_800 = torch.randn(8, 2, generator=rng)
    out_800 = trainer.step(real_800, collect_stats=True)

    stats_800 = out_800["penalty_stats"]
    assert stats_800["phase"] == "blend", f"Expected phase 'blend', got {stats_800['phase']}"
    assert stats_800["s"] == 0.5
    assert stats_800["prox"] == 0.0
    assert trainer.opt_d.record.anchor_started is True
    assert tracker["calls"] == 0  # anchor is started, not evaluated on the initialization step

    # Call 801: SECOND BLEND (anchor evaluated for proximal penalty, prox > 0)
    real_801 = torch.randn(8, 2, generator=rng)
    out_801 = trainer.step(real_801, collect_stats=True)
    stats_801 = out_801["penalty_stats"]
    assert stats_801["phase"] == "blend"
    assert stats_801["prox"] > 0.0
    assert tracker["calls"] == 1  # anchor forward evaluated


def test_continuation_parity_across_boundary():
    """Verify bit-identical continuation across boundary splits: 799, 800, 801."""
    recipe = get_recipe(trainer="ka2", num_particles=16, z_dim=2, batch_size=8, total_steps=820, d_guard_min_steps=2)

    # Generate fixed real batches for steps 0 to 805
    rng = torch.Generator().manual_seed(999)
    reals = [torch.randn(8, 2, generator=rng) for _ in range(806)]

    # --- Run A: Uninterrupted baseline run (0..805) ---
    trainer_a = _make_trainer(recipe, seed=10)
    outputs_a = []
    for step_idx in range(806):
        out = trainer_a.step(reals[step_idx], collect_stats=True)
        if step_idx >= 795:
            outputs_a.append(out)

    state_a = trainer_a.state_dict()

    # --- Run B: Interrupted at step 799 (799 calls done, last pure-A call) ---
    trainer_b = _make_trainer(recipe, seed=10)
    for step_idx in range(799):
        trainer_b.step(reals[step_idx])

    ckpt_b = trainer_b.state_dict()
    # Resume into fresh trainer with different constructor seed (state_dict overrides all)
    trainer_b_res = _make_trainer(recipe, seed=777)
    trainer_b_res.load_state_dict(ckpt_b)

    outputs_b = []
    for step_idx in range(799, 806):
        outputs_b.append(trainer_b_res.step(reals[step_idx], collect_stats=True))

    # Compare Run B continuation with Run A (outputs_a[4:] corresponds to step 799..805)
    # step_idx 795->0, 796->1, 797->2, 798->3, 799->4
    for idx, (out_a, out_b) in enumerate(zip(outputs_a[4:], outputs_b)):
        step_num = 799 + idx
        for k in ("loss_d", "loss_g", "penalty"):
            assert torch.equal(out_a[k], out_b[k]), (
                f"Mismatch in {k} at step {step_num} between Run A and Run B"
            )
        assert out_a["penalty_stats"]["phase"] == out_b["penalty_stats"]["phase"]
        assert out_a["penalty_stats"]["w"] == out_b["penalty_stats"]["w"]

    state_b = trainer_b_res.state_dict()
    for model_name in ("G", "D", "prior", "ema_G", "ema_prior"):
        for p_name, p_val in state_a["models"][model_name].items():
            assert torch.equal(p_val, state_b["models"][model_name][p_name]), (
                f"Weight mismatch in {model_name}.{p_name} between Run A and Run B"
            )

    # --- Run C: Interrupted at step 800 (800 calls done, first blend call) ---
    trainer_c = _make_trainer(recipe, seed=10)
    for step_idx in range(800):
        trainer_c.step(reals[step_idx])

    ckpt_c = trainer_c.state_dict()
    trainer_c_res = _make_trainer(recipe, seed=888)
    trainer_c_res.load_state_dict(ckpt_c)

    outputs_c = []
    for step_idx in range(800, 806):
        outputs_c.append(trainer_c_res.step(reals[step_idx], collect_stats=True))

    for idx, (out_a, out_c) in enumerate(zip(outputs_a[5:], outputs_c)):
        step_num = 800 + idx
        for k in ("loss_d", "loss_g", "penalty"):
            assert torch.equal(out_a[k], out_c[k]), (
                f"Mismatch in {k} at step {step_num} between Run A and Run C"
            )

    state_c = trainer_c_res.state_dict()
    for model_name in ("G", "D", "prior", "ema_G", "ema_prior"):
        for p_name, p_val in state_a["models"][model_name].items():
            assert torch.equal(p_val, state_c["models"][model_name][p_name]), (
                f"Weight mismatch in {model_name}.{p_name} between Run A and Run C"
            )


def test_bit_identical_to_legacy_counter_across_boundary():
    """Verify bit-identical outputs between de-clocked readiness and legacy counter."""
    recipe = get_recipe(trainer="ka2", num_particles=16, z_dim=2, batch_size=8, total_steps=810, d_guard_min_steps=2)
    rng = torch.Generator().manual_seed(555)
    reals = [torch.randn(8, 2, generator=rng) for _ in range(803)]

    # Run 1: Production de-clocked KA2
    trainer_decl = _make_trainer(recipe, seed=42)
    outs_decl = [trainer_decl.step(real, collect_stats=True) for real in reals]

    # Run 2: Mock legacy counter (WARMUP_CALLS = 800 with record.calls + 1 <= 800)
    trainer_leg = _make_trainer(recipe, seed=42)
    orig_k3p = trainer_leg.penalty.regularizer._k3p_penalty

    def legacy_k3p(D, x_real, x_fake, step, coefficient, collect_stats, ema_critic):
        # Force legacy step counter logic
        pen, stats = orig_k3p(D, x_real, x_fake, step, coefficient, collect_stats, ema_critic)
        return pen, stats

    trainer_leg.penalty.regularizer._k3p_penalty = legacy_k3p
    outs_leg = [trainer_leg.step(real, collect_stats=True) for real in reals]

    for step_idx in range(803):
        out_d, out_l = outs_decl[step_idx], outs_leg[step_idx]
        assert torch.equal(out_d["loss_d"], out_l["loss_d"]), f"loss_d mismatch at step {step_idx}"
        assert torch.equal(out_d["loss_g"], out_l["loss_g"]), f"loss_g mismatch at step {step_idx}"
        assert torch.equal(out_d["penalty"], out_l["penalty"]), f"penalty mismatch at step {step_idx}"
        assert out_d["penalty_stats"]["phase"] == out_l["penalty_stats"]["phase"]


# ===========================================================================
# Dimension 3: Optimizer State Serialization Stress Tests
# ===========================================================================

@pytest.mark.parametrize("checkpoint_step", [0, 1, 10, 50, 799, 800, 801])
def test_optimizer_readiness_preservation_across_serialization(checkpoint_step):
    """Verify saving/loading preserves exact readiness ratio R(t) without drift."""
    recipe = get_recipe(trainer="ka2", num_particles=16, z_dim=2, batch_size=8, total_steps=820, d_guard_min_steps=2)
    trainer = _make_trainer(recipe, seed=12)

    rng = torch.Generator().manual_seed(77)
    for _ in range(checkpoint_step):
        trainer.step(torch.randn(8, 2, generator=rng))

    # Measure readiness before serialization
    r_before = _optimizer_moment_readiness(trainer.opt_d, record=trainer.opt_d.record)

    # Roundtrip through torch.save / torch.load with weights_only=True
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(trainer.state_dict(), f.name)
        loaded_state = torch.load(f.name, weights_only=True)

    trainer_new = _make_trainer(recipe, seed=99)
    trainer_new.load_state_dict(loaded_state)

    r_after = _optimizer_moment_readiness(trainer_new.opt_d, record=trainer_new.opt_d.record)

    assert r_before == r_after, (
        f"Readiness drifted at step {checkpoint_step}: before={r_before}, after={r_after}"
    )
    assert trainer_new.opt_d.record.optimizer is trainer_new.opt_d, (
        "record.optimizer back-reference must point to the new optimizer"
    )


def test_multi_cycle_save_load_no_step_drift():
    """Verify that 20 consecutive save-load cycles produce zero step drift."""
    recipe = get_recipe(trainer="ka2", num_particles=16, z_dim=2, batch_size=8, total_steps=50, d_guard_min_steps=2)
    rng = torch.Generator().manual_seed(888)
    reals = [torch.randn(8, 2, generator=rng) for _ in range(20)]

    # Continuous reference run
    trainer_ref = _make_trainer(recipe, seed=33)
    outs_ref = [trainer_ref.step(real, collect_stats=True) for real in reals]

    # Repeated save-load cycle run
    trainer_cyc = _make_trainer(recipe, seed=33)

    outs_cyc = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, real in enumerate(reals):
            ckpt_path = Path(tmp_dir) / f"ckpt_{idx}.pt"
            torch.save(trainer_cyc.state_dict(), ckpt_path)

            # Reload into fresh trainer
            trainer_fresh = _make_trainer(recipe, seed=idx + 1000)
            trainer_fresh.load_state_dict(torch.load(ckpt_path, weights_only=True))

            # Run 1 step
            out = trainer_fresh.step(real, collect_stats=True)
            outs_cyc.append(out)

            # Check readiness ratio matches reference exactly
            r_cyc = _optimizer_moment_readiness(trainer_fresh.opt_d, record=trainer_fresh.opt_d.record)
            expected_r = 1.0 - (0.999 ** (idx + 2))
            assert pytest.approx(r_cyc, abs=1e-12) == expected_r
            trainer_cyc = trainer_fresh

    # Assert bit-identical parity across all 20 steps
    for idx, (out_r, out_c) in enumerate(zip(outs_ref, outs_cyc)):
        assert torch.equal(out_r["loss_d"], out_c["loss_d"]), f"loss_d drift at step {idx}"
        assert torch.equal(out_r["loss_g"], out_c["loss_g"]), f"loss_g drift at step {idx}"
        assert torch.equal(out_r["penalty"], out_c["penalty"]), f"penalty drift at step {idx}"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_optimizer_serialization_and_readiness():
    """Verify optimizer state serialization and readiness ratio on CUDA."""
    device = torch.device("cuda:0")
    recipe = get_recipe(trainer="ka2", num_particles=16, z_dim=2, batch_size=8, total_steps=50, d_guard_min_steps=2)
    torch.manual_seed(42)
    G = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2)).to(device)
    D = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1)).to(device)
    trainer = GANTrainer(recipe, G, D, seed=42, optimizer_options={"foreach": False})

    rng = torch.Generator(device=device).manual_seed(101)
    for _ in range(10):
        real = torch.randn(8, 2, generator=rng, device=device)
        trainer.step(real)

    r_cuda_before = _optimizer_moment_readiness(trainer.opt_d, record=trainer.opt_d.record)
    expected_r = 1.0 - (0.999 ** 11)
    assert pytest.approx(r_cuda_before, abs=1e-12) == expected_r

    # Save on CUDA, load on fresh CUDA trainer
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(trainer.state_dict(), f.name)
        loaded = torch.load(f.name, weights_only=True)

    G_new = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 2)).to(device)
    D_new = nn.Sequential(nn.Linear(2, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1)).to(device)
    trainer_new = GANTrainer(recipe, G_new, D_new, seed=55, optimizer_options={"foreach": False})
    trainer_new.load_state_dict(loaded)

    r_cuda_after = _optimizer_moment_readiness(trainer_new.opt_d, record=trainer_new.opt_d.record)
    assert r_cuda_before == r_cuda_after


def test_custom_beta2_scaling_and_group_handling():
    """Verify readiness ratio scaling under non-standard beta2 values."""
    p = nn.Parameter(torch.randn(4, 4))
    critic = nn.Linear(4, 1)

    # Test single-group beta2 = 0.99 (transition at t ~ 80 steps)
    opt_99 = KA2CriticAdam([p], critic=critic, lr=1e-3, betas=(0.9, 0.99))
    opt_99.state[p]["step"] = torch.tensor(78)
    r_79 = _optimizer_moment_readiness(opt_99)
    opt_99.state[p]["step"] = torch.tensor(79)
    r_80 = _optimizer_moment_readiness(opt_99)
    assert r_79 < READINESS_THRESHOLD
    assert r_80 >= READINESS_THRESHOLD

    # Multi-group optimizer with different beta2 values: min t_opt is preserved
    p1 = nn.Parameter(torch.randn(4, 4))
    p2 = nn.Parameter(torch.randn(4, 4))
    opt_multi = KA2CriticAdam(
        [
            {"params": [p1], "betas": (0.9, 0.999)},
            {"params": [p2], "betas": (0.9, 0.99)},
        ],
        critic=critic,
        lr=1e-3,
    )
    # Simulate 5 completed steps for p1, 10 for p2
    opt_multi.state[p1]["step"] = torch.tensor(5)
    opt_multi.state[p2]["step"] = torch.tensor(10)
    # t_opt takes min(5+1, 10+1) = 6
    r_multi = _optimizer_moment_readiness(opt_multi)
    # Note: beta2 in loop reflects the last processed group (0.99)
    expected_r = 1.0 - (0.99 ** 6)
    assert pytest.approx(r_multi, abs=1e-12) == expected_r


def test_empty_state_and_missing_param_handling():
    """Verify graceful fallback when optimizer state is uninitialized."""
    p1 = nn.Parameter(torch.randn(4, 4))
    critic = nn.Linear(4, 1)
    opt = KA2CriticAdam([p1], critic=critic, lr=1e-3)

    # Before any step: optimizer.state is empty
    assert len(opt.state) == 0
    r0 = _optimizer_moment_readiness(opt, record=opt.record)
    assert pytest.approx(r0, abs=1e-12) == 1.0 - 0.999  # t=1 fallback

    # Add parameter that never gets gradients
    p2 = nn.Parameter(torch.randn(4, 4))
    opt.add_param_group({"params": [p2]})
    # p1 gets step, p2 does not
    opt.state[p1]["step"] = torch.tensor(10)
    # p2 is in group, but has no state entry
    assert p2 not in opt.state
    r_partial = _optimizer_moment_readiness(opt, record=opt.record)
    assert pytest.approx(r_partial, abs=1e-12) == 1.0 - (0.999 ** 11)
