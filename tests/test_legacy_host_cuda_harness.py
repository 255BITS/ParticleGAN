"""CPU-identical host fixtures, and the CUDA smoke for the four repaired toys.

CPU checksums are the pre-change streams (two updates, RNG state, Fourier bank).
CUDA cases skip when this process has no GPU. The A6000 command is
``python -m benchmarks.locked_shared.cuda_host_smoke``.
"""

import copy
import hashlib
import math

import pytest
import torch

from benchmarks.locked_shared.hosts import (
    ae_gan_hold, cover_leftover, unipolar, unused_token_hold,
)
from benchmarks.toy100.device import device_policy_scope, experiment_generator
from benchmarks.toy100.train import train as train_toy100
from benchmarks.toy_suite import _check_toy100_policy
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy


CPU_STREAMS = {
    "unipolar": "21243cc2156cc1bfefb3f0a2d2266d00c7cd75d1b80809a4e9783ca5968ac100",
    "unipolar_rng": "dc43658172e648417b6fd58ce4ff18aa577f15ae7fa601cda1f16f9a55a22db5",
    "ae": "a1ad80742249265f6f8b550aa838492151d6f23427215b9cbd8ceea776dd2c4b",
    "ae_rng": "0d879dffd6a9dd0b952ed343cb12b183233c82bdb19c4a18d0c41140e282a770",
    "cover": "1b62214f3a7720ccf789e0489d02a41fa64d9998673f045ac2d39eeaec01e8f4",
    "cover_rng": "36427c5f61de0a6ecbdc4c55378c7a9afcf85b4bb657b2354176ad6d6c675c6f",
    "cover_bank": "45b2c545841d59a52782c3bd992474783243eec3928e405d3036873d7684106e",
    "unused": "79ecb3e9c67206df51c963cd4e20d87b89e4a0d4c1d508fd1bce39540f34887f",
    "unused_rng": "9545186d2d51bc84e496fcfae41796cc0727036c3fcc0c6cea564407b1ab162f",
}

COVER_FLOAT_KEYS = (
    "content_kept", "cover_weight", "even_norm", "fm_weight", "kappa", "leak_frac",
    "leak_ratio", "odd_norm", "on_content", "on_e", "on_u", "particle_l2",
    "particle_rms", "pole_rel_err_minus", "pole_rel_err_plus", "reg_coeff",
    "same_dir", "u_kept",
)
UNUSED_FLOAT_KEYS = (
    "concept_cos", "concept_delta_norm", "concept_move", "scale0_err",
    "unused_dist", "unused_hold",
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA harness smoke")


def _sha_floats(values):
    blob = torch.tensor(list(values), dtype=torch.float64).numpy().tobytes()
    return hashlib.sha256(blob).hexdigest()


def _rng_sha():
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def _finite(values):
    assert values
    assert all(math.isfinite(float(value)) for value in values)


def test_cpu_two_step_streams_match_the_prechange_bytes():
    torch.manual_seed(0)
    row = unipolar.run_arm("locked_rpgan", steps=2, seed=0)
    assert _sha_floats([
        row["cover"], row["off_caption"], row["neu_hold"], row["cos_plus"],
        row["mag_ratio"], row["half_norm_ratio"], row["origin_norm"],
        row["odd_norm"], row["even_norm"],
    ]) == CPU_STREAMS["unipolar"]
    assert _rng_sha() == CPU_STREAMS["unipolar_rng"]

    torch.manual_seed(0)
    out = ae_gan_hold.train(ae_gan_hold.HoldConfig(
        name="parity", steps=2, batch=16, n_particles=8,
    ))
    assert _sha_floats([out["recon_mse"], out["hold"], out["init_recon_mse"]]) == CPU_STREAMS["ae"]
    assert _rng_sha() == CPU_STREAMS["ae_rng"]

    torch.manual_seed(0)
    cover = cover_leftover.fit_cover_leftover(cover_leftover.CoverRecipe(steps=2))
    assert _sha_floats([cover[key] for key in COVER_FLOAT_KEYS]) == CPU_STREAMS["cover"]
    assert _rng_sha() == CPU_STREAMS["cover_rng"]
    bank = cover_leftover._FourierCritic(4, n_rand=16, hidden=8, seed=3).bank
    assert hashlib.sha256(bank.detach().cpu().contiguous().numpy().tobytes()).hexdigest() == (
        CPU_STREAMS["cover_bank"]
    )

    torch.manual_seed(0)
    unused = unused_token_hold.train(unused_token_hold.UnusedHoldRecipe(steps=2))
    assert _sha_floats([unused[key] for key in UNUSED_FLOAT_KEYS]) == CPU_STREAMS["unused"]
    assert _rng_sha() == CPU_STREAMS["unused_rng"]


def test_cpu_fixtures_are_not_resampled():
    student = unipolar.FreeOriginResidual(unipolar.DIM)
    assert unipolar._placed(unipolar.PLUS, student.odd) is unipolar.PLUS
    holder = unused_token_hold.SharedSlotStudent()
    assert torch.equal(holder.neu, unused_token_hold.NEU)
    assert holder.neu.data_ptr() != unused_token_hold.NEU.data_ptr()
    assert unused_token_hold._placed(
        unused_token_hold.CONCEPT_DIR, holder.shared,
    ) is unused_token_hold.CONCEPT_DIR


def test_ae_evaluation_leaves_the_training_rng_untouched():
    torch.manual_seed(4)
    cfg = ae_gan_hold.HoldConfig(name="rng", steps=1, batch=8, n_particles=4)
    recipe = ae_gan_hold.make_recipe(cfg)
    prior = recipe.make_prior()
    encoder, decoder = ae_gan_hold.MLP(2, 4), ae_gan_hold.MLP(2, 2)
    before = experiment_generator().get_state().clone()
    ae_gan_hold.evaluate(encoder, decoder, prior, recipe)
    assert torch.equal(before, experiment_generator().get_state())
    expected = torch.randn(5)
    experiment_generator().set_state(before)
    assert torch.equal(expected, torch.randn(5))


def _tiny_grid(tmp_path):
    config = dict(
        problem="grid100", seed=31, steps=2, device="cpu",
        z_dim=2, num_particles=8, batch_size=4, d_hidden=4, n_hidden=1,
        fourier=0, eval_samples=16, snapshot_samples=4, eval_interval=2,
        snapshot_interval=2, log_interval=2, threads=1,
        output_noise_std=0.029, output_noise_learnable=True,
        input_noise_std=0.5, toy100_model="affine_square_v1",
    )
    directory = tmp_path / "grid"
    summary = train_toy100(config, directory)
    resolved = summary["config"]
    policy = declared_model_policy(config)
    return directory, summary, resolved, policy


def test_replay_rejects_tampered_seed_device_config_or_noise(tmp_path):
    directory, summary, resolved, policy = _tiny_grid(tmp_path)
    assert _check_toy100_policy(directory, summary, resolved, policy)

    seeded = dict(resolved)
    seeded["seed"] = resolved["seed"] + 1
    with pytest.raises(ValueError, match="seeded affine initialization differs"):
        _check_toy100_policy(directory, summary, seeded, policy)

    moved = dict(resolved)
    moved["device"] = "cuda"
    with pytest.raises(ValueError, match="recorded device|seeded affine initialization differs"):
        _check_toy100_policy(directory, summary, moved, policy)

    resized = dict(resolved)
    resized["num_particles"] = resolved["num_particles"] + 8
    with pytest.raises(ValueError, match="model policy receipt differs"):
        _check_toy100_policy(directory, summary, resized, policy)

    missing_learned = copy.deepcopy(summary)
    missing_learned.pop("learnable_output_noise")
    with pytest.raises(ValueError, match="learned output-scale receipt is absent"):
        _check_toy100_policy(directory, missing_learned, resolved, policy)

    tampered_noise = copy.deepcopy(summary)
    tampered_noise["learnable_output_noise"]["parameter"] = "not-the-generator-scale"
    with pytest.raises(ValueError, match="learned output scale is not bound"):
        _check_toy100_policy(directory, tampered_noise, resolved, policy)

    isolated = dict(resolved)
    isolated["output_noise_rng"] = "isolated"
    with pytest.raises(ValueError, match="model policy receipt differs|isolated output-noise"):
        _check_toy100_policy(directory, summary, isolated, policy)


def test_candidate_episode_rejects_a_missing_noise_receipt(tmp_path, monkeypatch):
    from tests.test_toy_suite import _write_candidate_episode
    import benchmarks.toy_suite as toy_suite

    monkeypatch.setattr(toy_suite, "test_verdict", lambda spec, result: dict(
        status="PASS", passed=True, convergence=dict(passing_suffix=5),
    ))
    name = _write_candidate_episode(
        tmp_path / "missing", lambda record: record.pop("noise_receipt"),
    )
    grade = toy_suite._episode_rows(tmp_path / "missing", (name,), candidate=True)
    assert grade["status"] == "INVALID"
    assert "noise receipt is absent" in grade["reason"]


def _cuda_scope():
    import benchmarks.toy100.device as device

    scope = device_policy_scope()
    scope.__enter__()
    device._SELECTED = None
    device._RECEIPT = None
    device.apply_device_policy("cuda")
    return scope


@CUDA
@pytest.mark.parametrize("host", ("unipolar", "ae_gan_hold", "cover_leftover", "unused_token_hold"))
def test_cuda_two_finite_updates_and_noise_receipt(host):
    scope = _cuda_scope()
    try:
        policy = NoisePolicy(0.029, 0.5, 0.5, 2, seed=0)
        if host == "unipolar":
            row = unipolar.run_arm("locked_rpgan", steps=2, seed=0, noise_policy=policy)
            _finite([row["cover"], row["off_caption"], row["neu_hold"], row["odd_norm"]])
        elif host == "ae_gan_hold":
            row = ae_gan_hold.train(ae_gan_hold.HoldConfig(
                name="cuda", steps=2, batch=16, n_particles=8,
            ), noise_policy=policy)
            _finite([row["recon_mse"], row["hold"], row["init_recon_mse"]])
        elif host == "cover_leftover":
            row = cover_leftover.fit_cover_leftover(
                cover_leftover.CoverRecipe(steps=2), noise_policy=policy,
            )
            _finite([row[key] for key in ("u_kept", "leak_ratio", "odd_norm")])
        else:
            row = unused_token_hold.train(
                unused_token_hold.UnusedHoldRecipe(steps=2), noise_policy=policy,
            )
            _finite([row[key] for key in UNUSED_FLOAT_KEYS])
        receipt = policy.receipt()
        assert receipt["step_calls"] == 2
        assert receipt["train_output_applied"] is True
        assert receipt["train_input_applied"] is True
    finally:
        scope.__exit__(None, None, None)


@CUDA
def test_cuda_fresh_process_init_and_draw_match():
    import os
    import subprocess
    import sys

    script = """
import hashlib, json, torch
from benchmarks.toy100.device import apply_device_policy
from benchmarks.locked_shared.hosts import unipolar, ae_gan_hold, cover_leftover, unused_token_hold
apply_device_policy("cuda")
torch.manual_seed(0)
uni = unipolar.run_arm("locked_rpgan", steps=2, seed=0)
ae = ae_gan_hold.train(ae_gan_hold.HoldConfig(name="cuda", steps=2, batch=16, n_particles=8))
cover = cover_leftover.fit_cover_leftover(cover_leftover.CoverRecipe(steps=2))
unused = unused_token_hold.train(unused_token_hold.UnusedHoldRecipe(steps=2))
blob = json.dumps(dict(
    uni=[uni["cover"], uni["odd_norm"]],
    ae=[ae["recon_mse"], ae["hold"]],
    cover=[cover["u_kept"], cover["odd_norm"]],
    unused=[unused["unused_hold"], unused["concept_move"]],
), sort_keys=True)
print("SMOKE " + hashlib.sha256(blob.encode()).hexdigest())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    digests = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script], text=True, capture_output=True, env=env, check=True,
        )
        line = next(row for row in proc.stdout.splitlines() if row.startswith("SMOKE "))
        digests.append(line.split()[1])
    assert digests[0] == digests[1]


@CUDA
def test_cuda_save_restore_repeats_the_next_update():
    scope = _cuda_scope()
    try:
        torch.manual_seed(0)
        student = unipolar.FreeOriginResidual(unipolar.DIM)
        target = unipolar._placed(unipolar.PLUS, student.odd)
        opt = torch.optim.Adam(student.parameters(), lr=1e-2)
        def update():
            loss = (student.delta(1.0) - target).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            return student.odd.detach().clone()
        update()
        weights = copy.deepcopy(student.state_dict())
        moments = copy.deepcopy(opt.state_dict())
        rng = experiment_generator().get_state().clone()
        first = update()
        student.load_state_dict(weights)
        opt.load_state_dict(moments)
        experiment_generator().set_state(rng)
        second = update()
        assert torch.equal(first, second)
        assert first.device.type == "cuda"
        assert torch.equal(target.detach().cpu(), unipolar.PLUS)
    finally:
        scope.__exit__(None, None, None)


@CUDA
def test_cuda_evaluation_leaves_the_training_rng_untouched():
    scope = _cuda_scope()
    try:
        torch.manual_seed(4)
        cfg = ae_gan_hold.HoldConfig(name="cuda-rng", steps=1, batch=8, n_particles=4)
        recipe = ae_gan_hold.make_recipe(cfg)
        prior = recipe.make_prior()
        encoder, decoder = ae_gan_hold.MLP(2, 4), ae_gan_hold.MLP(2, 2)
        before = experiment_generator().get_state().clone()
        ae_gan_hold.evaluate(encoder, decoder, prior, recipe)
        assert torch.equal(before, experiment_generator().get_state())
        assert before.device.type == "cuda"
    finally:
        scope.__exit__(None, None, None)


@CUDA
def test_cuda_fourier_and_unused_fixtures_keep_cpu_values():
    scope = _cuda_scope()
    try:
        critic = cover_leftover._FourierCritic(4, n_rand=16, hidden=8, seed=3)
        assert critic.bank.device.type == "cuda"
        assert hashlib.sha256(critic.bank.detach().cpu().contiguous().numpy().tobytes()).hexdigest() == (
            CPU_STREAMS["cover_bank"]
        )
        holder = unused_token_hold.SharedSlotStudent()
        assert holder.neu.device.type == "cuda"
        assert torch.equal(holder.neu.detach().cpu(), unused_token_hold.NEU.cpu())
        real = unused_token_hold._placed(unused_token_hold.CONCEPT_DIR, holder.shared)
        assert real.device.type == "cuda"
        assert torch.equal(real.detach().cpu(), unused_token_hold.CONCEPT_DIR.cpu())
    finally:
        scope.__exit__(None, None, None)
