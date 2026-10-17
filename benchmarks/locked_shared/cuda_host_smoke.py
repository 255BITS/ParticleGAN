"""CUDA smoke for the four repaired hosts.

This command does not use pytest. From the repository root, on a machine with CUDA:

    python -m benchmarks.locked_shared.cuda_host_smoke

Exits 2 when CUDA is unavailable. CPU replay checks stay in
``tests/test_legacy_host_cuda_harness.py`` and that file does need pytest.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import os
import subprocess
import sys
from pathlib import Path

import torch

from benchmarks.locked_shared.hosts import (
    ae_gan_hold, cover_leftover, unipolar, unused_token_hold,
)
from benchmarks.toy100.device import (
    apply_device_policy, device_policy_scope, experiment_generator,
)
from benchmarks.transfer_suite.legacy_noise_adapters import NoisePolicy


ROOT = Path(__file__).resolve().parents[2]
HOSTS = ("unipolar", "ae_gan_hold", "cover_leftover", "unused_token_hold")
UNUSED_FLOAT_KEYS = (
    "concept_cos", "concept_delta_norm", "concept_move", "scale0_err",
    "unused_dist", "unused_hold",
)


def _finite(values) -> None:
    values = [float(value) for value in values]
    if not values or not all(value == value and abs(value) != float("inf") for value in values):
        raise AssertionError(f"non-finite update: {values}")


@contextmanager
def _cuda():
    """Select CUDA for one check and restore the process afterwards."""
    import benchmarks.toy100.device as device

    with device_policy_scope():
        device._SELECTED = None
        device._RECEIPT = None
        apply_device_policy("cuda")
        yield


def two_finite_updates(host: str) -> None:
    """Two finite updates, with a noise receipt, on one repaired host."""
    with _cuda():
        if experiment_generator().device.type != "cuda":
            raise AssertionError("CUDA training is not using the CUDA generator")
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
        elif host == "unused_token_hold":
            row = unused_token_hold.train(
                unused_token_hold.UnusedHoldRecipe(steps=2), noise_policy=policy,
            )
            _finite([row[key] for key in UNUSED_FLOAT_KEYS])
        else:
            raise AssertionError(host)
        receipt = policy.receipt()
        if receipt["step_calls"] != 2 or not receipt["train_output_applied"] or not receipt["train_input_applied"]:
            raise AssertionError(f"incomplete noise receipt: {host}")
    print(f"SMOKE host={host} check=two_updates status=PASS", flush=True)


def fresh_process_reproducible() -> None:
    """Two fresh processes draw the same two-step metrics."""
    script = """
import hashlib, json, torch
from benchmarks.toy100.device import apply_device_policy, experiment_generator
from benchmarks.locked_shared.hosts import unipolar, ae_gan_hold, cover_leftover, unused_token_hold
apply_device_policy("cuda")
if experiment_generator().device.type != "cuda":
    raise SystemExit("CUDA generator was not selected")
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
    env["PYTHONPATH"] = str(ROOT)
    digests = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script], text=True, capture_output=True, env=env,
            cwd=str(ROOT), check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(proc.stderr[-2000:] or proc.stdout[-2000:])
        line = next(row for row in proc.stdout.splitlines() if row.startswith("SMOKE "))
        digests.append(line.split()[1])
    if digests[0] != digests[1]:
        raise AssertionError(f"fresh-process draws differ: {digests}")
    print("SMOKE check=fresh_process status=PASS", flush=True)


def save_restore_next_update() -> None:
    """Saving the unipolar step and restoring it repeats the next update."""
    with _cuda():
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
        if not torch.equal(first, second):
            raise AssertionError("restored update differs")
        if first.device.type != "cuda" or not torch.equal(target.detach().cpu(), unipolar.PLUS):
            raise AssertionError("saved update left the CUDA fixture")
    print("SMOKE check=save_restore status=PASS", flush=True)


def evaluation_leaves_training_rng() -> None:
    """Evaluation restores the CUDA experiment generator.

    ``Generator.get_state()`` is a CPU byte tensor even for a CUDA generator.
    The generator's own device is the check that training is on CUDA.
    """
    with _cuda():
        generator = experiment_generator()
        if generator.device.type != "cuda":
            raise AssertionError(f"experiment generator is {generator.device}, not cuda")
        torch.manual_seed(4)
        cfg = ae_gan_hold.HoldConfig(name="cuda-rng", steps=1, batch=8, n_particles=4)
        recipe = ae_gan_hold.make_recipe(cfg)
        prior = recipe.make_prior()
        encoder, decoder = ae_gan_hold.MLP(2, 4), ae_gan_hold.MLP(2, 2)
        before = generator.get_state().clone()
        ae_gan_hold.evaluate(encoder, decoder, prior, recipe)
        if not torch.equal(before, generator.get_state()):
            raise AssertionError("evaluation advanced the training generator")
    print("SMOKE check=eval_rng status=PASS", flush=True)


def fixtures_keep_cpu_values() -> None:
    """CUDA modules use the historical CPU fixture values, moved and not redrawn."""
    with _cuda():
        gen = torch.Generator(device="cpu").manual_seed(3 + 17)
        expected = 2.0 * torch.randn(16, 4, generator=gen, device="cpu")
        critic = cover_leftover._FourierCritic(4, n_rand=16, hidden=8, seed=3)
        if critic.bank.device.type != "cuda" or not torch.equal(critic.bank.detach().cpu(), expected):
            raise AssertionError("Fourier bank was resampled onto CUDA")
        holder = unused_token_hold.SharedSlotStudent()
        concept = unused_token_hold._placed(unused_token_hold.CONCEPT_DIR, holder.shared)
        if holder.neu.device.type != "cuda" or concept.device.type != "cuda":
            raise AssertionError("unused fixtures stayed on CPU")
        if not torch.equal(holder.neu.detach().cpu(), unused_token_hold.NEU.cpu()):
            raise AssertionError("neu fixture changed value")
        if not torch.equal(concept.detach().cpu(), unused_token_hold.CONCEPT_DIR.cpu()):
            raise AssertionError("concept fixture changed value")
    print("SMOKE check=fixtures status=PASS", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is required for this smoke", file=sys.stderr)
        return 2
    os.chdir(ROOT)
    try:
        for host in HOSTS:
            two_finite_updates(host)
        fresh_process_reproducible()
        save_restore_next_update()
        evaluation_leaves_training_rng()
        fixtures_keep_cpu_values()
    except Exception as error:
        print(f"SMOKE FAIL {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print("SMOKE PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
