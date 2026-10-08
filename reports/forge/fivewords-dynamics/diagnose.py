"""Bounded CUDA saved-endpoint probes, through the archived public word host/API.

This is neither replay of the historical stochastic trajectory nor qualification.
See protocol.json for fixed scope and numerical controls. No CPU fallback.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
ROLES = ("G", "E", "prior", "D")
RUNTIME_COMMIT = "34db5abbf164229ee449d2862bba53e043da7f82"
RUNTIME_FILES = (
    "benchmarks/toy_audit/api_images.py", "benchmarks/toy_audit/definition_quality.py",
    "particlegan/recipes.py", "particlegan/optim/dualnorm.py", "particlegan/gan_loss.py",
    "particlegan/grad_regularizers.py", "particlegan/k3p.py", "particlegan/policy.py",
    "experiments/forge/api.py", "experiments/forge/word_adapter.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def runtime_receipt(root):
    receipt = {}
    for name in RUNTIME_FILES:
        expected = subprocess.check_output(["git", "show", f"{RUNTIME_COMMIT}:{name}"], cwd=root)
        actual = (root / name).read_bytes()
        if actual != expected:
            raise RuntimeError(f"Frozen numerical runtime differs: {name}")
        receipt[name] = hashlib.sha256(actual).hexdigest()
    return receipt


def modules(fixture):
    return dict(G=fixture.G, E=fixture.E, prior=fixture.prior, D=fixture.D)


def parameters(fixture):
    return {role: list(module.parameters()) for role, module in modules(fixture).items()}


def vector(parts):
    return torch.cat([part.reshape(-1) for part in parts])


def clone_parameters(fixture):
    return {role: [p.detach().clone() for p in ps] for role, ps in parameters(fixture).items()}


def restore_parameters(fixture, saved):
    with torch.no_grad():
        for role, ps in parameters(fixture).items():
            for p, value in zip(ps, saved[role]):
                p.copy_(value)


def field(fixture, *, dtype=None):
    """Expected raw role gradients; each original player's loss is minimized.

    The prior and real-word indices are independent. RpGAN is nonlinear in
    paired scores, so five same-index pairs are not the expected objective.
    """
    device = fixture.device
    ids = torch.arange(5, device=device)
    real = fixture.words[ids.repeat_interleave(5)]
    rows = ids.repeat(5)
    with torch.no_grad():
        encoded = fixture.E(real)
        latent = fixture.prior.z[rows]
        fake = fixture.G(latent)
    real_joint, fake_joint = join(real, encoded), join(fake, latent)
    adv_d = fixture.loss.d_loss(fixture.D(real_joint), fixture.D(fake_joint))
    penalty = fixture.penalty(fixture.D, real_joint, fake_joint)
    d_loss = adv_d + penalty
    pd = list(fixture.D.parameters())
    gd = torch.autograd.grad(d_loss, pd)
    encoded = fixture.E(real)
    latent = fixture.prior.z[rows]
    fake = fixture.G(latent)
    g_loss = fixture.loss.joint_g_loss(fixture.D(join(fake, latent)), fixture.D(join(real, encoded)))
    if fixture.recipe.prior_reg != 0:
        raise RuntimeError("This frozen diagnostic binds prior_reg0")
    ps = parameters(fixture)
    sizes = [len(ps[role]) for role in ROLES[:3]]
    gg = torch.autograd.grad(g_loss, [p for role in ROLES[:3] for p in ps[role]])
    result, start = {"D": [g.detach() for g in gd]}, 0
    for role, size in zip(ROLES[:3], sizes):
        result[role] = [g.detach() for g in gg[start:start + size]]
        start += size
    return result, dict(d_loss=float(d_loss.detach()), d_adversarial=float(adv_d.detach()),
                        d_penalty=float(penalty.detach()), joint_loss=float(g_loss.detach()))


def public_step(fixture, active):
    """One expected-batch role-masked public step, with original D-first order."""
    before = clone_parameters(fixture)
    raw_before, loss_before = field(fixture)
    ps = parameters(fixture)
    fixture.opt_d.zero_grad(set_to_none=True)
    if "D" in active:
        for p, grad in zip(ps["D"], raw_before["D"]):
            p.grad = grad.clone()
        fixture.opt_d.step()
    raw_after_d, loss_after_d = field(fixture)
    fixture.opt_g.zero_grad(set_to_none=True)
    for role in ROLES[:3]:
        if role in active:
            for p, grad in zip(ps[role], raw_after_d[role]):
                p.grad = grad.clone()
    fixture.opt_g.set_sampled_rows(fixture.prior.z, torch.arange(5, device=fixture.device))
    fixture.opt_g.step()
    after = clone_parameters(fixture)
    delta = {role: [b - a for a, b in zip(before[role], after[role])] for role in ROLES}
    used = {**raw_after_d, "D": raw_before["D"]}
    if any(not bool(torch.isfinite(x).all()) for parts in delta.values() for x in parts):
        raise RuntimeError("Nonfinite actual public optimizer update")
    return delta, used, loss_before, loss_after_d


def exact_metrics(fixture):
    """Public clean served output on all five rows; exact uniform support law."""
    with torch.no_grad():
        served = fixture.policy.served_model()
        rows = torch.arange(5, device=fixture.device)
        generated = served.generate(fixture.prior.z, rows=rows, output_noise=False)
        encoded = served.encoder(fixture.words)
        reconstruction = served.generate(encoded, output_noise=False)
        # Repeating each equal-mass row equally is deterministic; sample_count
        # is a representation of the exact law, not1025 independent samples.
        result = score_words(generated.repeat_interleave(205, 0).cpu().numpy(), reconstruction.cpu().numpy())
        labels = generated.argmax(1)
        target_labels = fixture.words.argmax(1)
        row_ids = []
        for row in labels:
            matched = [i for i, target in enumerate(target_labels) if bool((row == target).all())]
            row_ids.append(matched[0] if matched else -1)
        recon_ids = []
        for row in reconstruction.argmax(1):
            matched = [i for i, target in enumerate(target_labels) if bool((row == target).all())]
            recon_ids.append(matched[0] if matched else -1)
        distances = torch.cdist(encoded, fixture.prior.z)
        return dict(passed=result["passed"], failed_bounds=result["failed_bounds"],
                    metrics=result["metrics"], prior_row_word_assignment=row_ids,
                    reconstruction_word_assignment=recon_ids,
                    encoded_to_prior_nearest_rows=distances.argmin(1).tolist(),
                    encoded_to_prior_min_distance=distances.min(1).values.tolist(),
                    encoded_latents=encoded.tolist(), prior_latents=fixture.prior.z.tolist())


def load_fixture(root, source):
    from experiments.forge.word_adapter import word_context
    packet = torch.load(source / "state.pt", map_location="cuda:0", weights_only=False)
    request = json.loads((source / "request.json").read_text())["request"]
    task = request["tasks"]["five_word_joint_acquisition"]
    context = word_context(request, task, "cuda:0")
    fixture = WordFixture(device="cuda:0", seed=0, recipe_name=None,
                          max_steps=packet["fixture"]["max_steps"], components=context)
    # RNG state tensors stay on CPU as required by torch.Generator.set_state.
    packet = torch.load(source / "state.pt", map_location="cpu", weights_only=False)
    context.streams.load_state_dict(packet["streams"])
    fixture.policy.load_state_dict(packet["fixture"]["api_state"])
    fixture.data_generator.set_state(packet["fixture"]["data_generator"])
    actual = fixture.state_dict()
    expected = packet["fixture"]
    checks = {
        key: state_digest(actual["api_state"][key]) == state_digest(expected["api_state"][key])
        for key in ("models", "optimizers", "streams")
    }
    checks["data_stream"] = torch.equal(fixture.data_generator.get_state(), expected["data_generator"])
    checks["named_streams"] = state_digest(context.streams.state_dict()) == state_digest(packet["streams"])
    if not all(checks.values()) or fixture.completed_steps != 20001:
        raise RuntimeError(f"Archived checkpoint restore mismatch: {checks}")
    if any(p.device.type != "cuda" for ps in parameters(fixture).values() for p in ps):
        raise RuntimeError("CPU model fallback forbidden")
    return fixture, packet, context, checks


def reset(fixture, packet, context):
    context.streams.load_state_dict(packet["streams"])
    fixture.policy.load_state_dict(packet["fixture"]["api_state"])
    fixture.data_generator.set_state(packet["fixture"]["data_generator"])


@contextmanager
def arm_runtime(arm):
    original = dualnorm.polar_factor
    def polar(value, *, smoothing=0., **kwargs):
        return original(value, smoothing=smoothing, truncate=arm["truncation"])
    dualnorm.polar_factor = polar
    try:
        with torch.autograd.set_multithreading_enabled(arm["autograd_multithreading"]):
            yield
    finally:
        dualnorm.polar_factor = original


def expectation_control(fixture):
    """Independent one-pair loss enumeration, same expected population law."""
    with torch.no_grad():
        encoded = fixture.E(fixture.words)
        fake = fixture.G(fixture.prior.z)
        real_logits = fixture.D(join(fixture.words, encoded))
        fake_logits = fixture.D(join(fake, fixture.prior.z))
        d = torch.stack([fixture.loss.d_loss(real_logits[i:i+1], fake_logits[j:j+1])
                         for i in range(5) for j in range(5)]).mean()
        g = torch.stack([fixture.loss.joint_g_loss(fake_logits[j:j+1], real_logits[i:i+1])
                         for i in range(5) for j in range(5)]).mean()
    _, actual = field(fixture)
    errors = dict(d=abs(float(d) - actual["d_adversarial"]), g=abs(float(g) - actual["joint_loss"]))
    if any(error > 1e-4 * max(1., abs(actual["joint_loss"]), abs(actual["d_adversarial"])) for error in errors.values()):
        raise RuntimeError(f"Cartesian loss expectation mismatch: {errors}")
    return errors


def step_statistics(fixture, delta, used):
    ps = parameters(fixture)
    result = {}
    for role in ROLES:
        raw, change, weights = vector(used[role]), vector(delta[role]), vector([p.detach() for p in ps[role]])
        rn, dn = float(raw.norm()), float(change.norm())
        result[role] = dict(raw_gradient_norm=rn, actual_step_norm=dn,
            step_over_raw_norm=dn / max(rn, 1e-300), relative_parameter_step=dn / max(float(weights.norm()), 1e-300),
            cosine_descent_to_raw=float(torch.dot(-change, raw) / (change.norm() * raw.norm())) if rn and dn else None,
            layers=[dict(shape=list(p.shape), gradient_norm=float(g.norm()), actual_step_norm=float(d.norm()))
                    for p, g, d in zip(ps[role], used[role], delta[role])])
    return result


def projected_jacobian(fixture, directions):
    # Float64 CUDA perturbations separate arithmetic resolution from the
    # original float32 optimizer. No float64 optimizer or training is used.
    for module in modules(fixture).values():
        module.double()
    fixture.words = fixture.words.double()
    base = clone_parameters(fixture)
    basis = {}
    for role in ROLES:
        parts = [part.double() for part in directions[role]]
        norm = vector(parts).norm()
        if not bool(norm > 1e-30):
            return dict(identified=False, reason=f"Zero actual direction for {role}")
        basis[role] = [part / norm for part in parts]
    matrices = []
    for h in (1e-3, 3e-4):
        matrix = torch.zeros((4, 4), device="cuda:0", dtype=torch.float64)
        for j, perturbed_role in enumerate(ROLES):
            values = []
            for sign in (1, -1):
                restore_parameters(fixture, base)
                with torch.no_grad():
                    for p, e in zip(parameters(fixture)[perturbed_role], basis[perturbed_role]):
                        p.add_(e, alpha=sign * h)
                grads, _ = field(fixture)
                values.append([sum((g * e).sum() for g, e in zip(grads[role], basis[role])) for role in ROLES])
            matrix[:, j] = torch.stack([(a - b) / (2 * h) for a, b in zip(values[0], values[1])])
        matrices.append(matrix)
    restore_parameters(fixture, base)
    norm = float(matrices[-1].norm())
    relative = float((matrices[0] - matrices[1]).norm()) / max(norm, 1e-300)
    matrix = matrices[-1]
    skew, symmetric = .5 * (matrix - matrix.T), .5 * (matrix + matrix.T)
    offdiag = matrix.clone()
    offdiag.diagonal().zero_()
    offskew, offsym = .5 * (offdiag - offdiag.T), .5 * (offdiag + offdiag.T)
    eig = torch.linalg.eigvals(matrix)
    return dict(identified=relative <= .05 and norm > 1e-12,
        central_difference_scales=[1e-3, 3e-4], raw_gradient_units="loss gradient / parameter displacement",
        basis="Four disjoint unit parameter directions from first actual float32 sequential update",
        relative_scale_disagreement=relative, matrix_coarse=matrices[0].tolist(), matrix_fine=matrix.tolist(),
        matrix_norm=norm, skew_norm=float(skew.norm()), symmetric_norm=float(symmetric.norm()),
        skew_over_symmetric=float(skew.norm()) / max(float(symmetric.norm()), 1e-300),
        offdiagonal_skew_norm=float(offskew.norm()), offdiagonal_symmetric_norm=float(offsym.norm()),
        offdiagonal_skew_over_symmetric=float(offskew.norm()) / max(float(offsym.norm()), 1e-300),
        eigenvalues=[dict(real=float(e.real), imaginary=float(e.imag)) for e in eig],
        projected_circulation_dominates=relative <= .05 and norm > 1e-12 and bool(skew.norm() > symmetric.norm()))


def mathematical_controls():
    q = torch.tensor([.2, -.3], device="cuda:0", dtype=torch.float64)
    values = {}
    for name, expected in (("bilinear", [[0., 1.], [-1., 0.]]), ("quadratic", [[1., 0.], [0., 1.]])):
        target = torch.tensor(expected, device="cuda:0", dtype=torch.float64)
        def f(x):
            return torch.stack((x[1], -x[0])) if name == "bilinear" else x
        matrices = []
        for h in (1e-3, 3e-4):
            matrices.append(torch.stack([(f(q + h * e) - f(q - h * e)) / (2*h)
                           for e in torch.eye(2, device="cuda:0", dtype=torch.float64)], dim=1))
        error = max(float((m-target).abs().max()) for m in matrices)
        if error > 1e-10:
            raise RuntimeError(f"Projected Jacobian control failed: {name} {error}")
        values[name] = dict(max_absolute_error=error, passed=True, expected=expected)
    return values


def analyze_endpoint(args, cohort, arm_id, output):
    source = (args.unsmoothed_root / arm_id if cohort == "unsmoothed" else
              args.smoothed_root / ("five_word_joint_acquisition--" + arm_id))
    request = json.loads((source / "request.json").read_text())
    arm = request["arm"]
    started = time.monotonic()
    with arm_runtime(arm):
        fixture, packet, context, checks = load_fixture(args.runtime_root, source)
        stream_before = state_digest(context.streams.state_dict())
        base = exact_metrics(fixture)
        archived = torch.load(source / "observed-records.pt", map_location="cpu", weights_only=False)[-1]
        for key in ("quality_fraction", "modes", "reconstruction_exact", "minimum_reconstruction_token_probability"):
            if abs(float(base["metrics"][key]) - float(archived["metrics"][key])) > 1e-5:
                raise RuntimeError(f"Archived clean observer mismatch in {key}")
        expectation = expectation_control(fixture)
        delta, used, losses, after_d = public_step(fixture, set(ROLES))
        stats = step_statistics(fixture, delta, used)
        first_after = exact_metrics(fixture)
        reset(fixture, packet, context)
        probes = {}
        for mask in args.protocol["role_masks"]:
            reset(fixture, packet, context)
            observations = []
            for step in range(1, args.protocol["updates_per_probe"] + 1):
                public_step(fixture, set(mask))
                observed = exact_metrics(fixture)
                observations.append(dict(step=step, **observed))
            probes["+".join(mask)] = dict(first=observations[0], final=observations[-1],
                passing_checks=sum(x["passed"] for x in observations),
                first_failure_step=next((x["step"] for x in observations if not x["passed"]), None))
            torch.save(observations, output / (cohort + "--" + arm_id + "--" + "_".join(mask) + ".pt"))
        reset(fixture, packet, context)
        raw, _ = field(fixture)
        # Compare expected-before and actual-after-D raw gradients separately.
        sequential_change = {role: float((vector(raw[role])-vector(used[role])).norm()) /
                             max(float(vector(raw[role]).norm()), 1e-300) for role in ROLES[:3]}
        jacobian = projected_jacobian(fixture, delta)
        if state_digest(context.streams.state_dict()) != stream_before:
            raise RuntimeError("Diagnostic consumed a named RNG stream")
        result = dict(cohort=cohort, arm=arm, endpoint_step=fixture.completed_steps,
            input_files={name: dict(sha256=sha(source/name), bytes=(source/name).stat().st_size)
                         for name in ("state.pt", "observed-records.pt", "request.json", "source-manifest.json")},
            original_recipe=packet["fixture"]["recipe"], restore_checks=checks,
            original_endpoint_metrics=archived["metrics"], exact_uniform_endpoint=base,
            expected_loss_control=expectation, raw_losses=losses, losses_after_critic=after_d,
            expected_role_step=stats, relative_raw_gradient_change_from_critic_step=sequential_change,
            first_full_step=first_after, role_probes=probes, projected_raw_game_jacobian=jacobian,
            rng_draws_added=0, wall_seconds=time.monotonic()-started)
    write(output / (cohort + "--" + arm_id + ".json"), result)
    print(json.dumps(dict(event="endpoint_complete", cohort=cohort, arm=arm_id,
          original_pass=base["passed"], first_step_pass=first_after["passed"],
          all_final_pass=probes["G+E+prior+D"]["final"]["passed"],
          jacobian_identified=jacobian["identified"], wall_seconds=result["wall_seconds"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--unsmoothed-root", type=Path, required=True)
    parser.add_argument("--smoothed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Refuse to overwrite existing diagnostic")
    args.output.mkdir(parents=True, exist_ok=True)
    args.protocol = json.loads((HERE / "protocol.json").read_text())
    source_receipt = runtime_receipt(args.runtime_root)
    sys.path.insert(0, str(args.runtime_root))
    global torch, WordFixture, join, score_words, state_digest, dualnorm
    import torch
    from benchmarks.toy_audit.api_images import WordFixture, _join_words as join, score_words
    from experiments.forge.state import state_digest
    import particlegan.optim.dualnorm as dualnorm
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires reserved physical GPU1; no CPU fallback")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    controls = mathematical_controls()
    write(args.output / "execution.json", dict(protocol_sha256=sha(HERE/"protocol.json"),
        script_sha256=sha(__file__), source_files=source_receipt, runtime_commit=RUNTIME_COMMIT,
        python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0), visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        script_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()))
    results = []
    for label in args.protocol["cohorts"]:
        if time.monotonic()-started > args.protocol["maximum_runtime_seconds"]:
            raise RuntimeError("Bounded diagnosis exhausted declared runtime")
        cohort, arm = label.split("/")
        results.append(analyze_endpoint(args, cohort, arm, args.output))
    readout = dict(protocol_id=args.protocol["id"], qualification_input=False,
        historical_training_updates=0, diagnostic_logical_updates=384,
        finite_difference_optimizer_updates=0, sampling_draws=0,
        controls=controls, endpoints=results, wall_seconds=time.monotonic()-started,
        execution=json.loads((args.output/"execution.json").read_text()))
    write(args.output/"readout.json", readout)
    print(json.dumps(dict(event="complete", endpoints=len(results), wall_seconds=readout["wall_seconds"])), flush=True)


if __name__ == "__main__":
    main()
