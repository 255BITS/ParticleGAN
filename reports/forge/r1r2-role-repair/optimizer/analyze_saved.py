"""Read-only role moments and analytic critic fields from exact saved Ring16 states."""
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    values = values.double().flatten()
    return {"count": len(values), "min": float(values.min()),
            "median": float(values.median()), "mean": float(values.mean()),
            "p90": float(values.quantile(.9)), "max": float(values.max())}


def critic_field(points, state, means):
    """Exact analytic input derivative of the saved two-hidden-layer Fourier MLP.

    Float64 functional arithmetic; no module construction, samples or updates.
    Labels retain the original float32 nearest-component assignment.
    """
    labels = torch.cdist(points, points.new_tensor(means)).argmin(1)
    x = points.double()
    frequencies = state["freqs"].double()
    phase = x.unsqueeze(-1) * frequencies
    features = torch.cat([x, phase.sin().flatten(1), phase.cos().flatten(1)], 1)
    feature_jacobian = x.new_zeros(len(x), 10, 2)
    feature_jacobian[:, :2] = torch.eye(2, dtype=x.dtype)
    for coordinate in range(2):
        feature_jacobian[:, 2 + coordinate * 2:4 + coordinate * 2, coordinate] = (
            phase[:, coordinate].cos() * frequencies)
        feature_jacobian[:, 6 + coordinate * 2:8 + coordinate * 2, coordinate] = (
            -phase[:, coordinate].sin() * frequencies)
    w = lambda i: state[f"net.{i}.weight"].double()
    b = lambda i: state[f"net.{i}.bias"].double()
    h1 = features @ w(0).T + b(0)
    h2 = torch.nn.functional.leaky_relu(h1, .2) @ w(2).T + b(2)
    gradient_features = torch.einsum(
        "oi,bi,ij,bj,jk->bk", w(4),
        torch.where(h2 > 0, torch.ones_like(h2), torch.full_like(h2, .2)),
        w(2), torch.where(h1 > 0, torch.ones_like(h1), torch.full_like(h1, .2)), w(0))
    gradients = torch.einsum("bi,bij->bj", gradient_features, feature_jacobian)
    with torch.enable_grad():
        control = x[:16].clone().requires_grad_(True)
        control_phase = control.unsqueeze(-1) * frequencies
        control_features = torch.cat(
            [control, control_phase.sin().flatten(1), control_phase.cos().flatten(1)], 1)
        control_h1 = torch.nn.functional.leaky_relu(control_features @ w(0).T + b(0), .2)
        control_h2 = torch.nn.functional.leaky_relu(control_h1 @ w(2).T + b(2), .2)
        control_score = control_h2 @ w(4).T + b(4)
        reference = torch.autograd.grad(control_score.sum(), control)[0]
    derivative_error = float((reference - gradients[:16]).abs().max())
    assert derivative_error < 1e-12
    rows = []
    means = torch.tensor(means, dtype=torch.float64)
    for index, mean in enumerate(means):
        members, field = x[labels == index], gradients[labels == index]
        centered = members - members.mean(0)
        radial = mean / mean.norm()
        tangent = torch.stack([-radial[1], radial[0]])
        row = {"component": index, "count": len(members)}
        for name, direction in (("radial", radial), ("tangent", tangent)):
            offset, component = centered @ direction, field @ direction
            row[f"{name}_gradient_rms"] = float(component.square().mean().sqrt())
            row[f"{name}_restoring_slope"] = float(
                -(offset * component).mean() / offset.square().mean())
        rows.append(row)
    return {"analytic_vs_functional_autograd_max_abs_error": derivative_error,
            "input_gradient_norm": summarize(gradients.norm(dim=1)),
            "components": rows,
            "positive_radial_restoring_components": sum(r["radial_restoring_slope"] > 0 for r in rows),
            "positive_tangent_restoring_components": sum(r["tangent_restoring_slope"] > 0 for r in rows)}


def optimizer_moments(saved):
    rows = []
    for player, optimizer in zip(("generator", "critic"), saved["trainer"]["optimizers"], strict=True):
        for index, group in enumerate(optimizer["param_groups"]):
            role = "prior" if player == "generator" and index == 1 else player
            first, second, ratios, steps = [], [], [], []
            for parameter in group["params"]:
                state = optimizer["state"][parameter]
                m, v = state["exp_avg"].double(), state["exp_avg_sq"].double()
                maximum = state.get("max_exp_avg_sq", state["exp_avg_sq"]).double()
                step = int(state["step"])
                beta1, beta2 = group["betas"]
                denominator = (maximum / (1 - beta2 ** step)).sqrt() + group["eps"]
                displacement = -group["lr"] * m / (1 - beta1 ** step) / denominator
                first.append(m.flatten()); second.append(v.flatten())
                ratios.append(((maximum.sqrt() + group["eps"]) / (v.sqrt() + group["eps"])).flatten())
                steps.append(displacement)
            joined = torch.cat([v.flatten() for v in steps])
            row = {"role": role, "betas": list(group["betas"]), "amsgrad": group["amsgrad"],
                   "lr": group["lr"], "parameter_count": joined.numel(),
                   "first_moment_l2": float(torch.cat(first).norm()),
                   "second_moment_total": float(torch.cat(second).sum()),
                   "retained_max_vs_current_denominator_ratio": summarize(torch.cat(ratios)),
                   "reconstructed_last_native_adam_displacement_l2": float(joined.norm())}
            if role == "prior":
                displacement = steps[0]
                row["rows_with_nonzero_moment_displacement"] = int((displacement.norm(dim=1) > 1e-12).sum())
                row["row_displacement_norm"] = summarize(displacement.norm(dim=1))
            rows.append(row)
    return rows


def main():
    helper = load(ROOT / "reports/forge/recipe-prior-refactor/workflow.py", "role_saved_helpers")
    publisher = helper.load_publisher(ROOT)
    renderer = publisher._saved_renderer(ROOT)
    sources = [("constant", ROOT / "reports/forge/r1r2-tier1-search/publication/results.json"),
               ("amsgrad", ROOT / "reports/forge/r1r2-five-track-repair/publication/results.json"),
               ("momentum", ROOT / "reports/forge/r1r2-five-track-repair/publication/results.json")]
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    results = []
    with renderer.forbid_live_execution(), torch.no_grad():
        for role, publication_path in sources:
            publication = json.loads(publication_path.read_text())
            wanted = ("r1r2--169e9539b2dcfaab6b51e17f02907f28ea5b894f77155f7e688ab07a789de2b9"
                      if role == "constant" else
                      "r1r2-amsgrad-repair-v1" if role == "amsgrad" else "r1r2-five-track-momentum-v1")
            compact = next(row for row in publication["certified_results"]
                           if row["candidate_id"] == wanted and row["task_id"] == "ring16_acquisition")
            attempt = publisher.certified_attempt(ROOT, compact["attempt_id"])
            helper.saved_grading(attempt)
            row = attempt["result"]["task_results"][0]
            saved, proof = publisher.checkpoint(row)
            panels, inputs, _ = publisher.saved_outputs(attempt["request"]["tasks"]["ring16_acquisition"],
                                                       row["evidence"], attempt["local"])
            assert panels[-1]["step"] == 1600
            task = attempt["request"]["tasks"]["ring16_acquisition"]
            assert sha(ROOT / "benchmarks/locked_shared/mlp.py") == attempt["request"]["source"]["files"]["benchmarks/locked_shared/mlp.py"]
            results.append({"role": role, "attempt_id": compact["attempt_id"],
                            "candidate_id": wanted, "candidate_revision": compact["candidate_revision"],
                            "source_digest": compact["source_digest"], "checkpoint": proof,
                            "publication_sha256": sha(publication_path), "saved_sample_inputs": inputs,
                            "recorded_endpoint_metrics": compact["metrics"],
                            "optimizer_groups": optimizer_moments(saved),
                            "saved_critic_functional_input_gradient": critic_field(
                                panels[-1]["samples"], saved["trainer"]["models"]["D"],
                                task["execution"]["host_definition"]["means"])})
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    print(json.dumps({"schema_version": 1, "qualification_input": False,
                      "optimizer_updates_added": 0, "sampling_draws_added": 0,
                      "python_and_torch_rng_unchanged": True, "results": results,
                      "limits": "Endpoint role snapshots only. Beta1=0 exp_avg is the actual final parameter gradient; beta1=.5 exp_avg is its EMA, not a recovered raw gradient. Input derivatives are float64 functional recomputations on retained generated samples, not recorded training-time gradients. Restoring slopes are within-component descriptive covariances, not causal effects. Reconstructed Adam displacements approximate last native updates without float32 kernel rounding."},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
