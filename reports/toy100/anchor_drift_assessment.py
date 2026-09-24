"""Read-only drift and output-margin assessment of allocation warm/hold runs.

This never resumes a trainer, changes a proposal, or upgrades an experiment's
gate. It reads archived source/receipt bytes and the completed full snapshots.
Known ring means are used only for retrospective geometric interpretation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return dict(count=0, minimum=None, median=None, maximum=None, mean=None)
    if not all(math.isfinite(v) for v in values):
        raise FloatingPointError("nonfinite diagnostic value")
    return dict(count=len(values), minimum=values[0], median=statistics.median(values),
                maximum=values[-1], mean=statistics.mean(values))


def verify_bytes(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise RuntimeError(f"artifact hash changed: {path}")
    return raw


def optimizer_metrics(snapshot):
    import torch
    rows = {}
    for name in ("optimizer_d", "optimizer_g"):
        optimizer = snapshot[name]
        for index, group in enumerate(optimizer["param_groups"]):
            role = "d" if name == "optimizer_d" else "prior" if group.get("_comparison_prior") else "g"
            first, second, denominator, metric, proposal, steps = [], [], [], [], [], []
            beta1, beta2 = group["betas"]
            for key in group["params"]:
                state = optimizer["state"][key]
                step = int(state["step"])
                if step < 1:
                    raise ValueError("completed snapshot has no Adam moment update")
                m, v = state["exp_avg"].double(), state["exp_avg_sq"].double()
                if not torch.isfinite(m).all() or not torch.isfinite(v).all() or bool((v < 0).any()):
                    raise FloatingPointError("invalid completed Adam state")
                denom = (v / (1-beta2**step)).sqrt() + group["eps"]
                first.append(m.flatten()); second.append(v.flatten())
                denominator.append(denom.flatten())
                metric.append((group["lr"] / denom).flatten())
                proposal.append((group["lr"] * m / (1-beta1**step) / denom).flatten())
                steps.append(step)
            flat = lambda items: torch.cat(items)
            quantiles = lambda items: dict(zip(("min", "p50", "p90", "max"),
                torch.quantile(flat(items), torch.tensor([0., .5, .9, 1.], dtype=torch.float64)).tolist()))
            rows[role] = dict(group=index, parameters=flat(first).numel(), lr=group["lr"],
                eps=group["eps"], betas=group["betas"], moment_steps=sorted(set(steps)),
                first_moment_l2=float(flat(first).norm()), second_moment_l2=float(flat(second).norm()),
                bias_corrected_denominator=quantiles(denominator), coordinate_metric=quantiles(metric),
                last_moment_derived_unbounded_adam_proposal_l2=float(flat(proposal).norm()),
                scope="stored native Adam moments; not accepted joint-output parameter displacement")
    return rows


def correction_blocks(corrections, size=50, *, fit_start="native post-G parameters"):
    result = []
    for start in range(0, len(corrections), size):
        block = corrections[start:start+size]
        first = [row["fit"]["records"][0] for row in block if row["fit"]["records"]]
        all_records = [item for row in block for item in row["fit"]["records"]]
        result.append(dict(first_update=block[0]["step"], last_update=block[-1]["step"],
            fits=len(block), converged=sum(r["fit"]["status"] == "CONVERGED" for r in block),
            selection={name: sum(r["selected"] == name for r in block)
                       for name in ("joint_fit", "native_gan", "rest")},
            pre_loss=stats(r["pre_cost"] for r in block), native_loss=stats(r["native_cost"] for r in block),
            accepted_loss=stats(r["final_cost"] for r in block),
            native_to_pre_loss_ratio=stats(r["native_cost"] / r["pre_cost"] for r in block if r["pre_cost"] > 0),
            first_jacobian_singular_min=stats(r["singular_min"] for r in first),
            first_jacobian_singular_max=stats(r["singular_max"] for r in first),
            first_jacobian_condition=stats(r["singular_max"] / r["singular_min"] for r in first if r["singular_min"] > 0),
            observed_ranks=sorted(set(r["rank"] for r in all_records)),
            gn_iterations=stats(len(r["fit"]["records"]) for r in block),
            correction_parameter_displacement=stats(r["fit"]["actual_parameter_displacement"] for r in block),
            correction_output_displacement_max=stats(max(r["fit"]["actual_output_displacement"]) for r in block),
            correction_displacement_scope=f"{fit_start} to fitted state; selection may instead rest or accept native GAN"))
    return result


def output_margins(corrections, source_hashes):
    import torch
    from scipy.optimize import linear_sum_assignment
    from benchmarks.locked_shared import mode_hold
    from reports.toy100.anchor_invariant_region import population_hq_lower_bound
    if sha(Path(mode_hold.__file__).read_bytes()) != source_hashes["benchmarks/locked_shared/mode_hold.py"]:
        raise RuntimeError("offline host geometry source changed")
    means = mode_hold.ring_means().double()
    rows = []
    for row in corrections:
        centers = torch.tensor(row["centers"], dtype=torch.float64)
        target = torch.tensor(row["mm"]["target"], dtype=torch.float64)
        if len(centers) != len(means):
            rows.append(dict(step=row["step"], assessed=False, reason="inferred group count differs from host"))
            continue
        a, b = linear_sum_assignment(torch.cdist(centers, means).numpy())
        epsilon = float((centers[a]-means[b]).norm(dim=1).max())
        distance = torch.cdist(target, centers)
        target_error = float(distance.min(1).values.max())
        covered = len(set(distance.argmin(1).tolist())) == len(centers)
        rounded = float((target.float().double()-target).norm(dim=1).max())
        delta = row["fit"]["absolute_threshold"] + rounded + target_error
        tau = 64 * torch.finfo(torch.float64).eps * max(1., row["pre_cost"])
        eligible = row["fit"]["status"] == "CONVERGED" and covered
        radius = epsilon + math.sqrt(len(target) * (2*delta*delta+tau)) if eligible else None
        selection_valid = (row["selected"] == "rest" or
            eligible and row["final_cost"] <= row["fitted_cost"] + tau + 1e-15)
        rows.append(dict(step=row["step"], assessed=True, n_groups=len(centers), epsilon=epsilon,
            target_center_residual=target_error, target_covers_all_groups=covered,
            fit_eligible=eligible, selected=row["selected"], selection_consistent=selection_valid,
            delta_bound=delta, moving_radius_bound=radius,
            conditional_population_hq_lower_bound=None if radius is None else population_hq_lower_bound(radius)))
    moving = [r for r in rows if r.get("fit_eligible") and r["selected"] != "rest"]
    return dict(scope="posthoc true-center interpretation; moving proposals only; rest inherits its prior state",
        all_finite_center_errors_are_future_bounds=False, objective_rounding_error_assumed_zero=True,
        centroid_error=stats(r["epsilon"] for r in rows if r["assessed"]),
        moving_radius=stats(r["moving_radius_bound"] for r in moving),
        population_hq_lower_bound=stats(r["conditional_population_hq_lower_bound"] for r in moving),
        all_assessed=all(r["assessed"] for r in rows),
        selection_consistent=all(r.get("selection_consistent", False) for r in rows), rows=rows)


def assess_run(directory):
    import torch
    from reports.toy100.pr84_critic_refinement_capture import _sha
    declaration = json.loads((directory / "declaration.json").read_text())
    completed_gate = (directory / "summary.json").exists()
    summary_raw = (directory / ("summary.json" if completed_gate else "forks/summary.json")).read_bytes()
    summary = json.loads(summary_raw)
    if completed_gate and summary["source"] != declaration["source"]:
        raise RuntimeError("summary/source declaration mismatch")
    for name, digest in declaration["source"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe archived source path")
        verify_bytes(directory / "source" / name, digest)
    variants = {}
    for name in ("candidate", "disabled"):
        raw = (directory / "forks" / f"{name}.json").read_bytes()
        value = json.loads(raw)
        dynamics = value["dynamics_receipt"]
        state_path = directory / dynamics["final_state_file"]
        verify_bytes(state_path, dynamics["final_state_file_sha256"])
        state = torch.load(state_path, weights_only=True, map_location="cpu")
        if _sha(state) != dynamics["final_snapshot_sha256"]:
            raise RuntimeError("snapshot contents differ from accepted receipt")
        variants[name] = dict(raw_sha256=sha(raw), final_snapshot_sha256=_sha(state),
            final_state_file_sha256=dynamics["final_state_file_sha256"],
            status=value["status"], local_stability=value["local_stability"], long_hold=value["long_hold"],
            internal_state_blocks=dynamics["internal_state_blocks"], final_internal_state=dynamics["final_internal_state"],
            optimizer=optimizer_metrics(state), corrections=correction_blocks(
                dynamics.get("corrections", []), fit_start=dynamics.get("fit_start", "native post-G parameters")))
        if name == "candidate":
            variants[name]["output_margins"] = output_margins(dynamics["corrections"], declaration["source"])
    return dict(phase=declaration["phase"], method=declaration["method"], summary_sha256=sha(summary_raw),
        source=declaration["source"], gate_complete=completed_gate,
        status=summary["status"] if completed_gate else "INCOMPLETE_GATE",
        identity_cold_parity=summary["identity_cold_parity"],
        original_control_exact_parity=summary.get("original_control_exact_parity"),
        first200_training_parity=summary.get("first200_training_parity", summary.get("first200_parity")),
        first200_snapshot_reconciliation=summary.get("first200_snapshot_reconciliation"), variants=variants)


def affine_nullspace_example():
    """Exact fixed-J identity: correction at the native point retains ker J."""
    import torch
    j = torch.tensor([[1., 1., 0.], [0., 0., 1.]], dtype=torch.float64)
    inverse = torch.linalg.pinv(j)
    native = torch.tensor([1., -1., .2], dtype=torch.float64)
    target_delta = torch.tensor([.1, -.2], dtype=torch.float64)
    post_start = native + inverse @ (target_delta-j@native)
    pre_start = inverse @ target_delta
    inherited = (torch.eye(3, dtype=j.dtype)-inverse@j)@native
    if not torch.allclose(post_start, pre_start+inherited, atol=1e-14, rtol=0):
        raise AssertionError("affine nullspace decomposition failed")
    return dict(jacobian=j.tolist(), native=native.tolist(), target_delta=target_delta.tolist(),
        post_native_fit_displacement=post_start.tolist(), pre_state_fit_displacement=pre_start.tolist(),
        inherited_nullspace_displacement=inherited.tolist(),
        post_output=(j@post_start).tolist(), pre_output=(j@pre_start).tolist(),
        post_parameter_norm=float(post_start.norm()), pre_parameter_norm=float(pre_start.norm()),
        zero_target_pre_state_fit_displacement=(inverse@torch.zeros(2,dtype=j.dtype)).tolist(),
        scope="linear algebra example only; not attribution of the neural host")


def horizontal_rectangle(a=1., b=1., initial_z=0., repetitions=100):
    """Exact horizontal lift for F(x,y,z)=(x,y+x*z), not host training.

    dz=x*(du-z*dx)/(1+x²); therefore d(z*sqrt(1+x²))=x/sqrt(1+x²)*du.
    One closed output rectangle produces nonzero motion in the final fiber.
    """
    root = math.sqrt(1+a*a)
    z1 = initial_z/root
    z2 = (initial_z+a*b/root)/root
    final_z = initial_z+a*b/root
    vertices = [(0., 0., initial_z), (a, -a*z1, z1),
                (a, b-a*z2, z2), (0., b, final_z), (0., 0., final_z)]
    outputs = [(x, y+x*z) for x, y, z in vertices]
    return dict(output_map="F(x,y,z)=(x,y+xz)", a=a, b=b, parameter_vertices=vertices,
        output_vertices=outputs, fiber_shift_per_loop=a*b/root,
        repetitions=repetitions, final_z_after_repeated_loops=initial_z+repetitions*a*b/root,
        scope="continuous minimum-norm GN lift; bounded output loop, unbounded repeated parameter motion",
        not_a_claim_about_finite_step_host=True)


def fixed_readout_chart(snapshot):
    """Inspect a copied host's 12-code affine readout map; never optimize it.

    Native float32 hidden features are promoted only for the SVD. Full row
    rank gives representational feasibility at these fixed codes/features,
    not a training prescription, guarantee of good conditioning, or control
    of the critic. The saved state and caller's RNG must remain unchanged.
    """
    import torch
    from benchmarks.locked_shared import mode_hold
    from benchmarks.locked_shared.mlp import SimpleMLPGenerator
    from reports.toy100.pr84_critic_refinement_capture import _sha
    before, rng = _sha(snapshot), torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        model = SimpleMLPGenerator(mode_hold.Z_DIM, mode_hold.HIDDEN, mode_hold.N_HIDDEN, 2)
        model.load_state_dict({name.removeprefix("model."): value
                               for name, value in snapshot["generator"].items()})
        z = snapshot["prior"]["z"]
        hidden = model.net[:-1](z)
        features = torch.cat((hidden, torch.ones_like(hidden[:, :1])), dim=1).double()
        readout = torch.cat((model.net[-1].weight, model.net[-1].bias[:, None]), dim=1).double()
        original = model(z).double()
        reconstructed = features @ readout.T
        singular = torch.linalg.svdvals(features)
        rank = int((singular > 1e-6*singular[0]).sum())
        result = dict(state_sha256=before, rows=features.shape[0], columns=features.shape[1],
            native_feature_dtype=str(hidden.dtype), svd_dtype=str(features.dtype), rtol=1e-6,
            rank=rank, full_row_rank=rank == features.shape[0], singular_values=singular.tolist(),
            condition=float(singular[0]/singular[-1]),
            right_inverse_operator_norm=float(1/singular[-1]),
            right_inverse_frobenius_norm=float(singular.reciprocal().norm()),
            readout_parameter_l2=float(readout.norm()),
            reconstruction_max_abs_error=float((original-reconstructed).abs().max()),
            scope="fixed current hidden features and fixed 12 latent codes; affine last-layer weights plus bias")
        if not all(math.isfinite(result[key]) for key in (
                "condition", "right_inverse_operator_norm", "right_inverse_frobenius_norm")):
            raise FloatingPointError("fixed readout chart is numerically singular")
    if _sha(snapshot) != before or not torch.equal(rng, torch.get_rng_state()):
        raise RuntimeError("read-only feature chart changed saved state or caller RNG")
    result.update(saved_state_unchanged=True, caller_rng_unchanged=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--hold", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runs = [assess_run(args.warm)]
    if args.hold is not None:
        runs.append(assess_run(args.hold))
        if runs[0]["source"] != runs[1]["source"]:
            raise RuntimeError("warm/hold are not the same source")
    result = dict(scope="read-only internal-drift and conditional output assessment", shared_gate_eligible=False,
        source_sha256=sha(Path(__file__).read_bytes()), runs=runs, affine_example=affine_nullspace_example(),
        nonlinear_holonomy_example=horizontal_rectangle(),
        limitations=["no per-update accepted pre-to-post parameter vector is recorded",
                     "fit displacement is measured from the declared fit start, not necessarily the selected proposal",
                     "norm and singular-value growth alone do not identify nullspace drift",
                     "two final Adam snapshots do not identify the per-update source of drift",
                     "output-region bounds do not bound internal parameters or prove future sample-group recovery"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status="ASSESSMENT_COMPLETE", phases=[r["phase"] for r in runs],
                          output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
