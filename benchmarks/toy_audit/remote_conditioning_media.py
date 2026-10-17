"""PR246 goal GIF from retained observations, with no model execution.

The original full campaign's three observations and separate saved-endpoint
contrast diagnostic keep their own authorities. --fresh can export the same
public caller's new complete protocol; it never invokes that caller itself.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path

from PIL import Image

from .recent_toy_media import Inputs, finite_health, sha, write

ROOT = Path(__file__).resolve().parents[2]
HEAD = "b35dee597943c95499ca4a94d3cd141bd19f8fda"
CARD = "docs/routed_remote_conditioning_results_20261002.json"
CARD_SHA = "c0e70b1b9b47866b49ea487e97460de42301d1bce1b82f4c45bc9c80e21b0996"
PROTOCOL = "examples/routed_remote_conditioning_protocol.json"
PROTOCOL_SHA = "ccc7c3477cf7edc6a417044e321f583b9ddb989f86a4e1d9db27e38ef6da4c6c"
DRIVER = "examples/routed_remote_conditioning.py"
ARMS, MEDIA_STEPS, STEPS, SECONDS = ("G16", "G64"), (0, 100, 200), 200, 60
IDENTITY_GUARD = "sha256 of every frozen package/driver/contract/guide source"


def read(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


def number(value, *, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or positive
        and value <= 0
    ):
        raise ValueError("finite nonnegative numeric observation required")
    return value


def load_card():
    if sha(ROOT / CARD) != CARD_SHA:
        raise ValueError("PR246 public card differs from the reviewed exact head")
    return read(ROOT / CARD)


def active_sources():
    if sha(ROOT / PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("changed publication protocol identity")
    protocol = read(ROOT / PROTOCOL)
    hashes = protocol["source_hashes"]
    if not hashes or DRIVER not in hashes:
        raise ValueError("complete public caller source manifest required")
    for name, expected in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or sha(ROOT / path) != expected:
            raise ValueError(f"public source drift: {name}")
    return {"protocol_sha256": PROTOCOL_SHA, "source_hashes": deepcopy(hashes)}


def metric_checks(arms, variance):
    """Verify agreement with original endpoint inequalities; no model scorer."""
    checks = {}
    for step in MEDIA_STEPS[1:]:
        for name in ARMS:
            checks[f"{name}_converges_at{step}"] = (
                arms[name]["evaluations"][str(step)]["live_mask_error"]
                <= 0.9 * arms[name]["initial"]["live_mask_error"]
            )
        checks[f"G64_better_at{step}"] = (
            arms["G64"]["evaluations"][str(step)]["live_mask_error"]
            <= 0.9 * arms["G16"]["evaluations"][str(step)]["live_mask_error"]
        )
    for name in ARMS:
        checks[name + "_beats_codeblind_bound"] = (
            arms[name]["evaluations"][str(STEPS)]["live_mask_error"] <= 0.5 * variance
        )
    return checks


def verify_contrast(point, total, variance):
    finite_health(point)
    for key in (
        "pair_MSE",
        "midpoint_loss",
        "conditional_error",
        "conditional_error_over_V",
        "predicted_contrast_power_over_V",
        "target_contrast_power_over_V",
        "decomposition_identity_error",
    ):
        number(point[key])
    if type(point["alpha"]) not in (int, float) or not math.isfinite(point["alpha"]):
        raise ValueError("finite paired contrast coefficient required")
    if (
        abs(point["pair_MSE"] - total) > 1e-6
        or abs(point["pair_MSE"] - point["midpoint_loss"] - point["conditional_error"])
        > 1e-6
        or point["decomposition_identity_error"] > 1e-6
        or not math.isclose(
            point["conditional_error_over_V"],
            point["conditional_error"] / variance,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "paired contrast arithmetic differs from its recorded total/variance"
        )


def verify(runs, card, inputs, *, decomposition=None, fresh=False):
    runs = Path(runs)
    authority = card["provenance"]
    active = active_sources() if fresh else None
    protocol = inputs.read(
        runs / "protocol.json",
        active["protocol_sha256"]
        if fresh
        else authority["original_scientific_protocol_sha256"],
    )
    inputs.bind(
        runs / "source.py",
        active["source_hashes"][DRIVER]
        if fresh
        else authority["original_scientific_driver_sha256"],
    )
    if (
        protocol["external_steps_per_arm"] != STEPS
        or protocol["external_total_seconds"] != SECONDS
        or protocol["threads"] != 1
        or protocol["arms"]
        != {
            "G16": {"D_batch": 16, "G_batch": 16},
            "G64": {"D_batch": 16, "G_batch": 64},
        }
    ):
        raise ValueError("changed original API host or full400-update protocol")
    result = inputs.read(
        runs / "result.json", None if fresh else authority["original_result_sha256"]
    )
    finite_health(result)
    identity = result.get("identity", {})
    checkout = identity.get("checkout_git_sha")
    if (
        set(identity)
        != {
            "checkout_git_sha",
            "reference_package_git_sha",
            "identity_guard",
            "verified_source_files",
        }
        or identity.get("reference_package_git_sha") != protocol["package_git_sha"]
        or identity.get("identity_guard") != IDENTITY_GUARD
        or type(identity.get("verified_source_files")) is not int
        or identity.get("verified_source_files") != len(protocol["source_hashes"])
        or not isinstance(checkout, str)
        or len(checkout) != 40
        or any(char not in "0123456789abcdef" for char in checkout)
    ):
        raise ValueError("recorded caller source identity contradicts its protocol")
    if (
        result["failure"] is not None
        or result["status"] not in {"passed", "failed"}
        or set(result["arms"]) != set(ARMS)
        or result["committed_updates"] != {name: STEPS for name in ARMS}
    ):
        raise ValueError(
            "partial or invalid original campaign cannot export complete goal media"
        )
    if not 0 < number(result["elapsed_seconds"]) <= SECONDS:
        raise ValueError("original fixed wall budget exceeded")
    variance = number(result["teacher_nonlocal_variance"], positive=True)
    if (
        variance < 1e-8
        or variance != card["teacher_nonlocal_variance"]
        or any(
            result[key] != card[key]
            for key in ("fixture_hash", "teacher_hash", "calibration")
        )
    ):
        raise ValueError("fixed teacher, held fixture or fit-only units differ")
    curves, panels = {}, {}
    for name in ARMS:
        arm = result["arms"][name]
        path = inputs.bind(
            runs / (name + ".jsonl")
        )  # export-time identity; not in original card
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if (
            arm["steps"] != STEPS
            or [row["step"] for row in rows] != list(range(1, STEPS + 1))
            or any(
                row["d_batch"] != 16
                or row["g_batch"] != int(name[1:])
                or row["dense_rows"] != 128
                for row in rows
            )
        ):
            raise ValueError("missing or changed actual native update coverage")
        for row in rows:
            finite_health(row)
            for key in ("loss_g", "loss_d", "penalty", "sigma"):
                if type(row[key]) not in (int, float) or not math.isfinite(row[key]):
                    raise ValueError("nonfinite native health evidence")
            if len(row["quadratic_weights"]) != 2:
                raise ValueError("both finite quadratic channel weights required")
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in row["quadratic_weights"]
            ):
                raise ValueError("both finite quadratic channel weights required")
        panels[name] = [row["caller_panel_sha256"] for row in rows]
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
            for value in panels[name]
        ):
            raise ValueError("complete caller panel identities required")
        h, control = arm["health"], arm["control"]
        if (
            h["finite_steps"] != STEPS
            or h["min_dense_rows"] != 128
            or h["ka2_applied_calls"] <= 0
            or any(
                h.get(key) is not True
                for key in (
                    "ownership",
                    "frozen_host",
                    "critic_score_active_gradients",
                    "generator_live_gradients",
                )
            )
            or control["rows"]["counters"]["updates"] != STEPS
            or any(control["counters"][key] <= 0 for key in ("evals", "probes"))
        ):
            raise ValueError("full public native health prerequisite failed")
        if set(arm["evaluations"]) != {"100", "200"}:
            raise ValueError("exact actual evaluation checkpoints required")
        points = [
            {"step": 0, **arm["initial"]},
            *[
                {"step": step, **arm["evaluations"][str(step)]}
                for step in MEDIA_STEPS[1:]
            ],
        ]
        for point in points:
            for key in ("live_mask_error", "live_full_error", "served_mask_error"):
                number(point[key])
            if point["codeblind_mask_lower_bound"] != variance or point[
                "served_source"
            ] not in {"fast", "averaged"}:
                raise ValueError("changed original evaluation units or serving cohort")
        number(points[0]["live_mask_error"], positive=True)
        if not fresh and (
            arm["initial"] != card["arms"][name]["initial"]
            or arm["evaluations"] != card["arms"][name]["evaluations"]
        ):
            raise ValueError(
                "displayed observations differ from original public authority"
            )
        curves[name] = points
    if panels["G16"] != panels["G64"]:
        raise ValueError("actual common caller histories differ")
    checks = metric_checks(result["arms"], variance)
    prerequisite_keys = {
        "source_exact",
        "teacher_qualified",
        "initial_owners_match",
        "caller_streams_match",
        "fixed400_complete",
        "within60s",
        "G16_native_health",
        "G64_native_health",
    }
    if (
        set(result["checks"]) != set(checks) | prerequisite_keys
        or any(result["checks"][key] is not True for key in prerequisite_keys)
        or any(result["checks"][key] is not value for key, value in checks.items())
        or result["status"] != ("passed" if all(checks.values()) else "failed")
    ):
        raise ValueError(
            "recorded scientific verdict contradicts unchanged original gates"
        )
    contrast, contrast_points = {}, {}
    if fresh:
        if decomposition is not None:
            raise ValueError(
                "fresh observations cannot borrow the original endpoint diagnostic"
            )
        for name in ARMS:
            contrast_points[name] = []
            for observation in curves[name]:
                point = observation.get("paired_decomposition")
                if not isinstance(point, dict):
                    raise TypeError(
                        "fresh caller's existing paired observations are required"
                    )
                verify_contrast(point, observation["live_mask_error"], variance)
                contrast_points[name].append(
                    {"step": observation["step"], **deepcopy(point)}
                )
            contrast[name] = deepcopy(curves[name][-1]["paired_decomposition"])
        contrast_scope = "Descriptive algebra on this fresh caller's existing0/100/200 predictions; no extra forward."
    else:
        if decomposition is None:
            raise ValueError(
                "the separately attested saved200 contrast diagnostic is required"
            )
        directory = Path(decomposition)
        diagnostic = inputs.read(
            directory / "result.json",
            authority["saved_endpoint_decomposition_result_sha256"],
        )
        inputs.bind(directory / "decompose.py", diagnostic["source_sha256"])
        inputs.bind(directory / "protocol.json", diagnostic["protocol_sha256"])
        published = card["saved_endpoint_paired_decomposition"]
        if (
            any(diagnostic[key] != published[key] for key in diagnostic)
            or diagnostic["native_updates"] != 0
            or diagnostic["optimizer_steps"] != 0
            or diagnostic["autograd_calls"] != 0
            or diagnostic["scientific_training"] is not False
            or diagnostic["V"] != variance
            or set(diagnostic["arms"]) != set(ARMS)
        ):
            raise ValueError("changed separate saved-endpoint diagnostic authority")
        for name in ARMS:
            point = diagnostic["arms"][name]
            verify_contrast(point, curves[name][-1]["live_mask_error"], variance)
            if point["public_native_and_caller_replay_exact"] is not True:
                raise ValueError("saved-endpoint replay prerequisite missing")
            contrast[name] = deepcopy(point)
            contrast_points[name] = [{"step": STEPS, **deepcopy(point)}]
        contrast_scope = "Separate previously executed saved200 diagnostic (9 forwards,0 updates); endpoint only."
    return {
        "scientific_status": "PASS" if result["status"] == "passed" else "FAIL",
        "cohort": "fresh complete public protocol" if fresh else "original frozen V2",
        "actual_updates": list(MEDIA_STEPS),
        "native_updates_per_arm": STEPS,
        "original_checks": deepcopy(result["checks"]),
        "failed_original_checks": [k for k, v in checks.items() if not v],
        "fixture_hash": result["fixture_hash"],
        "teacher_hash": result["teacher_hash"],
        "calibration": deepcopy(result["calibration"]),
        "V": variance,
        "evaluations": curves,
        "contrast_endpoint": contrast,
        "contrast_observations": contrast_points,
        "contrast_scope": contrast_scope,
        "contrast_available_steps": list(MEDIA_STEPS) if fresh else [STEPS],
        "elapsed_seconds": result["elapsed_seconds"],
        "training_identity": deepcopy(result["identity"]),
        "source_manifest": deepcopy(protocol["source_hashes"]),
        "observation_limit": "No intermediate model image arrays retained; GIF uses recorded metric points only.",
    }


def render(data, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    curves, images = data["evaluations"], []
    maximum = max(
        point["live_mask_error"] / data["V"]
        for curve in curves.values()
        for point in curve
    )
    ratios = [
        b["live_mask_error"] / a["live_mask_error"] if a["live_mask_error"] else None
        for a, b in zip(curves["G16"], curves["G64"])
    ]
    ratio_values = [0.85, 1.025, *[value for value in ratios if value is not None]]
    fresh = data["cohort"] == "fresh complete public protocol"
    relative_max = max(
        point["live_mask_error"] / curve[0]["live_mask_error"]
        for curve in curves.values()
        for point in curve
    )
    contrast_max = max(
        point["conditional_error_over_V"]
        for curve in data["contrast_observations"].values()
        for point in curve
    )
    for index, step in enumerate(MEDIA_STEPS):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
        for arm, color in (("G16", "#75869b"), ("G64", "#be375d")):
            points = curves[arm][: index + 1]
            x, y = [p["step"] for p in points], [p["live_mask_error"] for p in points]
            axes[0, 0].plot(
                x,
                [v / curves[arm][0]["live_mask_error"] for v in y],
                marker="o",
                color=color,
                label=arm,
            )
            axes[1, 0].plot(
                x, [v / data["V"] for v in y], marker="o", color=color, label=arm
            )
        axes[0, 0].axhline(
            0.9, linestyle="--", color="#333333", label="Original 100+200 limit"
        )
        axes[0, 0].set(
            title="Each arm reduces held error at least 10%",
            ylabel="Error / own initial error",
            ylim=(0, max(1.08, relative_max * 1.08)),
        )
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].plot(
            MEDIA_STEPS[: index + 1], ratios[: index + 1], marker="o", color="#be375d"
        )
        axes[0, 1].axhline(
            0.9, linestyle="--", color="#333333", label="Original 100+200 limit"
        )
        axes[0, 1].axhline(1, color="#aaaaaa", linewidth=0.8)
        axes[0, 1].set(
            title="Does G64 gain at least 10% over G16?",
            ylabel="G64 error / G16 error",
            ylim=(min(ratio_values), max(ratio_values)),
        )
        if ratios[index] is None:
            axes[0, 1].text(
                0.5,
                0.55,
                "Zero G16 error: ratio undefined;\noriginal direct-error gate retained",
                transform=axes[0, 1].transAxes,
                ha="center",
                fontsize=8,
            )
        axes[0, 1].legend(fontsize=8)
        axes[1, 0].axhline(
            0.5, linestyle="--", color="#333333", label="Original 200 limit"
        )
        axes[1, 0].set(
            title="Original total-error marker-blind bound",
            ylabel="Held patch error / V",
            ylim=(0.25, maximum * 1.2),
            yscale="log",
        )
        axes[1, 0].legend(fontsize=8)
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
            ax.set(
                xlim=(0, STEPS), xticks=MEDIA_STEPS, xlabel="Actual API updates per arm"
            )
            ax.grid(alpha=0.15)
        ax = axes[1, 1]
        ax.axhline(
            0.5,
            linestyle="--",
            color="#333333",
            label="Separate contrast diagnostic limit",
        )
        ax.axhline(1, color="#aaaaaa", linewidth=0.8, label="Marker-blind reference")
        ax.set(
            title="Paired contrast: actual observed checkpoint"
            if fresh
            else "Paired contrast: saved 200 endpoint only",
            ylabel="Contrast error / V",
            ylim=(0, max(1.2, contrast_max * 1.15)),
            xticks=(0, 1),
            xticklabels=ARMS,
        )
        if step in data["contrast_available_steps"]:
            values = [
                next(
                    point
                    for point in data["contrast_observations"][arm]
                    if point["step"] == step
                )["conditional_error_over_V"]
                for arm in ARMS
            ]
            ax.bar((0, 1), values, color=("#75869b", "#be375d"), width=0.5)
            for x, value in enumerate(values):
                ax.text(x, value + 0.03, f"{value:.6f}", ha="center", fontsize=9)
        else:
            ax.text(
                0.5,
                0.58,
                "No retained contrast observation\nat this earlier checkpoint",
                transform=ax.transAxes,
                ha="center",
                fontsize=10,
            )
        ax.legend(fontsize=7, loc="lower right")
        fig.suptitle(
            "Does the routed global path learn the observed remote cue?", fontsize=13
        )
        fig.tight_layout(rect=(0, 0.22, 1, 0.94))
        failed = "; ".join(data["failed_original_checks"])
        fig.text(
            0.045,
            0.15,
            f"Full 400-update campaign {data['scientific_status']} | actual checkpoint {step}/200 per arm",
            fontsize=11,
            color="#b71c1c" if data["scientific_status"] == "FAIL" else "#187a45",
            weight="bold",
        )
        fig.text(
            0.045,
            0.106,
            f"{data['cohort']} | clean held 64 patch MSE; MSE never trained | V={data['V']:.9g}\n"
            "Arms ran sequentially; G64 uses 4x G rows. Only actual 0/100/200 observations; lines join recorded points.",
            fontsize=8,
        )
        fig.text(
            0.045,
            0.055,
            (
                "Fresh caller's existing paired algebra is available at the actual 0/100/200 observations.\n"
                if fresh
                else "Separate contrast algebra is available only at 200; earlier contrast is unknown.\n"
            )
            + "Total error can be dominated by midpoint error. Clean teacher capacity does not certify native noisy-code capacity.",
            fontsize=8,
        )
        fig.text(
            0.045,
            0.018,
            "Original failed metric checks: " + (failed or "none"),
            fontsize=7.2,
        )
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=100)
        plt.close(fig)
        images.append(Image.open(buffer).convert("RGB"))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=[1000, 1100, 2300],
        loop=0,
        optimize=False,
    )


def exporter_source():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    names = (
        Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "benchmarks/toy_audit/recent_toy_media.py",
    )
    files = {name: sha(ROOT / name) for name in names}
    for name, expected in files.items():
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(
                "commit the exact exporter/dependency before final media generation"
            )
    return {"commit": commit, "files_sha256": files}


def export(runs, output, *, decomposition=None, fresh=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("media export requires a new output directory")
    source = exporter_source()
    card = load_card()
    inputs = Inputs()
    inputs.bind(ROOT / CARD, CARD_SHA)
    data = verify(runs, card, inputs, decomposition=decomposition, fresh=fresh)
    output.mkdir(parents=True)
    goal = output / "goal.gif"
    render(data, goal)
    with Image.open(goal) as gif:
        frames = gif.n_frames
    if (
        frames != len(MEDIA_STEPS)
        or not inputs.unchanged()
        or exporter_source() != source
    ):
        raise ValueError(
            "export altered inputs/source or lost actual observation frames"
        )
    receipt = {
        "schema": "pr246_observation_goal_media_v1",
        "reviewed_pr_head": HEAD,
        "scientific_status": data["scientific_status"],
        "original_scientific_checks": data["original_checks"],
        "data": data,
        "input_files": inputs.files,
        "exporter_source": source,
        "original_inputs_unchanged": True,
        "training_updates": 0,
        "model_forwards": 0,
        "new_draws": 0,
        "metric_rescoring": False,
        "model_image_arrays_available": False,
        "goal_gif": {
            "path": "goal.gif",
            "sha256": sha(goal),
            "bytes": goal.stat().st_size,
            "frames": frames,
            "actual_updates": list(MEDIA_STEPS),
        },
    }
    write(output / "receipt.json", receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument(
        "--decomposition",
        type=Path,
        help="separate originally attested saved200 diagnostic",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="new complete output from the unchanged published public API caller",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = export(
        args.runs, args.output, decomposition=args.decomposition, fresh=args.fresh
    )
    print(
        json.dumps(
            {
                "export": "COMPLETE",
                "scientific_status": receipt["scientific_status"],
                "actual_frames": receipt["goal_gif"]["frames"],
                "training_updates": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
