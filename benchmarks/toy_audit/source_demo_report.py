"""Render real saved source-demo observations and compact final receipts."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .source_demos import digest, state_hash, write


WORDS = ("apple", "grape", "lemon", "melon", "berry")
CHARS = "abcdefghijklmnopqrstuvwxyz_ "
COLORS = {"live": "#ba1658", "ema": "#087e9b"}


def load(path):
    return json.loads(Path(path).read_text())


def rows(path):
    stream = path / "raw/observations.jsonl"
    return [json.loads(line) for line in stream.read_text().splitlines()] if stream.exists() else []


def final_status(summary):
    if not summary["measurement_complete"]:
        return f"{summary['status']}: incomplete evidence"
    return f"full-budget live {'PASS' if summary['live_pass'] else 'FAIL'} / EMA {'PASS' if summary['ema_pass'] else 'FAIL'}"


def attempt_summary(path, problem, profile="original"):
    """A missing final receipt cannot become a scientific endpoint."""
    if (path / "summary.json").exists():
        summary = load(path / "summary.json")
        summary.setdefault("profile", profile)
        summary["completed_update_pairs_exact"] = True
        return summary
    observed = rows(path)
    progress = load(path / "raw/progress.json") if (path / "raw/progress.json").exists() else {}
    markers = [("supervisor-error.json", "SUPERVISOR_ERROR"), ("interruption.json", "INTERRUPTED"),
               ("timeout.json", "TIMEOUT")]
    marker, status = next(((load(path / name), status) for name, status in markers if (path / name).exists()),
                          ({"cause": "No durable terminal execution receipt"}, "INTERRUPTED"))
    receipt = load(path / "source-receipt.json")
    recipe = None
    log = path / "execution.log"
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "recipe" in data:
                recipe = data["recipe"]
                break
    terminal = {cohort: {"passing_suffix": 0, "passed": False} for cohort in ("live", "ema")}
    for cohort in terminal:
        for shot in reversed(observed):
            if not shot[cohort]["passed"]:
                break
            terminal[cohort]["passing_suffix"] += 1
    final = observed[-1] if observed else None
    last_log_seconds = log.stat().st_mtime - (path / "source-receipt.json").stat().st_mtime if log.exists() else 0
    hard_timeout_lower_bound = receipt["wall_cap_seconds"] if status == "TIMEOUT" else 0
    return dict(problem=problem, profile=profile, status=status, error=marker,
                declared_budget=20000 if problem == "five_modes" else 1000,
                original_loop_update_count=20001 if problem == "five_modes" else 1000,
                completed_update_pairs=progress.get("completed_update_pairs", 0),
                completed_update_pairs_exact=False, seconds=None,
                recorded_wall_seconds_lower_bound=max(last_log_seconds, final["seconds"] if final else 0,
                                                      hard_timeout_lower_bound),
                wall_cap_seconds=receipt["wall_cap_seconds"], recipe=recipe,
                observations=len(observed), final=final, measurement_complete=False,
                live_pass=False, ema_pass=False, live_terminal=terminal["live"], ema_terminal=terminal["ema"])


def gate_status(summary, cohort):
    return "INCOMPLETE" if not summary["measurement_complete"] else "PASS" if summary[cohort + "_pass"] else "FAIL"


def diagnosis(summary, problem):
    if not summary["measurement_complete"]:
        return dict(status="INCOMPLETE", measured_failure="Execution did not reach the frozen full-budget endpoint",
                    optimizer_cause="Not established from a partial run")
    defects = {}
    for cohort in ("live", "ema"):
        metrics = summary["final"][cohort]
        failures = []
        if problem == "quickstart":
            if metrics["mean_error_sigma"] > .1:
                failures.append("Standardized mean error exceeds .10")
            if min(metrics["covariance_eigenvalues"]) < .85 or max(metrics["covariance_eigenvalues"]) > 1.15:
                failures.append("Standardized covariance eigenvalue is outside [.85,1.15]")
            if metrics["radial_ks"] > .075:
                failures.append("Radial CDF KS exceeds .075")
            if metrics["max_projection_ks"] > .06:
                failures.append("Maximum fixed projected CDF KS exceeds .06")
        if not summary[cohort + "_terminal"]["passed"]:
            failures.append("Five passing terminal observations are absent")
        defects[cohort] = failures
    return dict(status=gate_status(summary, "live"), measured_failure_components=defects,
                optimizer_cause="No optimizer mechanism is isolated by this single unchanged run")


def prefix_parity(interrupted, recovered):
    """Compare one identical saved prefix, without ranking scientific endpoints."""
    import torch
    prior = rows(interrupted)
    step = prior[-1]["step"]
    recovery = rows(recovered)
    index = next(i for i, shot in enumerate(recovery) if shot["step"] == step)
    checkpoints = (interrupted / f"raw/state-{len(prior)-1:03d}.pt",
                   recovered / f"raw/state-{index:03d}.pt")
    hashes = [state_hash(torch.load(path, weights_only=False, map_location="cpu")) for path in checkpoints]
    return dict(completed_update_pairs=step, exact_saved_state_match=hashes[0] == hashes[1],
                compared_state="All archived model/optimizer/RNG tensors and cursor fields",
                state_hashes=hashes, checkpoint_sha256=[digest(path) for path in checkpoints],
                purpose="Engineering prefix identity; no endpoint selection or resumability claim")


def image_of(figure):
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=80, facecolor="white")
    plt.close(figure)
    return Image.open(stream).convert("RGB")


def word_frame(shot, curve, raw, summary):
    figure = plt.figure(figsize=(11, 7))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.05, 1])
    table = figure.add_subplot(grid[0, :]); table.axis("off")
    lines = ["input     clean LIVE decode / min p(token)        EMA decode / min p(token)"]
    for i, word in enumerate(WORDS):
        values = []
        for cohort in ("live", "ema"):
            logits = raw[cohort + "_reconstruction_logits"][i].astype(np.float64)
            probabilities = np.exp(logits - logits.max(0, keepdims=True)); probabilities /= probabilities.sum(0, keepdims=True)
            decoded = "".join(CHARS[j] for j in probabilities.argmax(0))
            target = np.array([CHARS.index(c) for c in word + "_"])
            confidence = probabilities[target, np.arange(6)].min()
            values.append(f"{decoded:<6} / {confidence:6.3f}")
        lines.append(f"{word + '_':<8}  {values[0]:<28} {values[1]}")
    table.text(0.02, 0.85, "\n".join(lines), family="monospace", fontsize=12, va="top")
    table.text(0.02, 0.15, "All six characters are scored, including the '_' padding token.\n"
               "The original dashboard used EMA hard decoding; the new gate also tests confidence and mass.", fontsize=10)
    masses = figure.add_subplot(grid[1, 0])
    x = np.arange(6)
    for cohort, shift in (("live", -.17), ("ema", .17)):
        metrics = shot[cohort]
        masses.bar(x + shift, metrics["word_masses"] + [1 - metrics["quality_fraction"]],
                   width=.32, color=COLORS[cohort], label=cohort)
    masses.axhline(.2, color="black", linestyle="--", lw=1, label="target word mass .2")
    masses.set_xticks(x, WORDS + ("reject",), rotation=25); masses.set_ylim(0, 1)
    masses.set_title("Valid generated-word mass from actual prior draws"); masses.legend(fontsize=8)
    history = figure.add_subplot(grid[1, 1])
    for cohort in ("live", "ema"):
        history.plot([r["step"] for r in curve], [r[cohort]["mass_tv"] for r in curve],
                     color=COLORS[cohort], label=cohort + " mass TV")
        history.plot([r["step"] for r in curve], [r[cohort]["quality_fraction"] for r in curve],
                     color=COLORS[cohort], linestyle="--", label=cohort + " confident-word fraction")
    history.axhline(.1, color="black", linestyle=":", lw=1); history.axhline(.95, color="grey", linestyle=":", lw=1)
    history.set_xlim(0, 20001); history.set_ylim(-.03, 1.03); history.set_xlabel("completed D + GE update pairs")
    history.legend(fontsize=8); history.set_title("Fixed definition; complete terminal suffix required")
    checkpoint_gate = " / ".join(f"{cohort} {'PASS' if shot[cohort]['passed'] else 'FAIL'}" for cohort in ("live", "ema"))
    figure.suptitle(f"Five fixed words · observed update {shot['step']:,}/20,001\n"
                   f"Checkpoint gate: {checkpoint_gate}\nRun outcome: {final_status(summary)}", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .90))
    return image_of(figure)


def gaussian_frame(shot, curve, raw, summary, limits):
    figure = plt.figure(figsize=(12, 7))
    grid = figure.add_gridspec(2, 6, height_ratios=[1.4, 1])
    for i, cohort in enumerate(("live", "ema")):
        axis = figure.add_subplot(grid[0, i*3:(i+1)*3])
        cloud = raw[cohort]
        axis.scatter(cloud[:, 0], cloud[:, 1], s=3, alpha=.22, color=COLORS[cohort])
        for radius in (.2, .4, .6):
            axis.add_patch(plt.Circle((1, 1), radius, fill=False, color="black", lw=.7, alpha=.5))
        axis.scatter([1], [1], marker="+", color="black")
        axis.set_xlim(*limits); axis.set_ylim(*limits); axis.set_aspect("equal")
        axis.set_title(f"Clean {cohort}: actual 4,096 prior draws")
        metrics = shot[cohort]
        eig = metrics["covariance_eigenvalues"]
        axis.text(.03, .97, f"mean {metrics['mean_error_sigma']:.3f} σ; eig {eig[0]:.3f}–{eig[1]:.3f}\n"
                  f"radial KS {metrics['radial_ks']:.3f}; projected KS {metrics['max_projection_ks']:.3f}",
                  transform=axis.transAxes, va="top", fontsize=8,
                  bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    mean_axis = figure.add_subplot(grid[1, :2]); covariance_axis = figure.add_subplot(grid[1, 2:4]); ks_axis = figure.add_subplot(grid[1, 4:])
    for cohort in ("live", "ema"):
        steps = [r["step"] for r in curve]
        mean_axis.plot(steps, [r[cohort]["mean_error_sigma"] for r in curve], color=COLORS[cohort], label=cohort)
        eig = np.array([r[cohort]["covariance_eigenvalues"] for r in curve])
        covariance_axis.plot(steps, eig[:, 0], color=COLORS[cohort], label=cohort + " min")
        covariance_axis.plot(steps, eig[:, 1], color=COLORS[cohort], linestyle="--", label=cohort + " max")
        ks_axis.plot(steps, [r[cohort]["max_projection_ks"] for r in curve], color=COLORS[cohort], label=cohort + " projection")
        ks_axis.plot(steps, [r[cohort]["radial_ks"] for r in curve], color=COLORS[cohort], linestyle="--", label=cohort + " radial")
    mean_axis.axhline(.1, color="black", ls=":"); mean_axis.set_title("Mean error / sigma ≤.10")
    covariance_axis.axhspan(.85, 1.15, color="grey", alpha=.12); covariance_axis.set_title("Covariance eigenvalues .85–1.15")
    ks_axis.axhline(.06, color="black", ls=":"); ks_axis.axhline(.075, color="grey", ls=":"); ks_axis.set_title("Projection / radial KS ≤.06 / .075")
    for axis in (mean_axis, covariance_axis, ks_axis):
        axis.set_xlim(0, max(1, shot["step"])); axis.set_xlabel("completed updates"); axis.legend(fontsize=7)
    profile = "original batch2048" if summary.get("profile", "original") == "original" else "separate frozen batch128"
    checkpoint_gate = " / ".join(f"{cohort} {'PASS' if shot[cohort]['passed'] else 'FAIL'}" for cohort in ("live", "ema"))
    figure.suptitle(f"Gaussian quickstart · {profile} · observed update {shot['step']}/1,000\n"
                   f"Checkpoint gate: {checkpoint_gate}\nRun outcome: {final_status(summary)}", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .87))
    return image_of(figure)


def render(path, media, name, summary, observations):
    frames = []
    values = []
    if name == "quickstart":
        for i in range(len(observations)):
            with np.load(path / f"raw/cloud-{i:03d}.npz", allow_pickle=False) as raw:
                values.extend([raw["live"], raw["ema"]])
        limits = (min(-.6, float(min(v.min() for v in values)) - .1),
                  max(1.8, float(max(v.max() for v in values)) + .1))
    for i, shot in enumerate(observations):
        cloud_path = path / f"raw/cloud-{i:03d}.npz"
        if digest(cloud_path) != shot["cloud_sha256"]:
            raise ValueError("saved demo cloud identity differs")
        with np.load(cloud_path, allow_pickle=False) as raw:
            frames.append(word_frame(shot, observations[:i+1], raw, summary) if name == "five_modes"
                          else gaussian_frame(shot, observations[:i+1], raw, summary, limits))
    if not frames:
        return None
    stem = "source-five-modes" if name == "five_modes" else (
        "source-quickstart" if summary.get("profile", "original") == "original" else "source-quickstart-cpu128")
    gif, poster = media / (stem + ".gif"), media / (stem + ".png")
    media.mkdir(parents=True, exist_ok=True)
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=[140] * (len(frames)-1) + [1400], loop=0)
    frames[-1].save(poster)
    return dict(path="media/" + gif.name, poster="media/" + poster.name,
                gif_sha256=digest(gif), poster_sha256=digest(poster), frames=len(frames),
                actual_checkpoint_steps=[r["step"] for r in observations], interpolation=False)


def markdown(report):
    lines = ["# Original source-demo convergence evidence", "",
             "These two entries were source-only in the original 109-case review. This addendum runs the "
             "unaltered examples and package from develop `6ec7e5788e14ea15ddc3e16ac71110458108b6a6`, on CPU "
             "with one thread. Existing source-only receipts and quality ratings remain unchanged. No recipe, "
             "library or configuration was repaired. The strengthened evaluator is pinned separately; "
             "these are new diagnostic results, not Atlas or Forge qualification.", "",
             "| Example | Declared / actual full budget | Completed pairs / last scored | Live / EMA diagnostic | GIF |",
             "| --- | --- | --- | --- | --- |"]
    for case in report["cases"]:
        s, media = case["summary"], case["media"]
        outcome = final_status(s)
        gif = f"[actual checkpoints]({media['path']})" if media else "no scored checkpoint"
        lower_bound = "≥" if not s.get("completed_update_pairs_exact", True) else ""
        lines.append(f"| {case['name']} | {s['declared_budget']:,} / {s['original_loop_update_count']:,} | "
                     f"{lower_bound}{s['completed_update_pairs']:,} / {s['final']['step'] if s['final'] else 'none'} | {outcome} | {gif} |")
    lines.extend(["", "## Five-word interpretation", "",
                  "The original BiGAN-style joint critic loop has `range(total_steps + 1)`: its default "
                  "20,000 means **20,001 D/GE update pairs**, and dashboard step zero is already after the first "
                  "update. The source dashboard displays EMA reconstructions with hard decoding and strips "
                  "padding. The audit samples the actual prior distribution for both clean/live and EMA "
                  "cohorts, while separately reconstructing each canonical input in its correct paired row. "
                  "All six tokens, including underscore padding, enter the confidence and NLL measures. "
                  "It tests five-word mass, confidently decoded generated words and paired reconstruction; "
                  "it does not test language generation or held-out word/typo generalization.", ""])
    five = next(c for c in report["cases"] if c["problem"] == "five_modes")["summary"]
    if five["final"]:
        for cohort in ("live", "ema"):
            m = five["final"][cohort]
            lines.append(f"Final observed {cohort}: {m['modes']}/5 valid modes, confident-word fraction "
                         f"{m['quality_fraction']:.6f}, word/reject TV {m['mass_tv']:.6f}, paired reconstruction "
                         f"exact={m['reconstruction_exact']}, minimum correct-token probability "
                         f"{m['minimum_reconstruction_token_probability']:.6g}; "
                         f"{five[cohort + '_terminal']['passing_suffix']} consecutive passing terminal observations.")
    lines.extend(["", "## Gaussian interpretation", "",
                  "The quickstart targets `N((1,1), .04 I)` and has an original 1,000-update budget. The new "
                  "gate tests standardized mean, covariance eigenvalues, radial distribution and sixteen "
                  "fixed projected CDFs; a point at the mean or an equal-covariance circle cannot pass. "
                  "Clean live sampling is the original example's law. Clean EMA is an additional separate "
                  "diagnostic, and cannot replace a failed or incomplete live run.", ""])
    quick = next(c for c in report["cases"] if c["problem"] == "quickstart" and c["profile"] == "original")["summary"]
    if quick["final"]:
        for cohort in ("live", "ema"):
            m = quick["final"][cohort]
            lines.append(f"Last scored {cohort} at update {quick['final']['step']}: mean error "
                         f"{m['mean_error_sigma']:.6f} sigma, covariance eigenvalues {m['covariance_eigenvalues']}, "
                         f"radial KS {m['radial_ks']:.6f}, maximum projected KS {m['max_projection_ks']:.6f}.")
    if not quick["measurement_complete"]:
        lines.extend(["", "The original budget was not completed within the fixed 120-second cap. The default "
                      "batch is 2,048, and `BatchDistanceDiscriminator` computes differentiable batch-pair "
                      "distance features with four kernels, including higher-order penalty derivatives. "
                      "This is an execution-budget failure of this CPU cohort, not a demonstrated "
                      "1,000-update distribution failure. The GIF shows only saved actual checkpoints; "
                      "no final state or convergence is inferred beyond them. The failed attempt was not retried."])
    smaller = next(c for c in report["cases"] if c["profile"] == "cpu128")["summary"]
    lines.extend(["", "## Separate CPU-sized Gaussian profile", "",
                  "A separately frozen diagnostic keeps the original public example, networks, initializer, "
                  "seed, Gaussian data law and 1,000-update budget, changing only the public recipe batch "
                  "size from 2,048 to 128 through a wrapper around `particlegan.get_recipe`. Its complete "
                  "resolved recipe, exact wrapper, claim and analytic gates were registered before launch. "
                  "It has its own 120-second cap and is never substituted for the original timeout.", ""])
    if smaller["final"]:
        for cohort in ("live", "ema"):
            m = smaller["final"][cohort]
            lines.append(f"CPU128 {cohort}: {gate_status(smaller, cohort)}; mean error "
                         f"{m['mean_error_sigma']:.6f} sigma, covariance eigenvalues {m['covariance_eigenvalues']}, "
                         f"radial KS {m['radial_ks']:.6f}, maximum projected KS {m['max_projection_ks']:.6f}.")
    if smaller["measurement_complete"]:
        lines.extend(["", "At the full CPU128 budget, the live cloud misses the centering and projected-CDF "
                      "bounds. EMA misses the radial-CDF bound. These measurements identify the distribution "
                      "defects; this one unchanged run does not isolate an optimizer mechanism causing them."])
    lines.extend(["", "## Engineering interruptions", "",
                  "The first five-word attempt received external SIGTERM at a last valid 6,401-pair "
                  "checkpoint before the declared 900-second cap. The signal origin is **UNKNOWN**. The "
                  "heartbeat/child-ownership change improves supervision and does not prove why it happened. "
                  "Its source, raw checkpoints, last metrics and known costs remain archived. One corrected "
                  "full-budget attempt uses the same source/recipe/seed/budget; its endpoint is the evidence, "
                  "with no choice between attempts.", "",
                  "The first CPU128 launcher hit a concrete JSON-read race while progress was being "
                  "rewritten, ending after at least 38 update pairs and a valid update-20 checkpoint. "
                  "Atomic receipt replacement and tolerant heartbeat reads fixed this observer bug. One "
                  "engineering recovery used the identical frozen profile. Saved prefix identities are "
                  "checked separately for both recoveries. The genuine original "
                  "CPU timeout was preserved without retry. All five attempts and their receipts/costs "
                  "remain distinct.", ""])
    lines.extend(["", "## Purity and provenance", "",
                  "Short baseline/observed software probes compare exact model/optimizer tensors, training "
                  "inputs, owned RNG streams and caller Torch/Python/NumPy RNGs. Observation preserves module "
                  "modes and uses a separate fixed evaluation generator. Public `prior.sample` retains "
                  "MoG kernel noise when present; prior locations are never substituted for its sampling "
                  "law. Original five-word frame saves occur after EMA updates. No training loss or update "
                  "is replaced. The observers, pinned source files, evaluator and raw checkpoint clouds have "
                  "SHA-256 receipts. Bulk logs, streams, tensors and the original dashboard frames remain "
                  "outside Git at `/ml2/hypergan/toy-source-demos-20261001`.", "",
                  "Full-run caps are 900 seconds for five words and 120 seconds for quickstart. Repeated "
                  "scientific seed studies, confidence/mass tuning, best-checkpoint selection, longer budgets and default "
                  "promotion were not performed. The complete default budget plus five passing terminal "
                  "observations is required for a new diagnostic PASS.", "", "```sh",
                  "python -m benchmarks.toy_audit.source_demos --problem five_modes \\",
                  "  --evaluator /ml2/hypergan/toy-source-demos-20261001/five_modes/evaluator.py \\",
                  "  --output /ml2/hypergan/new-five-word-artifact --wall-seconds 900",
                  "python -m benchmarks.toy_audit.source_demos --problem quickstart \\",
                  "  --evaluator /ml2/hypergan/toy-source-demos-20261001/five_modes/evaluator.py \\",
                  "  --output /ml2/hypergan/new-quickstart-artifact --wall-seconds 120",
                  "python -m benchmarks.toy_audit.source_demos --problem quickstart --profile cpu128 \\",
                  "  --evaluator /ml2/hypergan/toy-source-demos-20261001/five_modes/evaluator.py \\",
                  "  --output /ml2/hypergan/new-quickstart-cpu128-artifact --wall-seconds 120",
                  "python -m benchmarks.toy_audit.source_demo_report \\",
                  "  --artifacts /ml2/hypergan/toy-source-demos-20261001 \\",
                  "  --output reports/toy_audit", "```", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = []
    definitions = (("five_modes", "source-family-15", "Five-word latent autoencoder demo", "original", "five_modes-recovered", "five_modes"),
                   ("quickstart", "source-family-16", "Single Gaussian quickstart: original batch2048", "original", "quickstart", None),
                   ("quickstart", "source-family-16", "Single Gaussian quickstart: separate CPU128 profile", "cpu128", "quickstart-cpu128-recovered", "quickstart-cpu128"))
    for problem, identifier, name, profile, folder, interrupted in definitions:
        path = args.artifacts / folder
        observations = rows(path)
        summary = attempt_summary(path, problem, profile)
        receipt = load(path / "source-receipt.json")
        media = render(path, args.output / "media", problem, summary, observations)
        attempts = []
        for item in ([interrupted] if interrupted else []) + [folder]:
            artifact = args.artifacts / item
            attempt = attempt_summary(artifact, problem, profile)
            if attempt.get("recipe") is None and summary.get("recipe") is not None:
                attempt["recipe"] = summary["recipe"]
                attempt["recipe_receipt"] = "Source-bound reconstruction from unchanged factory in the complete recovery"
            execution_receipts = {name: load(artifact / name) for name in
                                  ("launcher.json", "process.json", "timeout.json", "interruption.json", "supervisor-error.json")
                                  if (artifact / name).exists()}
            launcher_seconds = execution_receipts.get("launcher.json", {}).get("elapsed_seconds")
            known_seconds = (launcher_seconds if launcher_seconds is not None else
                             attempt.get("seconds") or attempt.get("recorded_wall_seconds_lower_bound", 0))
            attempts.append(dict(name=item, artifact_directory=str(artifact), summary=attempt,
                                 execution_receipts=execution_receipts,
                                 known_wall_seconds=known_seconds,
                                 known_wall_seconds_is_lower_bound=launcher_seconds is None,
                                 source_receipt=load(artifact / "source-receipt.json"),
                                 source_receipt_sha256=digest(artifact / "source-receipt.json"),
                                 observer_source_sha256=digest(artifact / "observer-source.py"),
                                 source_summary_sha256=digest(artifact / "summary.json") if (artifact / "summary.json").exists() else None,
                                 last_valid_frame=attempt["final"]["step"] if attempt["final"] else None,
                                 last_valid_frame_id=len(rows(artifact))-1 if rows(artifact) else None,
                                 resolved_protocol=load(artifact / "protocol.json") if (artifact / "protocol.json").exists() else None))
        cases.append(dict(id=identifier, catalog_id=identifier, problem=problem, profile=profile, name=name, historical_quality_rating=2,
                          original_source_only_receipt_unchanged=True, summary=summary,
                          original_model_gate=dict(status="NOT_DECLARED", explanation="The source demo has no scientific acceptance gate"),
                          original_scientific_status="NO_FROZEN_GATE", fresh_execution_status=summary["status"],
                          added_gate_status={cohort: gate_status(summary, cohort) for cohort in ("live", "ema")},
                          diagnostic_interpretation=diagnosis(summary, problem),
                          terminal_scientific_status=gate_status(summary, "live"), attempts=attempts,
                          engineering_prefix_parity=prefix_parity(args.artifacts / interrupted, path) if interrupted else None,
                          source_receipt=receipt, source_summary_sha256=digest(path / "summary.json") if (path / "summary.json").exists() else None,
                          media=media))
    report = dict(version="source-demo-convergence-v1", cases=cases, original_profile_attempts=3,
                  evaluator_version="toy-definition-quality-v1", evaluator_sha256=cases[0]["source_receipt"]["evaluator_sha256"],
                  known_attempt_wall_seconds_lower_bound=sum(a["known_wall_seconds"] for c in cases for a in c["attempts"]),
                  separately_frozen_profile_attempts=2, engineering_recoveries=2,
                  scientific_retries=0, production_configuration_changes=0,
                  new_test_profile_changed_factors={"quickstart-cpu128": ["batch_size:2048->128"]})
    write(args.output / "source-demos.json", report)
    (args.output / "SOURCE_DEMOS.md").write_text(markdown(report))
    print(json.dumps({c["problem"] + "/" + c["profile"]: final_status(c["summary"]) for c in cases}), flush=True)


if __name__ == "__main__":
    main()
