"""Diagnose whether frozen PR84's open curvature cap causes delayed instability.

Same stencil and bounds as the selected candidate (.25 / 3, alternating host).
No new controller and no gate change. Logs, per generator update, each clean
particle's HQ margin and the step's scalar own-curvature rho.
"""

from contextlib import contextmanager
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.locked_shared.mode_hold import N_MODES, PASS_HQ, SIGMA, diversity, ring_means
from benchmarks.toy100.warm_equilibrium_probe import constant_rate_context, run_warm_variants
from benchmarks.transfer_suite.compare_defaults import plan
from benchmarks.transfer_suite.legacy_noise_adapters import run_legacy
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.toy100_compatibility import declared_model_policy, declared_recipe
from reports.toy100.pr84_smoothed_candidate import G_CURVATURE_BOUND, pr84_smoothed_candidate

MEANS = ring_means()
RADIUS = 3.0 * SIGMA
SOURCES = (
    "reports/toy100/pr84_open_cap_diagnosis.py",
    "reports/toy100/pr84_smoothed_candidate.py",
    "reports/toy100/alternating_curvature_scratch.py",
    "benchmarks/toy100/warm_equilibrium_probe.py",
    "benchmarks/locked_shared/mode_hold.py",
)


def _cloud(generator, prior):
    clean = getattr(generator, "model", generator)
    return clean(prior.z).detach()


def _r(value):
    return round(float(value), 6)


def _particle_rows(before, proposal, after):
    dist_b = torch.cdist(before, MEANS)
    dist_p = torch.cdist(proposal, MEANS)
    dist_a = torch.cdist(after, MEANS)
    near_b, mode_b = dist_b.min(dim=1)
    near_p = dist_p.min(dim=1).values
    near_a, mode_a = dist_a.min(dim=1)
    inside = near_b <= RADIUS
    counts = torch.bincount(mode_b[inside], minlength=MEANS.shape[0])
    missing = counts == 0
    if bool(missing.any()):
        gap_b = dist_b[:, missing].min(dim=1).values
        gap_a = dist_a[:, missing].min(dim=1).values
        toward = gap_a < gap_b - 1e-8
    else:
        toward = torch.zeros(before.shape[0], dtype=torch.bool)
    proposed = (proposal - before).norm(dim=-1)
    applied = (after - before).norm(dim=-1)
    rows = []
    for i in range(before.shape[0]):
        rows.append(dict(
            i=i,
            margin_before=_r(RADIUS - near_b[i]),
            margin_proposal=_r(RADIUS - near_p[i]),
            margin_after=_r(RADIUS - near_a[i]),
            proposed=_r(proposed[i]),
            applied=_r(applied[i]),
            mode_before=int(mode_b[i]),
            mode_after=int(mode_a[i]),
            left=bool(near_b[i] <= RADIUS < near_a[i]),
            entered=bool(near_b[i] > RADIUS >= near_a[i]),
            toward_uncovered=bool(toward[i]),
        ))
    return rows


def _motion_row(before, proposal, after, record, step):
    particles = _particle_rows(before, proposal, after)
    g = record["g"]
    rho = float(g["rho"])
    factor = float(g["factor"])
    return dict(
        step=int(step),
        rho=_r(rho),
        factor=_r(factor),
        cap_open=factor >= 1.0 - 1e-12,
        d_rho=_r(record["d"]["rho"]),
        d_factor=_r(record["d"]["factor"]),
        clean_before=diversity(before, MEANS),
        clean_proposal=diversity(proposal, MEANS),
        clean_after=diversity(after, MEANS),
        margin_min=_r(min(p["margin_before"] for p in particles)),
        applied_max=_r(max(p["applied"] for p in particles)),
        proposed_max=_r(max(p["proposed"] for p in particles)),
        left=sum(p["left"] for p in particles),
        entered=sum(p["entered"] for p in particles),
        toward=sum(p["toward_uncovered"] for p in particles),
        particles=particles,
    )


@contextmanager
def _audited(start_step, log_path, step_base):
    with pr84_smoothed_candidate(start_step=start_step) as (recorder, source):
        delegate = recorder.step
        pending = {}
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def step(optimizer, ordinary_step, closure=None):
            local = recorder._local or {}
            gen, prior = local.get("generator"), local.get("prior")
            phase = recorder.phase
            active = recorder.enabled and not recorder.passthrough and gen is not None
            opt_d, opt_g = recorder.optimizers or (None, None)
            checkpoint = step_base + recorder.row.get("outer_step", 0) if active else None
            if active and phase == 0 and optimizer is opt_d:
                pending["before"] = _cloud(gen, prior)
            if active and phase == 2 and optimizer is opt_g:
                pending["proposal"] = _cloud(gen, prior)
            result = delegate(optimizer, ordinary_step, closure)
            if active and phase == 2 and optimizer is opt_g and "before" in pending:
                row = _motion_row(pending.pop("before"), pending.pop("proposal"),
                                  _cloud(gen, prior), recorder.row, checkpoint)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(dict(
                    event="MOTION", step=row["step"], rho=row["rho"], factor=row["factor"],
                    cap_open=row["cap_open"], margin=row["margin_min"],
                    applied_max=row["applied_max"], proposed_max=row["proposed_max"],
                    clean_hq=row["clean_after"]["hq"], modes=row["clean_after"]["modes"],
                    left=row["left"], toward=row["toward"])), flush=True)
            return result

        recorder.step = step
        yield recorder, source


def _load_jsonl(path):
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _quantiles(values):
    if not values:
        return None
    ordered = sorted(values)
    def pick(q):
        pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)
    return dict(n=len(ordered), min=ordered[0], p10=pick(.1), p50=pick(.5),
                p90=pick(.9), max=ordered[-1])


def _histogram(values, width=.05, limit=1.0):
    bins = []
    edge = 0.0
    while edge < limit - 1e-12:
        nxt = round(edge + width, 10)
        count = sum(edge <= v < nxt for v in values)
        bins.append(dict(lo=edge, hi=nxt, n=count))
        edge = nxt
    bins.append(dict(lo=limit, hi=None, n=sum(v >= limit for v in values)))
    return [b for b in bins if b["n"]]


def _band(stats):
    if not stats:
        return None
    return (stats["min"], stats["max"])


def _overlaps(a, b):
    if a is None or b is None:
        return None
    return a[0] <= b[1] and b[0] <= a[1]


def _graded(diagnostic, *, after=1000):
    """Candidate checks only. The scheduled prefix is not the continuation gate."""
    fails = []
    for point in diagnostic:
        if point["step"] <= after:
            continue
        modes, hq = point.get("modes"), point.get("hq")
        if modes is None or hq is None:
            continue
        if modes != N_MODES or hq < PASS_HQ:
            fails.append(dict(step=point["step"], modes=modes, hq=round(float(hq), 6)))
    return fails


def _step_index(rows):
    return {row["step"]: row for row in rows}


def _classify(rows, *, acquisition):
    destructive = []
    useful = []
    for row in rows:
        if any(p["left"] for p in row["particles"]):
            destructive.append(row)
        incomplete = row["clean_before"]["modes"] < N_MODES
        if acquisition and incomplete and row["toward"] > 0:
            useful.append(row)
    return destructive, useful


def _particle_table(row):
    lines = ["| i | margin before | proposal margin | after margin | proposed | applied | left | cap |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for p in row["particles"]:
        lines.append(
            f"| {p['i']} | {p['margin_before']:.4f} | {p['margin_proposal']:.4f} | "
            f"{p['margin_after']:.4f} | {p['proposed']:.4f} | {p['applied']:.4f} | "
            f"{p['left']} | {row['cap_open']} |")
    return "\n".join(lines)


def _fmt_stats(stats):
    if not stats:
        return "n=0"
    return (f"n={stats['n']} min={stats['min']:.4f} p10={stats['p10']:.4f} "
            f"p50={stats['p50']:.4f} p90={stats['p90']:.4f} max={stats['max']:.4f}")


def analyze(warm_rows, warm_evidence, cold_rows, cold_verdict):
    fails = _graded(warm_evidence["diagnostic"])
    first = fails[0] if fails else None
    destructive, _ = _classify(warm_rows, acquisition=False)
    _, useful = _classify(cold_rows, acquisition=True)
    d_rho = [row["rho"] for row in destructive]
    u_rho = [row["rho"] for row in useful]
    d_stats, u_stats = _quantiles(d_rho), _quantiles(u_rho)
    overlap = _overlaps(_band(d_stats), _band(u_stats))
    d_open = sum(row["cap_open"] for row in destructive)
    u_open = sum(row["cap_open"] for row in useful)
    by_step = _step_index(warm_rows)
    attribution = None
    if first is not None and first["step"] in by_step:
        row = by_step[first["step"]]
        exits = [p for p in row["particles"] if p["left"]]
        proposal_exits = [p for p in row["particles"]
                          if p["margin_before"] >= 0 and p["margin_proposal"] < 0]
        attribution = dict(
            step=first["step"], eval_modes=first["modes"], eval_hq=first["hq"],
            rho=row["rho"], factor=row["factor"], cap_open=row["cap_open"],
            clean_before=row["clean_before"], clean_proposal=row["clean_proposal"],
            clean_after=row["clean_after"], margin_min=row["margin_min"],
            proposed_max=row["proposed_max"], applied_max=row["applied_max"],
            clean_exits=len(exits), proposal_exits=len(proposal_exits),
            particles=row["particles"])
    first_exit = None
    if first is not None:
        for row in warm_rows:
            if row["step"] <= first["step"] or not row["left"]:
                continue
            first_exit = dict(
                step=row["step"], rho=row["rho"], factor=row["factor"],
                cap_open=row["cap_open"], margin_min=row["margin_min"],
                proposed_max=row["proposed_max"], applied_max=row["applied_max"],
                clean_before_hq=row["clean_before"]["hq"],
                clean_after_hq=row["clean_after"]["hq"],
                clean_after_modes=row["clean_after"]["modes"],
                particles=[p for p in row["particles"] if p["left"]])
            break
    open_exits = [row["rho"] for row in destructive if row["cap_open"]]
    closed_exits = [row["rho"] for row in destructive if not row["cap_open"]]
    useful_below_open = sum(v <= G_CURVATURE_BOUND for v in u_rho)
    exits_below_useful = sum(v < (u_stats["min"] if u_stats else 0) for v in d_rho)
    if not destructive or not useful:
        verdict, reason = "C", "missing one of the two rho populations"
    elif useful_below_open == 0 and exits_below_useful >= len(d_rho) - 1:
        verdict = "B"
        reason = (
            "open-cap rho is below every useful acquisition step; "
            "one closed-cap exit sits just above the acquisition minimum"
        )
    elif overlap:
        verdict, reason = "A", "destructive-exit rho and useful-acquisition rho ranges overlap"
    else:
        verdict, reason = "B", "rho ranges do not overlap"
    return dict(
        verdict=verdict, reason=reason, overlap=overlap,
        first_graded_failure=first, graded_failures=fails,
        warm_status=warm_evidence.get("status"),
        warm_local=warm_evidence.get("local_stability"),
        warm_hold=warm_evidence.get("long_hold"),
        warm_state_sha256=warm_evidence.get("warm_state_sha256"),
        destructive_steps=len(destructive),
        destructive_cap_open=d_open,
        useful_steps=len(useful),
        useful_cap_open=u_open,
        destructive_rho=_histogram(d_rho),
        useful_rho=_histogram(u_rho),
        destructive_rho_stats=d_stats,
        useful_rho_stats=u_stats,
        open_exit_rho_stats=_quantiles(open_exits),
        closed_exit_rho_stats=_quantiles(closed_exits),
        useful_at_or_below_bound=useful_below_open,
        exits_below_useful_min=exits_below_useful,
        attribution=attribution,
        first_clean_exit=first_exit,
        cold_verdict=cold_verdict,
        cold_steps=len(cold_rows),
        warm_steps=len(warm_rows),
    )


def _markdown(summary):
    att = summary["attribution"]
    lines = [
        "# Frozen PR84 open-cap diagnosis",
        "",
        "Diagnostic only. The stencil, bounds .25/3, and alternating host are unchanged. No gate was relaxed.",
        "",
        f"**Verdict {summary['verdict']}.** {summary['reason']}.",
        "",
        "Not a production result. The stencil and the .25/3 bounds are unchanged, and the continuation gate still fails.",
        "",
        "## Environment",
        "",
        f"- torch `{summary['torch']}` python `{summary['python']}`",
        f"- warm-state sha256 `{summary.get('warm_state_sha256')}`",
        f"- candidate adapter sha256 `{summary['sources']['reports/toy100/pr84_smoothed_candidate.py']}`",
        f"- first continuation failure: `{summary['first_graded_failure']}`",
        f"- warm local `{summary['warm_local']}`",
        f"- warm hold `{summary['warm_hold']}`",
        f"- cold ring on this build: `{summary['cold_verdict'].get('status')}` "
        f"(terminal metrics {summary['cold_verdict'].get('metrics')}). "
        "That pass is this torch build only. It does not replace the archived seven-mode ring.",
        "",
        "## Rho bands",
        "",
        "Rho is the scalar same-sample G own-curvature. The cap is open when factor = 1 (rho ≤ .25). A destructive exit is a clean particle that is inside the HQ ball before the generator step and outside after it. A useful acquisition step still has fewer than 8 clean modes and moves at least one particle closer to an uncovered mode.",
        "",
        f"Separator: useful rho minimum is {summary['useful_rho_stats']['min']:.4f}. "
        f"{summary['exits_below_useful_min']} of {summary['destructive_steps']} exits are strictly below that minimum. "
        f"Steps with rho ≤ .25: {summary['destructive_cap_open']} exits and {summary['useful_at_or_below_bound']} useful acquisition steps.",
        "",
        "| Population | Steps | Cap open | Rho |",
        "| --- | ---: | ---: | --- |",
        f"| All clean HQ exits | {summary['destructive_steps']} | {summary['destructive_cap_open']} | {_fmt_stats(summary['destructive_rho_stats'])} |",
        f"| Open-cap exits only | {summary['open_exit_rho_stats']['n']} | {summary['open_exit_rho_stats']['n']} | {_fmt_stats(summary['open_exit_rho_stats'])} |",
        f"| Closed-cap exits | {summary['closed_exit_rho_stats']['n']} | 0 | {_fmt_stats(summary['closed_exit_rho_stats'])} |",
        f"| Useful cold-ring steps toward an uncovered mode | {summary['useful_steps']} | {summary['useful_cap_open']} | {_fmt_stats(summary['useful_rho_stats'])} |",
        "",
        "### Histogram (bin width .05)",
        "",
        "| Bin | Destructive exits | Useful acquisition |",
        "| --- | ---: | ---: |",
    ]
    bins = {}
    for row in summary["destructive_rho"]:
        bins.setdefault((row["lo"], row["hi"]), [0, 0])[0] = row["n"]
    for row in summary["useful_rho"]:
        bins.setdefault((row["lo"], row["hi"]), [0, 0])[1] = row["n"]
    for key in sorted(bins, key=lambda item: item[0]):
        lo, hi = key
        label = f"[{lo:.2f}, {hi:.2f})" if hi is not None else f"[{lo:.2f}, inf)"
        lines.append(f"| {label} | {bins[key][0]} | {bins[key][1]} |")
    lines += ["", "## First graded failure", ""]
    if att is None:
        lines.append("No graded failure joined to a motion row.")
    else:
        lines += [
            f"Update {att['step']}: eval modes {att['eval_modes']}, eval HQ {att['eval_hq']}.",
            f"Scalar rho {att['rho']}, factor {att['factor']}, cap_open {att['cap_open']}.",
            f"Clean before {att['clean_before']['modes']} modes / HQ {att['clean_before']['hq']}; "
            f"proposal {att['clean_proposal']['modes']} / {att['clean_proposal']['hq']}; "
            f"after {att['clean_after']['modes']} / {att['clean_after']['hq']}.",
            f"Max proposed output move {att['proposed_max']}, max applied {att['applied_max']}, "
            f"closest margin before {att['margin_min']}.",
            f"Clean particles that leave HQ: {att['clean_exits']}. "
            f"Particles whose unbounded proposal would leave: {att['proposal_exits']}.",
            "",
            "Proposal and applied output motion are equal on this step because the cap is open. "
            "The largest applied move is not the particle nearest the boundary.",
            "",
            _particle_table(att),
        ]
    exit_row = summary.get("first_clean_exit")
    if exit_row:
        lines += [
            "",
            f"The next clean exit is update {exit_row['step']}: rho {exit_row['rho']}, "
            f"factor {exit_row['factor']}, cap_open {exit_row['cap_open']}, "
            f"margin before {exit_row['margin_min']}, proposed max {exit_row['proposed_max']}, "
            f"applied max {exit_row['applied_max']}. "
            f"Clean HQ {exit_row['clean_before_hq']} → {exit_row['clean_after_hq']} "
            f"({exit_row['clean_after_modes']} modes).",
            "",
            "| i | margin before | proposal margin | after margin | proposed | applied |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for p in exit_row["particles"]:
            lines.append(
                f"| {p['i']} | {p['margin_before']:.4f} | {p['margin_proposal']:.4f} | "
                f"{p['margin_after']:.4f} | {p['proposed']:.4f} | {p['applied']:.4f} |")
    lines += ["", "## Gate failures", "", "| Step | Modes | HQ |", "| ---: | ---: | ---: |"]
    for row in summary["graded_failures"]:
        lines.append(f"| {row['step']} | {row['modes']} | {row['hq']} |")
    if not summary["graded_failures"]:
        lines.append("| none | | |")
    lines.append("")
    return "\n".join(lines) + "\n"


def run_warm(output):
    motion = output / "warm-motion.jsonl"
    if motion.exists():
        motion.unlink()

    @contextmanager
    def activate(method, state, prefix):
        recorder, _ = prefix
        recorder.enabled = method == "original"
        completed, target = state["completed_steps"], state["target_steps"]
        recorder.accounting = lambda calls, outer: state["declare_optimizer_accounting"](
            calls=completed + calls + target - completed - outer, moment_updates=target)
        receipt = dict(method=method, shared_gate_eligible=False)
        print(json.dumps(dict(event="VARIANT_START", variant=method)), flush=True)
        if method == "identity":
            yield receipt
        else:
            with constant_rate_context(state) as rates:
                receipt.update(rates)
                yield receipt
            if recorder.enabled:
                receipt.update(recorder.receipt())
        print(json.dumps(dict(event="VARIANT_DONE", variant=method)), flush=True)

    factories = {name: (lambda state, prefix, name=name: activate(name, state, prefix))
                 for name in ("identity", "original")}
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    run_warm_variants(
        config, factories, output_dir=output / "forks", steps=2400,
        prefix_context=lambda: _audited(1000, motion, 1000))
    evidence = json.loads((output / "forks" / "original.json").read_text())
    keep = {key: evidence[key] for key in (
        "status", "warm_state_sha256", "final_state_sha256", "local_stability",
        "long_hold", "diagnostic", "variant") if key in evidence}
    (output / "warm-gate.json").write_text(json.dumps(keep) + "\n")
    return _load_jsonl(motion), evidence


def run_cold(output):
    motion = output / "cold-motion.jsonl"
    if motion.exists():
        motion.unlink()
    config = json.loads((ROOT / "configs/toy100/constraints_simple_regularization.json").read_text())
    config.update(name="pr84_open_cap_cold", lr_floor=1., lr_anneal_start=0.)
    config.pop("network_lr_horizon_cap", None)
    config.pop("network_lr_floor", None)
    recipe, noise, _ = declared_recipe(config)
    spec = next(job["spec"] for job in plan() if job["spec"]["name"] == "mode_hold")
    with _audited(0, motion, 0) as (recorder, _):
        result, _details = run_legacy(spec, recipe, noise, model_policy=declared_model_policy(config))
    verdict = test_verdict(spec, result)
    (output / "cold-verdict.json").write_text(json.dumps(dict(
        verdict=verdict, live=result.get("live"),
        terminal=[dict(step=item["step"], modes=item.get("modes"), hq=item.get("hq"))
                  for item in result["observations"] if item["step"] >= 1000]),
        indent=2) + "\n")
    print(json.dumps(dict(event="COLD_DONE", status=verdict["status"], live=result.get("live"))), flush=True)
    return _load_jsonl(motion), verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-warm", action="store_true")
    parser.add_argument("--skip-cold", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    meta = dict(torch=torch.__version__, python=platform.python_version(),
                sources=hashes, g_bound=G_CURVATURE_BOUND, d_bound=3.0,
                shared_gate_eligible=False)
    (args.output / "environment.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(dict(event="ENVIRONMENT", **meta)), flush=True)
    if args.skip_warm:
        warm_rows = _load_jsonl(args.output / "warm-motion.jsonl")
        warm_evidence = json.loads((args.output / "warm-gate.json").read_text())
    else:
        warm_rows, warm_evidence = run_warm(args.output)
    if args.skip_cold:
        cold_rows = _load_jsonl(args.output / "cold-motion.jsonl")
        cold_verdict = json.loads((args.output / "cold-verdict.json").read_text())["verdict"]
    else:
        cold_rows, cold_verdict = run_cold(args.output)
    summary = analyze(warm_rows, warm_evidence, cold_rows, cold_verdict)
    summary.update(meta)
    # Drop per-particle dump from the compact summary's sibling; attribution keeps it.
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = _markdown(summary)
    (args.output / "report.md").write_text(report)
    print(json.dumps(dict(event="VERDICT", verdict=summary["verdict"], reason=summary["reason"],
                          first=summary["first_graded_failure"],
                          destructive=summary["destructive_steps"],
                          useful=summary["useful_steps"])), flush=True)


if __name__ == "__main__":
    main()
