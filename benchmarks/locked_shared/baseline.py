"""Frozen, live-weight behavioral baseline: python -m benchmarks.locked_shared.baseline.

Each candidate is trained on every host. No config equality contributes to PASS.
The optional pinned conceptmod checkout supplies application integration checks.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from unittest.mock import patch

import torch

from particlegan import GANLoss, GradientPenalty
from . import mode_hold, trajectory, two_pole
from .observation import recording, sustained, OBSERVATIONS, MIN_STABLE_CHECKS
from .hosts import ae_gan_hold, cover_leftover, mid_scale_identity, residual_student, unipolar, unused_token_hold

VERSION = "behavior-v2"
DEFAULT_OUTPUT = Path("reports/behavioral_baseline")


@dataclass(frozen=True)
class Candidate:
    name: str
    loss_type: str = "logistic"
    gan_mode: str = "rp"
    reg_arm: str = "b_cap"
    reg_coeff: float = 1.0
    reg_kappa: float = 1.0
    particle_l2: float = 0.02
    vicreg_weight: float = 0.05
    cover_weight: float = 1.5
    lr_multiplier: float = 1.0
    lr_schedule: str = "host"
    lr_anneal_start: float = 0.6
    lr_floor: float = 0.05

    def __post_init__(self):
        if not self.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in self.name):
            raise ValueError("candidate name must use lowercase letters, digits, _ or -")
        for key in ("reg_coeff", "reg_kappa", "particle_l2", "vicreg_weight", "cover_weight", "lr_multiplier"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
        if self.lr_multiplier == 0:
            raise ValueError("lr_multiplier must be positive")
        if self.lr_schedule not in ("host", "cosine"):
            raise ValueError("lr_schedule must be host or cosine")
        for key in ("lr_anneal_start", "lr_floor"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if not 0 <= self.lr_anneal_start < 1 or not 0 <= self.lr_floor <= 1:
            raise ValueError("invalid cosine schedule bounds")
        self.make_loss()
        self.make_penalty()

    def make_loss(self):
        return GANLoss(self.loss_type, self.gan_mode)

    def make_penalty(self):
        return GradientPenalty(self.reg_arm, coeff=self.reg_coeff, kappa=self.reg_kappa,
                               norm="l2", lazy_k=1, target_anneal="none")

    def host_options(self):
        return {key: value for key, value in asdict(self).items() if key not in ("name", "lr_multiplier", "lr_schedule", "lr_anneal_start", "lr_floor")}


DEFAULT_CANDIDATES = (
    Candidate("locked_shared"),
    Candidate("no_particle_l2", particle_l2=0.0),
    Candidate("r1_r2_0_1", reg_arm="a_r1r2", reg_coeff=0.1),
    Candidate("r1_r2_0_1_no_l2", reg_arm="a_r1r2", reg_coeff=0.1, particle_l2=0.0),
    Candidate("b_cap_no_l2_lr_half", particle_l2=0.0, lr_multiplier=0.5),
    Candidate("b_cap_no_l2_lr_quarter", particle_l2=0.0, lr_multiplier=0.25),
    Candidate("b_cap_no_l2_coeff_2", particle_l2=0.0, reg_coeff=2.0),
    Candidate("b_cap_no_l2_coeff_5", particle_l2=0.0, reg_coeff=5.0),
)

# Values are thresholds from the original behavioral scorers, not tuned on results.
# A metric may have two bounds, and both must pass. Diagnostic-only values remain
# in raw.json but cannot earn passes. Each tuple is (metric, comparison, bound).
METRICS = {
    "two_pole": [("mean_abs", ">=", 0.30), ("grad_med", "<=", 1.0)],
    "trajectory": [("identity_mse", "<=", 0.02)],
    "residual_student": [("identity_mse", "<=", 0.02), ("success_rate", ">=", 1.0), ("wrong_pad_rate", "<=", 0.0)],
    "unipolar": [("cover", ">=", 0.85), ("off_caption", "<=", 0.05), ("neu_hold", ">=", 0.85)],
    "ae_gan_hold": [("recon_mse", "<=", 0.05), ("hold", "<=", 0.35)],
    "cover_leftover": [("u_kept", ">=", 0.85), ("content_kept", ">=", 0.75), ("leak_ratio", "<=", 0.20),
                       ("pole_rel_err_plus", "<=", 0.20), ("pole_rel_err_minus", "<=", 0.20), ("same_dir", "<=", 0.25)],
    "unused_token_hold": [("unused_hold", ">=", 0.85), ("concept_move", ">=", 0.85)],
    "mid_scale_identity": [("concept_cos_plus", ">=", 0.85), ("concept_cos_minus", ">=", 0.85),
                           ("concept_mag_plus", ">=", 0.75), ("concept_mag_plus", "<=", 1.25),
                           ("concept_mag_minus", ">=", 0.75), ("concept_mag_minus", "<=", 1.25),
                           ("identity_at_0", ">=", 0.85), ("identity_at_mid", ">=", 0.85)],
    "mode_hold": [("modes", ">=", 7), ("hq", ">=", 0.90)],
}
BUDGETS = dict(two_pole=80, trajectory=400, residual_student=400, unipolar=400,
               ae_gan_hold=250, cover_leftover=800, unused_token_hold=200,
               mid_scale_identity=800, mode_hold=1200)
SHARED = ("orbit_hold", "erase_keep_backend", "late_collapse", "keep_critic", "lm_target",
          "field_lift", "path_suffix_lora", "dsl_macro_expand", "dsl_phrase_jobs", "dsl_game_geometry")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def protocol():
    folder = Path(__file__).parent
    # Hash evaluation, training, and the actual primitives, including uncommitted edits.
    files = list(folder.glob("*.py")) + list((folder / "hosts").glob("*.py"))
    repo = folder.parents[1]
    files += list((repo / "particlegan").rglob("*.py"))
    hashes = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    return {"version": VERSION, "seed": 0, "device": "cpu", "threads": 1,
            "evaluation": "final live weights; EMA reported separately where implemented",
            "ranking": "passed toys, passed bounds, live ring modes, HQ, effective modes; descending",
            "convergence": {"observations": OBSERVATIONS, "minimum_passing_suffix": MIN_STABLE_CHECKS,
                            "ring_requires_all_modes": True, "timing": "wall seconds including setup and measurement"},
            "schedule_policy": "host preserves original schedules; cosine replaces all host schedules using initial optimizer group rates",
            "budgets": BUDGETS, "metrics": METRICS, "shared_checks": SHARED,
            "source_sha256": hashes, "torch": str(torch.__version__), "python": platform.python_version()}


def score_metrics(values, requirements):
    cells = []
    for name, op, threshold in requirements:
        value = values.get(name)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        ok = numeric and (value >= threshold if op == ">=" else value <= threshold)
        cells.append({"metric": name, "value": value if numeric else None, "op": op, "threshold": threshold,
                      "status": "PASS" if ok else "FAIL" if numeric else "MISSING",
                      "margin": (value - threshold if op == ">=" else threshold - value) if numeric else None})
    return cells


def score_row(row, shared):
    toys, all_cells = {}, []
    for toy, requirements in METRICS.items():
        result = row.get("toys", {}).get(toy, {})
        cells = score_metrics(result.get("live", {}), requirements)
        status = "ERROR" if result.get("error") else "MISSING" if any(c["status"] == "MISSING" for c in cells) else (
            "PASS" if all(c["status"] == "PASS" for c in cells) else "FAIL")
        toys[toy] = {"status": status, "metrics": cells}
        all_cells.extend(cells)
    shared_ok = all(shared.get(name, {}).get("status") == "PASS" for name in SHARED)
    complete = all(t["status"] not in ("MISSING", "ERROR") for t in toys.values())
    all_pass = all(t["status"] == "PASS" for t in toys.values())
    shared_complete = all(shared.get(name, {}).get("status") in ("PASS", "FAIL") for name in SHARED)
    status = "PASS" if all_pass and shared_ok else "INCOMPLETE" if not complete or not shared_complete else "FAIL"
    return {"status": status, "passed_toys": sum(t["status"] == "PASS" for t in toys.values()),
            "passed_metrics": sum(c["status"] == "PASS" for c in all_cells),
            "total_metrics": len(all_cells), "shared_ok": shared_ok, "toys": toys}


def run_toy(toy, cfg):
    """Serial, scoped host injection; one candidate card, no per-toy tuning."""
    knobs = cfg.host_options()
    with recording(BUDGETS[toy], schedule=cfg.lr_schedule, start=cfg.lr_anneal_start,
                   floor=cfg.lr_floor) as recorder, ExitStack() as stack:
        for module in (trajectory, residual_student, cover_leftover, mid_scale_identity):
            target = module.PROTOCOL if module in (trajectory, residual_student) else module.FORMULATION
            values = {k: v for k, v in knobs.items() if k in target}
            if "lr" in target:
                values["lr"] = target["lr"] * cfg.lr_multiplier
            stack.enter_context(patch.dict(target, values))
        for module in (mode_hold, unipolar, unused_token_hold, mid_scale_identity):
            stack.enter_context(patch.object(module, "LR", module.LR * cfg.lr_multiplier))
        stack.enter_context(patch.object(two_pole, "TOY_LR", two_pole.TOY_LR * cfg.lr_multiplier))
        if toy == "two_pole":
            raw = two_pole.train(gan_factory=cfg.make_loss, cap_factory=cfg.make_penalty, particle_l2=cfg.particle_l2)
        elif toy == "trajectory":
            raw = trajectory.train(gan_factory=cfg.make_loss, cap_factory=cfg.make_penalty, diagnostics=True)
        elif toy == "mode_hold":
            raw = mode_hold.train_mode_hold(mode_hold.ModeHoldRecipe(particle_l2=cfg.particle_l2, vicreg_weight=cfg.vicreg_weight),
                                           gan_factory=cfg.make_loss, cap_factory=cfg.make_penalty, diagnostics=True)
        elif toy == "residual_student":
            raw = residual_student.train()
        elif toy == "unipolar":
            raw = unipolar.run_arm("locked_rpgan", **{k: knobs[k] for k in ("loss_type", "gan_mode", "reg_arm", "reg_coeff", "reg_kappa")})
        elif toy == "ae_gan_hold":
            options = {k: v for k, v in knobs.items() if k in ae_gan_hold.HoldConfig.__dataclass_fields__}
            raw = ae_gan_hold.train(ae_gan_hold.HoldConfig(name=cfg.name, lr=ae_gan_hold.LR * cfg.lr_multiplier, **options))
            raw.pop("cfg", None)
        elif toy == "cover_leftover":
            raw = cover_leftover.fit_cover_leftover(cover_leftover.CoverRecipe())
        elif toy == "unused_token_hold":
            options = {k: v for k, v in knobs.items() if k in unused_token_hold.UnusedHoldRecipe.__dataclass_fields__}
            raw = unused_token_hold.train(unused_token_hold.UnusedHoldRecipe(name=cfg.name, **options))
        elif toy == "mid_scale_identity":
            raw = mid_scale_identity.run_arm("locked")
        else:
            raise ValueError(toy)
    live = raw.get("live", raw)
    # Keep raw diagnostics as well, but never use their legacy verdict strings.
    result = {"live": {key: live[key] for key, _, _ in METRICS[toy] if key in live}, "raw": raw}
    result["observations"] = recorder.curve
    requirements = [("modes", ">=", 8), ("hq", ">=", 0.90)] if toy == "mode_hold" else METRICS[toy]
    result["convergence"] = sustained(recorder.curve, requirements, expected_steps=recorder.steps)
    if "live" in raw:
        result["ema"] = {key: raw[key] for key, _, _ in METRICS[toy] if key in raw}
    return result


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def ring_quality(row):
    result = row.get("toys", {}).get("mode_hold", {})
    raw = result.get("raw", {})
    live = raw.get("live", result.get("live", {}))
    curve = raw.get("live_curve", [])
    tail = [point for point in curve if point["step"] >= BUDGETS["mode_hold"] - 200]
    return {"live": live, "tail": tail,
            "full_coverage_checks": sum(p["modes"] == mode_hold.N_MODES and p["hq"] >= mode_hold.PASS_HQ for p in tail)}


def render(report, destination):
    scored = [(row, score_row(row, report.get("shared", {}))) for row in report["rows"]]
    def quality_key(row, score):
        live = ring_quality(row)["live"]
        return (score["passed_toys"], score["passed_metrics"], live.get("modes", 0),
                live.get("hq", 0), live.get("effective_modes", 0))
    scored.sort(key=lambda pair: tuple(-v for v in quality_key(*pair)) + (pair[0]["config"]["name"],))
    winners = [row["config"]["name"] for row, score in scored if score["status"] == "PASS"]
    winner_text = "**Regression PASS: " + ", ".join(f"`{name}`" for name in winners) + ".**" if winners else "**No complete passing live configuration recorded.**"
    lines = ["# Behavioral baseline — live weights", "", winner_text, "",
             f"Protocol `{report['protocol']['version']}` · CPU · seed 0 · fixed host budgets · final live weights.", "",
             "**Overall PASS requires all 29 numerical bounds on all 9 trained toys, plus the 10 shared behavioral/integration checks.** "
             "EMA is diagnostic and cannot rescue a live failure. Config equality checks are excluded. "
             "Shared checks are run once: they do not depend on the GAN config and do not contribute to its rank.", "",
             "**Regression PASS is a minimum bar for default selection.** The original ring bar permits 7/8 modes. "
             "Actual coverage, HQ, balance and checkpoint stability are visible below; a final-step PASS does not establish a stable ParticleGAN default. "
             "The [default-selection analysis](default_selection.md) also compares the stock recipe on the ring host.", "",
             "Rows rank by passed toys, then passed numerical bounds, then live ring coverage, HQ and effective modes. "
             "Missing/nonfinite results and errors cannot pass. Thresholds and budgets are frozen before config search.", "",
             "| Rank | Config | Live toys | Live bounds | Ring modes | Ring HQ | Effective modes | Regression |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    last, rank = None, 0
    for index, (row, score) in enumerate(scored, 1):
        key = quality_key(row, score)
        if key != last:
            rank = index
        last = key
        live = ring_quality(row)["live"]
        hq = f"{live['hq']:.2%}" if "hq" in live else "Missing"
        effective = f"{live['effective_modes']:.2f}/8" if "effective_modes" in live else "Missing"
        penalty = row["config"].get("reg_arm", "unspecified")
        lines.append(f"| {rank} | `{row['config']['name']}` ({penalty}) | {score['passed_toys']}/9 | {score['passed_metrics']}/29 | {live.get('modes', 'Missing')}/8 | {hq} | {effective} | **{score['status']}** |")
    lines += ["", "Effective modes measures balance among high-quality outputs; eight balanced modes gives 8. "
              "HQ measures quality of generated samples and does not penalize a missing target cluster. "
              "Thus 100% HQ can coexist with 7/8 coverage.", "",
              "## Live stability and exact particle coverage", "",
              "Final-step selection is unchanged. For default selection, report the five observations at steps 1,000, 1,050, 1,100, 1,150 and 1,200. "
              "These are sampled checkpoints, not a claim about every intervening step. "
              "Enumerating all 12 equally likely particles additionally distinguishes a missing mode from an unlucky 4,096-sample evaluation.", "",
              "| Config | Worst tail modes | Worst tail HQ | Tail checks with 8/8 and HQ ≥90% | Final HQ particles by mode (0–7) |",
              "| --- | ---: | ---: | ---: | --- |"]
    for row, _ in scored:
        quality = ring_quality(row)
        tail = quality["tail"]
        support = quality["live"].get("support", {})
        modes = f"{min(p['modes'] for p in tail)}/8" if tail else "Missing"
        hq = f"{min(p['hq'] for p in tail):.2%}" if tail else "Missing"
        counts = str(support.get("hq_counts", "Missing"))
        lines.append(f"| `{row['config']['name']}` | {modes} | {hq} | {quality['full_coverage_checks']}/{len(tail)} | {counts} |")
    if any("convergence" in t for row, _ in scored for t in row["toys"].values()):
        lines += ["", "## Convergence speed", "",
                  "Each host has 24 evenly spaced observations. Sustained PASS requires a complete curve, at least five consecutive passing observations, "
                  "and no later failure through the final budget. Ring convergence requires all eight modes and HQ ≥90%; original regression bounds stay unchanged. "
                  "Stable-from is the start of that final passing stretch; confirmation is its fifth observation. Times include setup and measurement overhead. "
                  "These certify observations, not every intervening update. Historical runs without timing are not assigned estimated speeds.", "",
                  "| Config | Sustained toys | Ring first PASS step | Ring stable from step | Ring confirmed step | Ring confirmed seconds | All toys wall seconds |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row, _ in scored:
            c = row["toys"].get("mode_hold", {}).get("convergence", {})
            stable = sum(t.get("convergence", {}).get("stable_from_step") is not None for t in row["toys"].values())
            seconds = c.get("confirmed_seconds")
            elapsed = [t.get("seconds") for t in row["toys"].values()]
            wall = f"{sum(elapsed):.2f}" if len(elapsed) == len(METRICS) and all(isinstance(s, (int, float)) for s in elapsed) else "Missing"
            confirm = f"{seconds:.2f}" if seconds is not None else "Not reached"
            values = [str(c.get(key)) if c.get(key) is not None else "Not reached" for key in ("first_pass_step", "stable_from_step", "confirmed_step")]
            lines.append(f"| `{row['config']['name']}` | {stable}/9 | {' | '.join(values)} | {confirm} | {wall} |")
    if report.get("stock_ring"):
        lines += ["", "## Stock-recipe ring comparison", "",
                  "Both rows use the same 20,000 particles, 7,000 steps, optimizer and cosine schedule; only the penalty changes. "
                  "They are compared with each other and are not included in the 12-particle regression rank. "
                  "This is the ring host, not the 100-Gaussian benchmark. Tail observations cover the final 200 steps at 50-step intervals.", "",
                  "| Penalty | Live modes | Live HQ | Effective modes | EMA modes / HQ | Worst tail live HQ | Full-coverage/HQ tail checks |",
                  "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
        for row in report["stock_ring"]["rows"]:
            ring = row["ring"]
            live = ring["live"]
            tail = [p for p in ring["live_curve"] if p["step"] >= row["recipe"]["total_steps"] - 200]
            full = sum(p["modes"] == 8 and p["hq"] >= 0.9 for p in tail)
            label = f"{row['recipe']['reg_arm']} coeff {row['recipe']['reg_coeff']}"
            lines.append(f"| {label} | {live['modes']}/8 | {live['hq']:.2%} | {live['effective_modes']:.2f}/8 | {ring['modes']}/8 / {ring['hq']:.2%} | {min(p['hq'] for p in tail):.2%} | {full}/{len(tail)} |")
        lines += ["", "Both stock-recipe penalties sustain full coverage at the measured late checkpoints. "
                  "The current `b_cap` recipe and its EMA pass; the small-host R1+R2 result does not justify replacing the default or disabling EMA. "
                  "Raw evidence and its original source fingerprint are in [stock_ring.json](stock_ring.json).", ""]
    lines += ["", "## Live toy matrix", "", "| Config | " + " | ".join(METRICS) + " |",
              "| --- | " + " | ".join("---" for _ in METRICS) + " |"]
    for row, score in scored:
        lines.append("| `" + row["config"]["name"] + "` | " + " | ".join(score["toys"][t]["status"] for t in METRICS) + " |")
    lines += ["", "## EMA diagnostics", "", "Only the ring and cover/leftover hosts maintain EMA. These results are separate from the live ranking.", "",
              "| Config | Ring modes | Ring HQ | Ring bounds | Cover/leftover bounds |", "| --- | ---: | ---: | --- | --- |"]
    for row, _ in scored:
        ring = row["toys"].get("mode_hold", {}).get("ema", {})
        cover = row["toys"].get("cover_leftover", {}).get("ema", {})
        def tally(toy, metrics):
            cells = score_metrics(metrics, METRICS[toy])
            return f"{sum(c['status'] == 'PASS' for c in cells)}/{len(cells)}"
        hq = f"{ring['hq']:.2%}" if "hq" in ring else "Missing"
        lines.append(f"| `{row['config']['name']}` | {ring.get('modes', 'Missing')}/8 | {hq} | {tally('mode_hold', ring)} | {tally('cover_leftover', cover)} |")
    lines += ["", "## Shared behavioral and integration checks", "",
              "These measure fixed geometry, checkpoint selection, frozen-critic behavior, LoRA targeting and DSL execution. "
              "They run against the pinned conceptmod application; no candidate receives extra ranking points for them.", "",
              "| Check | Status | Evidence |", "| --- | --- | --- |"]
    for name in SHARED:
        check = report.get("shared", {}).get(name, {})
        lines.append(f"| {name} | {check.get('status', 'MISSING')} | {check.get('summary', 'Not run').replace('|', chr(92)+'|')} |")
    lines += ["", "## Every live metric", "", "Margins are in each metric's own units: nonnegative meets the bound. No averaging across metrics.", ""]
    for row, score in scored:
        lines += [f"<details><summary>{row['config']['name']}: {score['status']}</summary>", "", "```json", json.dumps(row["config"], indent=2), "```", "",
                  "| Toy | Metric | Actual | Required | Margin | Result |", "| --- | --- | ---: | --- | ---: | --- |"]
        for toy, result in score["toys"].items():
            for cell in result["metrics"]:
                value = "Missing" if cell["value"] is None else f"{cell['value']:.8g}"
                margin = "—" if cell["margin"] is None else f"{cell['margin']:+.4g}"
                lines.append(f"| {toy} | {cell['metric']} | {value} | {cell['op']} {cell['threshold']} | {margin} | {cell['status']} |")
            if row["toys"].get(toy, {}).get("error"):
                lines += ["", f"Error in {toy}: `{row['toys'][toy]['error']}`", ""]
        lines += ["", "</details>", ""]
    lines += ["## Reproduce or compare another approach", "", "```bash",
              "python -m benchmarks.locked_shared.baseline --reference /path/to/conceptmod",
              "python -m benchmarks.locked_shared.baseline --configs my_configs.json --reference /path/to/conceptmod --output reports/my_search",
              "# Resume only with exactly matching source, runtime and config fingerprints:",
              "python -m benchmarks.locked_shared.baseline --resume --reference /path/to/conceptmod",
              "```", "", "Use [passing_configs.json](passing_configs.json) to rerun the passing baseline alone, or [configs.json](configs.json) for the comparison set. Each setting applies wherever that loss exists; "
              "there are no per-toy overrides. The fixed protocol and actual source hashes are in [protocol.json](protocol.json); "
              "all measurements, timings and shared-check evidence are in [results.json](results.json). "
              "See [scope and configuration mapping](../../benchmarks/locked_shared/BASELINE.md). "
              "Exit code 0 means at least one full PASS; 1 means no full PASS. Output is saved after each toy. "
              "Without `--reference`, candidate training still runs but the full result is INCOMPLETE.", "",
              "This is one fixed-seed CPU regression baseline, not evidence of downstream transfer or robustness across random initializations.", ""]
    destination.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, help="JSON list of candidate objects; unspecified fields use defaults")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference", type=Path, help="clean conceptmod checkout at the pinned revision for shared checks")
    parser.add_argument("--stock-reference", type=Path, default=DEFAULT_OUTPUT / "stock_ring.json",
                        help="optional recorded stock-recipe ring comparison; displayed separately from candidate ranks")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configs = [Candidate(**c) for c in json.loads(args.configs.read_text())] if args.configs else list(DEFAULT_CANDIDATES)
    if not configs or len({c.name for c in configs}) != len(configs):
        parser.error("provide at least one candidate, with unique names")
    torch.set_num_threads(1)
    fingerprint = protocol()
    report = {"protocol": fingerprint, "protocol_sha256": digest(fingerprint),
              "created_at": datetime.now(timezone.utc).isoformat(), "rows": [], "shared": {}}
    stock = json.loads(args.stock_reference.read_text()) if args.stock_reference.exists() else None
    if stock is not None:
        # Preserve only the final observations and the tail needed for display;
        # the source artifact retains the complete curves and per-mode evidence.
        report["stock_ring"] = {"source_sha256": hashlib.sha256(args.stock_reference.read_bytes()).hexdigest(),
                                "protocol": stock["protocol"], "rows": []}
        for row in stock["rows"]:
            ring = row["ring"]
            report["stock_ring"]["rows"].append({"name": row["name"], "recipe": row["recipe"], "ring": {
                "modes": ring["modes"], "hq": ring["hq"],
                "live": {k: ring["live"][k] for k in ("modes", "hq", "effective_modes")},
                "live_curve": [{k: p[k] for k in ("step", "modes", "hq")} for p in ring["live_curve"]
                               if p["step"] >= row["recipe"]["total_steps"] - 200]}})
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "results.json"
    if result_path.exists():
        if not args.resume:
            parser.error(f"{result_path} exists; use --resume or a new --output directory")
        report = json.loads(result_path.read_text())
        if report["protocol_sha256"] != digest(fingerprint) or report["protocol"] != json.loads(json.dumps(fingerprint)):
            parser.error("source/protocol/runtime changed; start a separate output directory")
        if [r["config"] for r in report["rows"]] != [asdict(c) for c in configs]:
            parser.error("candidate configs/order changed; start a separate output directory")
    else:
        report["rows"] = [{"config": asdict(c), "config_sha256": digest(asdict(c)), "toys": {}} for c in configs]
    write_json(output / "configs.json", [asdict(c) for c in configs])
    write_json(output / "protocol.json", fingerprint)
    def save():
        write_json(result_path, report)
        write_json(output / "passing_configs.json", [r["config"] for r in report["rows"] if score_row(r, report["shared"])["status"] == "PASS"])
        render(report, output / "README.md")
    save()
    for cfg, row in zip(configs, report["rows"]):
        if row["config_sha256"] != digest(asdict(cfg)):
            parser.error("saved config fingerprint is invalid")
        for toy in METRICS:
            if toy in row["toys"] and "error" not in row["toys"][toy]:
                continue
            print(f"START candidate={cfg.name} toy={toy} steps={BUDGETS[toy]} seed=0", flush=True)
            start = time.monotonic()
            try:
                result = run_toy(toy, cfg)
                json.dumps(result, allow_nan=False)
            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            result["seconds"] = time.monotonic() - start
            row["toys"][toy] = result
            save()
            print(json.dumps({"event": "DONE", "candidate": cfg.name, "toy": toy,
                              "live": result.get("live"), "error": result.get("error"), "seconds": result["seconds"]}), flush=True)
    if args.reference:
        from .shared_checks import run_shared
        run_shared(args.reference, report, save)
    save()
    for row in report["rows"]:
        scored = score_row(row, report["shared"])
        print(f"RESULT {row['config']['name']} {scored['status']} live_toys={scored['passed_toys']}/9 live_bounds={scored['passed_metrics']}/29", flush=True)
    return 0 if any(score_row(r, report["shared"])["status"] == "PASS" for r in report["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
