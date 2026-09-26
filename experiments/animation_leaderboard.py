#!/usr/bin/env python
"""Rank sprite animation world model runs by test dream score and write a README leaderboard."""
import argparse
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_SOURCES = ("lib/animation_evaluation.py", "lib/sprite_animation.py")
TRAINING_FIELDS = ("steps", "batch_size", "seed", "lr")  # num_particles is a candidate switch
FINDINGS_PLACEHOLDER = "_Filled in after the runs complete._"


def load_rows(results):
    rows = []
    for path in sorted(Path(results).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        summary["name"] = path.parent.name
        summary["dir"] = path.parent
        rows.append(summary)
    if not rows:
        raise FileNotFoundError(f"No */summary.json under {results}")
    return rows


def check_comparable(rows):
    def same(values, what):
        distinct = {json.dumps(v, sort_keys=True) for v in values.values()}
        if len(distinct) > 1:
            detail = ", ".join(f"{k}={json.dumps(v, sort_keys=True)}" for k, v in values.items())
            raise ValueError(f"Rows differ in {what}: {detail}")

    same({r["name"]: r["provenance"]["dataset"] for r in rows}, "dataset sha256")
    for source in EVALUATION_SOURCES:
        same({r["name"]: r["provenance"]["sources"].get(source) for r in rows}, f"source hash {source}")
    trained = [r for r in rows if r["config"]["arm"] != "persistence"]
    for field in TRAINING_FIELDS:
        same({r["name"]: r["config"].get(field) for r in trained}, f"training field {field}")
    return trained


def fmt(value):
    return "—" if value is None else f"{value:.5f}"


def describe(cfg):
    arm = cfg["arm"]
    if arm == "persistence":
        return "No training: st+1 = st with the exact render of st. The baseline any useful model must beat."
    if arm == "direct":
        return ("Supervised st -> (st+1, gt) baseline with the same frame decoder, MSE on next state and frame; "
                "no prior, encoder or critics.")
    parts = ["GAN world model: G1/G2/G3 from the MoG prior, E(st) -> z dream loop",
             "joint + marginal critics" if cfg["joint_d"] else "marginal critics only (no joint D)",
             "E anchored on st/st+1/gt" if cfg["anchor"] == "full" else "E anchored on st+1 only (no G1/G3 anchor)",
             "detached synthetic composition" if cfg["detach_synthetic"] else "live synthetic composition"]
    if cfg.get("num_particles", 1024) != 1024:
        parts.append(f"MoG{cfg['num_particles']} prior")
    if cfg.get("routing_temperature", .25) != .25:
        parts.append(f"routing temperature {cfg['routing_temperature']:g}")
    for key, label in (("real_encoding_weight", "real encoding weight"),
                       ("synthetic_reconstruction_weight", "synthetic reconstruction weight")):
        if cfg.get(key, 1.) != 1.:
            parts.append(f"{label} {cfg[key]:g}")
    return "; ".join(parts) + "."


def existing_findings(readme):
    if not readme.exists():
        return FINDINGS_PLACEHOLDER
    text = readme.read_text()
    marker = "\n## Findings\n"
    if marker not in text:
        return FINDINGS_PLACEHOLDER
    body = text.split(marker, 1)[1].split("\n## ", 1)[0]
    return body.strip("\n") or FINDINGS_PLACEHOLDER


def build(results, out):
    results, out = Path(results), Path(out)
    rows = load_rows(results)
    trained = check_comparable(rows)
    rows.sort(key=lambda r: r["evaluation"]["test"]["dream_score"])
    out.mkdir(parents=True, exist_ok=True)
    readme = out / "README.md"
    findings = existing_findings(readme)
    meta_path = results / "data" / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    for row in rows:
        for source, suffix in (("config.yaml", "config.yaml"), ("dream.gif", "dream.gif")):
            if (row["dir"] / source).exists():
                shutil.copyfile(row["dir"] / source, out / f"{row['name']}_{suffix}")

    if trained:
        cfg = trained[0]["config"]
        budget = (f"Every trained arm uses {cfg['steps']:,} updates, batch {cfg['batch_size']}, "
                  f"lr {cfg['lr']:g} and training seed {cfg['seed']}; GAN rows use MoG1024 unless the row says "
                  f"otherwise (the direct arm has no prior). Persistence is untrained. No seed-only repeats.")
    else:
        budget = "No trained arms yet."
    lines = ["# Animation world model leaderboard", "",
             "Primary score: test dream score, the mean standardized-state MSE against the exact simulator at "
             "dream horizons 1, 5, 20 and 50, rolled out closed-loop from 3 start offsets (0, 15, 30) in every "
             "test episode. Each state's error is capped at 10 so one divergent rollout cannot dominate; "
             "Diverged is the fraction of rollouts at the cap at step 50. Lower is better; the exact simulator "
             "scores 0. First fail is the median first step "
             "whose standardized-state MSE exceeds 1. One-step MSE is teacher-forced over every test transition. "
             "Frame self compares dreamed frames with renders of the dreamed states (self-consistency); frame "
             "true compares them with renders of the simulator states.", "",
             budget, "",
             "Rows are refused unless dataset hashes, the evaluation/simulator source hashes and the training "
             "budget fields match. Checkpoint selection is minimum validation dream score.", "",
             "| Rank | Run | Arm | Dream score ↓ | 1 | 5 | 20 | 50 | First fail (median step) | OOD dream score ↓ "
             "| OOD/ID ratio | OOD diverged | One-step ↓ | Frame self ↓ | Frame true ↓ | G / E / D params | Train s |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows, 1):
        t, o, p = row["evaluation"]["test"], row["evaluation"]["ood_test"], row["parameters"]
        ratio = o["dream_score"] / t["dream_score"] if t["dream_score"] > 0 else None
        g = p.get("G", 0) or p.get("direct", 0)
        lines.append(f"| {rank} | {row['name']} | {row['config']['arm']} | {fmt(t['dream_score'])} | "
                     + " | ".join(fmt(t[f"dream_mse_{h}"]) for h in (1, 5, 20, 50))
                     + f" | {t['first_fail_median']:g} | {fmt(o['dream_score'])} | "
                     f"{'—' if ratio is None else f'{ratio:.2f}'} | {o.get('diverged_50', 0):.1%} | "
                     f"{fmt(t['one_step_mse'])} | "
                     f"{fmt(t['frame_self_mse'])} | {fmt(t['frame_true_mse'])} | "
                     f"{g:,} / {p.get('E', 0):,} / {p.get('D', 0):,} | {row['train_seconds']:.1f} |")
    lines += ["", "Simulator floor: **0** on every dream metric. G params for the direct arm are its predictor.",
              "", "## Findings", "", findings, "", "## OOD animation", "",
              "ood_test starts higher (y0 ∈ [0.75, 0.9]) with speeds 1.2–1.6 launched upward, so the sprite "
              "hits the ceiling, which never happens in training (every in-distribution state stays below "
              "y = 0.75)."]
    if meta:
        lines[-1] += (f" {meta['ood_outside_train_box']:.1%} of all OOD states and "
                      f"{meta['ood_first50_outside_train_box']:.1%} of states in the first 50 steps fall outside "
                      f"the per-coordinate training box; {meta['ood_ceiling_bounce_episodes']:.1%} of OOD episodes "
                      f"bounce off the ceiling within 50 steps.")
    lines += ["", "| Run | OOD dream score ↓ | 1 | 5 | 20 | 50 | First fail (median step) | Diverged | Frame self ↓ | Frame true ↓ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        o = row["evaluation"]["ood_test"]
        lines.append(f"| {row['name']} | {fmt(o['dream_score'])} | "
                     + " | ".join(fmt(o[f"dream_mse_{h}"]) for h in (1, 5, 20, 50))
                     + f" | {o['first_fail_median']:g} | {o.get('diverged_50', 0):.1%} | {fmt(o['frame_self_mse'])} "
                     f"| {fmt(o['frame_true_mse'])} |")
    gan = [r for r in rows if r["config"]["arm"] == "gan"]
    if gan:
        lines += ["", "## Prior and encoder health", "",
                  "Prior dynamics/frame MSE score G2 and G3 on prior samples against the exact step and render of "
                  "the sampled G1 state. Encoder health is measured on every test state.", "",
                  "| Run | Prior dynamics MSE ↓ | Prior frame MSE ↓ | Offset at bound ↓ | Components used | Effective components |",
                  "|---|---:|---:|---:|---:|---:|"]
        for row in gan:
            pr, en = row["evaluation"]["prior"], row["evaluation"]["encoder"]
            lines.append(f"| {row['name']} | {fmt(pr['prior_dynamics_mse'])} | {fmt(pr['prior_frame_mse'])} | "
                         f"{en['offset_at_bound']:.3f} | {en['components_used']:,} | {en['components_effective']:.1f} |")
    lines += ["", "## Runs", ""]
    for row in rows:
        links = []
        if (out / f"{row['name']}_config.yaml").exists():
            links.append(f"[Config]({row['name']}_config.yaml)")
        if (out / f"{row['name']}_dream.gif").exists():
            links.append(f"[Dream]({row['name']}_dream.gif)")
        lines.append(f"- **{row['name']}:** {describe(row['config'])} " + " · ".join(links))
    readme.write_text("\n".join(lines).rstrip() + "\n")

    table = [dict(rank=i, name=r["name"], config=r["config"], parameters=r["parameters"],
                  train_seconds=r["train_seconds"], best_step=r.get("best_step"), evaluation=r["evaluation"],
                  provenance=r["provenance"]) for i, r in enumerate(rows, 1)]
    (out / "leaderboard.json").write_text(json.dumps(table, indent=2) + "\n")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(ROOT / "results/animation/bouncing_sprite"))
    parser.add_argument("--out", default=str(ROOT / "reports/animation/leaderboard"))
    args = parser.parse_args()
    build(args.results, args.out)


if __name__ == "__main__":
    main()
