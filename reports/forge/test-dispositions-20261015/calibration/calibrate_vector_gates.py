#!/usr/bin/env python
"""DECLARED MEASUREMENT CALIBRATION (CPU only; no training, no GPU, no recipe, no generator).

Question: how often does a perfect sampler -- Monte Carlo draws from the task's own true target
mixture -- pass each sub-gate of Forge's vector_unequal_mass (and, cheaply, vector_unequal_width)?
Controls confirm the gates still reject real failures (collapsed rare component, missing mass).

Faithfulness:
  * spec      = experiments.forge.vectorprofiles.resolve_vector_spec(task) with
                spec["thresholds"] = task["evaluation"]["thresholds"]  (exactly as adapters._vector)
  * draws     = benchmarks.transfer_suite.vector_tasks.sample_target (the repository's own target
                sampler) or a bit-identical replica that also returns component labels (asserted)
  * scoring   = benchmarks.transfer_suite.vector_tasks.score_samples, unchanged, 4096 samples per
                observation (adapters._vector: run.sample(4096)); its internal seed-991 reference and
                seed-992 projections are untouched
  * verdict   = experiments.forge.views._transfer (Forge grading path -> protocol.test_verdict),
                24 observations at steps 50..1200, terminal suffix of >= 5 passing checks
  * panels other than "perfect" simulate only the 5 terminal checks: for any complete curve the
    verdict is PASS iff the last 5 checks all pass (asserted on the 24-check panels)

Named RNG seeds: seed(label) = int(sha256(label).hexdigest()[:16], 16) % 2**63 with
label = "<task>|<panel>|rep=<r>|obs=<o>" (or "|centers" for frozen particle centers).

Usage (CPU only, about 5 minutes for the full 200-repetition run):
  python reports/forge/test-dispositions-20261015/calibration/calibrate_vector_gates.py [--quick]
writes calibration-<full|quick>-summary.json next to this script (or in --output-dir), plus a tail-able
calibration-<full|quick>-panels.jsonl progress log. The summary repeats every panel record, so only the
summary is committed; like other JSONL traces, the log stays outside Git. The per-check rows (25,600 for
the full run, about 30 MB) are written only to an explicit --observations path; keep that file outside Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

torch.set_num_threads(2)

from benchmarks.transfer_suite.vector_tasks import score_samples, sample_target, passes, EVAL_SAMPLES  # noqa: E402
from experiments.forge.views import _transfer  # noqa: E402
from experiments.forge.vectorprofiles import resolve_vector_spec  # noqa: E402

OUT = Path(__file__).resolve().parent
STEPS = 1200
CHECKS = [math.ceil(i * STEPS / 24) for i in range(1, 25)]
PARTICLES = 256


def named_seed(label: str) -> int:
    return int(hashlib.sha256(label.encode()).hexdigest()[:16], 16) % (2 ** 63)


def gen(label: str) -> torch.Generator:
    return torch.Generator().manual_seed(named_seed(label))


def load(task_id):
    task = json.load(open(f"{REPO}/configs/forge/tasks/{task_id}.json"))
    spec = resolve_vector_spec(task)
    spec["thresholds"] = task["evaluation"]["thresholds"]
    return task, spec


def draw_mixture(means, cov, masses, count, rng):
    """Bit-identical replica of vector_tasks.sample_target (gaussian_mixture) that keeps labels."""
    means = torch.as_tensor(means, dtype=torch.float32)
    cov = torch.as_tensor(cov, dtype=torch.float32)
    index = torch.multinomial(torch.as_tensor(masses, dtype=torch.float32), count, replacement=True, generator=rng)
    noise = torch.randn(count, 2, generator=rng)
    points = means[index] + torch.bmm(torch.linalg.cholesky(cov)[index], noise.unsqueeze(2)).squeeze(2)
    return points, index


def per_component_full_eigen(fake, spec):
    """Replica of score_samples' full-component shape(): per-component min whitened eigenvalue."""
    means = torch.tensor(spec["means"])
    cov = torch.tensor(spec["covariances"])
    assignment = torch.cdist(fake, means).argmin(1)
    ratios = []
    for k in range(len(means)):
        pts = fake[assignment == k]
        if len(pts) < 10:
            ratios.append(0.0)
            continue
        x = pts - pts.mean(0)
        emp = x.T @ x / len(pts)
        inv = torch.linalg.inv(torch.linalg.cholesky(cov[k]))
        ratios.append(float(torch.linalg.eigvalsh(inv @ emp @ inv.T).min()))
    return ratios


def gate_results(metrics, thresholds):
    out = {}
    for key, op, bound in thresholds:
        v = metrics.get(key)
        out[key] = isinstance(v, (int, float)) and math.isfinite(v) and (v >= bound if op == ">=" else v <= bound)
    return out


def record(task_id, panel, rep, obs, seed_label, fake, spec, extra=None):
    m = score_samples(fake, spec, STEPS)
    eig = per_component_full_eigen(fake, spec)
    # The replica must reproduce the scorer's own minimum exactly.
    assert abs(min(eig) - m["component_min_eigen_ratio"]) < 1e-6, (min(eig), m["component_min_eigen_ratio"])
    gates = gate_results(m, spec["thresholds"])
    row = dict(task=task_id, panel=panel, rep=rep, obs=obs, seed_label=seed_label, seed=named_seed(seed_label),
               passed=all(gates.values()), gates=gates,
               metrics={k: m[k] for k in ("sw1_normalized", "mass_tv", "hq", "component_covariance_error",
                                          "component_min_eigen_ratio", "min_mass_ratio", "mean_error",
                                          "covariance_error")},
               component_counts=m["component_counts"], component_mass=m["component_mass"],
               component_full_eigen=eig, component_covariance_errors=m["component_covariance_errors"],
               component_core_eigen=m["component_core_eigen_ratios"], component_spill=m["component_spill"])
    if extra:
        row.update(extra)
    return m, row


# --------------------------------------------------------------------------- samplers
def perfect_sampler(spec):
    def sample(label, rep_state):
        return sample_target(spec, EVAL_SAMPLES, gen(label), STEPS)
    return sample


def modified_sampler(spec, *, masses=None, shrink=None, halo=None):
    """True-law draws with one declared modification (controls)."""
    means, cov = spec["means"], spec["covariances"]
    base_masses = masses if masses is not None else spec["masses"]

    def sample(label, rep_state):
        rng = gen(label)
        pts, idx = draw_mixture(means, cov, base_masses, EVAL_SAMPLES, rng)
        if shrink is not None:  # (component, (sx, sy)): contract deviations from the component mean
            k, (sx, sy) = shrink
            mu = torch.tensor(means[k], dtype=torch.float32)
            sel = idx == k
            d = pts[sel] - mu
            d = d * torch.tensor([sx, sy], dtype=torch.float32)
            pts[sel] = mu + d
        if halo is not None:  # (components, fraction, halo_sigma): replace a fraction by a wide halo
            comps, frac, hs = halo
            for k in comps:
                sel = torch.nonzero(idx == k).squeeze(1)
                n = int(round(frac * len(sel)))
                if n:
                    pick = sel[torch.randperm(len(sel), generator=rng)[:n]]
                    mu = torch.tensor(means[k], dtype=torch.float32)
                    pts[pick] = mu + hs * torch.randn(n, 2, generator=rng)
        return pts
    return sample


def largest_remainder(masses, total):
    raw = [m * total for m in masses]
    base = [int(math.floor(r)) for r in raw]
    order = sorted(range(len(raw)), key=lambda k: raw[k] - base[k], reverse=True)
    for k in order[: total - sum(base)]:
        base[k] += 1
    return base


def particle_sampler(spec, f):
    """Emulates Forge's served law: 256 equally weighted particles (uniform index draws, as
    ParticlePrior.sample_indices) and a locally linear generator whose Gaussian kernel around each
    particle has covariance f^2 * Sigma_k; frozen centers ~ N(mu_k, (1 - f^2) Sigma_k). Averaged over
    centers each component is exactly the target Gaussian; f=1 is the target law with masses
    quantized to 141/77/33/5 particles, f=0 is the finite-atom law of the upstream oracle."""
    means = torch.tensor(spec["means"], dtype=torch.float32)
    chol = torch.linalg.cholesky(torch.tensor(spec["covariances"], dtype=torch.float32))
    alloc = largest_remainder(spec["masses"], PARTICLES)
    comp_of = torch.repeat_interleave(torch.arange(len(alloc)), torch.tensor(alloc))

    def centers(label):
        rng = gen(label)
        e = torch.randn(PARTICLES, 2, generator=rng)
        return means[comp_of] + math.sqrt(max(0.0, 1 - f * f)) * torch.bmm(chol[comp_of], e.unsqueeze(2)).squeeze(2)

    def sample(label, rep_state):
        rng = gen(label)
        j = torch.randint(0, PARTICLES, (EVAL_SAMPLES,), generator=rng)
        eps = torch.randn(EVAL_SAMPLES, 2, generator=rng)
        return rep_state["centers"][j] + f * torch.bmm(chol[comp_of[j]], eps.unsqueeze(2)).squeeze(2)
    return sample, centers, alloc


# --------------------------------------------------------------------------- panel runner
def run_panel(task_id, spec, panel, sampler, reps, *, full_curve=False, centers=None, log=None, rows=None):
    t0 = time.time()
    verdicts, last5_rule, obs_pass, gate_pass = [], [], [], {k: [] for k, _, _ in spec["thresholds"]}
    checks = CHECKS if full_curve else CHECKS[-5:]
    for rep in range(reps):
        state = {}
        if centers is not None:
            state["centers"] = centers(f"{task_id}|{panel}|rep={rep:03d}|centers")
        points = []
        rep_pass = []
        for o, step in enumerate(checks):
            obs_index = CHECKS.index(step)
            label = f"{task_id}|{panel}|rep={rep:03d}|obs={obs_index:02d}"
            fake = sampler(label, state)
            m, row = record(task_id, panel, rep, obs_index, label, fake, spec)
            rows.append(row)
            rep_pass.append(row["passed"])
            obs_pass.append(row["passed"])
            for k, v in row["gates"].items():
                gate_pass[k].append(v)
            points.append({"step": step, **{k: m[k] for k, _, _ in spec["thresholds"]}})
        rule = all(rep_pass[-5:])
        last5_rule.append(rule)
        if full_curve:
            task = json.load(open(f"{REPO}/configs/forge/tasks/{task_id}.json"))
            grade = _transfer(task, {"observations": points, "live": points[-1]})
            verdict = grade["status"] == "PASS"
            assert verdict == rule, (grade, rep_pass)
            verdicts.append(verdict)
        else:
            verdicts.append(rule)
    n = len(obs_pass)
    summary = dict(task=task_id, panel=panel, reps=reps, checks_per_rep=len(checks), observations=n,
                   official_curve_grader=full_curve,
                   per_check_pass_rate=sum(obs_pass) / n,
                   verdict_pass_rate=sum(verdicts) / reps,
                   per_gate_pass_rate={k: sum(v) / len(v) for k, v in gate_pass.items()},
                   seconds=round(time.time() - t0, 1))
    if log:
        log.write(json.dumps(summary) + "\n")
        log.flush()
    print(f"[{time.strftime('%H:%M:%S')}] {task_id:21s} {panel:28s} check-pass={summary['per_check_pass_rate']:.4f} "
          f"verdict-pass={summary['verdict_pass_rate']:.4f} gates={ {k: round(v, 4) for k, v in summary['per_gate_pass_rate'].items()} } "
          f"({summary['seconds']}s)", flush=True)
    return summary


def quantiles(values, qs=(0.0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)):
    v = sorted(values)
    out = {}
    for q in qs:
        i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
        out[f"q{q:g}"] = v[i]
    out["mean"] = sum(v) / len(v)
    return out


def distribution_summary(rows, task_id, panel):
    rs = [r for r in rows if r["task"] == task_id and r["panel"] == panel]
    target = json.load(open(f"{REPO}/configs/forge/tasks/{task_id}.json"))["execution"]["host_definition"]["masses"]
    k = len(rs[0]["component_counts"])
    out = {"n": len(rs)}
    out["component_min_eigen_ratio"] = quantiles([r["metrics"]["component_min_eigen_ratio"] for r in rs])
    out["min_mass_ratio"] = quantiles([r["metrics"]["min_mass_ratio"] for r in rs])
    for key in ("sw1_normalized", "mass_tv", "hq", "component_covariance_error"):
        out[key] = quantiles([r["metrics"][key] for r in rs])
    out["per_component"] = []
    for c in range(k):
        out["per_component"].append(dict(
            component=c,
            count=quantiles([r["component_counts"][c] for r in rs]),
            mass_ratio=quantiles([r["component_mass"][c] / target[c] for r in rs]),
            full_eigen=quantiles([r["component_full_eigen"][c] for r in rs]),
            covariance_error=quantiles([r["component_covariance_errors"][c] for r in rs]),
            spill=quantiles([r["component_spill"][c] for r in rs])))
    # Which component sets the minimum eigen ratio / minimum mass ratio.
    out["argmin_eigen_component"] = {str(c): sum(1 for r in rs if min(range(k), key=lambda j: r["component_full_eigen"][j]) == c) for c in range(k)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--observations", type=Path, help="also write every per-check row here (large; keep outside Git)")
    args = ap.parse_args()
    reps = 4 if args.quick else args.reps
    tag = "quick" if args.quick else "full"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = []
    log = open(out / f"calibration-{tag}-panels.jsonl", "w")
    summaries = []
    meta = dict(declared="measurement calibration; CPU-only Monte Carlo from the true target law; no training",
                torch=torch.__version__, python=sys.version.split()[0], reps=reps, eval_samples=EVAL_SAMPLES,
                checks=CHECKS, seed_rule="int(sha256(label).hexdigest()[:16],16) % 2**63",
                scorer_sha256={p: hashlib.sha256(open(f"{REPO}/{p}", "rb").read()).hexdigest() for p in (
                    "benchmarks/transfer_suite/vector_tasks.py", "benchmarks/transfer_suite/protocol.py",
                    "benchmarks/locked_shared/observation.py", "experiments/forge/views.py",
                    "configs/forge/tasks/vector_unequal_mass.json", "configs/forge/tasks/vector_unequal_width.json")})

    # ---------------- vector_unequal_mass
    task_id = "vector_unequal_mass"
    task, spec = load(task_id)
    # Replica check: draw_mixture reproduces sample_target bit-for-bit.
    a = sample_target(spec, EVAL_SAMPLES, gen("replica-check"), STEPS)
    b, _ = draw_mixture(spec["means"], spec["covariances"], spec["masses"], EVAL_SAMPLES, gen("replica-check"))
    assert torch.equal(a, b), "replica sampler differs from sample_target"
    meta["replica_sampler_bit_identical"] = True

    summaries.append(run_panel(task_id, spec, "perfect", perfect_sampler(spec), reps, full_curve=True, log=log, rows=rows))
    m = spec["masses"]
    rare = 3
    others = [x / (1 - m[rare]) for x in m[:rare]]

    def with_rare(r):
        return [x * (1 - r) for x in others] + [r]
    controls = {
        "ctrl_rare_line_collapse": modified_sampler(spec, shrink=(rare, (1.0, 0.1))),
        "ctrl_rare_point_collapse": modified_sampler(spec, shrink=(rare, (0.1, 0.1))),
        "ctrl_rare_missing": modified_sampler(spec, masses=with_rare(0.0)),
        "ctrl_rare_mass_1_of_256": modified_sampler(spec, masses=with_rare(1 / 256)),
        "ctrl_rare_mass_2_of_256": modified_sampler(spec, masses=with_rare(2 / 256)),
        "ctrl_uniform_masses": modified_sampler(spec, masses=[0.25] * 4),
    }
    for name, smp in controls.items():
        summaries.append(run_panel(task_id, spec, name, smp, reps, log=log, rows=rows))
    for f in (1.0, 0.5, 0.25, 0.1, 0.0):
        smp, cen, alloc = particle_sampler(spec, f)
        meta["particle_allocation"] = alloc
        summaries.append(run_panel(task_id, spec, f"particles_f{f:g}", smp, reps, centers=cen, log=log, rows=rows))

    # ---------------- vector_unequal_width (cheap check)
    task_id = "vector_unequal_width"
    task, spec = load(task_id)
    summaries.append(run_panel(task_id, spec, "perfect", perfect_sampler(spec), reps, full_curve=True, log=log, rows=rows))
    widths = [math.sqrt(c[0][0]) for c in spec["covariances"]]
    common = math.sqrt(sum(w * w for w in widths) / len(widths))
    common_spec_cov = [[[common ** 2, 0.0], [0.0, common ** 2]] for _ in widths]

    def common_width_sampler(label, state):
        return draw_mixture(spec["means"], common_spec_cov, spec["masses"], EVAL_SAMPLES, gen(label))[0]
    summaries.append(run_panel(task_id, spec, "ctrl_common_width", common_width_sampler, reps, log=log, rows=rows))
    for frac in (0.005, 0.01, 0.02, 0.05):
        summaries.append(run_panel(task_id, spec, f"sens_narrow_halo_{frac:g}",
                                   modified_sampler(spec, halo=([0, 1], frac, 0.5)), reps, log=log, rows=rows))
    meta["common_width"] = common

    # ---------------- distributions
    dists = {}
    for s in summaries:
        dists[f"{s['task']}|{s['panel']}"] = distribution_summary(rows, s["task"], s["panel"])
    if args.observations:
        with open(args.observations, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    meta["seconds"] = round(time.time() - started, 1)
    json.dump(dict(meta=meta, panels=summaries, distributions=dists), open(out / f"calibration-{tag}-summary.json", "w"), indent=1)
    print(f"done in {meta['seconds']}s; wrote calibration-{tag}-summary.json and calibration-{tag}-panels.jsonl", flush=True)


if __name__ == "__main__":
    main()
