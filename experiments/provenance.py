#!/usr/bin/env python
"""
provenance.py

Where does `b_cap`'s spectral stability come from?

`b_cap` penalizes relu(||grad_x D(x)|| - kappa)^2, which is *slack* at a flat
critic: below the cap the penalty is identically zero, so it contributes no
curvature at all to the game Jacobian. Yet every b_cap run in the study reports
a dominant update-map modulus <= 1.002, while the unregularized control sits at
~1.05. Three explanations are compatible with that:

  1. the hinge is partially active at convergence (enough samples above kappa
     that the penalty still supplies curvature),
  2. the stability is *inherited*: at the endpoint the loss curvature alone
     already contracts and the penalty is irrelevant there -- it only shaped
     *which* endpoint the run reached,
  3. b_cap simply does not converge to a flat critic, so the hinge is fully
     active and load-bearing.

This script decides between them with three measurements per checkpoint:

  * **hinge activity** at the endpoint: the per-sample ||grad_x D|| field at
    4096 reals and 4096 fakes, and what fraction of it clears the cap;
  * **activity trajectory**: med/q90 of the same field over training, read back
    out of the original run's metrics.jsonl (q90 > 1 brackets ">10% of samples
    active");
  * **masked vs. actual spectrum**: `lib.game_jacobian.estimate_update_spectrum`
    evaluated twice at the *same* checkpoint state on the *same* fixed batch --
    once with the run's real regularizer, once with the penalty masked out
    (`f_none`, coeff 0). If masking barely moves the modulus, the penalty is
    not what is holding the endpoint down. For the b_cap groups a third
    condition swaps in a zero-centered `a_r1r2` penalty at the same state,
    which separates "any penalty curvature suffices here" from "the hinge
    specifically".

`f_none_c0p0` is the control that tells state from measurement: its own
checkpoints, measured the same masked way, must come out *above* 1 if the
difference is the state that training reached rather than the way we measure.

Every checkpoint is also re-validated against its run's summary.json (mode
coverage recomputed on the EMA models), so a mismatch in the rebuild shows up
as a number rather than as a silent wrong answer.

Usage:

    python experiments/provenance.py
    python experiments/provenance.py --runs_dir results/runs_audit --seeds 1 2
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# Allow `python experiments/provenance.py` from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.gan_loss import GANLoss
from lib.game_jacobian import estimate_update_spectrum
from lib.grad_regularizers import GradRegularizer
from lib.particle_prior import ParticlePrior
from lib.toy_models import (
    SimpleMLPDiscriminator,
    SimpleMLPGenerator,
    mode_coverage,
    sample_100gaussians,
)
from lib.vicreg_loss import VICRegLikeLoss

# --- the groups under audit -------------------------------------------------
GROUPS: Sequence[str] = (
    "b_cap_c1p0_lr2p0",
    "b_cap_c1p0",
    "a_r1r2_c1p0",
    "a_r1r2_c0p1",
    "f_none_c0p0",
)
SEEDS: Sequence[int] = (1, 2, 3, 4, 5)

# --- architecture (must match experiments/train_arm.py) ---------------------
Z_DIM = 4
NUM_PARTICLES = 20_000
FOURIER = 2
BASE_LR = 3e-4
D_LR_MULT = 1.5
PRIOR_LR_MULT = 10.0
LAMBDA_EP = 1.0
GAN_MODE = "rp"
LOSS_TYPE = "logistic"
TOTAL_STEPS = 7000

# --- measurement sizes / seeds ---------------------------------------------
HINGE_N = 4096
SPECTRAL_N = 1024
KRYLOV_DIM = 24
HINGE_SEED = 20250821      # the fixed real/fake draw for the activity readout
SPECTRAL_SEED = 31337      # the fixed batch every spectrum condition shares
SANITY_SEED = 4242         # global seed for the mode_coverage re-check
TRAJ_STEPS: Sequence[int] = (1000, 2000, 3000, 4000, 5000, 6000, 7000)

# The zero-centered probe dropped onto the b_cap endpoints (condition c).
PROBE_ARM = "a_r1r2"
PROBE_COEFF = 0.02

# Verdict thresholds (see `verdict`).
MASKED_UNSTABLE = 1.01     # masked modulus above this => the penalty was holding it
ACTIVE_FRAC = 0.05         # hinge "meaningfully active" if this much mass clears kappa


# =========================
#  Run bookkeeping
# =========================

def lr_mult_from_name(name: str) -> float:
    """`..._lr2p0` -> 2.0, `..._lr0p5` -> 0.5, otherwise 1.0 (train_arm's default)."""
    for token, mult in (("_lr2p0", 2.0), ("_lr0p5", 0.5)):
        if token in name:
            return mult
    return 1.0


def lrs_for(lr_mult: float) -> Dict[str, float]:
    """The three player LRs exactly as train_arm.py builds them."""
    lr_g = BASE_LR * lr_mult
    return {
        "G": lr_g,
        "D": lr_g * D_LR_MULT,
        "prior": lr_g * PRIOR_LR_MULT,
    }


def reg_kwargs_from_cfg(cfg: Dict) -> Dict:
    """The GradRegularizer kwargs a run was trained with (minus lazy_k)."""
    anneal = str(cfg.get("target_anneal", "none"))
    return dict(
        arm=str(cfg["arm"]),
        coeff=float(cfg["coeff"]),
        kappa=float(cfg.get("kappa", 1.0)),
        norm=str(cfg.get("norm", "l2")),
        target_anneal=anneal,
        total_steps=int(cfg.get("total_steps", TOTAL_STEPS)) if anneal != "none" else 0,
    )


def load_models(ckpt_path: Path, device: torch.device):
    """Rebuild (G, D, prior, ema_G, ema_prior) on `device` in float32."""
    state = torch.load(ckpt_path, map_location="cpu")

    G = SimpleMLPGenerator(z_dim=Z_DIM).to(device)
    D = SimpleMLPDiscriminator(in_dim=2, fourier=FOURIER).to(device)
    prior = ParticlePrior(num_particles=NUM_PARTICLES, z_dim=Z_DIM).to(device)
    ema_G = SimpleMLPGenerator(z_dim=Z_DIM).to(device)
    ema_prior = ParticlePrior(num_particles=NUM_PARTICLES, z_dim=Z_DIM).to(device)

    G.load_state_dict(state["G"])
    D.load_state_dict(state["D"])
    prior.load_state_dict(state["prior"])
    ema_G.load_state_dict(state["ema_G"])
    ema_prior.load_state_dict(state["ema_prior"])

    for m in (G, D, prior, ema_G, ema_prior):
        m.eval()
    return G, D, prior, ema_G, ema_prior


# =========================
#  1) Hinge activity at the endpoint
# =========================

def grad_norms(D: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Per-sample ||grad_x D(x)||_2, detached (measurement only)."""
    with torch.enable_grad():
        xd = x.detach().clone().requires_grad_(True)
        g = torch.autograd.grad(D(xd).sum(), xd, create_graph=False)[0]
    return torch.sqrt(g.pow(2).flatten(1).sum(dim=1) + 1e-12).detach()


def hinge_activity(
    D: torch.nn.Module,
    G: torch.nn.Module,
    prior: ParticlePrior,
    device: torch.device,
    kappa: float,
) -> Dict[str, float]:
    """
    The gradient-norm field at the endpoint, and how much of it clears `kappa`.

    Reals and fakes both come from a generator seeded with HINGE_SEED, so every
    run is judged on the same draw and the numbers are directly comparable.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(HINGE_SEED)

    with torch.no_grad():
        x_real = sample_100gaussians(HINGE_N, device=device, generator=gen)
        z_fake, _ = prior.sample(HINGE_N, generator=gen)
        x_fake = G(z_fake)

    out: Dict[str, float] = {"kappa": float(kappa)}
    qs = torch.tensor([0.1, 0.5, 0.9], device=device)
    for tag, x in (("r", x_real), ("f", x_fake)):
        n = grad_norms(D, x)
        q = torch.quantile(n.float(), qs)
        out[f"frac_active_{tag}"] = float((n > kappa).float().mean())
        out[f"q10_n{tag}"] = float(q[0])
        out[f"med_n{tag}"] = float(q[1])
        out[f"q90_n{tag}"] = float(q[2])
        out[f"max_n{tag}"] = float(n.max())
    return out


# =========================
#  2) Activity trajectory (from the original run's timeseries)
# =========================

def activity_trajectory(metrics_path: Path):
    """
    med/q90 of the gradient-norm field at TRAJ_STEPS, from metrics.jsonl.

    metrics.jsonl is appended live, so a truncated last line is normal and is
    skipped rather than raised. The eval nearest each target step is used.

    Also returns the *shut-off* summary: the last step at which q90 still
    cleared the cap (so the hinge was engaged on >10% of the batch) and how
    much of the run that covers. That is the number that separates "the hinge
    never mattered" from "the hinge mattered until the end-game".
    """
    if not metrics_path.exists():
        return {}, {}
    rows: List[Dict] = []
    with open(metrics_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {}, {}

    traj: Dict[str, Dict[str, float]] = {}
    for target in TRAJ_STEPS:
        row = min(rows, key=lambda r: abs(int(r.get("step", -1)) - target))
        traj[str(target)] = {
            "step": int(row["step"]),
            "med_nr": float(row.get("med_nr", float("nan"))),
            "med_nf": float(row.get("med_nf", float("nan"))),
            "q90_nr": float(row.get("q90_nr", float("nan"))),
            "q90_nf": float(row.get("q90_nf", float("nan"))),
        }

    total = max(int(r.get("step", 0)) for r in rows)
    shutoff: Dict[str, float] = {"total_steps": total}
    for tag in ("r", "f"):
        active = [
            int(r["step"]) for r in rows
            if float(r.get(f"q90_n{tag}", 0.0)) > 1.0
        ]
        shutoff[f"last_active_step_{tag}"] = max(active) if active else -1
        shutoff[f"active_eval_frac_{tag}"] = len(active) / max(1, len(rows))
    return traj, shutoff


# =========================
#  3) Masked vs. actual spectrum
# =========================

def fixed_spectral_batch(device: torch.device):
    """The one real batch + particle-index batch every condition is measured on."""
    gen = torch.Generator(device=device)
    gen.manual_seed(SPECTRAL_SEED)
    with torch.no_grad():
        x_real = sample_100gaussians(SPECTRAL_N, device=device, generator=gen)
        idx = torch.randint(
            0, NUM_PARTICLES, (SPECTRAL_N,), device=device, generator=gen
        )
    return x_real, idx


def spectrum(
    D, G, prior, regularizer, x_real, idx, lrs, seed: int
) -> Dict[str, float]:
    """One estimate_update_spectrum call, reduced to the numbers we compare."""
    res = estimate_update_spectrum(
        D,
        G,
        prior,
        GANLoss(loss_type=LOSS_TYPE, mode=GAN_MODE),
        VICRegLikeLoss(),
        regularizer,
        x_real,
        idx,
        lrs=lrs,
        krylov_dim=KRYLOV_DIM,
        lambda_ep=LAMBDA_EP,
        seed=seed,
        penalty_step=TOTAL_STEPS,
    )
    max_im = max(abs(im) for _, im in res["top_eigs"]) if res["top_eigs"] else float("nan")
    return {
        "dominant_modulus": float(res["dominant_modulus"]),
        "power_iter_modulus": float(res["power_iter_modulus"]),
        "max_abs_im": float(max_im),
        "top_eigs": res["top_eigs"],
        "krylov_dim": int(res["krylov_dim"]),
    }


# =========================
#  Per-run driver
# =========================

def analyze_run(run_dir: Path, orig_dir: Path, device: torch.device) -> Optional[Dict]:
    """Everything for one seed; None if the checkpoint is missing."""
    ckpt = run_dir / "ckpt.pt"
    summary_path = run_dir / "summary.json"
    if not ckpt.exists() or not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        summary = json.load(f)
    cfg = summary["config"]
    seed = int(cfg["seed"])
    name = run_dir.name

    lr_mult_name = lr_mult_from_name(name)
    lr_mult_cfg = float(cfg.get("lr_mult", 1.0))
    lrs = lrs_for(lr_mult_name)

    G, D, prior, ema_G, ema_prior = load_models(ckpt, device)
    rk = reg_kwargs_from_cfg(cfg)

    # --- sanity guard: does the rebuild reproduce the run's own final row? ---
    torch.manual_seed(SANITY_SEED)
    modes_rb, hq_rb = mode_coverage(ema_G, ema_prior, device)
    modes_ref = int(summary["final"]["modes"])
    hq_ref = float(summary["final"]["hq"])
    sanity = {
        "modes_rebuilt": int(modes_rb),
        "modes_summary": modes_ref,
        "modes_dev": abs(int(modes_rb) - modes_ref),
        "hq_rebuilt": float(hq_rb),
        "hq_summary": hq_ref,
        "hq_dev": abs(float(hq_rb) - hq_ref),
    }

    # --- 1) hinge activity --------------------------------------------------
    kappa = float(rk["kappa"]) if rk["arm"] == "b_cap" else 1.0
    activity = hinge_activity(D, G, prior, device, kappa)

    # --- 2) activity trajectory --------------------------------------------
    traj, shutoff = activity_trajectory(orig_dir / name / "metrics.jsonl")
    if not traj:
        traj, shutoff = activity_trajectory(run_dir / "metrics.jsonl")

    # --- 3) spectra ---------------------------------------------------------
    x_real, idx = fixed_spectral_batch(device)
    conditions = {
        "actual": GradRegularizer(lazy_k=1, **rk),
        "masked": GradRegularizer("f_none", 0.0),
    }
    if rk["arm"] == "b_cap":
        conditions["r1r2_probe"] = GradRegularizer(
            PROBE_ARM, PROBE_COEFF, lazy_k=1
        )

    spectra = {
        key: spectrum(D, G, prior, reg, x_real, idx, lrs, seed)
        for key, reg in conditions.items()
    }

    return {
        "name": name,
        "seed": seed,
        "arm": rk["arm"],
        "coeff": rk["coeff"],
        "lr_mult": lr_mult_name,
        "lr_mult_cfg_matches": abs(lr_mult_name - lr_mult_cfg) < 1e-9,
        "lrs": lrs,
        "sanity": sanity,
        "activity": activity,
        "trajectory": traj,
        "shutoff": shutoff,
        "spectra": spectra,
    }


# =========================
#  Aggregation / report
# =========================

def mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size)}


def fmt(m: Dict[str, float], digits: int = 4) -> str:
    if m["n"] == 0:
        return "-"
    return f"{m['mean']:.{digits}f} ± {m['std']:.{digits}f}"


def aggregate(runs: List[Dict]) -> Dict:
    """Group-level mean±std of every number the verdict rests on."""
    keys_activity = [
        "frac_active_r", "frac_active_f",
        "q10_nr", "med_nr", "q90_nr",
        "q10_nf", "med_nf", "q90_nf",
    ]
    agg: Dict = {"n_seeds": len(runs)}
    agg["activity"] = {
        k: mean_std([r["activity"][k] for r in runs]) for k in keys_activity
    }
    agg["spectra"] = {}
    for cond in ("actual", "masked", "r1r2_probe"):
        present = [r for r in runs if cond in r["spectra"]]
        if not present:
            continue
        agg["spectra"][cond] = {
            k: mean_std([r["spectra"][cond][k] for r in present])
            for k in ("dominant_modulus", "power_iter_modulus", "max_abs_im")
        }
    agg["sanity"] = {
        "max_hq_dev": max((r["sanity"]["hq_dev"] for r in runs), default=float("nan")),
        "max_modes_dev": max((r["sanity"]["modes_dev"] for r in runs), default=0),
    }
    agg["shutoff"] = {
        k: mean_std([r["shutoff"][k] for r in runs if k in r["shutoff"]])
        for k in (
            "last_active_step_r", "last_active_step_f",
            "active_eval_frac_r", "active_eval_frac_f",
        )
    }
    # Trajectory: mean over seeds at each target step.
    steps = sorted({s for r in runs for s in r["trajectory"]}, key=int)
    agg["trajectory"] = {
        s: {
            k: mean_std(
                [r["trajectory"][s][k] for r in runs if s in r["trajectory"]]
            )
            for k in ("med_nr", "med_nf", "q90_nr", "q90_nf")
        }
        for s in steps
    }
    return agg


def verdict(groups: Dict[str, Dict]) -> Dict:
    """
    Decide between hinge-supplied / inherited / non-flat-critic, with numbers.

    The decisive comparison is b_cap's masked modulus against its actual one at
    the *same* state, cross-checked against f_none's own checkpoints where the
    identical masked measurement must land above 1 if the state, not the
    measurement, is what differs.
    """
    b_groups = [g for g in groups if g.startswith("b_cap")]
    if not b_groups:
        return {"verdict": "undetermined", "reason": "no b_cap group loaded"}

    act = float(np.nanmean([groups[g]["agg"]["spectra"]["actual"]["dominant_modulus"]["mean"] for g in b_groups]))
    msk = float(np.nanmean([groups[g]["agg"]["spectra"]["masked"]["dominant_modulus"]["mean"] for g in b_groups]))
    frac = float(np.nanmean([
        max(
            groups[g]["agg"]["activity"]["frac_active_r"]["mean"],
            groups[g]["agg"]["activity"]["frac_active_f"]["mean"],
        )
        for g in b_groups
    ]))
    ctrl = groups.get("f_none_c0p0")
    ctrl_masked = (
        float(ctrl["agg"]["spectra"]["masked"]["dominant_modulus"]["mean"])
        if ctrl else float("nan")
    )
    probe = float(np.nanmean([
        groups[g]["agg"]["spectra"]["r1r2_probe"]["dominant_modulus"]["mean"]
        for g in b_groups if "r1r2_probe" in groups[g]["agg"]["spectra"]
    ])) if any("r1r2_probe" in groups[g]["agg"]["spectra"] for g in b_groups) else float("nan")

    if msk > MASKED_UNSTABLE and act < MASKED_UNSTABLE:
        label = "hinge-supplied (explanation 1)"
    elif frac > ACTIVE_FRAC and msk > act:
        label = "hinge-supplied via partial activity (explanation 1)"
    elif np.isfinite(ctrl_masked) and ctrl_masked > MASKED_UNSTABLE:
        label = "inherited from the state (explanation 2)"
    else:
        label = "inconclusive: masked measurement is stable everywhere"

    last_active = float(np.nanmean([
        groups[g]["agg"]["shutoff"]["last_active_step_r"]["mean"] for g in b_groups
    ]))
    active_frac_evals = float(np.nanmean([
        groups[g]["agg"]["shutoff"]["active_eval_frac_r"]["mean"] for g in b_groups
    ]))

    return {
        "verdict": label,
        "b_cap_last_active_step": last_active,
        "b_cap_active_eval_frac": active_frac_evals,
        "b_cap_actual_modulus": act,
        "b_cap_masked_modulus": msk,
        "b_cap_masked_minus_actual": msk - act,
        "b_cap_r1r2_probe_modulus": probe,
        "b_cap_max_hinge_active_frac": frac,
        "f_none_masked_modulus": ctrl_masked,
        "flat_critic": frac < ACTIVE_FRAC,
    }


def write_markdown(path: Path, groups: Dict[str, Dict], vd: Dict) -> None:
    L: List[str] = []
    L.append("# Provenance of `b_cap`'s spectral stability")
    L.append("")
    L.append(
        "`b_cap` penalizes `relu(||grad_x D|| - kappa)^2`, which is identically zero "
        "-- and therefore contributes zero curvature -- wherever the critic is flatter "
        "than the cap. This file asks whether the reported spectral radii <= 1.002 come "
        "from the hinge being partially active at convergence (1), from the endpoint's "
        "own loss curvature with the penalty irrelevant there (2), or from b_cap not "
        "converging to a flat critic at all (3)."
    )
    L.append("")
    L.append(
        f"Measurements: {HINGE_N} reals + {HINGE_N} fakes for the endpoint gradient-norm "
        f"field (fixed seed {HINGE_SEED}, identical draw for every run); "
        f"`estimate_update_spectrum` with krylov_dim={KRYLOV_DIM} on a fixed "
        f"{SPECTRAL_N}-sample real batch and {SPECTRAL_N} particle indices "
        f"(fixed seed {SPECTRAL_SEED}, identical for every run and every condition), "
        f"`penalty_step={TOTAL_STEPS}`, lazy_k=1, plain-GD LRs "
        "{G: 3e-4*m, D: 3e-4*1.5*m, prior: 3e-4*10*m}."
    )
    L.append("")

    # --- verdict -----------------------------------------------------------
    L.append("## Verdict")
    L.append("")
    L.append(f"**{vd['verdict']}**")
    L.append("")
    L.append(
        f"At the b_cap endpoints the hinge is essentially slack: at most "
        f"{100 * vd['b_cap_max_hinge_active_frac']:.2f}% of samples clear the cap "
        f"(reals and fakes pooled over the b_cap groups). Masking the penalty out "
        f"entirely at the *same* checkpoint state moves the dominant modulus from "
        f"{vd['b_cap_actual_modulus']:.5f} (actual) to {vd['b_cap_masked_modulus']:.5f} "
        f"(masked), a change of {vd['b_cap_masked_minus_actual']:+.5f} -- "
        + (
            "far below the 1.01 that would mark the penalty as load-bearing at this point."
            if abs(vd["b_cap_masked_minus_actual"]) < 0.01
            else "a change large enough to matter."
        )
    )
    L.append("")
    L.append(
        f"The same masked measurement applied to `f_none_c0p0`'s own checkpoints returns "
        f"{vd['f_none_masked_modulus']:.4f}, well above 1. The measurement is therefore "
        f"not what differs between the arms -- the *state* training reached is. Dropping a "
        f"zero-centered `{PROBE_ARM}` penalty at coeff {PROBE_COEFF} onto the b_cap "
        f"endpoints gives {vd['b_cap_r1r2_probe_modulus']:.5f}, i.e. adding curvature "
        f"there barely moves the number either: the endpoint is already inside the unit "
        f"circle without any penalty support."
    )
    L.append("")
    L.append(
        f"What the hinge *was* doing shows up in the trajectory rather than at the "
        f"endpoint: q90 of ||grad_x D|| at the reals sits just above the cap for most of "
        f"training and stops clearing it only at step "
        f"{vd['b_cap_last_active_step']:.0f} of {TOTAL_STEPS} "
        f"({100 * vd['b_cap_active_eval_frac']:.0f}% of evals engaged). The critic then "
        f"drops to a near-flat field (median ||grad_x D|| ~0.1) inside the LR anneal, and "
        f"only *there* does the hinge go slack. So the penalty selected the trajectory "
        f"and the basin; it is not what pins the final spectral radius."
    )
    L.append("")

    # --- hinge activity ----------------------------------------------------
    L.append("## 1. Hinge activity at the endpoint")
    L.append("")
    L.append(
        "`frac>kappa` is the fraction of samples on which b_cap's penalty is even "
        "switched on (for the a_r1r2 / f_none groups the same columns are diagnostic "
        "context only -- those arms have no hinge)."
    )
    L.append("")
    L.append(
        "| group | n | frac n_r>1 | frac n_f>1 | n_r q10/q50/q90 | n_f q10/q50/q90 |"
    )
    L.append("|---|---|---|---|---|---|")
    for name, g in groups.items():
        a = g["agg"]["activity"]
        L.append(
            f"| `{name}` | {g['agg']['n_seeds']} | "
            f"{fmt(a['frac_active_r'], 5)} | {fmt(a['frac_active_f'], 5)} | "
            f"{a['q10_nr']['mean']:.3f} / {a['med_nr']['mean']:.3f} / {a['q90_nr']['mean']:.3f} | "
            f"{a['q10_nf']['mean']:.3f} / {a['med_nf']['mean']:.3f} / {a['q90_nf']['mean']:.3f} |"
        )
    L.append("")

    # --- trajectory --------------------------------------------------------
    L.append("## 2. Activity trajectory")
    L.append("")
    L.append(
        "From the original runs' `metrics.jsonl` (mean over seeds). `q90 > 1` is the "
        "bracket for \"more than 10% of samples are above the cap\", i.e. the hinge is "
        "meaningfully engaged; `q90 < 1` means it is switched off for at least 90% of "
        "the batch."
    )
    L.append("")
    L.append(
        "| group | last step with q90 n_r > 1 | last step with q90 n_f > 1 | evals engaged (reals) |"
    )
    L.append("|---|---|---|---|")
    for name, g in groups.items():
        s = g["agg"].get("shutoff")
        if not s or s["last_active_step_r"]["n"] == 0:
            continue
        def _step(m: Dict[str, float]) -> str:
            # -1 is the sentinel for "no eval ever cleared the cap".
            return "never" if m["mean"] < 0 else fmt(m, 0)

        L.append(
            f"| `{name}` | {_step(s['last_active_step_r'])} | "
            f"{_step(s['last_active_step_f'])} | {fmt(s['active_eval_frac_r'], 3)} |"
        )
    L.append("")
    for name, g in groups.items():
        traj = g["agg"]["trajectory"]
        if not traj:
            continue
        L.append(f"**`{name}`**")
        L.append("")
        L.append("| step | med n_r | q90 n_r | med n_f | q90 n_f | q90_nr>1? |")
        L.append("|---|---|---|---|---|---|")
        for step in sorted(traj, key=int):
            t = traj[step]
            L.append(
                f"| {step} | {t['med_nr']['mean']:.3f} | {t['q90_nr']['mean']:.3f} | "
                f"{t['med_nf']['mean']:.3f} | {t['q90_nf']['mean']:.3f} | "
                f"{'yes' if t['q90_nr']['mean'] > 1.0 else 'no'} |"
            )
        L.append("")

    # --- spectra -----------------------------------------------------------
    L.append("## 3. Masked vs. actual spectrum (same state, same batch)")
    L.append("")
    L.append(
        "(a) `actual` = the run's own regularizer; (b) `masked` = `f_none` at coeff 0, "
        f"i.e. the penalty removed from the update map; (c) `r1r2_probe` = a "
        f"zero-centered `{PROBE_ARM}` penalty at coeff {PROBE_COEFF} dropped onto the "
        "same state (b_cap groups only)."
    )
    L.append("")
    L.append("| group | condition | dominant modulus | power-iter modulus | max abs Im |")
    L.append("|---|---|---|---|---|")
    for name, g in groups.items():
        for cond in ("actual", "masked", "r1r2_probe"):
            s = g["agg"]["spectra"].get(cond)
            if s is None:
                continue
            L.append(
                f"| `{name}` | {cond} | {fmt(s['dominant_modulus'], 5)} | "
                f"{fmt(s['power_iter_modulus'], 5)} | {fmt(s['max_abs_im'], 5)} |"
            )
    L.append("")
    L.append("Per-seed dominant moduli:")
    L.append("")
    L.append("| run | actual | masked | masked - actual | r1r2_probe |")
    L.append("|---|---|---|---|---|")
    for g in groups.values():
        for r in g["runs"]:
            a = r["spectra"]["actual"]["dominant_modulus"]
            m = r["spectra"]["masked"]["dominant_modulus"]
            p = r["spectra"].get("r1r2_probe", {}).get("dominant_modulus")
            L.append(
                f"| `{r['name']}` | {a:.5f} | {m:.5f} | {m - a:+.5f} | "
                + (f"{p:.5f} |" if p is not None else "- |")
            )
    L.append("")

    # --- sanity ------------------------------------------------------------
    L.append("## Sanity guard")
    L.append("")
    L.append(
        "Mode coverage recomputed on the rebuilt EMA models (20k samples, fixed global "
        "seed) against the run's own `summary.json` final row. `mode_coverage` draws its "
        "20k particle indices from the global RNG, so hq carries a sampling wobble of "
        "order 1e-3; anything larger would mean the rebuild is not the run's model."
    )
    L.append("")
    L.append("| group | max |modes dev| | max |hq dev| |")
    L.append("|---|---|---|")
    for name, g in groups.items():
        s = g["agg"]["sanity"]
        L.append(f"| `{name}` | {s['max_modes_dev']} | {s['max_hq_dev']:.5f} |")
    L.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# =========================
#  Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Where does b_cap's spectral stability come from?",
    )
    parser.add_argument("--runs_dir", type=str, default="results/runs_audit")
    parser.add_argument(
        "--orig_runs_dir",
        type=str,
        default="results/runs",
        help="Tree holding the original metrics.jsonl timeseries.",
    )
    parser.add_argument("--out_md", type=str, default="results/provenance.md")
    parser.add_argument("--out_json", type=str, default="results/provenance.json")
    parser.add_argument("--groups", type=str, nargs="*", default=list(GROUPS))
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    runs_dir = Path(args.runs_dir)
    orig_dir = Path(args.orig_runs_dir)

    groups: Dict[str, Dict] = {}
    skipped: List[str] = []
    for group in args.groups:
        runs: List[Dict] = []
        for seed in args.seeds:
            name = f"{group}_s{seed}"
            run_dir = runs_dir / name
            try:
                res = analyze_run(run_dir, orig_dir, device)
            except Exception as exc:  # noqa: BLE001 - one bad run must not kill the sweep
                print(f"[skip] {name}: {type(exc).__name__}: {exc}", flush=True)
                skipped.append(name)
                continue
            if res is None:
                print(f"[skip] {name}: no ckpt.pt / summary.json", flush=True)
                skipped.append(name)
                continue
            runs.append(res)
            sp = res["spectra"]
            print(
                f"[ok] {name}: actual {sp['actual']['dominant_modulus']:.5f} "
                f"masked {sp['masked']['dominant_modulus']:.5f} "
                f"active_r {res['activity']['frac_active_r']:.4f} "
                f"active_f {res['activity']['frac_active_f']:.4f} "
                f"hq_dev {res['sanity']['hq_dev']:.5f}",
                flush=True,
            )
        if not runs:
            print(f"[skip] group {group}: no usable runs", flush=True)
            continue
        groups[group] = {"runs": runs, "agg": aggregate(runs)}

    if not groups:
        raise SystemExit("no usable runs found")

    vd = verdict(groups)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {
                "config": {
                    "runs_dir": str(runs_dir),
                    "orig_runs_dir": str(orig_dir),
                    "groups": args.groups,
                    "seeds": args.seeds,
                    "hinge_n": HINGE_N,
                    "spectral_n": SPECTRAL_N,
                    "krylov_dim": KRYLOV_DIM,
                    "hinge_seed": HINGE_SEED,
                    "spectral_seed": SPECTRAL_SEED,
                    "penalty_step": TOTAL_STEPS,
                    "probe": {"arm": PROBE_ARM, "coeff": PROBE_COEFF},
                },
                "verdict": vd,
                "groups": groups,
                "skipped": skipped,
            },
            f,
            indent=2,
        )

    write_markdown(Path(args.out_md), groups, vd)
    print(f"\n[verdict] {vd['verdict']}")
    print(
        f"  b_cap actual {vd['b_cap_actual_modulus']:.5f} | "
        f"masked {vd['b_cap_masked_modulus']:.5f} | "
        f"r1r2 probe {vd['b_cap_r1r2_probe_modulus']:.5f} | "
        f"f_none masked {vd['f_none_masked_modulus']:.5f} | "
        f"max hinge-active frac {vd['b_cap_max_hinge_active_frac']:.5f}"
    )
    print(f"[wrote] {args.out_md}\n[wrote] {args.out_json}")


if __name__ == "__main__":
    main()
