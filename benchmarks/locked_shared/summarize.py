"""Render the recorded formulation comparisons without rerunning training."""

import json
from pathlib import Path


LABELS = {
    "base_core": "Locked / base GAN core",
    "r1_r2": "R1+R2, coefficient 0.02",
    "eikonal": "Eikonal",
    "interpolation_cap": "Interpolation cap",
    "relativistic_average": "Ra logistic",
    "rp_hinge": "Rp hinge",
    "no_vicreg": "Remove VICReg",
    "base_prior_weight": "VICReg weight 1",
    "no_particle_l2": "Remove particle L2",
    "no_cover": "Remove trajectory cover",
    "music_cover": "Trajectory cover 1.0",
    "stronger_cap": "Cap coefficient 10",
    "base_regularization": "Base regularization on tiny host",
    "no_prior_regularization": "Remove both prior penalties",
    "r1_r2_0_1": "R1+R2, coefficient 0.1",
}


def main():
    root = Path("reports/locked_shared")
    initial = json.loads((root / "investigation.json").read_text())
    followup = json.loads((root / "followup.json").read_text())
    stock = json.loads((root / "base_recipe.json").read_text())
    rows = initial["rows"] + followup["rows"]
    lines = ["# Which existing formulation works on these toys?", "",
             "**Removing the host particle L2 term is the strongest small-change candidate in this fixed-seed comparison:** "
             "all three measured targets pass, trajectory MSE is 0.002835, and the ring holds 8/8 modes at 100% HQ. "
             "The RpGAN loss, b_cap, models and training budgets stay unchanged. "
             "This is a candidate for these toys, not a universal replacement for particle L2.", "",
             "## Same-budget comparison", "",
             "All rows use seed 0. Training budgets remain two-pole 80, trajectory 400, ring 1,200 steps. "
             "Default: cover 1.5 (trajectory only), particle L2 0.02, VICReg 0.05 (trajectory/ring only). "
             "Only the named term changes unless the row explicitly combines settings. "
             "Two-pole has no VICReg or cover training loss, so those ablations leave that host unchanged.", "",
             "| Formulation | Two-pole slope / result | Trajectory MSE / result | Ring modes; HQ / result | Targets passed |",
             "| --- | --- | --- | --- | --- |"]
    order = {name: i for i, name in enumerate(("no_particle_l2", "no_vicreg", "base_regularization", "base_core"))}
    for row in sorted(rows, key=lambda r: order.get(r["name"], 100)):
        pole, traj, ring = row.get("two_pole"), row.get("trajectory"), row.get("ring")
        values = [pole, traj, ring]
        count = sum(v is not None and v["verdict"] == "PASS" for v in values)
        total = sum(v is not None for v in values)
        p = f"{pole['grad_med']:.4f} / {pole['verdict']}" if pole else "Not run"
        t = f"{traj['identity_mse']:.6f} / {traj['verdict']}" if traj else "Not run"
        m = f"{ring['modes']}/8; {ring['hq']:.2%} / {ring['verdict']}" if ring else "Not run"
        lines.append(f"| {LABELS[row['name']]} | {p} | {t} | {m} | **{count}/{total}** |")
    lines += ["", "Two-pole also requires travel ≥ 0.30; every candidate above reaches that travel threshold. "
              "Its slope limit is 1.0. Trajectory MSE must be ≤ 0.02. Ring requires ≥ 7 modes and HQ ≥ 90%. "
              "INCONCLUSIVE is preserved. Not-run combinations earn no pass.", "",
              "The base-core row calls `get_recipe('gan').make_loss()` and `.make_gradient_penalty()`. "
              "It is numerically identical to locked_shared on these hosts. "
              "Base regularization means VICReg 1, particle L2 0, trajectory cover 0; "
              "the small cloud, host optimizers and budgets stay fixed in that row.", "",
              "## Stock recipe on the ring host", "",
              "Here the stock recipe supplies its 20,000-particle prior, VICReg 1, no particle L2, "
              "batch 256, learning rate 0.0006, discriminator LR multiplier 1.5, prior multiplier 10, "
              "betas (0, 0.999), EMA 0.995 and cosine decay. The original 8-mode ring data, "
              "96-wide host networks/initialization and 4,096-sample evaluation remain. "
              "This applies the recipe to the ring; it is not a run of the separate 100-Gaussians trainer.", "",
              "| Budget | EMA modes / HQ | EMA result | Live modes / HQ | Live result |",
              "| --- | --- | --- | --- | --- |"]
    for row in stock["rows"]:
        r = row["ring"]
        live = r["live"]
        lines.append(f"| {r['step']:,} | {r['modes']}/8; {r['hq']:.2%} | {r['verdict']} | {live['modes']}/8; {live['hq']:.2%} | {live['verdict']} |")
    lines += ["", "The 7,000-step run uses the stock budget and more particles; it is not an equal-compute win over the 1,200-step tiny-cloud runs.", "",
              "## What the measurements suggest", "",
              "1. **No implementation drift was found.** PR #36, PyPI 0.5.0, and the base commit of the original ParticleGAN toy PRs "
              "use identical GAN loss, gradient penalty, particle-prior and VICReg source files. The current conceptmod suite reproduces "
              "19 PASS and 2 FAIL after excluding cover-posture columns; the two failures are the same trajectory/ring cases.", "",
              "2. **The auxiliary objectives interact badly in this small host.** Removing either particle L2 or VICReg clears both missed "
              "targets; removing both fails trajectory again (MSE 0.263834). L2 contracts the cloud while VICReg encourages spread. "
              "These interventions establish sensitivity to their combination, not a proof that one term is always harmful.", "",
              "3. **Set coverage does not guarantee paired identity.** In the locked trajectory run only 6/12 predictions are nearest "
              "to their own target. Removing cover or reducing it from 1.5 to 1.0 restores 12/12 and passes identity MSE. "
              "The set-cover loss can reward the right collection of arcs under the wrong assignment; it competes with the identity objective.", "",
              "4. **EMA and training stability matter.** The no-L2 ring's reported EMA is 8/8 at 100% HQ, but its live model is "
              "8/8 at 74.05% HQ. The stock 7,000-step run passes with both EMA and live weights. The no-L2 candidate is promising, "
              "but its live dynamics are not solved. R1+R2 at 0.02 passes trajectory/ring, yet misses the two-pole slope requirement "
              "(1.0976 > 1); increasing it to 0.1 fixes that bound but loses the ring EMA target.", "",
              "**Recommendation:** keep the current production defaults unchanged in this verification PR. "
              "Use the no-particle-L2 arm as the next candidate for hosts where VICReg already supplies the prior regularization; "
              "retain the measured identity, diversity and slope tests. Keep the stock recipe as the longer-budget baseline. "
              "There is no evidence here that a new GAN formulation is needed, and no candidate has been shown to win every behavioral toy "
              "or downstream task. No seed search was performed.", "",
              "## Reproduce", "", "```bash",
              "python -m benchmarks.locked_shared.investigate --only base_core r1_r2 eikonal interpolation_cap relativistic_average rp_hinge no_vicreg base_prior_weight no_particle_l2 no_cover music_cover stronger_cap",
              "python -m benchmarks.locked_shared.investigate --only base_regularization no_prior_regularization r1_r2_0_1 --output reports/locked_shared/followup.json",
              "python -m benchmarks.locked_shared.base_recipe",
              "python -m benchmarks.locked_shared.summarize", "```", "",
              "Raw metrics and training curves: [initial comparisons](investigation.json), [follow-up controls](followup.json), "
              "[stock recipe](base_recipe.json). These are actual trained outcomes, not configuration acceptance checks.", ""]
    (root / "comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
