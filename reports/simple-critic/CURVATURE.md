# Curvature arms: does a curvature floor at the reals stop the seesaw?

Hypothesis: secant arms flip because D's peak at the reals is a flat plateau (R1 zeroes the slope,
nothing sets a curvature floor), so nothing restores the equilibrium. The ideal D would be a
Huber-rounded negative distance to the nearest real.

**Verdict: rejected.** A higher curvature at the reals goes with *more* departures, both across
arms and within a run. The one change that helped was loosening the gradient cap (c=3), and it
barely changes the curvature.

Code: `curvature_worker.py` wraps `lr_grid.py` and `worker.py` without editing them. The
launcher is `curvature.sh`. Setup: seed 0, no noise, constant LRs (critic ×0.5), Dβ2 0.9, A2
off. Logs are in `logs/<arm>.log`, one line per observation with `curv= wide= gfmed=`
appended. With every new term off, the worker reproduces `lr_c0.5_g1` bit-for-bit. It matches
on all metrics.jsonl fields over 100 updates and on every scored point over 4600 updates.

**κ = 7.14** (1/κ = 0.14 = 2 × the ring mode std 0.07), so the rounded cap spans the mode
(86% of its mass in 2D). It was fixed once and not swept. Both the margin and the Huber terms
use offsets δ in a uniform direction with |δ| ~ U(0, 1/κ), drawn from their own RNG stream.
Each has weight 10.

Curvature diagnostic (observation-only, own RNG): `curv` = mean over 4096 probe reals of
(2D(r) − D(r+e) − D(r−e))/|e|², at |e| = 0.01 in a random direction. This equals 2·mean(D(r) −
D(r±e))/|e|², and positive means peaked. `wide` is the same quantity at |e| = 1/κ. Medians are
taken over all 460 observations.

| arm | formulation delta vs B | prehold | arrival | post-arrival | departures | longest fail streak | fails outside transit | final HQ | median grad (fakes) | max grad | median curv at reals (wide) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **B_cap3** | cap c=3 | 82/120 | +140 | 207/207 | 10 | 23 | **38** | 0.990 | 0.07 | 5.94 | 0.19 (0.37) |
| B = lr_c0.5_g1 | — | 96/120 | +260 | 177/195 | 17 | 21 | 42 | 0.993 | 0.05 | 2.83 | 0.17 (0.27) |
| k3p_constant (ref) | K3P v0.8.0, noise off | 88/120 | +360 | 167/185 | 5 | 32 | 50 | 0.989 | 0.03 | 5.81 | 0.14 (0.28) |
| B_cap2 | cap c=2 | 89/120 | +280 | 170/193 | 13 | 23 | 54 | 0.992 | 0.08 | 4.31 | 0.26 (0.44) |
| B_nnpair | cap points on fake→nearest real | 81/120 | +250 | 124/196 | 31 | 47 | 111 | 0.967 | 0.09 | 5.79 | 0.41 (0.54) |
| B_margin | + peak margin(10) | 85/120 | +240 | 114/197 | 13 | 60 | 118 | 0.935 | 0.08 | 3.05 | 0.48 (0.65) |
| wgan_huber | wgan + Huber profile only | 0/120 | none | – | – | – | 120 | 0.792 | 1.51 | 29.7 | 20.9 (17.5) |
| wgan_margin | wgan + peak margin only | 0/120 | none | – | – | – | 120 | 0.011 | 3177 | 1.8e6 | diverged |

For B and k3p_constant, the curvature numbers come from observation-only reruns in `diag/`.
Every scored field of those reruns is identical to `runs/lr_c0.5_g1` and `runs/k3p_constant`.
`summarize.py` now ranks B_cap3 #1 overall (B was #2).

No combo was run (step 6). B_cap3 was the only arm from steps 1–3 that improved on B.
B_margin and B_nnpair more than doubled the fails. B_cap2 lowered departures but raised fails.

## Hypothesis test
- **The plateau is real but not the cause.** B's curvature at the reals (0.17) is about 40×
  below κ. k3p_constant has the fewest departures (5) and has the *lowest* curvature (0.14).
- **Across the 6 converging arms,** the Spearman correlation between curvature and departures
  is +0.49 (p = 0.32). Between curvature and fails it is +0.77 (p = 0.07). Both are the wrong
  sign for the hypothesis.
- **Within runs,** at observations that pass, curvature relative to the arm's median is higher
  just before a departure than before a hold: 1.08 vs 0.66 (n = 58 vs 1411, Mann–Whitney
  p = 2e-5). So D gets sharper as a flip approaches. Sharpening is a symptom or a precursor of
  the flip, not a missing restoring force.
- **Forcing curvature hurts.** The margin term tripled curvature but left the peaks lopsided
  (fails 42 → 118, post-arrival 114/197). The margin was met in the U(0, 1/κ) band (residual
  term ≈ 0.009) mostly through a cone rather than a quadratic cap: small-δ curvature reached
  only 0.48. nnpair concentrates the cap's points next to the reals and also raised curvature
  (0.41), with 31 departures.
- **Without a Lipschitz constraint, the Huber profile does not work.** With plain wgan it never
  holds 8 modes (curvature 21, max grad 30). The margin alone diverges, since nothing bounds
  D's gap.

## Recommendation
Next base: **B_cap3** (B with c=3). It has the fewest fails (38), zero fails after arrival
(207/207) and the fastest arrival (+140). Its costs are a lower prehold (82 vs 96) and a max
grad of 5.9. Drop curvature-floor terms. Since loosening the cap helps, the next single
question is whether the cap is needed at all on the reals and fakes: cap-interp only at c=3,
versus no cap (`no_cap` already exists for the older base).
