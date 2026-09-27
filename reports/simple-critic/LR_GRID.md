# LR grid on `sec_nodamp`

Base: `sec_nodamp` = wgan + R1(1) + secant path (10, t .5) + cap-all (10, c=1), critic Adam β2 .9, A2 off; no noise, constant LRs, no controller.
Base LRs: G .00425, critic .00425, prior .0085. Arm `lr_c{mc}_g{mg}` sets critic × mc and G and prior × mg, so prior stays 2×G. Seed 0 only.

Tooling: `lr_grid.py` wraps `worker.py` without editing it. It sets `recipe.lr` = base·mg and `d_lr_mult` = mc/mg. The multipliers are powers of 2, so every rate is exact. It checks the requested LRs on every update (4600/4600 for every arm).
The 1/1 cell reproduces `sec_nodamp` bit-for-bit over updates 1–100, matching all 10 stored observations including losses and LRs.
Launcher: `lr_grid.sh`. Logs: `logs/lr_c*.log`.

## Fails outside transit (lower is better; refs: sec_nodamp 50, k3p_constant 50)

| mc \ mg | 0.5 | 1 | 2 |
|---|---|---|---|
| 0.5 | 81 | **42** | 61 |
| 1   | 78 | 50 (sec_nodamp) | 106 |
| 2   | 133 | 194 | 225 |

Joint scale: c0.25/g0.25 → 56; c4/g4 → 120 (never passes: 0/120 prehold, collapsed).

## Arrival, updates after shift (lower is better; refs: sec_nodamp 230, k3p_constant 360)

| mc \ mg | 0.5 | 1 | 2 |
|---|---|---|---|
| 0.5 | 560 | 260 | 740 |
| 1   | **20** | 230 (sec_nodamp) | 340 |
| 2   | 140 | 240 | 350 |

Joint scale: c0.25/g0.25 → 240; c4/g4 → none.

## Full rows (`summarize.py`)

| arm | prehold | arrival | post-arrival | departures | longest fail streak | final suffix | final HQ | max abs D(real) | max grad-norm | fails outside transit |
|---|---|---|---|---|---|---|---|---|---|---|
| ref: k3p_constant | 88/120 | 360 | 167/185 | 5 | 32 | 123 (from 3380) | 0.989 | 2.63 | 5.81 | 50 |
| lr_c0.5_g1 | 96/120 | 260 | 177/195 | 17 | 21 | 123 (from 3380) | 0.993 | 1.87 | 2.83 | 42 |
| sec_nodamp (1/1) | 97/120 | 230 | 171/198 | 21 | 24 | 3 (from 4580) | 0.989 | 2.26 | 2.64 | 50 |
| lr_c0.25_g0.25 | 87/120 | 240 | 174/197 | 16 | 15 | 86 (from 3750) | 0.980 | 3.05 | 2.55 | 56 |
| lr_c0.5_g2 | 100/120 | 740 | 106/147 | 9 | 36 | 23 (from 4380) | 0.991 | 4.20 | 3.32 | 61 |
| lr_c1_g0.5 | 85/120 | 20 | 176/219 | 26 | 20 | 5 (from 4560) | 0.979 | 2.02 | 2.57 | 78 |
| lr_c0.5_g0.5 | 97/120 | 560 | 107/165 | 29 | 22 | 33 (from 4280) | 0.992 | 2.99 | 3.05 | 81 |
| lr_c1_g2 | 72/120 | 340 | 129/187 | 20 | 51 | 0 | 0.226 | 6.00 | 4.41 | 106 |
| lr_c2_g0.5 | 49/120 | 140 | 145/207 | 67 | 23 | 10 (from 4510) | 0.961 | 2.65 | 3.15 | 133 |
| lr_c2_g1 | 34/120 | 240 | 89/197 | 36 | 76 | 2 (from 4590) | 0.967 | 4.11 | 3.99 | 194 |
| lr_c2_g2 | 6/120 | 350 | 75/186 | 28 | 102 | 6 (from 4550) | 0.974 | 7.85 | 3.93 | 225 |
| lr_c4_g4 | 0/120 | none | 0/0 | 0 | 0 | 0 | 0.048 | 32.60 | 30.68 | 120 |

Diagnostics (`summarize.py --diag`): observations with gmax > 2 are 12/460 for c0.5_g1 vs 14 for sec_nodamp, 22–97 for the mc=2 row, and 458 for k3p_constant.

## Reading

- **Critic LR** is the destabilizer. At mc=2, every mg has 133–225 fails and 28–67 departures, prehold collapses to ≤49/120, and max grad-norm rises to 3.2–4.0.
  Halving the critic LR (mc=0.5) helps at mg=1 and mg=2: departures fall to 17 and 9, and max abs D(real) is lowest at c0.5_g1.
- **G LR** mainly sets arrival. At mc ≥ 1, raising mg slows arrival monotonically (20 → 230 → 340 at mc=1; 140 → 240 → 350 at mc=2). mg=2 also inflates D(real) to 4–8 and gmax to 3.3–4.4, and c1_g2 collapses at the end (final HQ .23).
  Low mg (0.5) arrives fast but departs often (26–67).
- The critic and G LRs interact. At mc=0.5, mg=0.5 is poor (arrival 560, 29 departures), so a lower critic LR needs the normal G LR.
- The joint scales bracket the stable region. At ×0.25 the run is calm (streak 15, grad 2.55) but has more prehold fails (56). At ×4 it diverges.
- Best cell: `lr_c0.5_g1`, with 42 fails vs 50 for both sec_nodamp and k3p_constant. Its arrival is +260: slightly worse than sec_nodamp (+230) but better than k3p (+360). Its 17 departures are still far above k3p's 5.

## Recommendation

Adopt critic LR ×0.5 (critic .002125, G .00425, prior .0085) as the new base. The gain is modest (−8 fails) and comes from a single seed.
Next distinct settings: mc ∈ {0.35, 0.7} at mg=1, and mg=0.7 at mc=0.5. The last one probes the fast-arrival corner, since c1_g0.5 arrives in 20.
LR alone does not bring departures down to k3p's level: the best cell has 17 departures vs 5. Closing that gap needs a mechanism change, not more LR tuning.
