# t2-dv12reg: critic-side factor probes on dv12-ams-rc3 (stage A only)
Base dv12-ams-rc3: fails only img_intensity2 (0/24, 2m hq.84). Each variant = base overrides + one factor.

| cand | change | mode_hold | intens2 | bars4 | v.mass | A |
|---|---|---|---|---|---|---|
| k1p5 | reg_kappa 1.5 | PASS 5/24 | FAIL 0/24 hq.84 (bitwise = base traj) | PASS 15 | PASS 17 | 3/4 |
| k2 | reg_kappa 2.0 | FAIL 5m | FAIL 0/24 hq.84 | PASS 15 | FAIL 4/24 | 1/4 |
| k0p75 | reg_kappa 0.75 | FAIL 8m hq.84 | FAIL 0/24 hq.84 | PASS 15 | PASS 17 | 2/4 |
| re2 | reg_every 2 | PASS 5/24 | FAIL 2/24 hq.84 | FAIL 3m | PASS 15 | 2/4 |
| dg3 | d_guard_ratio 3 | PASS 9/24 | FAIL 0/24 hq.88 | PASS 15 | PASS 18 | 3/4 |
| aw0p5 | reg_anchor_weight .5 | PASS 9/24 | FAIL 0/24 hq.84 | PASS 15 | PASS 18 | 3/4 |
| rc3p5 | reg_coeff 3.5 | FAIL sfx2 | FAIL 1/24 hq.88 | PASS 16 | FAIL cov | 1/4 |

No variant passed stage A, so no quick-gate or ring runs were submitted.

Findings
- The kappa cap never binds on images: k1p5's intensity2 trajectory is identical to the base. Kappa only changes vector/mode_hold, and both directions made them worse.
- intensity2 under rc3 converges slowly and oscillates: hq .41 -> .75 -> .47 -> .88 -> .78 over 600 updates.
  rc1 (dv12-ams) reaches hq .91-1.0 from update 375. The problem is image convergence speed/stability, not a critic quality cap.
- The d_guard_ratio 3 and anchor 0.5 results matched the base on the other gates (the guard and anchor are near-inactive); dg3 only moved the final hq to .88.
- reg_every 2 gets the earliest intensity2 passes (2 obs) but breaks bars4.
- reg_coeff is non-monotone and fragile: 2.0, 2.5 and 3.5 all break vectors.
Next: the fix probably is not on the critic reg side. Try G/prior-side image speed (e.g. prior_lr_mult, ema_decay) on rc3.
