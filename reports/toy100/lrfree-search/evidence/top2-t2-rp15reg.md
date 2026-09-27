# t2-rp15reg: critic-regularization strength on API-RP15 (plain Adam)

Base API-RP15 (11/12 quick; fails only img_bars4 3m hq.72). One factor = critic penalty (reg_coeff / reg_kappa).
Stage A = img_intensity2, img_bars4, vector_unequal_mass, mode_hold. Priority 50. No variant passed all 4 -> no quick/ring stages.

| cand | change | intens2 | bars4 final | v.mass | mode_hold |
|---|---|---|---|---|---|
| API-RP15 (base) | - | PASS 5 | 3m hq.72 | PASS 21 | PASS 14 |
| t2-rp15reg-rc1p25 | reg_coeff 1.25 | PASS 5 | 4m hq.81 | PASS 20 | PASS 12 |
| t2-rp15reg-rc1p5 | reg_coeff 1.5 | FAIL sfx2 | 4m hq.88 | PASS 20 | FAIL 8m hq.88 |
| t2-rp15reg-rc2 | reg_coeff 2 | FAIL sfx2 | 4m hq.78 | PASS 20 | PASS 7 |
| rp15-rc3 (other run) | reg_coeff 3 | FAIL sfx1 | 4m hq.81 | - | FAIL |
| t2-rp15reg-k0p75 | reg_kappa .75 | PASS 5 | 3m hq.72 | PASS 20 | FAIL 7m |
| t2-rp15reg-k0p5 | reg_kappa .5 | PASS 6 | 3m hq.75 | PASS 13 | PASS 10 |
| t2-rp15reg-rc1p5-k0p5 | rc1.5+k.5 | FAIL sfx2 | 4m hq.88 | PASS 19 | PASS 9 |
| t2-rp15reg-rc2-k0p5 | rc2+k.5 | FAIL sfx2 | 4m hq.78 | PASS 21 | FAIL 8m hq.80 |

Findings
- reg_coeff >1 reliably fixes bars4 *coverage* (3 -> 4 modes, arrives ~@150-300) but HQ plateaus .78-.88 (<.90): 
  ~4 of 32 particles stay off-mode. Non-monotone in coeff (1.5 best HQ .88).
- Cost: reg_coeff >=1.5 delays intensity2 (final HQ ok, suffix only 1-2 -> arrival too late) and destabilizes mode_hold.
- reg_kappa is nearly inert on bars4 (k.5 bars4 trajectory ~= base; rc1.5+k.5 bars4 identical to rc1.5).
  k.5 speeds intensity2 slightly (6 @450) but costs v.mass margin (13/24).
- Best keeper: t2-rp15reg-rc1p25 = base on 3 gates (no regression) and bars4 3m/.72 -> 4m/.81. Still 11/12-ish.
- Critic strength alone does not close bars4 HQ; the remaining gap is particle precision, not critic Lipschitz.
  Next lever should be generator/particle side (e.g. output noise / latent damping) on top of rc1.25-1.5.
