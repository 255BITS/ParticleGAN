# Way 1: raise prior_lr_mult on API-DV12 (p2-plm3/4/6), 2026-09-27

Base API-DV12, with only `prior_lr_mult` changed (base 2.0). Stage A was run at priority 50. Override files are in the scratchpad `plm/plm{3,4,6}.json`.

| task | API-DV12 (2.0) | p2-plm3 | p2-plm4 | p2-plm6 |
|---|---|---|---|---|
| v.mass | FAIL 0 [mmr .232, ccov .879] | PASS 5 @1000 (mmr .342, ccov .765) | FAIL 0 [mmr .233, ccov .798] | FAIL 0 [ccov .876, mmr .391] |
| intens2 | PASS 6 @425 | FAIL 4 @475 (2m hq .81) | FAIL 7 @425 sfx3 | PASS 9 @400 |
| mode_hold | PASS 12 @650 | FAIL 3 @1100 sfx3 | FAIL 1 @1200 sfx1 | FAIL 0 (7m hq .86) |
| bars4 | PASS 5 @500 | FAIL 0 (4m hq .66) | PASS 14 @250 | FAIL 0 (3m hq .88) |

Result: **no variant passes all 4 stage-A tasks**, so none went on to the other 7 gates or to ring.

Findings:
- Raising the multiplier does not change v.mass in a consistent direction. At 4 the allocation stays frozen: min_mass_ratio .233 is the same value as the base.
- At 3 and 6 the smallest component gets more particles (mmr .34-.39), but the per-component covariance error stays at about .77-.88. The component with error ~2.5 is still the smallest one.
  - At 3 this barely passes, with a late arrival (5/24 @1000).
  - At 6 it fails on covariance error.
- mode_hold gets worse at every multiplier (12/24 down to 3, 1 and 0). The reason: `prior_lr_mult` is global, so it also speeds up the 12-particle mode_hold prior and the image priors.
- The image results scatter from one multiplier to the next (bars4 passes only at 4, intens2 only at 6). That looks like chaotic sensitivity, not a trend.
- Conclusion: a global prior-LR multiplier is the wrong lever. A fix has to act only when the allocation is frozen. Examples: a mobility floor or a slower mobility decay tied to data_drive/pe^2 = 0 on the vector cards, or a prior step kept separate from the controller cut.
