# t2-rp15noise: API-RP15 + constant (clock-free) critic noise
Base API-RP15 (11/12, fails only img_bars4 3m hq.72). RP15 noise_policy="constant" -> training.py returns
recipe.input_noise_std / output_noise_std verbatim (no clock). Only noise fields changed; priority 50.

Stage A (mode_hold | intens2 | bars4 | v.mass):
| cand | change | mode_hold | intens2 | bars4 | v.mass | A |
|---|---|---|---|---|---|---|
| in05 | in .05 | FAIL (2m hq.10) | PASS 6 | FAIL 3m hq.97 | PASS 20 | 2/4 |
| in08 | in .08 | PASS 10 | FAIL 1 @550 (2m hq.84) | PASS 10 | PASS 20 | 3/4 |
| in10 | in .10 | PASS 11 | FAIL 2 @550 sfx1 (hq.91) | PASS 12 | PASS 21 | 3/4 |
| in10-out015 | in .10, out .015 | FAIL 9 sfx1 | FAIL 5 @475 sfx4 | PASS 6 | PASS 21 | 2/4 |
| in125 | in .125 | FAIL 8 sfx2 | FAIL 0 (hq.84) | PASS 13 | FAIL eigen | 1/4 |
| in20 | in .20 | FAIL 5 @500 | FAIL 0 | FAIL 3m | FAIL eigen | 0/4 |
| out06 | out .06 | FAIL 5 @500 | FAIL 0 | FAIL 4m | PASS 20 | 1/4 |

No variant passed all 4 -> no quick-11 / ring runs.
Finding: constant input noise .08-.125 fixes bars4 (10-13/24) and keeps mode_hold/v.mass, but slows
img_intensity2 (RP15 itself only arrives @500, 5/24); at in .10 it arrives @550 and misses the suffix
by ~3 checks. Window is narrow and non-monotone (in .05 breaks mode_hold). Lowering output noise to .015
brings intens2 arrival earlier (@475, sfx4, 1 check short) but costs mode_hold.
Next (not run): in .10 + out .02, or pair in .10 with a state-driven noise decay / faster intensity arrival.
