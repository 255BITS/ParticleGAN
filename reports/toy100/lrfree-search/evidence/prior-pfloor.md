# p2-pf: DV12 + Recipe.continuous_prior_floor (Way 2)
Package: candidates/dv12-pfloor/package (copy of API-DV12). Change: `Recipe.continuous_prior_floor: float = 0.05`
(appended last, validated finite in [0,1], bool rejected); GANTrainer passes it to `DataDriftController(prior_floor=)`;
`current_scales()` prior = (f + (1-f)*m)*game_trust (x prior_lr_mult downstream). Overrides: candidates/dv12-pfloor/overrides-*.json.

Identity (p2-pf-id, f default): compare.py vs runs/API-DV12 -> bitwise true on mode_hold (264/264) and
vector_unequal_mass (1272/1272); rates.jsonl byte-identical. (1-.05 == .95 exactly in float64.)

Stage A (vec.mass / intens2 / mode_hold / bars4):
| cand | f | v.mass | intens2 | mode_hold | bars4 |
|---|---|---|---|---|---|
| API-DV12 | .05 | FAIL mmr .232, cce .879 | PASS | PASS | PASS |
| p2-pf25 | .25 | FAIL mmr .232 (cce .809 ok) | FAIL sfx4 | FAIL hq .84 | FAIL 4m |
| p2-pf50 | .50 | FAIL mmr .146 | PASS | FAIL 5m hq .63 | FAIL 3m |
| p2-pf100 | 1.0 (diag, v.mass only) | FAIL sfx (final mmr .256, cce .684; 10/24) | - | - | - |

Prior LR at update 1200 (rates): .00069 (base) -> .0023 (f .25) -> .0044 (f .5); raised as intended.
Allocation still freezes: smallest (2%) component mass ~.003-.005 (~1 of 256 particles) from update ~400-500
in every f, while mobility is still .2-.4, i.e. BEFORE the prior scale drops below .5. f=1 (constant prior LR)
only moves mmr .232->.256 by quantization (same ~1 particle) and still misses the final suffix.
=> The "prior-scale clock" diagnosis is refuted as the binding cause: allocation to the 2% mode is set during the
early high-mobility phase; larger late prior LR does not re-allocate particles, and f>=.25 costs mode_hold
(hq .84 / 5 modes) and bars4. No variant passed stage A; nothing promoted to other gates/ring.
Next ideas: act on early allocation (mass-aware reg/reg_coeff schedule-free, particle split/birth), not prior LR floor.
