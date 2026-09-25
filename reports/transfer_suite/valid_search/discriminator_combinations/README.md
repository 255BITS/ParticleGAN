# Shared-recipe D architecture combinations

Fixed recipe: logistic RP; cap3/kappa1.25; prior regularization.05, no particle L2; Adam(0,.999); G LR.00075, D multiplier2, prior multiplier30; 1:1 cosine. Only D architecture varies among rows. This is an explicitly adaptive width-focused follow-up using prior screen evidence; the previous frozen study remains unchanged. Original G/data/particle count/budgets and every metric bound remain unchanged. Six valid data tasks only; no intentionally poor diagnostics or forced D/batch/capacity stress cases enter selection.

| D architecture | Unequal mass | Unequal width | Overlap | Broad | Anisotropic | Spiral | Sustained / measured |
|---|---|---|---|---|---|---|---:|
| d128_l3_f2 | FAIL (0/24) | FAIL (0/24) | FAIL (3/24) | unmeasured | unmeasured | unmeasured | 0/3 |
| d64_l4_f3 | FAIL (0/24) | FAIL (0/24) | PASS (23/24) | unmeasured | unmeasured | unmeasured | 1/3 |
| d64_l2_f3 | FAIL (0/24) | FAIL (0/24) | PASS (21/24) | unmeasured | unmeasured | unmeasured | 1/3 |

Cells use sustained live verdicts, complete24-point curves and at least5 final passing observations. EMA is recorded separately. The original-D recipe already solves unequal_mass and overlap but fails unequal_width; its exact episodes are retained in references/. A full-six result is reported only after the other3 cases are run. No union of different D architectures is a shared winner.

[Frozen plan](frozen_plan.json.gz), [full results](results.json.gz), [screen leaderboard](screen/README.md), [exact script](run.py), [log](progress.log).
