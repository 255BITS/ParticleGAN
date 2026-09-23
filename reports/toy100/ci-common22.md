# GitHub CI common-22 trained-gate audit

[GitHub Actions run 35898491581](https://github.com/255BITS/ParticleGAN/actions/runs/35898491581) executed the single `benchmarks.toy_suite run` command in [the workflow](../../.github/workflows/toy100.yml) against `configs/toy100/shared_candidate.json`. Its downloaded `common22-trained-gate` evidence is retained locally at `artifacts/toy100-accuracy/ci-common22-fixed/artifacts/toy-suite-ci/`, with the console log at `artifacts/toy100-accuracy/ci-common22-fixed/toy-suite-ci.log`. The runner launched all three 7,000-step 100-mode trainings, their separate accuracy gate, and all 19 frozen transfer tasks. `command_returns.json` records exit codes `1`, `1`, and `0` for those three commands: both 100-mode gates failed, while the transfer runner completed. The workflow's training step therefore failed as expected; artifact upload still ran.

The saved `compatibility.json` reports **INCOMPLETE, 16/22**, with `global_recipe_identical: false`. That identity flag is a comparison bug: the resolved in-memory `Recipe.to_dict()` contains tuple-valued Adam betas, while the saved transfer protocol contains their JSON list representation. Their values and every other global recipe/noise field agree. After normalizing this comparison to JSON values, an independent strict regrade of the *unchanged* downloaded `toy100` and `candidate19` evidence reports **valid FAIL, 16/22**: recipe identity true, noise actually applied on all 19 hosts, 0/3 native passes, 16/19 transfer passes, and 4/6 vector passes. The [regression test](../../tests/test_toy_suite.py) covers the real shared config's tuple-to-JSON-list round trip; all 35 `test_toy_suite.py` tests pass. The correction changes the classification from incomplete evidence to a fully tested failing recipe. It does not turn a failed model into a pass.

The strict regrader checked the copied config and 19 compressed episodes against their hashes, frozen host specs, optimizer and noise receipts, 24-checkpoint schedules, stored verdicts, and archived source. All 105 transfer source digests and the ten native source digests match PR source commit `6365997`; they also match the CI synthetic merge commit `e7235826b01c4dc64257117e28b882fab6f3c0b9` recorded in native provenance. No hashed benchmark or training source changed in that merge. The optional installed-wheel public-default control was not requested and is correctly `MISSING`; it is not one of the 22 candidate cases.

| 100-mode problem | Final live modes / 100 | Final live precision | Final live mass TV | Independent 100k holdout precision / mass TV | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| grid100 | 47 | .6250 | .28505 | .62935 / .28237 | Coverage and accuracy FAIL |
| rotated100 | 11 | .1819 | .22455 | .18442 / .23487 | Coverage and accuracy FAIL |
| staggered100 | 45 | .78445 | .45270 | .78215 / .44630 | Coverage and accuracy FAIL |

The frozen coverage gate requires 100 modes, precision at least .97, and mass TV at most .10, among its other conditions. The additional accuracy gate requires mass TV at most .06 and adequate per-mode data for center/shape checks. On grid100, the 100k holdout has center RMS error **1.274 target sigmas** and radial KS **.25755**, versus limits .20 and .04. For rotated100 and staggered100, conditional center and shape estimates are undefined because too few modes qualify. All five terminal accuracy checks fail for each problem; the frozen coverage gate and every independent holdout fail as well.

The 19-host transfer replay passes 16 frozen live gates. Its three misses are:

| Host | Saved final metric(s) | Frozen failure |
| --- | --- | --- |
| trajectory | Identity MSE .01920, below the .020 limit | Only the last of 24 observations passes; five consecutive passing checks are required. |
| vector_unequal_mass | Conditional covariance error 1.3316; minimum component eigenvalue ratio .0280 | Exceeds the .85 covariance-error limit and falls below the .15 eigenvalue-ratio limit. Its rarest component has 29 of 4,096 samples versus an expected ~82; final passing suffix is zero. |
| vector_overlap | Sliced W1 .0862, mean error .0472, covariance error .3879, all within limits at the final check | Covariance error reaches .5301 at the penultimate check, above the .45 limit; final passing suffix is one, short of five. |

This matches the earlier local best 16/19 replay at `artifacts/toy100-accuracy/compatibility/exact-beta999-priorlr3-end02-all19/`. The global substantive recipe fields, noise fields, and all transfer source hashes are identical; only the recipe's display name differs. All 19 pass/fail verdicts and every saved observation, action, convergence result, loss, and live/EMA metric agree exactly after excluding elapsed-time fields and that display name. Thus the downloaded CI record shows **no observed numerical platform difference** in the transfer replay. It offers one complete 22-case execution and a reproducible failure, not evidence of 22/22 success.

## Repeat at learned-noise infrastructure commit

[GitHub Actions run 35901036817](https://github.com/255BITS/ParticleGAN/actions/runs/35901036817) repeated the same unchanged fixed-noise candidate at `8f44980`. It again trained all 22 and uploaded its evidence. Regrading the downloaded artifact with the JSON identity fix gives **valid FAIL, 16/22**, with shared-recipe identity verified. The local artifact is `artifacts/toy100-accuracy/ci-common22-learned-support/artifacts/toy-suite-ci`. The optional learned-noise feature is disabled in this candidate; its separate failed trials do not replace these fixed-noise results.
