# Selected family diagnostic coverage

The selected formulations, numerical gates and qualification are unchanged. Four existing source-matched Tier 2 diagnostic panels supply 84 measurements without rerunning them. The new strict BCAP clock-free audit passes all four comparisons in 24 updates; it used a separate Torch 2.13 runtime and grants no credit to the selected Torch 2.14 cohort.

New training: **1 PASS**, **8.570 seconds**. The runner was configured for at most four tests total and two per GPU. Only one missing independent test was compatible. All formulations remain unchanged. [Actual-training audit GIF](media/clockfree_audit.gif) uses the saved proof states and adds no optimizer update or sampling draw.

| Family | Existing Tier 2 diagnostics |
| --- | ---: |
| release07-gan-v3 | 9/21 (12 fail) |
| k3p | 11/21 (10 fail) |
| ka2 | 10/21 (11 fail) |
| r1r2 | 9/21 (12 fail) |

The strict clock-free audits for K3P, KA2, R1/R2 and released GAN require removing declared clock dependencies, so they were skipped as incompatible. Atlas and E22 lack the required policy/task contracts for every ordinary question. The long ring, target-shift and native continuations retain their explicit parent dependencies: the recorded mode-hold and native acquisition gates fail. Those tests remain not run; a diagnostic view does not manufacture a passing parent or a checkpoint.

The table below counts 35 distinct ordinary questions once per selected recipe. PASS and FAIL are measurements; `incompatible` and `not run` are separate. Existing Tier 2 diagnostics stay outside ordinary qualification. The new BCAP audit is shown separately because its runtime differs.

| Task | BCAP | Atlas | E22 | GAN v3 | K3P | KA2 | R1/R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ae_gan_hold` | PASS | incompatible | incompatible | PASS | PASS | PASS | PASS |
| `clockfree_audit` | not run | incompatible | incompatible | incompatible | incompatible | incompatible | incompatible |
| `clockfree_audit_measurement_v1` | PASS | incompatible | incompatible | FAIL | FAIL | FAIL | FAIL |
| `cover_leftover` | PASS | incompatible | incompatible | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) |
| `five_word_joint_hold` | PASS | incompatible | incompatible | FAIL (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | FAIL (diagnostic) |
| `five_word_joint_smoke` | PASS | incompatible | incompatible | PASS | PASS | PASS | PASS |
| `gaussian1d_smoke` | PASS | incompatible | incompatible | PASS | PASS | PASS | PASS |
| `gaussian1d_stability` | FAIL | incompatible | incompatible | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) |
| `grid100` | FAIL | incompatible | incompatible | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) |
| `grid100_14k` | not run | incompatible | incompatible | not run | not run | not run | not run |
| `img_bars4` | FAIL | incompatible | incompatible | PASS (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | PASS (diagnostic) |
| `img_blobs4` | FAIL | incompatible | incompatible | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | PASS (diagnostic) |
| `img_intensity2` | FAIL | incompatible | incompatible | PASS (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | PASS (diagnostic) |
| `img_stripes2` | PASS | incompatible | incompatible | FAIL (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | FAIL (diagnostic) |
| `mid_scale_identity` | PASS | incompatible | incompatible | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) |
| `mode_hold` | FAIL | incompatible | incompatible | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) |
| `residual_student` | PASS | incompatible | incompatible | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | FAIL (diagnostic) |
| `ring16_acquisition` | PASS | incompatible | incompatible | FAIL | FAIL | FAIL | FAIL |
| `ring_extension` | not run | incompatible | incompatible | not run | not run | not run | not run |
| `ring_hold` | not run | incompatible | incompatible | not run | not run | not run | not run |
| `rotated100` | FAIL | incompatible | incompatible | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) |
| `rotated100_14k` | not run | incompatible | incompatible | not run | not run | not run | not run |
| `staggered100` | FAIL | incompatible | incompatible | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) |
| `staggered100_14k` | not run | incompatible | incompatible | not run | not run | not run | not run |
| `target_shift_recovery` | not run | incompatible | incompatible | not run | not run | not run | not run |
| `trajectory` | PASS | incompatible | incompatible | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | FAIL (diagnostic) |
| `two_pole` | PASS | incompatible | incompatible | PASS | PASS | PASS | PASS |
| `unipolar` | PASS | incompatible | incompatible | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) |
| `unused_token_hold` | PASS | incompatible | incompatible | PASS | PASS | PASS | PASS |
| `vector_anisotropic` | FAIL | incompatible | incompatible | FAIL (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) |
| `vector_overlap` | FAIL | incompatible | incompatible | FAIL (diagnostic) | PASS (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) |
| `vector_spiral` | PASS | incompatible | incompatible | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) |
| `vector_two_broad` | PASS | incompatible | incompatible | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) | PASS (diagnostic) |
| `vector_unequal_mass` | FAIL | incompatible | incompatible | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) |
| `vector_unequal_width` | FAIL | incompatible | incompatible | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) | FAIL (diagnostic) |

[Exact statuses and skip reasons](results.json) · [Source-bound preflight](preflight.json) · [New audit receipt](clockfree-receipt.json) · [Original numerical diagnostics](../recipe-prior-refactor/publication/README.md)

Reproduce the diagnostic with `bcap-clockfree-coverage-v1` on the frozen source `a0493ca3c213d559523eadc76e244ec9f9c1f81f`, preserving its exact task and recipe. Raw proof tensors and logs remain local under `/ml2/hypergan/fill-tier-values-20261014-run/queue/selected-family-coverage-20261014/a61d4c1b615e49d4b6423bb7f6937923/`. The compact receipt binds their certified result.

Refresh the compatibility and readout without training:

```sh
python reports/forge/tier-coverage-20261014/preflight.py --source-root <exact-source-checkout>
python reports/forge/tier-coverage-20261014/collect.py
python reports/forge/regenerate_technique_inventory.py --refresh-publication
```

Next work should address the failed acquisition/retention gates as separately declared research. No recipe tuning, seed variants or promotion follows from these coverage results.
