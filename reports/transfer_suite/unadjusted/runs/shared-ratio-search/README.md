# Shared learning-rate ratio search

**Best new complete score:15/19. No improved shared default.** The strongest ratio ties the existing pass count but has larger normalized final shortfall (.687944 versus .422626 for `lr00425_prior2`). It passes overlap while losing anisotropic data; the other completed ratio scores13/19. No production default changed.

One fixed recipe per candidate across every test. This bounded study changes only global G:D:particle learning-rate ratios; all six candidates use Rp logistic, b_cap3/κ1.25, spread .05, Adam(0,.99), no particle L2, and the same delayed cosine (60% hold,5% floor). The canonical19-case reference architecture profile, targets, resources, initialization, budgets and metric bounds are unchanged. Seed0, CPU, one Torch thread.

Six cards were frozen before42 screening episodes. Two selected cards complete the other12 cases each, without reruns or altered settings:66 episodes total. A complete PASS requires19/19 live cases, every metric sustained for at least five final observations in a complete24-point curve. EMA is separate. Partial candidates cannot compete as complete defaults.

| Candidate | G / D / particle LR | Required | Data | Images | Live pass / attempted | Overall |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ratio_g34_d68_p85 | 0.0034 / 0.0068 / 0.0085 | 8/9 (9 tried) | 3/6 (6 tried) | 4/4 (4 tried) | 15/19 | FAIL |
| ratio_g425_d6375_p85 | 0.00425 / 0.006375 / 0.0085 | 9/9 (9 tried) | 2/6 (6 tried) | 2/4 (4 tried) | 13/19 | FAIL |
| ratio_g25_d15_p85 | 0.0025 / 0.0015 / 0.0085 | 1/9 (2 tried) | 0/6 (4 tried) | 1/4 (1 tried) | 2/7 | INCOMPLETE |
| ratio_g425_d2125_p85 | 0.00425 / 0.002125 / 0.0085 | 0/9 (2 tried) | 1/6 (4 tried) | 1/4 (1 tried) | 2/7 | INCOMPLETE |
| ratio_g25_d375_p100 | 0.0025 / 0.00375 / 0.01 | 1/9 (2 tried) | 1/6 (4 tried) | 0/4 (1 tried) | 2/7 | INCOMPLETE |
| ratio_g34_d255_p85 | 0.0034 / 0.00255 / 0.0085 | 0/9 (2 tried) | 0/6 (4 tried) | 1/4 (1 tried) | 1/7 | INCOMPLETE |

**Frozen finalist rule:** After all42 screen episodes, select two by descending sustained screen passes, then mean final normalized bound shortfall, then mean confirmation fraction, then name. Complete the other12 unique tests with the exact same recipe. No additional numerical candidate search in this bounded study.

| Candidate | Failed measured tests |
| --- | --- |
| ratio_g34_d68_p85 | mode_hold, vector_unequal_mass, vector_unequal_width, vector_anisotropic |
| ratio_g425_d6375_p85 | vector_unequal_mass, vector_unequal_width, vector_anisotropic, vector_overlap, img_blobs4, img_bars4 |
| ratio_g25_d15_p85 | two_pole, vector_unequal_mass, vector_unequal_width, vector_anisotropic, vector_overlap |
| ratio_g425_d2125_p85 | two_pole, mode_hold, vector_unequal_mass, vector_unequal_width, vector_overlap |
| ratio_g25_d375_p100 | mode_hold, vector_unequal_mass, vector_unequal_width, vector_overlap, img_blobs4 |
| ratio_g34_d255_p85 | two_pole, mode_hold, vector_unequal_mass, vector_unequal_width, vector_anisotropic, vector_overlap |

## Complete candidates, every test

| Test | ratio_g34_d68_p85 | ratio_g425_d6375_p85 |
| --- | --- | --- |
| two_pole | [PASS (15/24 final streak)](screen/episodes/ratio_g34_d68_p85__two_pole.json.gz) | [PASS (15/24 final streak)](screen/episodes/ratio_g425_d6375_p85__two_pole.json.gz) |
| trajectory | [PASS (23/24 final streak)](completion/episodes/ratio_g34_d68_p85__trajectory.json.gz) | [PASS (24/24 final streak)](completion/episodes/ratio_g425_d6375_p85__trajectory.json.gz) |
| residual_student | [PASS (17/24 final streak)](completion/episodes/ratio_g34_d68_p85__residual_student.json.gz) | [PASS (22/24 final streak)](completion/episodes/ratio_g425_d6375_p85__residual_student.json.gz) |
| unipolar | [PASS (17/24 final streak)](completion/episodes/ratio_g34_d68_p85__unipolar.json.gz) | [PASS (18/24 final streak)](completion/episodes/ratio_g425_d6375_p85__unipolar.json.gz) |
| ae_gan_hold | [PASS (21/24 final streak)](completion/episodes/ratio_g34_d68_p85__ae_gan_hold.json.gz) | [PASS (16/24 final streak)](completion/episodes/ratio_g425_d6375_p85__ae_gan_hold.json.gz) |
| cover_leftover | [PASS (13/24 final streak)](completion/episodes/ratio_g34_d68_p85__cover_leftover.json.gz) | [PASS (15/24 final streak)](completion/episodes/ratio_g425_d6375_p85__cover_leftover.json.gz) |
| unused_token_hold | [PASS (10/24 final streak)](completion/episodes/ratio_g34_d68_p85__unused_token_hold.json.gz) | [PASS (14/24 final streak)](completion/episodes/ratio_g425_d6375_p85__unused_token_hold.json.gz) |
| mid_scale_identity | [PASS (16/24 final streak)](completion/episodes/ratio_g34_d68_p85__mid_scale_identity.json.gz) | [PASS (17/24 final streak)](completion/episodes/ratio_g425_d6375_p85__mid_scale_identity.json.gz) |
| mode_hold | [FAIL (0/24 final streak)](screen/episodes/ratio_g34_d68_p85__mode_hold.json.gz) | [PASS (5/24 final streak)](screen/episodes/ratio_g425_d6375_p85__mode_hold.json.gz) |
| vector_two_broad | [PASS (20/24 final streak)](completion/episodes/ratio_g34_d68_p85__vector_two_broad.json.gz) | [PASS (23/24 final streak)](completion/episodes/ratio_g425_d6375_p85__vector_two_broad.json.gz) |
| vector_unequal_mass | [FAIL (0/24 final streak)](screen/episodes/ratio_g34_d68_p85__vector_unequal_mass.json.gz) | [FAIL (0/24 final streak)](screen/episodes/ratio_g425_d6375_p85__vector_unequal_mass.json.gz) |
| vector_unequal_width | [FAIL (0/24 final streak)](screen/episodes/ratio_g34_d68_p85__vector_unequal_width.json.gz) | [FAIL (0/24 final streak)](screen/episodes/ratio_g425_d6375_p85__vector_unequal_width.json.gz) |
| vector_anisotropic | [FAIL (0/24 final streak)](screen/episodes/ratio_g34_d68_p85__vector_anisotropic.json.gz) | [FAIL (0/24 final streak)](screen/episodes/ratio_g425_d6375_p85__vector_anisotropic.json.gz) |
| vector_overlap | [PASS (5/24 final streak)](screen/episodes/ratio_g34_d68_p85__vector_overlap.json.gz) | [FAIL (3/24 final streak)](screen/episodes/ratio_g425_d6375_p85__vector_overlap.json.gz) |
| vector_spiral | [PASS (24/24 final streak)](completion/episodes/ratio_g34_d68_p85__vector_spiral.json.gz) | [PASS (17/24 final streak)](completion/episodes/ratio_g425_d6375_p85__vector_spiral.json.gz) |
| img_stripes2 | [PASS (18/24 final streak)](completion/episodes/ratio_g34_d68_p85__img_stripes2.json.gz) | [PASS (12/24 final streak)](completion/episodes/ratio_g425_d6375_p85__img_stripes2.json.gz) |
| img_bars4 | [PASS (10/24 final streak)](completion/episodes/ratio_g34_d68_p85__img_bars4.json.gz) | [FAIL (0/24 final streak)](completion/episodes/ratio_g425_d6375_p85__img_bars4.json.gz) |
| img_blobs4 | [PASS (12/24 final streak)](screen/episodes/ratio_g34_d68_p85__img_blobs4.json.gz) | [FAIL (0/24 final streak)](screen/episodes/ratio_g425_d6375_p85__img_blobs4.json.gz) |
| img_intensity2 | [PASS (17/24 final streak)](completion/episodes/ratio_g34_d68_p85__img_intensity2.json.gz) | [PASS (11/24 final streak)](completion/episodes/ratio_g425_d6375_p85__img_intensity2.json.gz) |

These are inspected development cases. A better score in this search is not proof of a production or out-of-distribution default. No candidate mixes per-task optimizer/formulation settings and no production default was changed.

[Predeclared study and resolved recipes](study_plan.json) · [Screen plan](screen_plan.json) · [Frozen-rule selection](selection.json) · [Completion plan](completion_plan.json) · [Full summary/metrics](leaderboard.json) · [Audit](audit.json) · [Screen raw episodes/source](screen/index.json) · [Completion raw episodes/source](completion/index.json) · [Screen log](screen.log) · [Completion log](completion.log).

Exact commands, run from the isolated checkout at78c872236b70f6e5527169db7143559536e052a1:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_default_search --plan /tmp/pr38-shared-ratio-search/screen_plan.json --output /tmp/pr38-shared-ratio-search/screen > /tmp/pr38-shared-ratio-search/screen.log 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.shared_default_search --plan /tmp/pr38-shared-ratio-search/completion_plan.json --output /tmp/pr38-shared-ratio-search/completion > /tmp/pr38-shared-ratio-search/completion.log 2>&1
```

Use new output directories when reproducing; the runner refuses to overwrite archived runs.
