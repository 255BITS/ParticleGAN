# Five-track R1/R2 Tier 1 repair campaign

The five further subagent proposals found **no complete Tier 1 repair**. All five pass clock independence and sustained pole transport; every recipe fails Ring16. **AMSGrad passes 5/6 ordinary tasks plus clock**, leaving only density unresolved. Momentum improves the best observed covariance error to **1.098351**, still above **0.85**, and also fails Gaussian and word acquisition. Retain the recorded baseline; these provisional screening results confer no default adoption.

This separately authorized, single-round campaign follows the [closed constant-policy/half-rate search](../r1r2-tier1-search/README.md). The [plan](PLAN.md), [manifest](manifest.json), [registration](registration.json) and [independent preflight review](ttur/review.json) preserve its scope and source identities. The earlier search remains closed under its original stopping rules.

## Whole global recipes

Every new recipe changes exactly one field from the same measured constant parent: beta1=0, beta2=0.99, gamma=0.1, LR=0.0085. Each uses paired R1/R2 every update, learned prior locations, zero prior regularization/noise/EMA and unchanged task conditions. The transport arm explicitly adds a generator local density loss to R1/R2. Prior mobility is inapplicable to the separate direct-coordinate pole and unsampled token fixtures. The scheduled baseline is reused exactly; the constant parent retains its original 4/6 ordinary result plus clock PASS in the preceding publication.

PASS/FAIL links open actual-training GIFs. Gaussian and word gates certify independent same-state acquisition; pole, token, AE and Ring16 retain sustained requirements. Clock remains a separate measurement. A repair requires every ordinary gate plus its own clock gate.

| Whole recipe | Gaussian | Pole | Token hold | AE hold | Ring16 | Word acquisition | Clock | Ordinary |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| Cached scheduled baseline | [PASS](publication/media/baseline-gaussian1d_smoke.gif) | [PASS](publication/media/baseline-two_pole.gif) | [PASS](publication/media/baseline-unused_token_hold.gif) | [PASS](publication/media/baseline-ae_gan_hold.gif) | [FAIL](publication/media/baseline-ring16_acquisition.gif) | [PASS](publication/media/baseline-five_word_joint_smoke.gif) | [FAIL](publication/media/baseline-clockfree_audit_measurement_v1.gif) | 5/6 |
| [Critic LR ×2](ttur/README.md) | [PASS](publication/media/ttur-gaussian1d_smoke.gif) | [PASS](publication/media/ttur-two_pole.gif) | [PASS](publication/media/ttur-unused_token_hold.gif) | [PASS](publication/media/ttur-ae_gan_hold.gif) | [FAIL](publication/media/ttur-ring16_acquisition.gif) | [FAIL](publication/media/ttur-five_word_joint_smoke.gif) | [PASS](publication/media/ttur-clockfree_audit_measurement_v1.gif) | 4/6 |
| [Prior LR ×0.25](prior/README.md) | [PASS](publication/media/prior-gaussian1d_smoke.gif) | [PASS](publication/media/prior-two_pole.gif) | [PASS](publication/media/prior-unused_token_hold.gif) | [PASS](publication/media/prior-ae_gan_hold.gif) | [FAIL](publication/media/prior-ring16_acquisition.gif) | [FAIL](publication/media/prior-five_word_joint_smoke.gif) | [PASS](publication/media/prior-clockfree_audit_measurement_v1.gif) | 4/6 |
| [AMSGrad](optimizer/README.md) | [PASS](publication/media/optimizer-gaussian1d_smoke.gif) | [PASS](publication/media/optimizer-two_pole.gif) | [PASS](publication/media/optimizer-unused_token_hold.gif) | [PASS](publication/media/optimizer-ae_gan_hold.gif) | [FAIL](publication/media/optimizer-ring16_acquisition.gif) | [PASS](publication/media/optimizer-five_word_joint_smoke.gif) | [PASS](publication/media/optimizer-clockfree_audit_measurement_v1.gif) | 5/6 |
| [R1/R2 + local transport 0.1](transport/proposal.md) | [PASS](publication/media/transport-gaussian1d_smoke.gif) | [PASS](publication/media/transport-two_pole.gif) | [PASS](publication/media/transport-unused_token_hold.gif) | [PASS](publication/media/transport-ae_gan_hold.gif) | [FAIL](publication/media/transport-ring16_acquisition.gif) | [FAIL](publication/media/transport-five_word_joint_smoke.gif) | [PASS](publication/media/transport-clockfree_audit_measurement_v1.gif) | 4/6 |
| [Adam beta1=0.5](momentum/proposal.md) | [FAIL](publication/media/momentum-gaussian1d_smoke.gif) | [PASS](publication/media/momentum-two_pole.gif) | [PASS](publication/media/momentum-unused_token_hold.gif) | [PASS](publication/media/momentum-ae_gan_hold.gif) | [FAIL](publication/media/momentum-ring16_acquisition.gif) | [FAIL](publication/media/momentum-five_word_joint_smoke.gif) | [PASS](publication/media/momentum-clockfree_audit_measurement_v1.gif) | 3/6 |

[Certified final metrics and provenance](publication/results.json) · [Saved density analysis](density-readout.md) · [Numerical density audit](density-readout.json). The [current technique inventory](../technique-inventory.md) remains the single generated leaderboard; failed proposed replacements do not change its incumbent.

## What fails

All five arms have **0/96 covariance-bound passes**, **0/96 full Ring16 passes** and terminal suffix **0/5**. Independent saved-sample analysis reproduces all six gate metrics exactly across all 480 observations, without new draws or updates. Endpoint density failures are:

| Recipe delta | Best covariance error (step) | Endpoint covariance | HQ | HQ modes | Min eigen ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Critic LR ×2](ttur/README.md) | 2.873421 (1534) | 3.609186 | 0.636719 | 16 | 0.197063 |
| [Prior LR ×0.25](prior/README.md) | 2.912019 (1534) | 3.175129 | 0.475342 | 12 | 0.098492 |
| [AMSGrad](optimizer/README.md) | 2.416893 (1434) | 2.960424 | 0.839355 | 16 | 0.310603 |
| [R1/R2 + local transport 0.1](transport/proposal.md) | 3.268189 (1584) | 4.653696 | 0.557617 | 12 | 0.777843 |
| [Adam beta1=0.5](momentum/proposal.md) | 1.098351 (1584) | 1.185580 | 0.936523 | 16 | 0.052811 |

The original bounds are covariance <=0.85, HQ >=0.85, all 16 HQ modes, mass TV <=0.15 and minimum eigen ratio >=0.15, with five terminal full passes. Every endpoint meets mass TV. Momentum has the strongest covariance and precision improvement, but endpoint covariance 1.185580 and minimum eigen ratio 0.052811 both fail. Its tangent variance remains 1.36–4.18 times target; radial variance ranges from 0.054 to 2.843 times target. Mean spread can look reasonable while individual component directions fail.

AMSGrad repairs the constant parent’s word acquisition: 19/24 confirmed checks pass, first at update 5,001, and the endpoint has all five words, exact inverse reconstruction, HQ 1 and NLL 0.000333. Its Ring16 covariance is 2.960424; HQ never reaches 0.85 in any saved observation. Gaussian acquisition passes at update 84, but its endpoint mean error 0.329296 sigma and CDF KS 0.126906 fail. Tier 1 acquisition PASS is not retention qualification.

TTUR, slower prior and local transport all fail word acquisition with 0/24 confirmed passes. Momentum also has 0/24 Gaussian and word passes; its Gaussian endpoint CDF KS 0.053078 misses <=0.05 despite mean error 0.060794 sigma. All five retain pole terminal suffixes of 10–13 checks against five required, token and AE hold passes, and own clock PASS.

Faster critic, slower prior and local transport do not retain the preceding clean-center geometry: 49, 104 and 86 of 256 mapped prior means lie outside target 3-sigma regions. AMSGrad and momentum have only three outside, but still show uneven local covariance. The saved Jacobian and between-mean calculations are first-order diagnostics, not an exact variance decomposition or proof of one causal explanation. The [density readout](density-readout.md) records those limitations and per-component results.

## Recommendation and stopping

Close this five-track campaign with all failures visible. AMSGrad is a useful source-bound word/clock result, but its density remains worse than its constant parent. Momentum is the best density lead and introduces separate acquisition and minimum-spread failures. Neither qualifies as a replacement. Preserve the scheduled baseline and its original failure report.

A next justified question would separate momentum by optimizer role or examine a generator correction that reduces tangential stretch while preserving radial spread. First use saved gradient/state evidence to distinguish those mechanisms. Such a comparison needs a new finite registration and one global recipe; the current results do not authorize more tuning, longer budgets, altered priors/sampling, seed repeats or Tier 2 work.

## Evidence, cost and reproduction

Execution commit **`9ad31157b8327044f9364b5fc458659cced0c8f0`**; common scientific source **`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`**. Runtime remains the exact Python 3.12.13 / torch 2.14.0 project environment, runtime hash `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`. Preparation and admission reject changed source/runtime/task/job bindings before spending. All six ordinary tasks match actual initial tensors and ordered consumed target batches across the six roles; every consumed named stream is checkpointed. Each clock receipt retains its own comparison cohort.

New paid work is **927.1709415850346 seconds across 35 attempts**. Seven scheduled-baseline jobs reuse exact certified results at zero new cost. Shared-queue total is **2731.7028826579044 seconds across 105 unique attempts**, including the earlier 70. The new campaign reserves 15,120 seconds for six roles and retains the 5,400-second aggregate paid ceiling. Both archives remain within the combined 8 GiB cap and 5 GiB free-space floor. These costs are accounting, not a speed ranking.

All 42 final cases have actual-training GIFs and saved provenance. Collection adds zero training updates or sampling draws. The [validation receipt](validation.json) and [independent collection audit](collection-audit.json) bind the checks. Thirty workflow cases pass, including cached-baseline identity, original charge ownership and full-reservation caps. Two pre-existing family-publication test failures remain documented in the [earlier validation receipt](../r1r2-baseline/validation.json); no unrelated presentation repair or unchanged scientific rerun follows.

Reproduce admission from a separate checkout at the execution commit, using the frozen declarations and original queue/runtime in the [plan commands](PLAN.md). Read-only collection and [density reconstruction](density_readout.py) use retained artifacts from the final report checkout. Bulk stdout, events, checkpoints and tensors remain outside Git:

```sh
 tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-five-track-repair-20261012/queue/events.jsonl
```

The new archive has its own admission/progress and coordinator logs; its queue symlink retains the original artifact tree and every prior charge. `run.log`, `collect.log` and validation stdout are in `/mnt/ml7tb/ParticleGAN-forge/r1r2-five-track-repair-20261012/`. No execution errors, scientific retries, seed variants, changed task laws, higher-tier launches or defaults changes occurred.
