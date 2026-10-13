# R1/R2 gradient direction and critic denominator readout

All three new global recipes pass 4/6 ordinary tasks and their own clock audit.
Every recipe fails sustained Ring16 and word acquisition. The existing
AdaNSGDA direction-preserving norm graft and critic epsilon 1e-4 therefore
supply no Tier 1 repair. Preserve the original scheduled baseline and all
failed replacements.

The [plan](PLAN.md), [manifest](manifest.json), [registration](registration.json),
[prospective optimizer preflight](optimizer-preflight.json) and
[independent review](preflight-review.json) freeze this separate finite campaign.
It follows the closed [optimizer-role comparison](../r1r2-role-repair/README.md).
The task/source/protocol, actual initial tensors and consumed target batches,
initial prior distribution/capacity, sampling, budgets, evaluation and gates
remain unchanged. All candidates use one whole global recipe across tasks.

## Frozen whole recipes

All new arms inherit the constant parent: paired R1/R2 coefficient .1 every
update, LR .0085, beta1=0/beta2=.99, epsilon1e-8, learned prior and zero extra
prior regularization/noise/EMA. D/prior beta overrides and prior epsilon remain
null. AdaNSGDA changes G/E, D and sampled prior update laws; direct pole uses
its generator law. Critic epsilon changes D only, including the word host.
In AdaNSGDA it affects both the adaptive denominator and gradient-norm graft.

PASS/FAIL links open actual-training GIFs. The table is an unranked frozen
comparison; [technique inventory](../technique-inventory.md) remains the single
current generated leaderboard.

| Recipe | Gaussian | Pole | Token | AE | Ring16 | Words | Clock | Ordinary |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: |
| Cached scheduled baseline | [PASS](publication/media/baseline-gaussian1d_smoke.gif) | [PASS](publication/media/baseline-two_pole.gif) | [PASS](publication/media/baseline-unused_token_hold.gif) | [PASS](publication/media/baseline-ae_gan_hold.gif) | [FAIL](publication/media/baseline-ring16_acquisition.gif) | [PASS](publication/media/baseline-five_word_joint_smoke.gif) | [FAIL](publication/media/baseline-clockfree_audit_measurement_v1.gif) | 5/6 |
| Cached constant Adam | [PASS](publication/media/adam-gaussian1d_smoke.gif) | [PASS](publication/media/adam-two_pole.gif) | [PASS](publication/media/adam-unused_token_hold.gif) | [PASS](publication/media/adam-ae_gan_hold.gif) | [FAIL](publication/media/adam-ring16_acquisition.gif) | [FAIL](publication/media/adam-five_word_joint_smoke.gif) | [PASS](publication/media/adam-clockfree_audit_measurement_v1.gif) | 4/6 |
| Adam + critic epsilon 1e-4 | [PASS](publication/media/adam_eps-gaussian1d_smoke.gif) | [PASS](publication/media/adam_eps-two_pole.gif) | [PASS](publication/media/adam_eps-unused_token_hold.gif) | [PASS](publication/media/adam_eps-ae_gan_hold.gif) | [FAIL](publication/media/adam_eps-ring16_acquisition.gif) | [FAIL](publication/media/adam_eps-five_word_joint_smoke.gif) | [PASS](publication/media/adam_eps-clockfree_audit_measurement_v1.gif) | 4/6 |
| AdaNSGDA | [PASS](publication/media/ada-gaussian1d_smoke.gif) | [PASS](publication/media/ada-two_pole.gif) | [PASS](publication/media/ada-unused_token_hold.gif) | [PASS](publication/media/ada-ae_gan_hold.gif) | [FAIL](publication/media/ada-ring16_acquisition.gif) | [FAIL](publication/media/ada-five_word_joint_smoke.gif) | [PASS](publication/media/ada-clockfree_audit_measurement_v1.gif) | 4/6 |
| AdaNSGDA + critic epsilon 1e-4 | [PASS](publication/media/ada_eps-gaussian1d_smoke.gif) | [PASS](publication/media/ada_eps-two_pole.gif) | [PASS](publication/media/ada_eps-unused_token_hold.gif) | [PASS](publication/media/ada_eps-ae_gan_hold.gif) | [FAIL](publication/media/ada_eps-ring16_acquisition.gif) | [FAIL](publication/media/ada_eps-five_word_joint_smoke.gif) | [PASS](publication/media/ada_eps-clockfree_audit_measurement_v1.gif) | 4/6 |

[Final metrics/provenance](publication/results.json) ·
[Independent collection audit](collection-audit.json) ·
[Saved density readout](density-readout.md) · [Density receipt](density-audit.json).

## What fails and what this rules out

All 288 saved Ring16 panels reproduce the original six gate metrics exactly.
Each new recipe has 0/96 covariance passes, 0/96 full passes and terminal suffix
0/5. Best observed covariance errors are 2.294437 for critic epsilon,
2.647227 for AdaNSGDA and 2.661903 for their combination, against <=.85.
Their endpoints are 2.635454, 4.570228 and 7.055577. AdaNSGDA's endpoint HQ
falls to .432861 with 14 modes; the joint arm has HQ .258057 and six modes.
These are broader density/acquisition failures, rather than a small miss of
the minimum-spread threshold. All three word gates have zero confirmed passes
and retain zero/one generated modes at the endpoint.

Gaussian acquisition still passes, but its endpoint CDF KS exceeds .05 in all
three recipes. Acquisition is not a retention claim. Pole, token/AE hold and
clock pass under each new recipe.

The saved endpoint optimizer-direction evidence did not predict a successful
trained trajectory. Removing within-tensor Adam rotation while retaining
its step norm is insufficient. Critic epsilon damping also fails, and its
combination with Ada worsens precision and coverage. This does not identify
one causal explanation for native Adam's density failure. Do not infer that
all conditioning methods are equivalent or fill a different task with one
of these partial successes.

## Recommendation and stopping

Close this campaign with all failures visible. The
[saved spectral-update note](optimizer/spectral-note.md) distinguishes the
existing polar update from Ada: it changes the retained singular-value spectrum
of the update rather than preserving raw gradient singular-value ratios.
An explicitly registered full-versus-critic-only comparison is a separate
bounded hypothesis; it cannot claim to constrain existing weight/Jacobian
conditioning or isolate generator updates from prior normalization.
The [transport review](transport/next-hypothesis.md) rejects immediate reuse of
local transport .1: its measured prior failure broadened radial/tangent density,
and its loss has no explicit directional covariance or eigen floor.
No rate grid, task relaxation, seed repetition, automatic refinement or default
promotion follows from these negative results.

## Evidence, accounting and reproduction

Execution commit `a7f141c9c73d7aead972f2d7a382b2f52a59ac69`; scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`;
runtime `640b92546b7cefb9a81f0bd8d7c20139fc353d7aff632548e34eec7736155ebc`.
Independent audits bind original stored grades, candidate/job/source/runtime,
actual optimizer family/epsilon/moment state, captured initial tensors,
ordered consumed target batches and every named RNG checkpoint. The original
zero/identity/direct-coordinate fixtures and clock proofs remain explicit
separate cohorts.

21 new attempts cost 586.6006355709687 seconds; 14 cached controls add zero
new cost. Shared accounting preserves all 154 prior charges and now contains
175 unique attempts totaling 4,587.580182052945 seconds. Full reservation is
12,600 seconds; aggregate paid cap remains 8,000. Combined archives keep the
8 GiB cap and 5 GiB free-space floor. Costs are accounting, not a speed ranking.

All 35 final cases have actual-training GIFs and compact provenance. Collection
adds zero optimizer updates or sampling draws. The
[validation receipt](validation.json) binds the checks. Seventy-one workflow
checks pass; qualification publication records and the current inventory remain
unchanged. Provisional screening does not qualify default promotion.

Reproduce admission from a checkout at the execution commit using the
[frozen plan commands](PLAN.md); use the final report checkout for read-only
collection against retained artifacts. Bulk stdout, JSONL, checkpoints and
states stay outside Git in
`/mnt/ml7tb/ParticleGAN-forge/r1r2-direction-floor-repair-20261012/`.
Coordinator logs are `run.log` and `collect.log`:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/r1r2-direction-floor-repair-20261012/queue/events.jsonl
```
