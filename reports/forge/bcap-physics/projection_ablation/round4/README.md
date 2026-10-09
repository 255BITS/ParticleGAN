# Projection repair: direction versus finite acceptance

**Direction alone is the smallest tested repair that retains BOTH complete identity passes and the mid-scale guardrail.** It finishes **5 PASS / 1 FAIL**, matching finite strict progress's **5 PASS / 1 FAIL**, against corrected nonascent's **3 PASS / 3 FAIL**. Finite acceptance supplies a same-batch protected-loss certificate but is unnecessary for these bounded quality gates. Gaussian stability remains FAIL.

This completed three-arm diagnostic runs corrected schema2 `nonascent`, the identical strict common-descent `direction_blend` at full scale without finite probes, and `strict_progress` with Armijo acceptance on one common source/runtime. [PR367](https://github.com/255BITS/ParticleGAN/pull/367) is stacked on [PR361](https://github.com/255BITS/ParticleGAN/pull/361). No fourth arm, sweep, seed variation, gate change or new supervision is added. These conditional cloud and Gaussian MoG cohorts do not confer ordinary Tier2 qualification or public-default adoption.

The primary question is which smallest mechanism retains BOTH trajectory and residual full sustained identity passes while preserving the mid-scale passing guardrail. A residual endpoint MSE forecast <= .02 is registered, but every original bound and required suffix decides scientific success. The finite-only boundary arm is degenerate because its protected derivative is zero and cannot meet strict acceptance.

The direction is unchanged from [PR361](https://github.com/255BITS/ParticleGAN/pull/361): for an actual conflicting base displacement d and normalized existing protected gradients n_j, q = -||d|| mean(n_j)/||mean(n_j)|| and p = (project_nonascent(d,a)+q)/2. The direction-only arm always applies p at scale1; opposed gradients restore the pre-step tensors. The finite arm tries scale1 through 1/256, accepting rounded strict derivatives and same-batch Armijo decrease with coefficient1e-4. The base optimizer clocks advance once. Inactive steps preserve updated tensors bitwise. Protected losses, unchanged host loss coefficients and target information are unchanged; no identity supervision is introduced.

For at most two nonzero unit normals, their mean is the minimum-norm convex combination. Its negative is common descent unless the normals oppose. This is a local directional property, consistent with [MGDA](https://mgda.inria.fr/mgda) and [Sener and Koltun](https://arxiv.org/abs/1810.04650), not a stochastic GAN convergence guarantee. Strict derivatives do not guarantee finite decrease; changing-batch finite decrease does not guarantee distribution fidelity.

Read-only [saved endpoint attribution](saved-attribution.json) restores six certified states: original winner and both PR361 arms for trajectory/residual. It includes adversarial, set coverage, latent L2/spread and the existing paired residual on the actual both-land mask. It transforms the summed gradients with the saved full-DualNorm rates/smoothing before attribution, and also reports normalized components and leave-one-out proposals. These CPU FP64 derivatives/public FP32 polar probes are a separate diagnostic cohort; they consume zero training steps or samples and do not prove an actual rounded step or unique causal culprit. [Reproduce](probe_saved.py).

All arms use the exact saved winning BCAP settings: nonsaturating, full DualNorm smoothing .001/momentum0/per_offset, G/E .012, D .018, prior .030, constant floors1, cap/coefficient1 each update, zero additive training output noise and clean/live scoring. `get_recipe('bcap')` alone is not this winner. The only trainer delta is `constraint_geometry_mode`.

The six unchanged tasks are trajectory400, residual400, mid-scale800, two-pole80, Gaussian smoke1000 and own-checkpoint Gaussian stability+5000. Each arm reserves 6,420 seconds; the campaign reserves 19,260 within the21,600-second ceiling. Failed own smoke blocks stability with zero spend. Gaussian/vector adapters retain their one-real-tensor D/G reuse; the fixed two-pole identity/zero/stored-weight fixture remains separate. Protocol seed0, public deterministic initialization, task architectures/priors/laws/seen batches/update budgets/scoring cadence and named checkpointed RNG streams stay fixed. Numerical scorer controls and saved actual-training GIFs accompany the report. The focused software/protocol suite passes 231 checks, including CPU/CUDA float32/64 parity, public factory, checkpoint resume and nonlinear overshoot without evaluator calls. This is a research diagnostic; profiles remain provisional and no default adoption or ordinary qualification follows.

Tail execution:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_ablation/queue/events.jsonl
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round4-20261009/projection_ablation/logs/driver.log
```

Source and ready declarations were pushed before admission. Public Queue/drain used one shared GPU0 worker and no full-compile callback. Bulk logs, JSONL, checkpoints and tensors stay outside Git. The parent owns the single current goal leaderboard; this report contains a scoped arm-comparison table only.

## Saved-state loss attribution

The conditional hosts own `L = A + 1.5 C + .02 mean(z²) + spread (weight .05)(z)`, plus the existing weight1 paired residual on both-land rows in residual student. Native winner `prior_reg=0` does not disable these host-owned terms. Mid-scale instead owns adversarial plus its existing paired cover. No term is removed in the trained ablation.

| Original winner saved endpoint | Normalized combined identity derivative | Without set coverage | Without adversarial | Without paired residual | Nonadditivity ratio |
| --- | --- | --- | --- | --- | --- |
| Trajectory | +.00578882 | −.0204396 | +.0198500 | Absent | .99691 |
| Residual | −.000425692 | −.0080514 | +.0066190 | +.0030832 | 1.21582 |

The nonadditivity ratio is `||N(sum grad L_j) − sum N(grad L_j)|| / ||N(sum grad L_j)||`. Gradient sums are verified, but the full-DualNorm mapping N depends nonlinearly on singular values and row norms. It includes each group's saved rate and smoothing. It is therefore invalid to add independently normalized loss displacements as if they were contributions to the combined applied step. The leave-one-out proposals are counterfactual derivatives, not independent training arms.

At these original endpoints, set coverage is the largest observed antagonist to identity progress. Removing latent L2 or spread has much smaller effects and does not change either sign; removing adversarial harms both, and removing the existing paired residual reverses residual's sign. Yet every saved endpoint's combined *protected* derivatives are already negative, including failing endpoints. The six endpoint probes do not identify which terms caused earlier conflict steps, nor prove that one term should be removed. At PR361's solved endpoints, removing coverage does not consistently improve the local identity derivative. Normalization, mixed effects and the moving critic prevent identifying a single global culprit. Original evidence identities and source digests remain in the saved receipt.

## Full numerical comparison

| Task | Corrected nonascent | Direction blend alone | Blend plus finite acceptance |
| --- | --- | --- | --- |
| Trajectory | **FAIL**, MSE .2437363565; suffix0 | **PASS**, MSE **.00025847997**; suffix**19** | **PASS**, identical MSE and suffix**19** |
| Residual student | **FAIL**, MSE .0626398921; success5/12, wrong7/12; suffix0 | **PASS**, MSE **.00024908854**; success**1**, wrong**0**; suffix**21** | **PASS**, MSE **.00036959240**; success**1**, wrong**0**; suffix**21** |
| Mid-scale identity | **PASS**, mid identity .99295571; suffix20 | **PASS**, mid identity **.99299656**; suffix**20** | **PASS**, mid identity **.98635834**; suffix**20** |
| Two-pole explicit fixture | **PASS**, mean_abs .95850247, grad_med .95234573; suffix17 | Identical **PASS** | Identical **PASS**, certified retry |
| Gaussian smoke | **PASS**, first confirmation375, 3/24 scheduled passes; endpoint KS .07184216 | Identical **PASS** | Identical **PASS** |
| Own Gaussian stability | **FAIL**, stationary2/72; deadline FAIL, shifted hold0/24; endpoint KS .32062289 | Identical **FAIL** | Identical **FAIL** |

The identity gates remain MSE <= .02, plus every correct landing and no wrong landing for residual. All learned conditional gates require five complete terminal passing checks. Both repaired identity arms have 19/24 trajectory passes and 21/24 residual passes; first/confirmed steps 100/167 and 67/134. All 12 saved endpoint identities are nearest their own target in both repaired tasks. Mid-scale requires concept cosine and identities >= .85 and magnitudes in [.75,1.25]; all eight bounds pass, with sustained confirmation 300 in all arms. The direction-only residual's lower endpoint error is a descriptive result of one deterministic comparison, not statistical superiority.

Smoke certifies any scheduled full pass plus an independent same-state confirmation and completes1,000 updates; its endpoint KS need not pass. Stability resumes each arm's own confirmed state, adds 5,000 updates, requires every stationary check, deadline reacquisition and shifted hold. No control checkpoint is borrowed. Gaussian metrics retain 4,096 samples, finite fraction1, mean error <= .2sigma, width ratio[.8,1.2] and KS <= .05. Two-pole retains mean_abs >= .3 and grad_med <=1 with its separate stored-weight fixture. Acquisition cannot replace retention.

## What finite checks contribute

| Active task | Direction-only conflicts | Finite conflicts / accepted | Finite probes / reductions | Smallest accepted scale | Mean conflict displacement/base norm: direction / finite |
| --- | --- | --- | --- | --- | --- |
| Trajectory | 12/400 | 12/12 | 24 / 0 | 1 | .701794 / .701794 |
| Residual | 55/400 | 186/186 | 997 / 625 | 1/64 | .780943 / .118141 |
| Mid-scale | 43/800 | 45/45 | 152 / 62 | 1/8 | .902208 / .468236 |

All direction-only conflicts apply the full blend; no evaluator, finite probe, rejection or Pareto stall occurs. Its maximum positive rounded protected derivative is zero, and maximum retained norm ratios are .70708 / .81597 / .97784. Nonascent control leaves small rounded violations up to 1.14e-8. Direction-only does not certify finite objective descent.

Finite strict progress certifies all 243 conflicts, with 1,173 probe evaluations and 687 scale reductions; no rejection or Pareto stall occurs and maximum Armijo violation is zero. Accepted same-batch decrease sums are trajectory adversarial 1.06403; residual adversarial .383408 / paired .122690; mid-scale adversarial .189609 / paired cover .0493175. These are comparisons against each update's fixed critic and batch, not a net changing-game objective or population guarantee. [Armijo's variable-step construction](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-s.pdf) motivates sufficient-decrease checks; its deterministic convergence assumptions are not established here.

Trajectory's zero reductions and the [bitwise state/output audit](inactive-trained-parity.json) show that removing finite checks leaves its entire scored trajectory and final model/base optimizer states unchanged. Residual and mid-scale take different paths after finite scaling, yet retain the same complete sustained PASSes without it. The three-arm comparison therefore isolates the finite-check layer: it changes local acceptance and motion, but is unnecessary for these gates. It does **not** separate the feasible projection half from the common-descent half of the direction, or identify a unique harmful training term. No finite-only strict-boundary arm is meaningful because the ideal projected derivative is zero.

## Protocol, execution and retained evidence

Measured source is **32a55b1a972695e70ae867231bc37d93fb0b58bc**, digest `09b10f621e20b7f7685c806e0f86320e01a41e273477290450ecd670f54409b0`. All three ready arms were frozen before one bounded drain. [Results](results.json), [compact receipts](receipts.json), [provenance](provenance.json) and [validation](validation.json) bind every final metric to its actual recipe, task gate, prior, initializer, budget, sampling, source and runtime. Public model initialization and every consumed named RNG stream match across each task's arms. Every exercised scientific source file remains unchanged after training; later code changes are publication/audit helpers only.

Gaussian tasks use learned 256-component MoG, sigma .1, unstandardized, fixed uniform weights. Conditional trajectory/residual enumerate their learned zero-width cloud directly; mid-scale uses a parameter cloud without latent sampling; two-pole uses direct sample coordinates. All cloud exceptions stay explicit on their unchanged task cards. The scalar adapter's one actual real tensor for D/G is retained. Fixed identity/zero/stored-weight two-pole is a separate cohort, never a replacement for public deterministic initialization. Named constructor/data/noise/evaluation streams and complete optimizer states are checkpointed. No noisy serving, EMA or target retargeting is introduced.

All 18 final jobs complete: **13 PASS / 5 FAIL**, zero final BLOCKED/INCOMPLETE/INVALID. They account for 23,040 certified task updates, plus the first two-pole attempt's log reaches its full 80-update allowance before timing out. There are 19 paid attempts and exactly **one execution-only retry**. Original strict two-pole attempt `2cccdd36a1ee4dc88b18dff63d7f08cb` exhausted its 300-second wall allowance before certified result publication; its **INCOMPLETE** result,302.150086 paid seconds, original three certificate hashes and log stay intact in [execution retry history](results.json). The unchanged frozen retry `83fe2f9510934b6e83f1643cea5df859` completes in7.930917 paid seconds. The timeout is superseded only for execution, not regraded as a scientific pass.

Total paid worker time is **1154.075927 seconds**, including the timeout and retry. Initial full reservations 19,260 plus retry 300 total **19560 seconds**, below the21,600-second track ceiling. No reservation, worker or watcher remains. The source-frozen studies are concluded ([nonascent readout](../../../records/readout-7cac8ddc90e2d8e580ac36b4.json), [direction_blend readout](../../../records/readout-e1716333811cd46df3150ed3.json), [strict_progress readout](../../../records/readout-45b4ad3bd3dbeaa2bc7ebe7b.json)); reporting uses summaries-only compilation and preserves archived qualification/automation hashes. Contended timing is accounting, not an optimizer-speed claim. Saved-state analysis adds zero training steps or sampling draws; software/scorer/media audits likewise add no training. No unchanged scientific run was added for publishing or merging.

[Oracle and destructive scorer controls](scorer-controls.json) verify both identities, Gaussian width/mean/CDF sensitivity, mid-scale identity sensitivity and the stored two-pole fixture gate. They are explicitly target-informed scoring witnesses, not trained arms or alternative initializers. Every [GIF](media/index.json) uses certified actual-training states and fixed target/output axes; selection changes no scoring check or grade. [Publication](publish.py), [parity audit](audit_saved_parity.py), [scorer controls](scorer_controls.py), [validation](validate_saved.py), [original execution](execute.py) and [bounded retry](retry_timeout.py) provide reproduction sources. Bulk stdout, JSONL, checkpoints and tensors stay under the local queue archive.

## Actual-training GIFs and recommendation

| Task | Nonascent | Direction alone | Finite acceptance |
| --- | --- | --- | --- |
| Trajectory | [GIF](media/nonascent-trajectory.gif) | [GIF](media/direction_blend-trajectory.gif) | [GIF](media/strict_progress-trajectory.gif) |
| Residual | [GIF](media/nonascent-residual_student.gif) | [GIF](media/direction_blend-residual_student.gif) | [GIF](media/strict_progress-residual_student.gif) |
| Mid-scale | [GIF](media/nonascent-mid_scale_identity.gif) | [GIF](media/direction_blend-mid_scale_identity.gif) | [GIF](media/strict_progress-mid_scale_identity.gif) |
| Two-pole | [GIF](media/nonascent-two_pole.gif) | [GIF](media/direction_blend-two_pole.gif) | [GIF](media/strict_progress-two_pole.gif) |
| Gaussian smoke | [GIF](media/nonascent-gaussian1d_smoke.gif) | [GIF](media/direction_blend-gaussian1d_smoke.gif) | [GIF](media/strict_progress-gaussian1d_smoke.gif) |
| Gaussian stability | [GIF](media/nonascent-gaussian1d_stability.gif) | [GIF](media/direction_blend-gaussian1d_stability.gif) | [GIF](media/strict_progress-gaussian1d_stability.gif) |

Prefer **direction_blend** as the smallest measured opt-in repair for this conditional scope. Retain **strict_progress** when an application needs its finite same-batch protected-loss certificate. Neither version repairs Gaussian continuous stability or establishes distribution fidelity beyond these frozen tasks. The result does not authorize broad native/image/word transfer, ordinary Tier2 qualification, a combined repair or public-default promotion. Stop this completed ablation; any transfer or further component isolation needs its own bounded question. The original winner's archived 7/21 and all earlier evidence remain under their original contracts.
