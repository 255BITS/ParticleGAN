# BCAP generator idle rule (`generator_idle_se`)

Closed PR [#366](https://github.com/255BITS/ParticleGAN/pull/366) proposed skipping the generator
step while the critic cannot tell real from fake. It targeted the constant-LR "stay" failure noted
in closed PR [#279](https://github.com/255BITS/ParticleGAN/pull/279), but was never scored. This
round rebuilds the rule on the recipe factories and scores PR #366's planned value, 1.0 standard
error, against the BCAP leader `recipe-prior-refactor-bcap-pure-v1`. Both PRs are cited as
motivation only, under their original head commits ([prior_evidence.json](prior_evidence.json)).

- **Trainer delta:** `generator_idle_se` None → 1.0, and nothing else ([spec](spec.json), [registration](registration.json)).
- **Held fixed:** protocol seed 0, `deterministic_orthogonal` initializer, revision-9 `discriminator_stability` tasks,
  architectures, data law, batch sequence, prior capacity, sampling, update budgets, cadence and gates.
- **Source:** executed commit `94a6d429de0d01abc9fa3f993ed2089b741ef111`, digest `2ffc95c341a7…`.
  Both arms ran on two RTX A6000s.

## Leaderboard

<!-- leaderboard:start -->
| Task | Tier | Control | idle1.0 | idle1.0 idle steps / G decisions |
| --- | ---: | --- | --- | ---: |
| gaussian1d_smoke | 1 | PASS | PASS | 429/1000 (42.9%) |
| two_pole | 1 | PASS | PASS | 1/80 (1.2%) |
| unused_token_hold | 1 | PASS | PASS | 0/200 (0.0%) |
| ae_gan_hold | 1 | PASS | PASS | 64/250 (25.6%) |
| ring16_acquisition | 1 | PASS | FAIL | 5/1600 (0.3%) |
| five_word_joint_smoke | 1 | INCOMPLETE | PASS | 10923/20001 (54.6%) |
| clockfree_audit_measurement_v1 | 1 | PASS | PASS | — |
| 21 Tier 2 tasks (gaussian1d_stability, five_word_joint_hold, trajectory, residual_student, unipolar, cover_leftover, mid_scale_identity, mode_hold, vector_two_broad, vector_unequal_mass, vector_unequal_width, vector_anisotropic, vector_overlap, vector_spiral, img_stripes2, img_bars4, img_blobs4, img_intensity2, grid100, rotated100, staggered100) | 2 | BLOCKED | BLOCKED | — |

Required passes: control 5/6 Tier 1, 0/21 Tier 2; idle1.0 5/6 Tier 1, 0/21 Tier 2 (the clock audit is an optional diagnostic and is listed but not counted).

Off-ladder target diagnostic (separate cohort, no qualification credit):

| Arm | Task | Diagnostic verdict | Sustained gate | Target metrics (final) | Idle steps / G decisions |
| --- | --- | --- | --- | --- | ---: |
| control | mode_hold | FAIL | terminal suffix 3/>=5 | modes 8, hq 0.989258 | — |
| control | gaussian1d_stability | FAIL | stationary 2/72, reacquisition FAIL, shifted hold 0/24 | mean_error_sigma 0.255238, std_ratio 0.66233, cdf_ks 0.320623 | — |
| idle1.0 | mode_hold | FAIL | terminal suffix 3/>=5 | modes 8, hq 0.989258 | 0/1200 (0.0%) |
| idle1.0 | gaussian1d_stability | FAIL | stationary 14/72, reacquisition FAIL, shifted hold 6/24 | mean_error_sigma 0.312055, std_ratio 0.864775, cdf_ks 0.126974 | 2844/6000 (47.4%) |
<!-- leaderboard:end -->

This table compares two arms on one frozen source. The goal's single current leaderboard is still
[technique-inventory.md](../technique-inventory.md). [measurements.md](measurements.md) and
[results.json](results.json) hold every metric with its target. [audit.json](audit.json) is the
saved-receipt audit, and every check passes. [media/](media/README.md) has actual-training GIFs
of every completed toy (13).

## Readout

**The registered repair criterion is not met.** The idle arm loses `ring16_acquisition`, which is
a Tier 1 gate. Its terminal passing suffix is 0, against at least 5. Its final
`component_covariance_error` is 3.09991, against a limit of .85; the control scored .496567 with a
suffix of 26. This happened after only 5 of 1600 G steps were skipped. Every Tier 2 cell
of the idle arm is therefore BLOCKED, including both target gates. An off-record software dry run
before admission predicted this same ring16 FAIL. That value was not changed, so no tuning was
done against the gate.

**The matched control is INCOMPLETE on one Tier 1 task.** Its `five_word_joint_smoke` reached
19,168 of 20,001 updates before the 900 s allowance expired. The two A6000s were shared with other
agents' jobs at the time. This is an infrastructure outcome, not a scientific FAIL. The plan did
not authorize retries, so none was run. The control's Tier 2 is BLOCKED too. Even so, the
candidate's verdict holds without it: a Tier 1 FAIL where the control passes cannot meet
"6/6 Tier 1".

**Idle fraction compared with repair.** The rule fires where the gap between real and fake critic
scores is already small and noisy:

- about half the G steps on the 1D Gaussian (42.9%) and the five-word smoke (54.6%)
- a quarter on `ae_gan_hold` (25.6%)
- almost never on the ring (0.3%), `two_pole` (1.2%) or `unused_token_hold` (0%)

So it acts as a hold brake on easy, already-matched tasks. On ring16, the few skips perturb a
chaotic trajectory instead of steadying it.

**Off-ladder diagnostic.** This is a separate cohort with no qualification credit. Both frozen
requests were replayed on `gaussian1d_stability`, each from its own certified smoke state, and on
`mode_hold`. The control replays the archived leader exactly: stationary 2/72 and std_ratio
.66233 on the Gaussian, suffix 3/5 on mode_hold.

- On `gaussian1d_stability` the idle arm skips 47.4% of G steps and moves the right way. Stationary
  passes go from 2 to 14 of 72, shifted-hold passes from 0 to 6 of 24, std_ratio from .662 to .865
  and cdf_ks from .321 to .127. The registered falsifier (std_ratio < .8) is not triggered, and the
  std_ratio half of the prediction holds. The other half, cdf_ks ≤ .05, fails. mean_error_sigma
  worsens from .255 to .312, and the gate stays FAIL.
- On `mode_hold` the rule never fires (0 of 1200 decisions). The run is identical to the control
  and stays FAIL, so this rule cannot repair mode_hold.

## Recommendation

Keep `generator_idle_se` opt-in, with default None, and do not adopt it. Its off path is
bitwise identical to develop (see `tests/test_generator_idle.py`). On this suite the 1-SE rule
costs a Tier 1 gate and leaves both targets FAIL. A follow-up should change the idle signal rather
than the threshold, and be registered as a new structural idea with a fresh matched control. Two
reasons: the rule never fires on mode_hold, and it fires about half the time on easy holds. This
round runs no threshold sweep, seed variant, retry or Tier 3.

## Reproduce

```sh
A=/path/outside/checkout/bcap-generator-idle-v1   # git-ignored bulk archive
python reports/forge/bcap-generator-idle/workflow.py prepare --archive $A
python reports/forge/bcap-generator-idle/workflow.py run --archive $A --source-commit <commit> --submit
python -u reports/forge/bcap-generator-idle/workflow.py run --archive $A --source-commit <commit> --drain --gpus 0,1
tail -f $A/queue/*/*/run.log                      # per-job, line-buffered
python reports/forge/bcap-generator-idle/workflow.py publish --archive $A
python reports/forge/bcap-generator-idle/diagnostic.py --archive $A   # off-ladder target diagnostic
python reports/forge/bcap-generator-idle/summarize.py --archive $A    # audit.json + leaderboard
```
