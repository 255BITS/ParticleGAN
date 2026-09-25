# 22-toy results and continuous stability

**Selected research base: [direct_particle_response](direct-particle-base/README.md).**
**15 PASS / 1 FAIL / 6 NOT_RUN** on its own GPU suite. Unequal-width covariance
fails (.981321 > .85, zero terminal passing checks). All seven initial gates
and eight further toys pass. Full22 and own-state retention are unqualified.

| Current research candidate | GPU measured | GPU unmeasured | Blocker | Own-state hold |
|---|---|---:|---|---|
| direct_particle_response (selected) | **15 PASS / 1 FAIL** | 6 | unequal width | NOT_RUN |
| dimension_rms_hybrid (parent) | 6 PASS / 1 FAIL | 15 | two_pole | NOT_RUN |

[Sixteen raw results and independent audit](direct-particle-base/audit.json) ·
[Latest completed round: 9 proposals, 34 gates, 25 PASS / 9 FAIL](direct-particle-base/round-summary.json).
These are partial candidate results, not directly ranked full22 scores. Do not
combine them with the historical native GPU16/22 control below.

## Round leader, marked not promoted: a2_bounded_damp

**19 PASS on the 22 GPU toys, own-state continuation 7/8, and the selected base's blocker
cleared: unequal-width covariance .0590 against a .85 maximum, from .9813.** This marks the
round's leading candidate. It is not promoted and is not a full-22 replacement.

The change is `latent.py` only; `config.json`, `mechanism.py` and `response.py` are
byte-identical to direct-particle-base. A sparse registered ParticlePrior row scales its
update by agreement with that row's last observed gradient, u = (.75 + .25*cos(g,h)) * g
bounded in [g/2, g], inactive rows motionless, scoped to tables with a missing row and a
cumulative observation rate below one half. Marked lineage:
`gan-attempts/claude-pool-20260925T024114Z/surviving_moment_weight/20260925T042840Z-2529991`.

| Evidence | Measured, CUDA |
|---|---|
| 22-toy suite | 19 PASS, three native problems failing or unrun |
| Replication | 19/22 independently re-earned in six attempt lineages across three lanes |
| Own-state continuation | 7/8 PASS; `vector_unequal_mass` FAIL, 2 PASS / 1 FAIL across lineages |
| unequal width | covariance .0590 against a .85 maximum |

An earlier version of this entry gave a grid100 collapse, and a no-op latent control that
reached 100/100 modes where a2 reached 17/100, as reasons against promotion. **That
attribution is withdrawn.** See the next section: both were single runs of a bistable gate.

## Native results are bistable: single-run attributions are void

Four seeds per arm on grid100, identical code, driver and device, only the config seed
differing. Full gate needs coverage and accuracy, both sustained over five terminal checks.

| Arm | final modes by seed | acquired 100 | full gate PASS |
|---|---|---:|---:|
| a2_bounded_damp, `reg_arm: a_r1r2` | 17, 100, 33, 100 | 2/4 | 0/4 |
| no-op latent control, `a_r1r2` | 100, 17, 17, 28 | 1/4 | 0/4 |
| no-op latent control, `b_cap` | 100, 100, 61, 70 | 2/4 | **2/4** |
| a2_bounded_damp, `b_cap` | — | — | 1/4 |

Acquisition is decided in one window near steps 500 to 750: a run either reaches 100 modes
there or stalls near 17 for the remaining 6250 steps. Seed 1234, the seed every earlier
single-run conclusion used, is the one where the `a_r1r2` control acquires and a2 does not;
seed 1235 reverses it. **Any native claim from one run, for or against a mechanism, is not
evidence.** Native results now require an acquisition rate over at least four declared seeds.

## `reg_arm` decides native precision; the latent rule does not

Of the runs that acquired all 100 modes, terminal live precision against the .97 floor:

| Arm | precision when acquired | center_rms_sigma vs .20 |
|---|---|---|
| `a_r1r2` | .9587, .9607, .9629 — never reaches .97 | .19 to .23 |
| `b_cap` | **.9865, .9851** — passes outright | **.1416, .1377** |

a2 against a no-op latent moved precision by .4 points. `reg_arm` moved it by 2.4. Measured
one config field apart with the same latent rule on the sixteen focused toys, `a_r1r2` scores
16/16 and `b_cap` 12/15, losing `vector_unequal_mass`, `mode_hold` and `img_stripes2`. Each
arm wins what the other loses, and `configs/100gaussians/default.toml`, the real target's
production recipe, uses `b_cap`.

## Learning-rate floors buy plasticity without losing stability

Ring hold to update 2400, then a target shift, with a matched frozen control. One run per
setting, so this needs seeds before it is load-bearing.

| floors network/prior | post-convergence hold | re-acquires after the shift | 16 focused toys |
|---|---|---|---:|
| .01/.05 (current base) | 1200/1200 | **never, 0/81 checks** | — |
| **.1/.1** | **1200/1200** | **yes, 110 updates, 81/81** | **16/16** |
| .3/.3 | 1200/1200 | partial, 68/81 | 16/16 |
| 1/1 (fully constant) | NOT_CONVERGED | — | — |

The current base holds perfectly and can never learn again; fully constant never converges.
The frozen control passes 0/81 after the shift, so the .1/.1 re-acquisition is real.

## The sparse-latent family is inert where tables are fully observed

On the 12-row `mode_hold` ring, a2 records zero scoped calls and runs bitwise as the parent.
Ring and hold results therefore cannot credit or blame it, and a latent rule cannot recover
`mode_hold`. It engages on the natives and the sparse vector hosts, which is where its
unequal-width result comes from.

Raw results, configs and drivers for every table above: `gan-attempts/session-findings/`.

**Yes, there is a recorded 22/22 PASS.** The original
`constraints_simple_regularization` recipe still passes when its saved CPU
evidence is independently regraded. Its preserved-recipe CUDA control scores
**16/22**, with all three native 100-mode gates passing.

| Recipe / scope | Backend | Toy PASS / 22 | Native PASS / 3 | Post-convergence hold |
|---|---|---:|---:|---|
| Original recipe, decay and original auxiliary host terms | Recorded CPU, regraded | **22/22 PASS** | 3/3 | Not a constant-rate claim |
| Same original recipe | CUDA control | **16/22** | 3/3 | Not a constant-rate claim |
| Shared column RMS, strict continuous variant | CUDA | 11/22 | 0/3 | 131 good updates, then FAIL |
| Shared RMS, strict continuous variant | CUDA | 11/22 | 0/3 | 35 good updates, then FAIL |
| H, strict continuous variant | CUDA | 11/22 | 0/3 | Not confirmed by 6000 |
| Epsilon, strict continuous variant | CUDA | 9/22 | 0/3 | 19 good updates, then FAIL |

[Original 22/22 evidence](simpler22/README.md) ·
[Same-recipe GPU control and all 22 results](gpu-known-winner-control/README.md) ·
[Continuous-learning GPU matrix](gpu-leaderboard/LEADERBOARD.md)

The first GPU report omitted the original winner and evaluated only eight
continuous-learning variants. Its statement that no candidate qualified applied
to that cohort. It did not invalidate the earlier passing recipe. The new
control supplies the missing same-recipe CPU/GPU comparison.

The original recipe uses decaying learning rates and the frozen auxiliary
losses in the autoencoder and unused-token hosts. The strict H-family variants
keep rates active and disable those auxiliary terms. Compare these scopes
explicitly: a finite-budget pass with decay is not evidence of continual
stability with active rates, and the historical 22/22 is not a claim that every
host uses an exclusively adversarial objective.

The preserved-recipe GPU failures are `mode_hold`, `trajectory`, `img_bars4`,
`img_blobs4`, `img_intensity2`, and `vector_unequal_mass`. The other sixteen pass,
including native coverage and accuracy on all three 100-mode problems. All 22
GPU control verdicts and all 105 earlier GPU variant results were regraded.

The earlier porting work started from the **original scheduled CPU recipe**.
The selected direct-particle formulation above now replaces it as the research base. [Twenty-four porting controls](cpu-recipe-gpu-port/README.md)
confirm 6/6 fresh CPU passes on the failed hosts. CPU initialization alone
recovers four on CUDA; ring and unequal mass remain failing. This partial
diagnostic is not a new 22-toy score.

Shared column RMS is retained as a historical **strict continuous-learning**
reference; it is not the active porting base. None of the
continuous variants passes the required 1,200-update hold after confirmation.
PR107/140 do not confirm by 6000; PR143 confirms at 1400 and fails after 11 good
hold checks. Their published adapters cover only two toys.

[GPU protocol and replay](gpu-leaderboard/README.md) ·
[Historical strict GPU reference](gpu-leaderboard/current-gpu-reference.json) ·
[Historical continuous-learning CPU board](continuous-practical-leaderboard-cpu-history.md)

[PyTorch 2.14 upgrade check](torch214-gpu-check/README.md): both remaining
blockers fail identically to 2.13 under native and CPU initialization. Four
full-budget CUDA runs; no change to the 16/22 full-suite result.

[Completed formulation round](formulation-round-20260924/README.md): 18 proposals,
36 primary GPU gates, 7 PASS / 29 FAIL. None passes both blockers. The full-suite
reference stays 16/22. [Measured follow-ups](formulation-round-20260924/FOLLOWUPS.md)
include two formulations that pass both blockers but fail trajectory; neither is
promoted. R1/R2 history is a reason to avoid unchanged repeats, not to reject a
candidate that passes its measured gates.
