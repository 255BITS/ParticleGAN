# Paired BD sigma-release follow-up

## Question

Can releasing the learnable output-noise floor after the ordinary generator
parameters settle improve the rotated100 accuracy failure of `cf1-bdpair`,
without changing its paired-only birth/death rule?

## Evidence motivating the experiment

`cf1-bdpair` passes grid100 and staggered100, but rotated100 fails. Its rotated
holdout has precision 0.95093, centre RMS 0.24201 sigma, absolute covariance
trace bias 0.10865, and radial KS 0.05649. The failure is not explained by mass
TV (0.02194). Output sigma stays at 0.029 throughout the run. Source inspection
shows the width-release rule includes both the sigma optimizer tester and the
prior tester; the prior remains active after ordinary generator parameters
settle. In the rotated trajectory, the ordinary generator reaches its 1/64
learning-rate floor at update 5112.

## Single declared change

Starting from the byte-identical `cf1-bdpair` package, change only the
learnable-width floor release condition. Release the floor when every
non-sigma generator optimizer group's stationarity scale is at or below 1/64.
Exclude the prior group and the sigma parameter's own tester from this
condition. Continue to compute the floor as `base_sigma * controller.mobility`
after release, and retain `max(exp(log_sigma), floor)`.

Keep the initial sigma (0.029), paired-only BD, all optimizer settings,
controller and critic rules, initialization, data/noise streams, and native
evaluation thresholds unchanged. This is a change to the width-control policy;
it is not a task-specific sigma setting.

## Run order and decision rule

1. Run the unchanged 7,000-step grid100 gate.
2. If grid100 passes, run rotated100 and staggered100 with the same frozen
   package and recipe. Do not tune from intermediate evaluation metrics.
3. A candidate solves the native problem only if all three unchanged native100
   gates pass, including terminal live accuracy and the independent holdout.

Log each checkpoint's raw learned sigma, applied sigma, width floor, ordinary
generator/prior/critic stationarity scales, sigma gradient-block status,
precision, centre RMS, trace bias, radial KS, and paired BD move count. The
declared intervention is active only if the floor falls after ordinary G
settlement. If the floor falls but sigma remains high, the learned sigma itself
is the limiting factor. If width and fidelity recover but centre error remains,
that indicates a separate spatial-dynamics failure.

## Astra review

Astra reviewed the candidate source and results before this experiment. It
identified the all-group sigma release dependency as the cleanest next test,
recommended excluding prior and sigma testers while retaining the 0.029
initialization, and advised keeping BD, optimizer rates, critic policy, and
evaluation rules fixed.

## D-tracking result: improved to 2/3, still not solved

The one-factor D-tracking candidate passed grid100 and rotated100, but failed
staggered100. The results were:

| Task | Gate | Terminal precision | Live centre RMS | Holdout |
|---|---:|---:|---:|---|
| grid100 | PASS, 19/34 | 0.9815 | 0.1857 sigma | PASS |
| rotated100 | PASS, 10/34 | 0.97225 | 0.1592 sigma | PASS |
| staggered100 | FAIL, 18/34 | 0.9749 | 0.2072 sigma | PASS |

The staggered failure is narrowly the live centre RMS threshold (.20); its
holdout passes, and mass TV, covariance trace bias, and radial KS pass. In the
last checkpoints the prior tester reports scale .5 while D's own scale is
below 1e-6. D tracking therefore materially changed the critic rate, but the
current `max(own, prior)` rule did not clear the final small center-error
margin on staggered. The candidate and raw evidence are archived under
`/ml2/hypergan/gan-attempts/formulations-20260928T040504Z/custom_follow/20260928T040504Z-4138988/`.

This is not a solved 3/3 native candidate yet. Astra reviewed the staggered
trajectory and selected an isolated follow-up; the result and plan follow.

## Predeclared follow-up: reassess after prior rate reductions

Astra's trajectory review found that staggered's prior made a strongly
stationary decision at update 5112 (the two tested scales had t statistics
-11.13 and -17.85), reduced its scale to 0.5, and retained a block duration of
64. No further prior decision occurred before step 7000; about 1,184 updates
remained until the next decision. At the end, median row movement was 0.216
sigma, compared with 0.111 for `cf1-bdpair`, while generator motion was only
0.0143 sigma RMS. Rotated reduced its prior scale to 0.125 and passed.

Starting from `cf1-bdpair-dtracks-prior`, make one change: whenever a prior
stationarity decision is `stationary` and lowers its scale, set that tester's
next block length to its existing `B0`. Preserve its new scale, accumulated
intrinsic-time remainder, and normal decision-boundary cleanup. Do not reopen
the scale at 1, restart the complete tester, or change sigma, BD, critic
tracking, or evaluation rules.

Run grid100 first; if it passes, run rotated100 and staggered100 with unchanged
gates. The expected evidence is additional supported prior decisions and lower
sustained prior row motion where the evidence supports another reduction. The
main risk is that short blocks overreact to local drift and cause scale
reversals. Faster decisions without improved task metrics do not count as
success.

Astra reviewed this follow-up after inspecting the full `dtracks-prior`
trajectories and recommended the B0-only reset, retaining the new scale and
existing boundary cleanup.

## Result: rejected on grid100

The frozen grid100 run completed with **FAIL**, 2/34 passing observations and
no terminal accuracy passes. At step 7000, live precision was 0.958, mass TV
0.0619, centre RMS 0.3506 sigma, minimum covariance eigenvalue ratio 0.074,
and maximum ratio 2.299. The independent holdout also failed: precision 0.9579,
centre RMS 0.3408 sigma, and mass TV 0.05525. This regressed against
`cf1-bdpair`, which passed grid100 and staggered100.

The rule activated: applied sigma was still 0.029 at step 2000, then fell to
0.0140 at 2500, 0.0060 at 3000, and 0.00113 at 7000. In this interval, paired
BD moves accelerated and mass TV rose as high as 0.0844. The result rejects
G-only settlement as a safe release signal while the prior remains active. No
transfer runs were started because the preregistered grid gate failed.

## Next isolated hypothesis

Return to the original `cf1-bdpair` width policy. The rotated baseline has a
large critic/prior scale gap late in training: prior scale 1.0 versus critic
scale 1/64, while the prior continues moving. The next one-factor experiment
will set the critic stationarity scale to `max(critic_own_scale, prior_scale)`
before the existing payoff damping. Keep the critic's own stationarity tester
running so it observes the actual applied learning rate. Preserve the original
sigma policy, paired BD, all other optimizer/controller settings, and gates.

Test grid100 first, then run both transfers only if grid passes. Log the critic
own and applied scales, prior scale and reversals, clean and noisy geometry,
precision, centre/width errors, and BD moves. This experiment is separately
predeclared; it does not reuse any change from the rejected width-release run.

## Result: rejected fast-recheck rule

The B0 reset candidate failed grid100 (18/34). Its final live centre RMS was
0.2469 sigma and holdout centre RMS was 0.2265 sigma, while precision and mass
TV passed. The prior tester made a stationary decision at 5112, reduced scale
to 0.5, then detected drift at 5160 and restored scale to 1; subsequent blocks
grew from 1 to 4, 16, and 64. This supports retaining long evidence blocks:
the reset discarded useful timescale evidence and increased prior movement.
No transfer runs were started.

## Next isolated experiment: damp applied prior rate after stationarity

After reviewing the fast-recheck failure, Astra recommended an event-based
rate intervention. Starting from `cf1-bdpair-dtracks-prior`, after the first
prior stationarity decision that lowers the tester scale, multiply the prior's
applied LR by 0.5 for the rest of the run. Leave the tester's scale and block
duration unchanged. Keep D's tracking floor tied to the prior tester's
unmodified scale, so only prior movement is directly reduced. Do not change
sigma, BD, D tracking, or the native evaluator.

Run grid first, then both transfers only if grid passes. This is a causal
diagnostic, not a proposed general policy. The expected signal is lower prior
motion and staggered centre error while retaining coverage and precision. If
staggered improves but rotated loses its pass, reject simple prior damping as a
general fix.

Astra recommended this after comparing the prior decision history across the
baseline D-tracking, staggered D-tracking, and failed fast-recheck grid runs.

## Result: half-strength D tracking fails rotated

The preregistered `max(D_own, 0.5 * prior_scale)` candidate passed grid100 but
failed rotated100 with 0/34 observations passing and no passing terminal
accuracy checks. At step 7000, live precision was 0.96015, centre RMS 0.2402
sigma, trace bias 0.0990, and radial KS 0.0524; the holdout also failed
(precision 0.96058, centre 0.2183 sigma). The prior tester had returned to
scale 1 after a drift decision at step 5750, leaving D at half that scale.
Staggered was not run because rotated failed its gate.

## Result: prior damping trades away rotated precision

The event-triggered 0.5 prior-rate intervention passed grid100 but failed
rotated100 (3/34 observations; terminal accuracy arrived only at 6500). This
was a holdout precision miss, 0.96953 versus the 0.97 threshold; holdout centre
(0.1272 sigma), trace bias, radial KS, and other accuracy terms passed. Final
live centre RMS was 0.1437 sigma. Damping also changed the controller path, so
the result is not a simple time-scaled version of the undamped run. Staggered
was not run because rotated failed the predeclared gate.

## Next isolated experiment: half-strength critic tracking

After reviewing the exact precision failure, Astra recommended returning to
`cf1-bdpair-dtracks-prior` and changing only the D tracking floor to
`max(D_own_scale, 0.5 * prior_tester_scale)`. The coefficient 0.5 is the
controller's existing dyadic step. Keep the prior's own rate, its tester and
block duration, sigma, BD, payoff damping, and all evaluation rules fixed.

Run grid100 first; if it passes, run rotated100 and staggered100. Record actual
D/prior rates, prior decisions, center movement, and precision. The hypothesis
is that half-strength tracking preserves enough critic response for rotated
while reducing coupled activity enough to clear staggered's small center miss.
This is a test, not an established effect; any task-specific tradeoff rejects
the interpolation as a general candidate.

Astra reviewed the prior-damping failure and specifically recommended this
single-parameter rate interpolation.

## Final bounded interpolation test

Astra's review found that half tracking caused feedback reversals in the prior
tester: it returned to scale 1 multiple times and ended with four times the
applied D rate of the full-tracking run. Full tracking remains the best
candidate. Astra approved one bounded empirical interpolation, `alpha=0.75`,
while cautioning that performance need not interpolate smoothly.

Starting from `cf1-bdpair-dtracks-prior`, change only the D scale floor to
`max(D_own_scale, 0.75 * prior_tester_scale)`. Keep all other controller, prior,
sigma, BD, payoff damping, and evaluation settings unchanged. Use the frozen
grid100 → rotated100 → staggered100 sequence, proceeding only when the prior
gate passes. Record prior decisions and applied rates to detect the same
feedback cascade. If this fails, end the multiplier search and retain full
tracking as the best measured candidate.

## Alpha=.75 completed result and Astra review

Candidate: `cf1-bdpair-dtracks-075-prior`, package SHA-256
`8cdcc534c348d51afaad8b65b1058cb2018beb656af6b8fb842733c886731d90`.
Only the D tracking floor differs from full tracking:
`max(D_own_scale, 0.75 * prior_tester_scale)`.

| Task | Gate | Terminal precision | Live centre RMS | Holdout |
|---|---:|---:|---:|---|
| grid100 | PASS, 23/34 | 0.98320 | 0.18265 sigma | PASS |
| rotated100 | PASS, 8/34 | 0.97225 | 0.1592 sigma | PASS |
| staggered100 | FAIL, 18/34 | 0.97445 | 0.19904 sigma | PASS |

Staggered's last five live centre checks are 0.1771, 0.1771, **0.2083**,
0.1950, and 0.1990 sigma. The 6500 checkpoint is the sole terminal live
accuracy miss; the final point and independent holdout pass. Native requires all
five terminal checks, so this remains **2/3, unresolved**.

Astra's review of the completed run found that alpha=.75 did not improve the
underlying staggered centre margin: the holdout centre is 0.18733 sigma versus
0.17955 with full tracking, and median prior motion is 0.236 versus 0.216.
Both runs make the same prior-scale decisions and finish at scale .5 with block
length 64. Astra recommends ending the multiplier search. The aggregate traces
do not identify whether the centre residual comes from systematic per-mode
bias, coherent particle oscillation, or a small set of moving rows, so another
controller change would be speculative. A row- or mode-resolved diagnostic is
needed before proposing a different mechanism.

Saved prior-motion diagnostics add a candidate-specific signal: at step 7000,
alpha=.75 staggered has row-motion p50/p90/p99 of 0.236/0.581/1.354 sigma,
with the top 1% of rows accounting for 96.0% of motion energy. Full-tracking
staggered is 0.216/0.543/1.133 sigma, with 38.9% of energy in the top 1%.
Alpha=.75 staggered's mode-mean and within-mode motion are 0.260 and 1.747
sigma; its generator RMS motion is 0.0083 sigma. This flags a sparse-motion
concentration that tracks with the weak alpha=.75 result, but does not attribute
that motion to birth/death transport rather than prior gradients. The saved
diagnostics report invalid row lineage across the interval. Astra therefore
still recommends no training intervention. The next defensible step is an
instrumentation-only replay that retains row identities, measuring per-mode
output-mean contributions from BD departures/arrivals, rows that stay in each
mode, and generator movement during the failing interval.

## Remaining custom-host checks

The eight custom22 hosts (`two_pole`, `trajectory`, `residual_student`,
`unipolar`, `ae_gan_hold`, `cover_leftover`, `unused_token_hold`, and
`mid_scale_identity`) were invoked on this exact package. All eight stop before
training with the same harness parity refusal: `critic input noise > 0 is not
bound for custom hosts`. This is recorded as **8 ERROR**, not as model failures
or passes. Raw receipts are under
`/ml2/hypergan/gan-attempts/formulations-20260928T040504Z/custom_follow/20260928T040504Z-4138988/runs/cf1-bdpair-dtracks-075-prior-*/result.json`.

## How the D-tracking rule works

Each optimizer group keeps its own stationarity tester. At rate application,
the trainer reads the prior tester's scale and applies the critic scale
`max(s_D, 0.75 * s_prior)`, then multiplies by the existing payoff damping. The
critic rate therefore cannot fall below 75% of the prior's current stationarity
scale when its own scale is smaller. As the prior settles and lowers its scale,
that floor lowers too. The critic tester continues measuring the applied rate.

This is a rate coupling, not a direct change to the prior rate, sigma policy, or
birth/death rule. Its feedback can still change later decisions in both groups;
0.75 is an empirical interpolation, not a derived optimum. It passes grid100
and rotated100 but does not satisfy the staggered100 terminal accuracy rule.

## Row-motion and optimizer attribution replay

A temporary diagnostic-only trainer copy was run on staggered100 with the exact
saved overrides. It retained output snapshots at 6250, 6500, 6750, and 7000,
plus a shadow table that received every realized optimizer update but skipped
birth/death moves. The replay's `rates.jsonl` and `native100-diagnostics.jsonl`
are byte-identical to the archived alpha=.75 run; `metrics.jsonl` differs only
in elapsed-seconds fields. It therefore preserves the frozen training path.

For the failing 6250→6500 interval, clean center RMS rises from 0.16247 to
0.18661 sigma. Holding initial latents fixed while updating G changes center
RMS only from 0.15867 to 0.15941. The no-BD shadow ends at 0.18328, and the
actual table also ends at 0.18328: the direct BD contribution is zero at this
checkpoint. The noisy scorer's fixed 20k draw gives 0.20830, crossing the .20
limit. Over later intervals BD has a sparse pointwise effect (row-output RMS
1.528 and 1.198 sigma), but center RMS matches the no-BD shadow to five digits
(.18372 in 6500→6750; .19325 in 6750→7000).

The table's prior group uses beta1=0, beta2=.999 and AMSGrad, with sparse A2 row
damping. Across the three 250-step windows, the median per-row cosine between
cumulative raw gradient and cumulative optimizer displacement is about -.985
(negative is the descent direction); the full-table cosine is -.938 to -1.000.
This points to adversarial-gradient-driven center drift, not first-moment
momentum reversal. It does not establish whether the gradient drift is harmful
noise near equilibrium or a systematic objective mismatch.

## Final bounded prior-rate test

The diagnostic motivated one compromise test: change only
`prior_lr_mult` from 2.0 to 1.5, keeping alpha=.75 D tracking and all other
settings fixed. Grid100 fails (14/34; no terminal live accuracy checks pass).
Final live center RMS is 0.28919 sigma and independent holdout center RMS is
0.27910 sigma. Per the frozen run order, rotated100 and staggered100 were not
run after the grid gate failed. A smaller global prior rate is not the fix.

The best measured candidate remains full D tracking at 2/3 native100. The
alpha=.75 and prior-rate=1.5 alternatives are archived as rejected; custom22
remains 8 parity ERRORs before training. The evidence does not support another
training change without distinguishing noisy near-equilibrium gradients from a
systematic mismatch in the adversarial objective.
