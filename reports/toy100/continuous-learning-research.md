# Removing learning-rate decay: mechanism and validation

This study targets PR #60's remaining learning-rate schedules. A replacement
must learn with time-independent optimizer hyperparameters and preserve live
sample quality during uninterrupted continuation on a fixed target distribution.
Passing a final checkpoint is insufficient.
Stationary weights and arbitrarily small steps are acceptable when there is
no useful acquisition signal. The requirement is to acquire an initially
unlearned target and maintain its quality. A positive amount of parameter
movement at an already matched distribution is not required.
The existing shared 22-case recipe remains the control until a replacement
passes its unchanged production gates. No training-seed sweep is used.

**Scope clarified September 23, 2026:** adaptation to a changed target
distribution is a separate problem. Shift experiments and their original
bounds remain preserved as diagnostics; they no longer gate PR #60 promotion.
The existing acquisition and stationary-quality failures still apply.

## Isolation before research or candidate search

The frozen `mode_hold` production host is already a small, faithful filter:
1,200 updates take approximately 7–9 seconds on one CPU thread. It retains
the actual learned prior, three-layer generator, Fourier critic, relativistic
logistic objective, noise mechanism, and strict eight-mode/HQ≥.9 metric.
Its 24 checkpoints include a required five-check terminal suffix.

Using the simpler PR winner, the scheduled control passes with terminal HQ
approximately [.9998, 1, 1, 1, 1]. Removing only LR decay fails with terminal
HQ [.2029, .6267, .4441, .2437, .3694], and final coverage 7/8.

The diagnostic observer measures paired clean outputs before and after the
same update, without additional training RNG draws. Over updates 1,000–1,200,
ordinary constant-rate Adam at .00425 moves samples .4062 units per generator
update versus .0372 from the learned particles. The HQ radius is .21. The
scheduled control moves .01971 and .00204 respectively. These are measured
functional displacements, not estimates from parameter norms.

At the same passing scheduled state at update 1,000, restoring constant base
rates destroys quality within 50 updates. Freezing only the generator keeps
all eight modes and HQ≥.956 over the next four checks while particles keep
learning; freezing only the prior fails badly. Freezing is a diagnostic
intervention, never a candidate continuous-learning solution.

The failed run's cap is active on about 23% of sampled critic input gradients;
the scheduled control's fraction is about .13%. Therefore a claim that the
penalty is simply always inactive in the failing run is unsupported. The
stronger measured issue is generator-output movement that exceeds the quality
radius. This does not establish a mathematical limit cycle.

## Research checked after the failure was isolated

* [Tang et al., C-CHAIN, May 2025](https://arxiv.org/html/2506.00592v1)
  studies output changes on data outside the training batch, relates these
  changes to neural-tangent-kernel rank, and regularizes them in continual RL.
  It motivates measuring and controlling functional movement here. Applying a
  fixed output-movement bound to a GAN optimizer is our experimental adaptation,
  not a reproduction or a GAN convergence guarantee from that paper.
* [Giagtzoglou et al., January 2026](https://arxiv.org/html/2601.13920v1)
  studies discriminator-only regularization through variational inequalities,
  including zero-centered real/fake input-gradient penalties. Its smoothness,
  curvature, and identifiability assumptions are not verified for these
  LeakyReLU networks and Adam updates. The repository already implements the
  relevant R1+R2 loss arm was tested directly with constant rates. Its apparent
  short-run survivor failed both transfer and continuation. Subsequent work
  prioritizes training dynamics that preserve the original game's stationary
  points, following the user's preference; no further zero-pull grid is planned.
* [Tang, Chen and Liu, September 16, 2026 preprint](https://arxiv.org/html/2609.18314v1)
  examines Adam instability through moment timescales and loss geometry at
  the scale of an update. It supports inspecting actual steps and denominator
  scales rather than inferring stability from nominal LR. It does not establish
  that this GAN exhibits the paper's spike mechanism, or prove that increasing
  epsilon will solve it.
* [NVIDIA's Adam epsilon documentation](https://docs.nvidia.com/nemo/emerging-optimizers/latest/primer/epsilon.html)
  explains how a fixed epsilon comparable to the second-moment denominator
  reduces the normalized update. We test fixed network and generator-only
  epsilon values. There is no elapsed-time schedule, but the preconditioner
  remains state dependent; constant nominal rates do not mean constant effective
  steps.
* [Chavdarova et al., Lookahead-Minmax, Algorithm 1](https://arxiv.org/html/2006.14567v3)
  motivates jointly interpolating both players after a fixed number of game
  rounds. Our test includes the learned prior and preserves Adam moments. All
  live intermediate iterates are scored, preventing synchronization boundaries
  from concealing excursions. The paper does not guarantee convergence for our
  nonlinear stochastic host.
* [Yoon and Loizou, August 2026](https://arxiv.org/abs/2608.06182)
  analyzes same-sample and independent-sample stochastic extragradient under
  explicit operator assumptions. It motivates testing both sampling choices;
  same-sample replay is not assumed to be universally better. Our experimental
  Adam-metric variant is not covered by a convergence claim from this paper.
* [Feng, Ou and Wang, May 2026](https://arxiv.org/html/2605.19392v1)
  derives continuous-time models for Adam in zero-sum games and studies the
  distinct roles of its two moment timescales. Its local analysis primarily
  concerns deterministic simultaneous dynamics. We do not transfer those
  guarantees to alternating stochastic relativistic GAN training.

## Matched passing-state filter

One scheduled prefix reaches update 1,000. Linux forks preserve its complete
live state: all three models, both Adam optimizers, EMA, and every training
random stream. Each candidate initially receives the same next 200 updates,
with quality checked after every update. An unchanged child must exactly match
a separate uninterrupted control's final weights, moments, EMA, and RNG hash.
This is a passing-state local stability test, not a claim of exact equilibrium:
the host's 12 equally weighted particles cannot exactly represent eight equally
weighted modes, and its output-noise scale differs from the target width.

The unchanged scheduled continuation passes 200/200 checks. Restoring constant
LR .00425 passes only 6/200. Joint Lookahead with alpha .5 and k=2, 5, 10 passes
1, 28, and 8 checks respectively; none advances. All six corresponding cold
Lookahead configurations also fail the original host gate.

Constant LR .001 passes all 200 local checks, with or without Lookahead. But
uninterrupted continuation to update 2,400 fails at update 1,950 (HQ .7903),
despite ending at eight modes/HQ .9990. It passes 119/120 extended checks;
the scheduled identity passes all 120. Thus local survival is only a cheap
rejection filter, not proof of continuous learning.

Fixed-metric joint extragradient at .00425 fails 0/200 warm checks with either
same-sample or independent-sample evaluations. The same-sample branch replays
all 200 RNG sequences exactly and advances Adam moments only once per outer
update. Its correction amplifies a second gradient through the first gradient's
small denominator; at update 1,055 a particle ratio reaches about 3.3e11, followed
by second-moment overflow. This rejects this specific unguarded metric variant,
not extragradient methods generally. A follow-up uses reversible trials
and a local secant acceptance condition to control that measured failure.

With `P=diag(lr/Adam_denominator)`, the follow-up accepts a trial only when
`norm(F(trial)-F(base), P) <= c * norm(F(base), P)`. Failed trials halve the
step factor; the next update tries twice the last accepted factor, capped at
one. The metric is updated once per outer step and all trials replay the same
training random draws. This is state-dependent step adaptation, explicitly
recorded in addition to the unchanged nominal rates.

Tolerances .25, .5 and .9 each pass 200/200 warm checks; their mean accepted
factors are .112, .267 and .315. All three then fail the cheaper cold trajectory
gate: final identity MSE .2872, .3540 and .1332 versus the required .02.
During the last 50 trajectory updates, their mean step factors are only
4.27e-6, .00407 and .0519. Small factors alone are not a failure: sitting still
can be appropriate at a matched target. These rows are rejected for their
measured cold acquisition failures, not for having small factors. A shift test
can separately assess adaptation after changing the target distribution.
No cold mode-hold or larger-host tests are run for these rejected candidates.
The next bounded mechanism tests an implicit linearized joint response. This
is related to game-aware coupled updates discussed by
[Schäfer and Anandkumar](https://f-t-s.github.io/projects/cgd/), but a full game
Jacobian solve includes the own-player blocks omitted by their CGD update;
our experimental variant must be identified separately.

The bounded cross-player-only follow-up omits the own-player Jacobian blocks
and checks a mixed-point cross response. It passes 196/200 warm checks, with
minimum HQ .8894, so it fails the local gate. An unplanned cold trajectory
follow-up, launched before consuming that verdict, reaches MSE .020058 but
has no passing checkpoint at the fixed .02 threshold. It is retained as a
diagnostic protocol deviation and receives no promotion credit. This improves
acquisition relative to the full-J implicit method without solving stability.
The first failed warm step is 1194; replaying that transition is a more
targeted next investigation than expanding a threshold grid.

An independent cheap controller scales only the generator network's ordinary
Adam proposal by `max(.02, min(1, advantage / threshold))`, where `advantage`
is twice the current training batch's mean `sigmoid(D(real)-D(fake))` minus one.
Moments and the particle update remain unchanged; there is no added loss or
elapsed-time input. Thresholds .25 and 1 each pass all 200 warm checks, but both
fail all five terminal checks during cold training. The unchanged observer
exactly reproduces the ordinary constant-rate control. Neither candidate
advances to more expensive hosts. This is further evidence that protecting an
already passing state and acquiring all modes are separate requirements.

## Validation order

First reproduce the scheduled pass and constant-rate failure on the exact
1,200-update mode-hold host. A numerical survivor then faces inexpensive,
historically discriminating hosts, ordered by the mechanism being tested:
trajectory, residual-student, stripes, bars, overlap, blobs, intensity,
unequal mass, unequal width. For R1+R2 interactions, trajectory goes first
because its known failure costs approximately one second. Historical rejection
rates for later stages are conditional on preceding passes, not unconditional
probabilities.

A promising stationary result also receives uninterrupted continuation to
2,400 updates with dense live checks. Noise burn-in remains tied to the original
1,200-update horizon. Neither models, moments, nor RNG state reset. Survivors
still require fresh full 19, the three strict native
100-mode problems, a production common-22 replay, and longer continuation.
Distribution-shift experiments with matched frozen controls are now optional
and separate from this fixed-target promotion sequence.

All scratch-policy outputs are explicitly marked ineligible for the production
common gate. Actual optimizer-group rates, epsilon values, source/configuration
hashes, full-budget episodes, and independent saved-evidence regrades accompany
the screens. Missing evidence is invalid; an unattempted later stage is skipped.

## Why dense continuation matters

Constant Adam LR .001 with R1+R2 coefficient .1 passes the original mode-hold
gate with terminal HQ [1, 1, .9214, .9678, .9990]. It fails the next cheap
trajectory test (identity MSE .2610), disqualifying it as a shared recipe.
It also fails 92/120 mode-hold checks during updates 1,210–2,400, including zero
modes/HQ=0 at update 1,400. Its final checkpoint recovers to eight modes/HQ=1.
The continued test catches a failure that a final-only test would hide.

The exact continued run reproduces its initial 1,200-update trace. G and D
rates remain .001 and the particle rate remains .002 throughout; the output
noise and its initial warmup remain fixed to the original horizon.
