# Removing learning-rate decay: mechanism and validation

This study targets PR #60's remaining learning-rate schedules. A replacement
must learn with time-independent optimizer hyperparameters, preserve live
sample quality during uninterrupted continuation, and remain responsive when
the real distribution changes. Passing a final checkpoint is insufficient.
The existing shared 22-case recipe remains the control until a replacement
passes its unchanged production gates. No training-seed sweep is used.

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
  relevant R1+R2 loss arm; we test it directly with constant rates.
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
1,200-update horizon. Neither models, moments, nor RNG state reset. Distribution
shift is tested only after sustained hold, with a matched frozen negative
control. Survivors still require fresh full 19, the three strict native
100-mode problems, a production common-22 replay, and longer continuation.

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
