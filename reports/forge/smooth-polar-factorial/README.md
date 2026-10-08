# Fixed smoothing across the four runtime combinations

**One fixed smoothing scale repairs serial Gaussian acquisition, but does not
repair both regressions. Words fail the original terminal stability gate in
all four combinations; Ring16 passes in all four. Keep smoothing opt-in.**

This user-requested diagnostic is based on develop
`83b099d1e4330dda953d5fce4f68ce00f75fa9a6`. All twelve CUDA runs executed the
same frozen commit `34db5abbf164229ee449d2862bba53e043da7f82`, source digest
`a30e99d3be7da3c5550270138daca165a96911f3086c9eb1a5f8e22eeb2a354f`.
The [preregistered protocol](protocol.json) retains predictions, falsifiers,
budgets and exact identities of the unsmoothed controls in
[Gaussian PR #342](https://github.com/255BITS/ParticleGAN/pull/342) and
[word PR #343](https://github.com/255BITS/ParticleGAN/pull/343).
The [compact readout](readout.json) contains final metrics, failed-bound counts,
input checks and source receipts. Bulk traces and tensors are archived locally.

## Results

GIF links use saved, actually scored outputs from training. Numerical gates
supply every verdict. Serial means autograd multithreading disabled; threaded
means enabled, with the actual graph and optimizer modes audited. These are
process-local research overrides; the project policy remains disabled.

| Truncation / threading | Gaussian smoke: confirmed updates | Gaussian KS at update 1000 | Words: verdict, passing checks / 24, terminal streak | Ring16: verdict, terminal streak / 96 |
|---|---|---:|---|---|
| Truncated / serial — current defaults | [PASS: 459, 625, 750](media/gaussian1d_smoke--truncated-serial.gif) | 0.137062 | [FAIL: 12/24, streak 0](media/five_word_joint_acquisition--truncated-serial.gif) | [PASS: 35/96](media/ring16_acquisition--truncated-serial.gif) |
| Full / serial | [PASS: 459](media/gaussian1d_smoke--full-serial.gif) | 0.225477 | [FAIL: 5/24, streak 0](media/five_word_joint_acquisition--full-serial.gif) | [PASS: 47/96](media/ring16_acquisition--full-serial.gif) |
| Truncated / threaded | [PASS: 250](media/gaussian1d_smoke--truncated-threaded.gif) | 0.148651 | [FAIL: 11/24, streak 3](media/five_word_joint_acquisition--truncated-threaded.gif) | [PASS: 45/96](media/ring16_acquisition--truncated-threaded.gif) |
| Full / threaded | [FAIL: no confirmed update](media/gaussian1d_smoke--full-threaded.gif) | 0.145326 | [FAIL: 14/24, streak 0](media/five_word_joint_acquisition--full-threaded.gif) | [PASS: 45/96](media/ring16_acquisition--full-threaded.gif) |

The Gaussian gate requires KS <= .05, mean error <= .2 target standard
deviations and standard deviation ratio in [.8, 1.2], on both a primary and an
independent 4,096-sample confirmation at the same state. Under current defaults,
update459 scores primary KS .029583 / confirmation .035877; update625 scores
.045474 / .041520; update750 scores .040774 / .036120. **All four Gaussian
endpoints fail**, including the three acquisition passes. This is no evidence
of settling, continued hold or response to a changed target.

Compared with the retained unsmoothed controls, both serial Gaussian failures
become acquisition passes. Truncated/threaded remains an acquisition pass;
full/threaded changes from PASS to FAIL. Smoothing does not remove dependence
on the runtime combination.

Words retain the original requirement of five terminal passing checks. The
unsmoothed word terminal streaks, in table order, were 2, 0, 3, 6; smoothed
streaks are 0, 0, 3, 0. The previous full/threaded word PASS becomes FAIL.
At the current-default endpoint, generation covers all five words with mass
TV .018945, but reconstruction is wrong and minimum token probability is only
.000675, below .90. Full/serial ends with three words and TV .404688.
Truncated/threaded ends at a full passing state but only holds the last three
checks. Full/threaded reconstructs correctly but generates only four words,
with TV .217188. Coverage and paired inversion are separate failure modes.

**All four word runs acquire the full joint generation-and-inverse goal at
least once.** The task's name includes acquisition, but its current gate also
includes stability. This comparison preserves that gate; converting a word
acquisition question into a smoke test would be a separate task change and
would not demonstrate a continuous learner that stays at the solution.

## What changed and what stayed fixed

The [paper's smoothed matrix-polar rule](https://arxiv.org/html/2608.01911v1)
uses singular weights `s / sqrt(s*s + epsilon)`. We freeze one absolute
`lambda=1e-4`, so the paper's `epsilon=lambda**2=1e-8`, for every task, parameter
and update. Matrices use `s / hypot(s, lambda)`. Truncated arms additionally
retain the existing numerical-rank mask; full arms smooth every singular value.
Biases and actually sampled prior rows use `g / hypot(norm(g), lambda)`.
Those vector extensions are our adaptation. The paper studies continuous-time
minimization, which does not provide a convergence guarantee for this discrete
stochastic adversarial game.

The global recipe retains BCAP/DualNorm, rates G/E .012, D .018, prior .03,
zero momentum and prior regularization, and constant effective rates. No
annealing, seed change, scale sweep, restart or continuation is used.

| Task | Fixed host, prior and batch | Budget and checks |
|---|---|---|
| Gaussian | Target N(2, .5²); z2, width32/depth2, critic Fourier2; 256 learned MoG locations, sigma_rel .1; batch128 | 1,000 updates; 24 paired checks |
| Words | Five finite words with paired inverse; z2; G 2→64→128→168, E 168→128→64→2, joint D 170→256→128→1; five learned particle-cloud rows; batch256 | 20,001 updates; 24 checks, five terminal passes |
| Ring16 | Sixteen equally weighted 2D clusters, radius3, sigma .1; z4, width64/depth2, critic Fourier2; 256 learned MoG locations, sigma_rel .1; batch128 | 1,600 updates; 96 checks, five terminal passes |

Each task uses protocol seed0 and the public named deterministic initializer.
Constructor, data, training noise and evaluation streams are isolated and
checkpointed. Initial model/prior hashes, actual data sequences, recipes with
smoothing removed and consumed stream states match across new arms and their
Gaussian/word controls. The word fixture's complete sampled training-index
sequence and data-generator state also match. No new seed experiment is included.

Archived controls retain their own source and instrumentation. New runs add
sparse post-update SVD-value measurements at steps1/2 and 24 checkpoints.
Their input and recipe matches support this finite comparison; the comparison
is not an identical-instrumentation runtime replay. It does not isolate a
PyTorch accumulation mechanism or establish robustness to all numerical changes.
Ring16 has four new protection runs; this study does not claim four matched
historical Ring16 controls.

## Interpretation and recommendation

The current-default word trace gives a useful lead. At update16668, the three
G matrices have leading singular values 2.15e-6, 8.15e-6 and 1.06e-5. Smoothing
turns their leading-direction weights into .0215, .0812 and .1053, whereas the
leading E and D directions remain almost1. The same absolute scale therefore
changes the relative pace of the players. This is a retrospective observation
from saved gradients, not a causal role-isolation result. Bias and prior
extensions have not been independently isolated.

Do not adopt this exact all-role scale as the shared default. It falsifies the
preregistered Gaussian-and-word repair prediction and does not deliver continuous
Gaussian retention. Retain the optional API and the failed evidence. Before
another paid study, inspect the saved G/E/prior movement around word collapses;
a bounded matrix-only versus all-role smoothing comparison could then isolate
the added vector rules. No further training is launched by this recommendation.

This is a standalone task diagnostic: qualification_input/reuse are false,
and no default, task threshold, ordinary qualification or Tier2 credit changes.
The single canonical technique inventory and its original evidence remain intact.

## Validation, cost and reproduction

All twelve full-budget runs finish once, with finite state and no unintended
RNG deviations. The exporter verifies exact committed execution source, retained
raw artifact hashes, original gates and input matching, and recomputes every
saved Gaussian primary/confirmation and word metric. Twelve goal GIFs add zero
model updates or model sampling draws.

There are 191 distinct successful software checks: 103 existing optimizer
checks, 13 new CUDA analytic/checkpoint checks, one separate public-CUDA default
parity fixture and 74 recipe/Forge compatibility checks. Zero smoothing matches
three pinned-develop public updates and optimizer/stream checkpoint packets
bitexact in that explicitly scoped software fixture. New numerical tests require
CUDA and skip on CPU-only CI. [Software receipts](software-validation.json)
retain the initial constructor-only reference-fixture correction; it consumed
no training updates or scientific retry. Later edits document the operator and
adjust the CUDA-unavailable test guard; measured numerical code is unchanged.

The study spends 90,404 logical research updates and 3,064.254 measured arm
seconds against 5,280 reserved seconds. Gaussian/Ring16 run sequentially on
physical A6000 GPU0; words run sequentially on GPU1. Actual costs include
instrumentation and workstation activity and supply no speed ranking.
The [archive receipt](archive.json) pins every raw member, tensor, log, JUnit
file, executed source and selected task declaration; all archive bytes are
reopened and verified. Bulk evidence stays out of Git. The earlier export's
identity is retained when exact task/software sources are supplemented.

For historical training reproduction, use an empty checkout of the executed
commit above. Run each controller in its own terminal using the project Python
environment; each controller pins one physical GPU and refuses to overwrite
existing runs. These are the commands used for the closed study:

```sh
mkdir -p runs/forge/smooth-polar-factorial-v1
/home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/smooth-polar-factorial/controller.py --slot 0 \
  > runs/forge/smooth-polar-factorial-v1/controller-0.log 2>&1
/home/martyn/dev/ParticleGAN/.venv/bin/python -u \
  reports/forge/smooth-polar-factorial/controller.py --slot 1 \
  > runs/forge/smooth-polar-factorial-v1/controller-1.log 2>&1
tail -F runs/forge/smooth-polar-factorial-v1/five_word_joint_acquisition--full-threaded.log
```

For saved-evidence verification/export only, restore the pinned control archives
and fetch `origin/report/bcap-gaussian-regression` and
`origin/report/bcap-words-regression` if their commit objects are absent:

```sh
CUDA_VISIBLE_DEVICES=0 /home/martyn/dev/ParticleGAN/.venv/bin/python \
  reports/forge/smooth-polar-factorial/publish.py --media \
  --gaussian-controls /home/martyn/dev/ParticleGAN-bcap-gaussian-regression/runs/bcap-gaussian-regression \
  --word-controls /home/martyn/dev/ParticleGAN-bcap-words-regression/runs/forge/bcap-word-regression
CUDA_VISIBLE_DEVICES=0 /home/martyn/dev/ParticleGAN/.venv/bin/python -m pytest -q \
  reports/forge/smooth-polar-factorial/test_default_parity.py
```
