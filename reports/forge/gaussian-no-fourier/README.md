# Scalar Gaussian: remove critic Fourier features

**Removing Fourier features fails the new Tier 1 smoke question:** 0/24 primary
full passes and no independently confirmed passing state within 1,000 updates.
Continued learning also fails: 3/72 stationary hold checks and 1/24 shifted hold
checks pass, with failed deadline reacquisition. Keep the original Fourier-2
critic; this ablation does not repair acquisition or stability.

[PR #323](https://github.com/255BITS/ParticleGAN/pull/323) targets develop and
contains this negative result. The shared smoke/Tier-2 split is
[PR #322](https://github.com/255BITS/ParticleGAN/pull/322).

This is the independently declared Fourier-only architecture ablation. It asks
whether a raw-input critic makes Gaussian acquisition and continued learning
easier under the unchanged winning BCAP recipe. The depth ablation retains its
original Fourier features and has its own PR and evidence.

The new Tier 1 question is whether training produces at least one full passing
Gaussian state within 1,000 updates, confirmed by an independent sample draw at
the same state. All 24 paired observations and all 1,000 updates still execute.
Tier 2 tests every remaining stationary check through 4,000, then shifts the
target mean from 2 to 3, requires five terminal passing checks by 5,000, and
requires every subsequent check through 6,000 to pass. Historical five-terminal
acquisition verdicts remain a separately reported comparison.

## Fixed comparison

[Protocol](protocol.json) and explicit [smoke](smoke.json) /
[stability](stability.json) cards bind the architecture change: critic Fourier
count 2 becomes 0. Its first layer receives raw scalar input instead of raw
input plus sine/cosine features at frequencies pi and 2pi. Generator and critic
remain width 32 with two LeakyReLU hidden layers; latent dimension remains 2.
Generator parameter count remains 1,185; critic count drops from 1,281 to 1,153.

The public named deterministic initializer preserves every generator and prior
tensor. The critic first weight, its fan-in dependent initial bias and the
Fourier frequency buffer change as expected. Every deeper critic tensor matches
the archived initial state. Only the critic constructor stream changes; training
streams and initial optimizer state/rates match exactly. This is recorded in
[the zero-update CUDA proof](initial-proof.json), with [reproduction](preflight.py).
The unchanged generator's archived capacity control remains applicable as a
representation diagnostic; it is not a trained pass.

The learned prior remains 256 uniform MoG locations with sigma .1, without
standardization; target N(2,.5²), batch 128 and protocol seed 0 are fixed. Trainer
delta is zero: alternating BCAP dualnorm, constant G .012, D .012×1.5 and prior
.012×2.5, zero momentum, no prior regularizer, EMA, annealing or additive output
noise. Clean live public sampling retains MoG kernel noise.

Every distribution check uses 4,096 samples and the original full bounds: mean
error ≤ .2 target sigma, standard deviation ratio [.8,1.2], KS ≤ .05 and finite
fraction 1. The independent confirmation stream runs at every smoke check, so
the sampling schedule does not depend on outcomes. All 72 stationary hold and
48 shifted checks execute; the last 24 shifted checks form the hold phase. A
no-update copy of the own 4,000 checkpoint uses matched shifted evaluation draws.

The single finite trial reserves 720 seconds: 120 for smoke and 600 for the
5,000-update own-state continuation. It spends at most 6,000 new updates and
768,000 real examples, with zero scientific retries or post-result tuning.
All neural initialization, training, sampling and checkpoint restores use CUDA.
Inherited CPU target generation, saved-output scoring and media rendering are
explicit exceptions. The public shared `GANTrainer` host performs every update;
the architecture adapter validates its frozen inputs and contains no training
loop.

[Archived controls](controls.json) retain the original Fourier-2 alternating
4000/6000 evidence and verdicts under their actual source identities, with zero
new cost. Original sigma-.025 Gaussian acquisition remains another task. This
sigma-.1 architecture diagnostic does not confer ordinary qualification or
retroactively change previous failures. The architecture change is Gaussian-only;
the passing ring recipe and its source-bound evidence require no new ring run.

## Completed readout

The single CUDA run completed exactly 6,000 updates and 768,000 real examples
from scientific commit `7c3409a194fc7d7f130306e8f03d571d6711c757`, with no retries.
Measured adapter loops total **49.887372 seconds**: 8.809735 for smoke and
41.077637 for continuation. This excludes publication and is cost accounting,
not a speed ranking. The final 6,000-update endpoint passes all distribution
bounds, but it cannot replace timely smoke acquisition or the failed hold gates.

| Phase / endpoint | Full checks passed | Mean error / sigma | Std ratio | KS | Declared result |
| --- | ---: | ---: | ---: | ---: | --- |
| Smoke / 1,000 | 0/24; no confirmed hits | .72182 | 1.36738 | .25776 | FAIL |
| Stationary hold / 4,000 | 3/72 | .36097 | 1.14665 | .14352 | FAIL |
| Shift reacquisition / 5,000 | 2/24; terminal suffix 0 | .22697 | 1.28000 | .09503 | FAIL |
| Shift hold / 6,000 | 1/24 | .08558 | 1.14201 | .04585 | FAIL; endpoint PASS |
| Frozen no-update shift control | 0/48 | — | — | — | No adaptation |

These are scoped diagnostic phase counts, not a second technique leaderboard.
The original five-terminal acquisition question also remains FAIL, with suffix 0.
[Compact results](results.json), [failure analysis](analysis.json), and
[provenance](provenance.json) retain the numerical gates and source bindings.
[Metric curves](metrics.svg) show every primary observation. Actual training is
illustrated in the nine-frame [smoke GIF](smoke.gif) and [stability GIF](stability.gif);
the GIFs do not determine the verdicts.

KS fails every smoke check; mean error fails 16/24 and excessive width fails
5/24. During stationary hold, KS fails 69/72, mean error 49/72 and width fails
5/72. These counts overlap; KS also responds to location and scale error, so
they do not establish a separate shape cause. The longest stationary full-pass
streak is two checks. Shift reacquisition and later hold have longest streak
one, despite the good final snapshot.

The archived Fourier-2 control has full primary passes at 917 and 1,000, compared
with zero for this arm. Its old acquisition/hold verdicts retain their original
scope and do not gain a new confirmed-smoke verdict from this comparison. More
time yields occasional valid states here, but does not make the smaller critic
reliably converge. Changing feature count also changes first-layer geometry and
the public fan-in dependent initialization; this single ablation does not isolate
which part causes the regression.

**Recommendation:** retain Fourier 2, stop this exact Fourier-0 revision, and
assess the independent depth ablation. Separating acquisition smoke from
continuous stability remains useful: it exposes a late passing state without
treating it as a completed basic test. No ordinary task/default is changed by
this negative architecture report. There is no extra seed, scale search or
training continuation in this PR.

Both actual data segments and the shifted sequence exactly match the predeclared
reference hashes. The actual saved step-0 generator, prior, optimizer state and
training streams match the archived baseline. Primary evaluation registers lazily
in the new shared host; its named seed matches the actual final manifest, and
the audit compares the unconsumed binding explicitly without model draws.

[Saved-output verification](verification.json) recomputes 219 sample sets:
146 primary (including two initial illustrations), 25 confirmation (including
the preserved initial illustration) and 48 frozen draws; both full grades match.
[Six exact CUDA restores](restore-proof.json) cover step 0, both 1,000 states,
the 4,000 pre-shift state, final 6,000 state and frozen 4,000 state. Finite-state,
actual three-role optimizer counts, mechanism activation and RNG isolation
guards pass. Saved checkpoint learning rates stay constant. Execution and
verification used the frozen scientific source above. The later develop merge
keeps the archived protocol/source bindings; reproduction requires that commit.

Raw evidence is in the ignored local archive
`artifacts/gaussian-no-fourier-v1.tar.gz`, SHA-256
`8c5515862f5afd51fb09fcf8bee963b013014aaaa04a80846a26974f9c44e812`.
The [archive receipt](archive-receipt.json) records its size and raw location.
This PR depends on the shared smoke/Tier-2 split; it adds an explicit negative
architecture variant and evidence, not an adopted default.

## Reproduction and logs

Use the frozen scientific commit recorded in the completed provenance. Choose a
fresh ignored raw path for an explicitly authorized reproduction:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 /usr/bin/python -u \
  -m benchmarks.toy_audit.gaussian_no_fourier \
  --device cuda:0 --output runs/api/gaussian-no-fourier-v1 \
  > runs/api/gaussian-no-fourier-v1.log 2>&1
tail -F runs/api/gaussian-no-fourier-v1.log
```

Raw stdout, observation arrays, RNG states, optimizer diagnostics and checkpoints
remain ignored locally or in the immutable archive. Publish saved outputs and
verify exact CUDA restores without training:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python \
  -m benchmarks.toy_audit.gaussian_architecture_publish \
  --raw runs/api/gaussian-no-fourier-v1 \
  --output reports/forge/gaussian-no-fourier \
  --protocol reports/forge/gaussian-no-fourier/protocol.json
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 /usr/bin/python \
  -m benchmarks.toy_audit.gaussian_architecture_publish \
  --raw runs/api/gaussian-no-fourier-v1 \
  --output reports/forge/gaussian-no-fourier --verify-state --device cuda:0
```
