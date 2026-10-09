# Information geometry: relative spectral-half BCAP diagnostic

**The relative spectral-half candidate fails as a global BCAP repair.** Both
candidate and matched winner finish **2 PASS / 4 FAIL**, with no incomplete runs
or blockers. Unequal-mass covariance and allocation improve, but local contraction
still fails; both image tasks regress and Gaussian retention does not improve.
The candidate is [administratively abandoned](../../records/lifecycle-f3bc2d33f89c35bf3364cd55.json).
Retain the incumbent. The comparison costs
**502.118434 paid worker seconds**, against 12,840 full reserved seconds and a
14,400-second ceiling. This diagnostic supplies no ordinary qualification or
default-adoption claim.

[Final metrics and complete gates](results.json), [certified provenance and
matched-condition proofs](provenance.json), and [actual-training GIF receipts](media/index.json)
retain the complete evidence. The parent owns the one current cross-track
comparison; the task table below is this fixed diagnostic's readout.

| Full unchanged task | Winning control | Spectral-half candidate | Exact measured difference |
| --- | --- | --- | --- |
| two_pole | PASS, suffix17 | PASS, suffix17 | Identical mean absolute coordinate .9585024714 and median gradient .9523457289 |
| gaussian1d_smoke | PASS, first confirmation375 | PASS, first confirmation459 | Endpoint KS .0718421618 → .1029597128; smoke accepts a confirmed scheduled state, not endpoint retention |
| gaussian1d_stability | FAIL | FAIL | Both stationary2/72, shifted hold0/24; final KS .3206228940 → .3526373411 |
| vector_unequal_mass | FAIL, full checks0/24 | FAIL, full checks0/24 | Covariance3.6916531473 → .7413333580; min mass .2075195313 → .7560220957; min eigen .0090729063 → .0969907194 (required≥.15) |
| img_blobs4 | FAIL, HQ .7500, modes2 | FAIL, HQ .5625, modes2 | Both full checks0/24; requiredHQ≥.9 and four genuine modes |
| img_intensity2 | FAIL, HQ .6875, modes2 | FAIL, HQ .4375, modes1 | Both full checks0/24; requiredHQ≥.9 and two genuine modes |

The [candidate study readout](../../records/readout-34d60f269493acf6d8d140cc.json)
and [control readout](../../records/readout-f74ba207f04d77da8e45dd82.json) are concluded.
The preregistered blobs-HQ prediction≥.9 is falsified. Better final unequal-mass
covariance does not replace the missing full gate or terminal suffix. Its rarest
component receives109/4,096 draws instead of17/4,096 in the control, while the
remaining minimum eigenvalue floor still fails. No task-specific winner is selected.

Two-pole is an explicitly fixed zero-particle/stored-critic fixture. Its direct
particle matrix is N×1 and critic matrices are32×1 and1×32: all have rank≤1.
Our rule exactly equals smoothed polar in that case, so its identical trajectory
checks public API compatibility but gives **no evidence of preserving learned
multidimensional passing behavior**. No additional experiment was added after
seeing this limitation. Gaussian smoke is learned but tests acquisition only.

Endpoint image decomposition uses the exact32 saved outputs and their nearest
templates. Background means target absolute pixel≤.001; other pixels are signal.
It adds no draws, training, or replacement gate:

| Pixel-region diagnostic | Control | Candidate |
| --- | ---: | ---: |
| Blobs background RMSE | .1027143512 | .0883120391 |
| Blobs signal RMSE | .0785586926 | .4542922028 |
| Blobs signal signed bias | −.0139758703 | −.2497308980 |
| Intensity background RMSE | .0008729852 | .0009848633 |
| Intensity signal RMSE | .1372261864 | .1643675214 |
| Intensity signal signed bias | −.0362130781 | −.0398471188 |

Lower blob background error accompanies a much larger signal error and negative
intensity bias. The results contradict a useful global repair from suppressing
weak singular directions; they are compatible with suppressing useful template
features. They do not prove that interpretation, since matrix direction and total
Frobenius motion change together and no true curvature is measured.

Next action: keep the winner, stop spectral-half, and inspect saved multidimensional
signal/gradient alignment before considering a metric that estimates actual
curvature. No exponent/rate search, extra steps, fresh seed, continuation,
qualification, or promotion follows from this round.

## Theory and measurable optimizer strength

Optimizer strength has no universal ordering from evolution to SGD to polar to
natural gradient. For a fixed smooth scalar objective with Hessian H and positive
preconditioner P, a relevant local property is the condition number
κ(P^(1/2) H P^(1/2)), which controls the optimally stepped quadratic contraction
factor (κ−1)/(κ+1). H must be positive definite for this argument; a GAN game
has a coupled, generally nonsymmetric field Jacobian, so this scalar guarantee
does not transfer. An evolutionary method can exploit discontinuous fitness
where gradient methods have no useful derivative, while an exact gradient can
have lower sampling cost on a smooth objective. Those are different strengths.

For density qθ, the statistical metric is
F = E_q[∇θ log qθ ∇θ log qθᵀ]. Natural gradient F^−1 ∇θ L defines an
infinitesimal direction invariant to smooth coordinate changes when the exact
metric is transformed consistently. SGD uses the Euclidean metric and lacks
that general invariance. Natural evolution strategies apply this construction
to a parameterized *search distribution*, rather than asserting that evolution
is inherently weaker. See [Amari (1998)](https://doi.org/10.1162/089976698300017746)
and [Wierstra et al. (2014)](https://jmlr.org/papers/v15/wierstra14a.html).

[K-FAC](https://proceedings.mlr.press/v37/martens15.html) approximates neural
Fisher blocks using activation/derivative factors; [Shampoo](https://proceedings.mlr.press/v80/gupta18a.html)
uses tensor gradient second moments and inverse-root preconditioning. These
methods provide concrete conditioning hypotheses, rather than a universal
hierarchy. Our candidate is a smaller instantaneous fractional Gram surrogate:
it neither estimates a Fisher matrix nor accumulates Shampoo history, and its
convergence is not established by either paper.

Write a matrix gradient G = U diag(s) Vᵀ, ε = .001, hᵢ = sqrt(sᵢ² + ε²).
The existing smoothed polar rule has weights sᵢ/hᵢ. The candidate has weights

    wᵢ = sᵢ / sqrt(hᵢ h_max).

Equivalently, apply (GGᵀ + ε²I)^−1/4 G / sqrt(h_max). Away from the
damping scale, weights become sqrt(sᵢ/s_max), rather than all one. The largest
singular step is exactly matched to smoothed polar; weaker singular steps
shrink by sqrt(hᵢ/h_max). This changes direction geometry and total matrix
motion together, while holding its spectral cap. It is not a pure direction
ablation. Numerical rank cutoff remains max(rows,cols)*machine_eps*s_max.

A measurable implementation property is orthogonal equivariance:
T(QGRᵀ) = QT(G)Rᵀ. Both polar and this rule satisfy it; neither has general
Fisher reparameterization invariance. A diagonal gradient [4,1,.04,0] must yield
[1,.5,.1,0], with identical top spectral norm to polar and smaller Frobenius
norm. These algebra checks distinguish the candidate's spectrum retention from
an unsupported claim of better curvature conditioning. Fixed damping also means
exact gradient-scale invariance holds only when ε is negligible.

Exact SVD costs match the incumbent's asymptotic cost; no full parameter Hessian
or new optimizer history is stored. Per-offset convolution matrices use the
existing explicit channel/group/transposed layout and aspect factors. This
scales to the current dense and convolution hosts, but supplies no large-model
throughput or speed claim, especially with shared GPUs.

## Evidence, hypothesis and competing explanation

The [original failure analysis](../../bcap-tier2-search/FAILURE_ANALYSIS.md) and
[saved-state diagnostics](../../bcap-tier2-search/failure-state-analysis.json)
retain original seed0, source and initializer identities. Native endpoints have
useful critic directions but excessive network movement. Unequal mass has just
one latent center assigned to the rarest target component; its minimum local
variance ratio .0091 and minimum mass ratio .2075 fail. Blobs has background
leakage and only two genuine modes; intensity has signal-region bias despite
both modes being present. These are distinct failures.

Hypothesis: near-rank-deficient minibatch gradients contain weak singular
directions that polar promotes disproportionately. Relative attenuation may
reduce incoherent background changes and density distortion, while preserving
strong transport. Prediction: blobs endpoint HQ ≥ .9 (also a falsifier if below
.9). Additional preregistered success criteria: unchanged full blob and intensity
five-terminal-check gates, unequal-mass covariance ≤ .85, minimum eigen ratio
≥ .15 and minimum mass ratio ≥ .25, and preservation of the two-pole guardrail.
No prediction replaces a task's complete gate.

Competing explanation: weak directions carry rare-component and low-intensity
signal. Shrinking them may lose precisely the behavior we seek. Sampled prior
motion, equal latent weights and coupled-game oscillation are unchanged; this
candidate cannot diagnose their separate contribution. A passing endpoint alone
cannot establish retention. A failure stops this revision without more steps,
seed changes, cap search or automatic continuation.

[DualNorm pacing](../../dualnorm-pacing-v2/README.md) already tested global rates,
D/prior ratios and momenta, with tradeoffs among Gaussian, ring and words.
[Gaussian diagnosis](../../gaussian1d-diagnosis/README.md) shows acquired states
can drift and finite normalized prior motion remains substantial. Newer
[network magnitude PR320](https://github.com/255BITS/ParticleGAN/pull/320),
[prior magnitude PR319](https://github.com/255BITS/ParticleGAN/pull/319), and
[combined magnitude PR321](https://github.com/255BITS/ParticleGAN/pull/321)
fail continuous Gaussian gates in their sigma-.1 relativistic cohorts, including
ring regressions. Their absolute cap .1 changes bias and top-matrix response;
our relative fractional rule preserves bias/prior laws and the strongest
smoothed-polar step. Those studies are motivation, never compatible control
credit. No past extrapolation or fresh extragradient is introduced here.

## Frozen comparison and stopping

Control is `bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`,
original revision `dfe88a2ee15fb9d83ffdc5c8a25d698686b73d4e7c63b6b0e35efb0d64e94359`.
The new matched control resolves that exact declaration in this executed source.
Its recipe is non-saturating BCAP coefficient/cap1/every-update, G/E .012,
D .018, sampled prior .030, momentum0, smoothing .001, constant rates,
per-offset convolution and no training noise or EMA. Only matrix weighting
changes through public `Recipe.optimizer_family='information_geometry_spectral_half'`.
Bias and sampled-row prior updates remain identical, including ownership masks.

| Unchanged task | Updates | Full reservation per recipe | Role |
| --- | ---: | ---: | --- |
| two_pole | 80 | 300 s | Passing fixed-fixture common guardrail |
| gaussian1d_smoke | 1,000 | 120 s | Common learned-MoG anchor, independent same-state confirmation |
| gaussian1d_stability | 6,000 total | 600 s | Own smoke checkpoint required; strict hold and target shift |
| vector_unequal_mass | 1,200 | 1,800 s | Rare component allocation and covariance |
| img_blobs4 | 600 | 1,800 s | Background fidelity and four-mode coverage |
| img_intensity2 | 600 | 1,800 s | Signal intensity fidelity |

Total reservation is 6,420 s per recipe, 12,840 s for the matched pair, within
14,400 s. No extra candidate, scientific retries, capacity probes or media
training is authorized by this study. Full original budgets, scoring cadences,
priors, sampling and numerical gates remain unchanged. The diagnostic view
places independent specialist tasks together without ordinary-tier qualification;
Gaussian checkpoint dependency remains mandatory. Seed0, public deterministic
initialization and isolated checkpointed constructor/data/training/evaluation
streams are bound by the existing public Forge adapters. The fixed two-pole
fixture is an explicit separate task cohort and cannot substitute for learned
initialization evidence.

The two ready studies share `information_geometry_campaign_v1`, with a 14,400 s
campaign ceiling and 7,200 s per-candidate ceiling. The control study names the
candidate as its comparison reference; this reverses comparison direction only,
and adds no third trainer. Enqueue does not launch either reference implicitly.

## Reproduction and validation

Use the frozen scientific commit recorded in the final provenance receipt.
The shared parent coordinator owns execution on GPU0/1 with three workers per
GPU. Do not start another drain. Both commands below only enqueue:

```sh
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue information_geometry_spectral_half_v1 --study information_geometry_candidate_v1
.venv/bin/python -m experiments.forge --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue enqueue bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36 --study information_geometry_control_v1
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue/events.jsonl
```

Software validation: **179 tests pass**, including spectral contrast,
orthogonal equivariance, unchanged bias/sampled-prior laws, public grouped
Conv2d/ConvTranspose2d construction, exact optimizer resume and family mismatch
rejection. The candidate plan reports READY, zero preflight blockers and exactly
one consumed optimizer-family delta on every task. Certified endpoint optimizer
packets contain the actual candidate/control families, smoothing .001 and
momentum0. All six matched task contracts agree. Public initializer hashes, initial
named RNG bindings and consumed data-stream endpoints match on the five learned
task cohorts; two-pole retains its explicit fixed source-defined fixture. Both Gaussian
continuations use their own actual smoke endpoints; their stationary/shift batch
digests also match. Vector/image receipts omit a batch-sequence digest, so their
proof uses identical frozen source/data draws, named-stream bindings and final
consumption hashes rather than claiming a nonexistent digest. Every consumed
stream is retained in its provenance checkpoint.

Training source commit is `2c182d552100a562cd9f7c9d8f3d226d855b26c8`, executed digest
`a31696abf0d63393a34795022aa730be575944d390bf79a0992017306a9a10f0`.
The shared runtime is CPython3.12.13, Torch2.14.0, NumPy2.5.2, SciPy1.17.1,
RTX A6000 with deterministic algorithms and TF32 disabled. Candidate paid cost
is258.014341164 s and control244.104093079 s, with zero retries, zero residual
reservation and16,960 new updates including both5,000-update Gaussian
continuations. These shared-GPU supervised costs include startup and independent
grading and are accounting, not throughput evidence.

Twelve GIFs show saved training outputs/measurements, with nine frames each.
All96 image metric observations are independently recomputed from saved arrays.
Publication adds zero training or sampling. Bulk stdout, JSONL, checkpoints and
state tensors stay outside Git in the durable shared queue campaign, while
[the publisher](information_geometry_publish.py) verifies original receipt hashes,
source/runtime equality, checkpoint metadata, matched conditions and media inputs.
Render and inspect those same completed records with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u \
  reports/forge/bcap-physics/information_geometry/information_geometry_publish.py \
  --queue-root /mnt/ml7tb/ParticleGAN-forge/bcap-physics-20261009/queue \
  --certificate-root /home/martyn/dev/ParticleGAN
```

| Task | Candidate actual-training GIF | Control actual-training GIF |
| --- | --- | --- |
| two_pole | [GIF](media/candidate/two_pole.gif) | [GIF](media/control/two_pole.gif) |
| Gaussian smoke | [GIF](media/candidate/gaussian1d_smoke.gif) | [GIF](media/control/gaussian1d_smoke.gif) |
| Gaussian stability | [GIF](media/candidate/gaussian1d_stability.gif) | [GIF](media/control/gaussian1d_stability.gif) |
| Unequal mass | [GIF](media/candidate/vector_unequal_mass.gif) | [GIF](media/control/vector_unequal_mass.gif) |
| Blobs | [GIF](media/candidate/img_blobs4.gif) | [GIF](media/control/img_blobs4.gif) |
| Intensity | [GIF](media/candidate/img_intensity2.gif) | [GIF](media/control/img_intensity2.gif) |
The parent retains the one current cross-track comparison; this report adds no
second generated leaderboard.
