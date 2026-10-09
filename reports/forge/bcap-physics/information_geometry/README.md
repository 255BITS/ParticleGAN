# Information geometry: relative spectral-half BCAP diagnostic

Status: preregistered, training pending. One candidate and the exact winning
BCAP control will run in the shared campaign. This is a mechanism diagnostic,
with no ordinary-tier qualification or default-adoption claim.

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

Equivalently, normalize (GGᵀ + ε²I)^−1/4 G by h_max^−1/2. Away from the
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

Pretraining software validation: 144 tests pass, including spectral contrast,
orthogonal equivariance, unchanged bias/sampled-prior laws, public grouped
Conv2d/ConvTranspose2d construction, exact optimizer resume and family mismatch
rejection. The candidate plan reports READY, zero preflight blockers and exactly
one consumed optimizer-family delta on every task. Final measured results,
source/runtime receipts and actual-training GIFs will replace this pending status.
The parent retains the one current cross-track comparison; this report adds no
second generated leaderboard.
