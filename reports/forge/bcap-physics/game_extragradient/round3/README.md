# Round 3: same-batch correction of the alternating BCAP game

Preregistered bounded mechanism diagnostic. Scientific outcome pending; no
ordinary qualification or default adoption. The parent owns the sole current
goal leaderboard. This report compares task metrics within this frozen source.

## Mechanism and mathematical scope

Let `w=(D,G,z)`, `h` be the current optimizer/policy histories, and `xi` be the
complete per-update data and noise draws. Define the **actual alternating**
public update displacement `U(w;xi,h)`: D minimizes non-saturating logistic loss
plus real/fake BCAP, then G/prior minimize their original loss at that updated D.
Each network uses the winner's smoothed polar/bias rule and the prior uses its
actual sampled rows. The new public `Recipe.same_batch_extragradient=True` does

```
predict: w_tilde = w + U(w; xi, h)
correct: w_next = w + U(w_tilde; xi, h)
```

The preview is reversible. The correction begins with the original optimizer
histories, buffers and complete RNG state at predicted joint parameters. Its
displacement is rebased to the original parameters; inactive coordinates and
unsampled prior rows retain their exact original bits. Only the correction's
histories, buffers and consumed streams commit; there is one completed update
and one logical optimizer tick. EMA0 derived models are synchronized to the
committed state. Callable generator data is fetched once before prediction and
reused. Each field evaluation still alternates D then G/prior. This evaluates
the normalized learned-prior game, rather than extrapolating a stale network
direction or drawing another batch.

For a **different, simultaneous linear rotation model** `F(w)=Aw`, `A^T=-A`,
classical extragradient yields `w_next=(I-eta*A+eta^2*A^2)w`; a frequency block
has squared modulus `1-a^2+a^4`, with `a=eta*omega`, below one for `0<|a|<1`.
This is a scoped algebraic damping property. BCAP's nonlinear normalized field,
alternation and asynchronous sampled-prior response do not satisfy that model
or its monotonicity assumptions. The actual composition above is an
extragradient-inspired alternating-map correction, not an implementation of the
simultaneous theorem. There is no global guarantee and no physical conservation
law. A full-size preview can overshoot; its corrected norm can stay large.

[Gidel et al.](https://arxiv.org/abs/1802.10551) study extrapolation for GAN
variational inequalities; [Mishchenko et al.](https://proceedings.mlr.press/v108/mishchenko20a.html)
study same-sample stochastic extragradient and its approximation to implicit
updates. This track adapts established predictor/corrector ideas to the actual
public alternating normalized map and claims no novelty for extragradient.

## Evidence inspected before the candidate was frozen

The [original failure analysis](../../../bcap-tier2-search/FAILURE_ANALYSIS.md)
finds negligible Gaussian BCAP activation and persistent normalized movement
after acquisition. Its original source digest
`2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`, seed0 and
initializer remain intact. The
[failed optimism report](https://github.com/255BITS/ParticleGAN/blob/61b07b1c2769b5eb97bcbde70abc3cd4b4464ad9/reports/forge/bcap-physics/thermodynamic/README.md)
lost Gaussian confirmation and collapsed mode hold to 3/8. It stores no actual
antisymmetric-Jacobian census. Round two's
[comparison](../../round2/README.md) also retains Gaussian and width failures.

[Saved response](saved-response.json) and its
[reproduction source](probe_saved_response.py) verify original artifact hashes
at Gaussian updates 1000/4000/6000 and discard every local parameter proposal.
The deterministic FP64 full-table MoG cubature is a separate diagnostic cohort,
with analytic Gaussian quantiles, no random draws and zero optimizer updates.
Normalized simultaneous proposal/reevaluation G cosines are
`.866/.895/.772`, D `.777/.727/.817`, and prior `.669/.600/.719`.
Relative response norms range `.451–.874`. This shows substantial finite-step
response. It neither identifies rotational dominance nor predicts the actual
alternating stochastic update. Probe elapsed CPU time is recorded separately
and charged to the fresh ceiling.

## Frozen comparison and causal forecasts

One candidate and one primary matched **exact winner**. Both resolve the saved
`bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`
in this same scientific source/runtime. The sole delta is
`same_batch_extragradient=False -> True`. Non-saturating BCAP coefficient1/cap1
every update, full DualNorm smoothing .001/momentum0/per-offset convolution,
constant G/E .012, D .018, prior .030, floors1, and zero additive noise/EMA remain.
No archived winner or previous candidate result supplies third-arm causal credit.
The control study uses the winner's supported parent declaration solely as an
admission reference because the reverse candidate binding cannot execute
two-pole. It launches no parent run and supplies no third-arm comparison.

Before execution, forecast final Gaussian stability **`cdf_ks <= .05`**,
at least **60/72** stationary and **20/24** shifted-hold full passes, a mode-hold
terminal passing suffix **>=5**, and retention of the full broad-vector PASS.
Expect unequal-width full covariance error **<=.85** with every mode resolved;
the full five-check terminal gate still determines its grade. Gaussian final
`cdf_ks > .05`, guardrail loss, or failure of the complete sustained mode gate
rejects the proposed global repair. The machine study signature uses the actual
`cdf_ks` key; blocked/missing evidence cannot satisfy a prediction.

Competing explanations are excessive finite travel, covariance deformation,
allocation and stochastic normalized-field sensitivity, rather than correctable
circulation. Identical data can still produce a large direction change after a
preview. Improved final moments without retention are insufficient. This one
global rule has no per-task parameters and no automatic tuning/continuation.

The six unchanged tasks are two_pole, gaussian1d_smoke, its own checkpoint-bound
gaussian1d_stability, mode_hold, vector_two_broad and vector_unequal_width.
Two-pole retains its separate identity/zero/stored-weight fixture: candidate
execution is genuinely **BLOCKED** because this public-components host cannot
bind a reversible GANTrainer joint update. It is not silently adapted. Other
tasks retain seed0, public deterministic initialization, architecture, target/data,
priors/widths/masses, sampling, updates, full numerical gates and cadence.
The frozen vector/Gaussian adapters reuse one actual real tensor for D/G;
that law remains. All consumed named streams are checkpointed and replayed.
Clean live grading stays separate from noisy/EMA evidence.

Each arm declares 6420 full worker seconds (including two-pole); total 12840
under a fresh **14400-second** ceiling covering controls, retries and diagnostics.
No native task, seed repeat, sweep or subsequent candidate is admitted. Failed
smoke leaves its own dependent stability task BLOCKED. All other independent
jobs complete regardless of numerical failures. Finish and publish either result.

## Reproduction and live logs

Use shared `/home/martyn/dev/ParticleGAN/.venv/bin/python` (3.12), with
`PYTHONPATH=/home/martyn/dev/ParticleGAN-bcap-physics-game_extragradient`.
Candidate/control studies and diagnostic view use unique
`game-extragradient-round3-*` IDs. Scientific code and ready declarations are
pushed before enqueue. The independent Queue/drain runner uses GPU1, one worker,
sharing enabled, watch disabled, and `on_completion=None` to preserve archived
qualification. It stops after the bounded jobs finish.

```
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/game_extragradient/logs/drain.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/game_extragradient/queue/events.jsonl
```

Publication will contain final metrics, source/runtime/stream proofs, certificates,
actual-training GIFs and scoped recommendations. Bulk stdout, JSONL, checkpoints
and dumps stay in the local artifact archive. Summaries-only compilation is used;
historical qualification and archived evidence are never regraded.
