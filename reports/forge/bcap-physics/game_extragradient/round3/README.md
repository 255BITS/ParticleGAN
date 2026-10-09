# Round 3: same-batch correction of the alternating BCAP game

**Stop this exact global revision.** Same-batch correction improves Gaussian
acquisition and late unequal-width fidelity, but adds no sustained task PASS and
severely regresses mode hold. Candidate: **2 PASS, 3 FAIL, 1 BLOCKED**; exact
winner control: **3 PASS, 3 FAIL**. The five mutually executable tasks retain
2/5 PASS for both arms. All 11 runnable jobs complete, with zero scientific
retries. [PR364](https://github.com/255BITS/ParticleGAN/pull/364) publishes the
reusable public capability and this bounded comparison.

No ordinary qualification or default adoption follows. The parent owns the sole
current goal leaderboard; archived winner 7/21 remains unchanged. These are
within-source task comparisons, not a pooled ranking across specialist tracks.

## Completed numerical comparison

[Final metrics](results.json), [verified provenance](provenance.json), and the
[candidate](../../../records/readout-5231ef4446c335a94913008f.json) /
[control](../../../records/readout-d2e07346387dca9ac14348e8.json) readouts retain
all original gates and identities. Covariance below means the **full component**
covariance error, not global or core-only covariance.

| Unchanged task | Same-batch candidate | Matched winner control | Full-gate interpretation |
| --- | --- | --- | --- |
| Two-pole fixed fixture | **BLOCKED**, unsupported joint-step host, zero spend | **PASS**; mean .958502, gradient .952346, suffix 17 | Existing public-components law retained; no silent GANTrainer substitution |
| Gaussian smoke | **PASS**; 11/24 primary passes, 9 independently confirmed; final KS .036440 | **PASS**; 3/24 primary, 2 confirmed; final KS .071842 | Any complete primary plus independent same-state pass; all 1000 updates complete |
| Gaussian stability | **FAIL**; stationary 20/72, shift hold 6/24, deadline reacquisition FAIL; final KS .058208 | **FAIL**; stationary 2/72, shift hold 0/24, deadline FAIL; final KS .320623 | All 72 stationary/all 24 shift-hold plus reacquisition suffix must pass |
| Mode hold | **FAIL**; final 5/8 modes, quality .220459; 0/24 full passes, suffix 0 | **FAIL**; final 8 modes, quality .989258; 10/24 passes, suffix 3 | All 8 modes and quality >=.90 at five terminal checks |
| Broad-vector guardrail | **PASS**; covariance .174091, eigen .937665, SW .065050, TV .047607; suffix 24 | **PASS**; covariance .385581, eigen .400066, SW .134590, TV .062988; suffix 22 | Complete distribution/mass/quality/covariance bounds; suffix>=5 |
| Unequal width | **FAIL**; covariance .253042, eigen .575574, SW .080210, TV .085205, quality .994385; suffix 4 | **FAIL**; covariance 6.287559, eigen .307472, SW .295865, TV .291504, quality .978760; suffix 0 | Endpoint passes every bound, but only four terminal full checks |

Unequal width's update 1000 fails **only quality**: `.801025 < .85`.
Its covariance .309099, eigen .448377, SW .084621 and TV .073486 pass.
The next four states 1050/1100/1150/1200 pass all bounds; no added update or relaxed
suffix converts this late acquisition into a sustained PASS. Endpoint component
counts are 1013/686/1184/1213, all four components resolved. This is useful late
law fidelity under unchanged width and sampling, rather than certified repair.

Gaussian retention's endpoint mean error .119860 sigma and width ratio 1.009888
pass their individual bounds, while KS .058208 misses .05. Its 20 stationary
and 6 shifted-hold passes fall short of the 60/20 forecasts and full 72/24 gates.
Width/mean endpoints cannot replace the whole target CDF or continuous hold.
The candidate broad guardrail passes every scheduled check, establishing that
the failures do not amount to blanket numerical inability. Mode hold nevertheless
collapses late: qualities .650635/.747314/.277100/.291748/.220459 at the last
five observations; a universal damping interpretation is unsupported.

The actual-key Gaussian forecast is measured and **falsified** (`cdf_ks>.05`).
Forge's candidate aggregate decision remains **incomplete** because two-pole
is genuinely unsupported; the control decision is `falsified`. These immutable
decision receipts are preserved. Missing evidence is not a scientific pass or
authorization for an unchanged rerun.

## Recommendation and scope

Retain the exact winner as the experimental control; reject this exact
same-batch global candidate. Preserve the acquisition and late width findings,
but do not promote or continue training this revision. Before another separately
admitted mechanism, inspect saved predictor/corrector sensitivity at the mode
collapse and width-quality transition. This experiment does not identify whether
the critic, network deformation or sampled-prior movement causes those failures.
It does not measure a game Jacobian's antisymmetric part, prove monotonicity,
establish transfer to images/native tasks, or provide robustness across seeds.
No seed study, sweep, native job, extra candidate or follow-on training occurs.

## Provenance, checks and accounting

Both arms trained source commit `3b9bda7f0ff4f34414eaf2ddbaa292206a217240`, digest
`86e864be4656a32824e13464de7f6d726b0ffe2b825eedf48ac6ff2cdf6a02a3`.
Candidate revision is `88dc4cb3594f8847cf15aa2b0d33f4127af2274bf5dc03ba18d28ed156c4a227`;
control `955fa11d15efffaae1098a0f399b3a3f62019ad5df0fa73e6f78e546a25a7d21`.
CPython 3.12.13, Torch 2.14.0, NumPy 2.5.2 and SciPy 1.17.1 ran on the RTX A6000 CUDA
cohort. Scientific code and ready declarations were pushed before training.
Later additions are publication/analysis only; **all 1200 measured source files
still match their executed hashes**. No public trainer software fix after training
is attributed to these results.

All 11 certificate/result envelopes, saved checkpoint manifests and media inputs
verify. Five paired tasks have equal initial-model proofs and final named stream
hashes, with zero unintended RNG deviations. Gaussian consumed-data digests also
match for stationary/shift phases. Vector/ring final stream parity is a state
audit, not a stored bytewise consumed-batch digest. All candidate parameter clocks
are 1000/6000/1200 as appropriate, equal to completed logical updates; no doubled
clock is committed. Task-owned prior, initialization and clean-live laws remain.

**113 focused software tests pass**, covering the exact map composition, single
clock, same streams, checkpoint replay, callable-once data, failure rollback,
zero/unsampled-row preservation, stochastic layers/buffer counts, unsupported
components, default compatibility, Forge studies and field boundaries.
Forge validation and summaries-only memory checks pass. The
[scorer controls](scorer-controls.json) accept Gaussian/vector/ring oracle laws
and reject relevant collapse, wrong width/shift and missing-component controls;
their [source](scorer_controls.py) adds zero training. These controls do not
calibrate the failed screening profiles or grant ordinary qualification.

Declared full allowances total **12840 seconds**, executed full allowances **12540**
after the unsupported 300-second cell. Paid worker cost is **524.028745 seconds**:
candidate 354.228650, control 169.800094. Active reservation is **zero**, with zero
scientific retries and zero invalid/incomplete workers. Separate bounded CPU
saved-response/scorer probes took .937047/.156190 seconds; including these
compactly recorded diagnostics remains inside the 14400 fresh ceiling. Shared-GPU wall cost is accounting, not optimizer
speed or FLOP evidence. The independent drain has stopped; both subscriptions
are closed through normal readout. No watcher remains.

## Certified actual-training GIFs

[Media receipts](media-index.json) bind every frame to certified observations
and retained sample arrays where available. Export adds zero model calls,
sampling or updates. Ring/two-pole media show their actual numerical training
observations against the goal. The unsupported candidate cell has no fabricated
training media.

| Task | Candidate | Control |
| --- | --- | --- |
| Gaussian smoke | [GIF](media/candidate/gaussian1d_smoke.gif) | [GIF](media/control/gaussian1d_smoke.gif) |
| Gaussian stability | [GIF](media/candidate/gaussian1d_stability.gif) | [GIF](media/control/gaussian1d_stability.gif) |
| Mode hold | [GIF](media/candidate/mode_hold.gif) | [GIF](media/control/mode_hold.gif) |
| Broad vector | [GIF](media/candidate/vector_two_broad.gif) | [GIF](media/control/vector_two_broad.gif) |
| Unequal width | [GIF](media/candidate/vector_unequal_width.gif) | [GIF](media/control/vector_unequal_width.gif) |
| Two-pole | BLOCKED | [GIF](media/control/two_pole.gif) |

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
and one logical optimizer tick. EMA 0 derived models are synchronized to the
committed state. Callable generator data is fetched once before prediction and
reused. Each field evaluation still alternates D then G/prior. This evaluates
the normalized learned-prior game, rather than extrapolating a stale network
direction or drawing another batch.
There are two actual loss/gradient evaluations and two temporary optimizer
applications per logical update. The predictor's clocks and side effects are
rolled back. Receipt update counts describe committed updates, not arithmetic
work; extra compute and snapshot overhead are included in paid worker cost.

For a scoped **alternating bilinear** descending game `L_D=-xy`, `L_G=xy`,
let D/G steps be `a,b>0`. The ordinary map is
`M=[[1,a],[-b,1-ab]]`. Our correction is exactly `I-M+M^2`.
Since `det(M)=1` and `tr(M)=2-ab`, Cayley-Hamilton gives
`I-M+M^2=(1-ab)M`. For `0<ab<2`, the complex eigenvalues of M have modulus 1,
so the corrected spectral radius is `|1-ab|<1`. This is asymptotic linear
damping, not a monotonic Euclidean-energy claim. It directly accounts for
alternation in that two-variable model.

For the different simultaneous rotation `F(w)=Aw`, `A^T=-A`, classical
extragradient has frequency-block squared modulus `1-c^2+c^4`, with
`c=eta*omega`, below one for `0<|c|<1`. Neither scoped algebraic fact certifies
BCAP's nonlinear normalized field or asynchronous sampled-prior response.
Our actual composition above is an
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
`2e1d0e2704f3e8cff0845f46fe66e8fb641c32fd32b7d1929f05a680b4c3bbed`, seed 0 and
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
`same_batch_extragradient=False -> True`. Non-saturating BCAP coefficient 1/cap 1
every update, full DualNorm smoothing .001/momentum 0/per-offset convolution,
constant G/E .012, D .018, prior .030, floors 1, and zero additive noise/EMA remain.
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
tasks retain seed 0, public deterministic initialization, architecture, target/data,
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

This publication contains final metrics, source/runtime/stream proofs, certificates,
actual-training GIFs and scoped recommendations. Bulk stdout, JSONL, checkpoints
and dumps stay in the local artifact archive. Summaries-only compilation is used;
historical qualification and archived evidence are never regraded.


Re-export the saved observations without training:

```sh
PYTHONPATH=/home/martyn/dev/ParticleGAN-bcap-physics-game_extragradient /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/game_extragradient/round3/publish.py --queue /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/game_extragradient/queue --certificates /home/martyn/dev/ParticleGAN-bcap-physics-game_extragradient/reports/forge/attempts
```
