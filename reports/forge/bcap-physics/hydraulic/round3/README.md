# Hydraulic round three: finite local shape and mean transport

**Stop this exact combined revision as a global repair.** The candidate finishes
1PASS,2FAIL,2BLOCKED versus the matched winner’s3PASS,2FAIL. Native precision
falls .24072 → .03876 and Gaussian smoke is lost, while broad-vector PASS is
retained with improved covariance. [PR360](https://github.com/255BITS/ParticleGAN/pull/360)
preserves the complete negative readout and the useful scoped guardrail result.
One successor replaces the failed global secant contraction with a training-local
finite-shape controller. The primary matched control is the exact winning BCAP
recipe. The predecessor cannot be admitted as a control because its two-pole
public-components path is unsupported. This measures the combined package
against the winner, not the incremental causal effect relative to hydraulic-v2.
[Round two](../round2/README.md) retains its original identity and results.

## Evidence and mechanism

[Saved finite probes](saved-finite.json) restore the original round-two native
checkpoints (executed source digest `2ebd7377a68b5db8783006c3cb59ff84b7a625027974bf3387db722a940f8357`).
Independent seed0 diagnostic jitter, shared across both saved arms, gives mean
odd-response energy .009230 for hydraulic-v2 versus assigned real-neighborhood
capacity .001751. The winner gives .005825 versus .001655. Even-response energy
is much smaller (.000004924 for v2); endpoint network/prior displacements partly
cancel (normalized dot -.313783). Endpoint swaps are sensitivity diagnostics,
not causal per-update ablations. These 40,000 scalar diagnostic Gaussian draws
consume neither original training/evaluation streams nor qualification credit.
The saved observer real graph has133 components, not the ground-truth100 modes.

For the consumed generator-side real batch, let r be median nearest-distinct
spacing. Construct the epsilon-neighborhood graph with links of length at most
4r. A component's capacity v_k is its unbiased centered total variance. A
singleton/identical component uses r²; a constant finite batch uses a zero
G/prior ray. No component labels, target covariance, centers or scoring threshold
enters training. The graph is a deterministic training-data estimate.
[Ester et al. (1996)](https://aaai.org/papers/kdd96-037-a-density-based-algorithm-for-discovering-clusters-in-large-spatial-databases-with-noise/)
motivate epsilon neighborhoods; this simple connected-components rule has no
core-point filtering and is not the full DBSCAN algorithm.

Reuse each already consumed MoG center c_i and jitter epsilon_i. For a joint
G/prior state theta, define q_i(theta)=(G(c_i+epsilon_i)-G(c_i-epsilon_i))/2,
m_i(theta)=(G(c_i+epsilon_i)+G(c_i-epsilon_i))/2. Assign each old midpoint to
its nearest consumed real point's graph component and hold that assignment
fixed through the proposal. Let n_k count assigned probes and

`S(theta) = sum_k (n_k/N) max(mean_{i in k} ||q_i(theta)||² - v_k, 0)`.

The public zero-momentum DualNorm optimizer proposes its original adversarial
G/prior step once. At ray scale alpha, request
`S(theta_new) <= (1-alpha) S(theta_old) + numerical_tolerance`.
This permits legitimate initialization expansion within data capacity; it does
not freeze the random initial Jacobian. When correction is needed, compute
`g=grad_theta S`, `M=Jacobian_theta mean_i m_i`, and
`g_perp = g - M.T pinv(M M.T) M g`.
Apply `delta_theta=-2 (S-target)/||g_perp||² * g_perp` at most three times.
This correction preserves the sampled global midpoint mean to first order.
It does not preserve every mode mean or finite nonlinear mean. Actual rounded
`M delta_theta` is recorded. The full proposal must pass both exact finite
shape and same-latent output-travel checks (RMS at most r). Seven ray trials
halve alpha; exhaustion restores the old G/prior parameters, with optimizer
clocks advanced once. D and the original adversarial objective stay unchanged.

[Odena et al. (2018)](https://arxiv.org/abs/1802.08768) study generator Jacobian
conditioning. Our rule controls sampled finite response excess in inferred real
neighborhoods; it is not a condition-number or density/convergence guarantee.
Graph splitting, incorrect anchors, suppressed useful adversarial motion and
nonlinear mean drift are competing explanations if the full gates fail.

One fixed global rule uses link multiplier4, correction over-relaxation2,
three corrections per ray, seven rays and travel fraction1. The correction
acts jointly on G and learned prior locations. There is no global secant loss,
fresh training draw, target oracle, per-task tuning or second candidate.
Supported scope: positive-width nonstandardized MoG, one/two output coordinates,
clean stateless generator and zero-momentum DualNorm. Nonfinite proposals fail
explicitly after G/prior restoration. Two-pole remains genuinely unsupported.

## Frozen comparison and forecasts

The unchanged five-task `hydraulic_motion_diagnostic` view includes explicit
fixed-identity two-pole, Gaussian smoke and its own smoke-dependent stability,
full native100 7,000 updates and broad-vector passing guardrail. Both arms use
seed0, public deterministic initializer, fixed task model/data/prior/sampling,
updates and original complete numerical gates. Gaussian/vector adapters retain
one real tensor shared by D/G; every consumed named stream is checkpointed.
Clean/live grades remain separate from noisy or averaged serving.

Ready schema3 candidate `hydraulic-local-shape-v3` and ready studies
`hydraulic-local-shape-{candidate,control}-round3-v1` freeze campaign
`hydraulic-local-shape-round3-v1`:14,400-second ceiling,6,420 per-arm full
allowances,12,840 total reservation. Unsupported candidate two-pole costs0,
leaving12,540 runnable allowances if all own prerequisites pass. No scientific
retry, continuation, sweep, default adoption or ordinary qualification follows.
The exact winner's parent is an untrained admission reference, not a third arm.

Frozen forecast: native precision at least.60; falsifier below.48. Explanatory
forecasts: absolute covariance trace bias at most.50 and median local Jacobian
variance proxy at most1.5 times target total variance. Broad-vector must retain
full PASS; Gaussian/native success requires their original complete gates.
Forecast success alone cannot override a failed task or unsupported cell.

## Reproduction and local logs

From the assigned worktree with its path in PYTHONPATH, use the shared Python3.12
venv, never system Python. `run.py plan`, then commit/push, `run.py enqueue`,
and `run.py drain` use the public Queue with completion callback disabled,
one GPU0 worker, sharing allowed, no watcher. Both arms freeze identical current
scientific source/runtime before drain. `publish.py --media` verifies frozen
source and matched condition/stream receipts and renders retained actual-training
frames; it never reruns training or qualification. Root owns the one current
leaderboard. Bulk logs/checkpoints/event streams stay outside Git.

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/hydraulic/queue/hydraulic-local-shape-round3-v1/*/run.log
```

Pre-freeze software validation:159 checks pass across the finite-shape rule,
public Forge API, vector scorers, trainer integration and recipe defaults. Nine
unchanged hydraulic-v1/v2 checks also pass (168 distinct checks total). The new
checks exercise active CPU/CUDA finite correction, exact travel/progress bounds,
rounded first-order mean residual, legitimate full-proposal initial growth,
public checkpoint replay, consumed-stream parity and pre-mutation refusals.
An initial test harness used a config-only loader for the winner; it was fixed
to use the public declaration loader before source freeze. No training change
followed a scientific result. Final grades, costs, exact source identities, GIFs
and the stop/proceed recommendation are recorded below.


## Complete numerical readout

[Final metrics](results.json), [provenance](provenance.json) and
[study readouts](study-readouts.json) bind all eight completed workers. These are
full unchanged task gates, not shortened screening endpoints.

| Task | Finite-shape successor | Exact winner control | Decisive evidence |
| --- | --- | --- | --- |
| Fixed identity/zero two-pole cohort | BLOCKED, no attempt | PASS | Candidate extension requires public trainer, unavailable on public-components fixture; original fixture remains distinct |
| Gaussian smoke,1,000 updates | FAIL | PASS | Candidate0/24 passing scheduled observations; control3/24, first confirmed at375 |
| Own-smoke Gaussian stability | BLOCKED, no attempt | FAIL | Candidate’s own smoke FAIL prevents continuation; control stationary2/72, shifted hold0/24 and deadline FAIL |
| Native100,7,000 updates | FAIL | FAIL | Candidate/control precision .03876/.24072, mass TV .94548/.14718; both0/5 terminal full-accuracy checks and coverage FAIL |
| Broad vector,1,200 updates | PASS | PASS | Candidate/control terminal passing suffix21/22, exceeding unchanged minimum5 |

Only three tasks have two executed arms: candidate1PASS/2FAIL versus control
2PASS/1FAIL. The remaining cells retain actual applicability/dependency status;
no missing state receives a synthetic grade or GIF. Forge’s global dependency
job remains pending after its diagnostic subscription completes; publication
reports BLOCKED from the actual own-smoke FAIL, with no queue mutation, retry
or changed scientific source. The initial generic publisher called that missing
job INCOMPLETE; its report-only correction preserves the original failure.

Candidate smoke ends with mean−.2047531 against target2, standard deviation
.0357028 against target.5, standard-deviation ratio .0714056 and CDF KS .9999714
against limit.05. These fail severely. The control smoke grade asks whether any
scheduled state has independent same-state confirmation; its final KS .0718422
does not erase the confirmed scheduled PASS. Stability retains all72stationary
checks, deadline five-check suffix and24shifted hold checks; none is relaxed.

Native’s100,000-output holdout has3,876 within-radius samples, precision .03876
and mass TV .94548. Covariance/centering/radial metrics are **unavailable** because
of insufficient per-mode support, not zero or a favorable covariance forecast.
All five terminal scheduled accuracy checks fail. The matched winner reproduces
its .24072 precision and .14718 mass TV. The unchanged native coverage and full
accuracy gates reject both arms; the reference-law scorer control passes.

Broad-vector improves final normalized SW1 .134590 → .083331, covariance error
.385581 → .221603, minimum eigen ratio .400066 → .620014 and high-quality fraction
.988770 → .994873. Component counts2306/1790 →2300/1796 and mass TV .062988 →
.061523. The complete unchanged gates are SW1≤.18, mass TV≤.15, HQ≥.85, covariance
error≤.85 and minimum eigen ratio≥.15. The candidate first passes at200 and has
21terminal consecutive passes; the control first passes at150 and has22. This
supports a scoped passing result, without repairing the failing Gaussian/native
questions or establishing the incremental shape effect relative to archived v2.

The .60 precision prediction is missed and the .48 numerical falsifier is
observed. The absolute covariance-bias≤.50 explanatory forecast is unmeasurable.
The median Jacobian variance proxy≤1.5 forecast is observed, but by contraction:
.259211 times target total variance versus the winner’s2.976159. Candidate median
minimum eigen ratio is .043869, versus2.022163. Only .33% of candidate centers
have total variance above target, versus97.56% for the winner. This is an
uncensored all20,000-center FP64 derivative diagnostic with an independent
reverse-mode check, not served-law covariance or a task PASS. All restored
checkpoints and global RNG remain unchanged; no sampling/training is added.

The automatic study outcome remains **INCOMPLETE** for the candidate because
its declared five-task comparison contains two unexecuted cells. This is
separate from the observed numerical falsifier and rejection of the repair.
The control’s frozen forecast is **FALSIFIED**; its parent is an untrained
admission reference and supplies no causal baseline or qualification credit.
Both request subscriptions are concluded through the normal readout API.

## What the controller actually did

[Controller counters](controller-diagnostics.json) describe the full original
training, with no new draws or updates.

| Training task | Mean graph components | Mean assigned capacity | Shape-active updates | Mean ray scale | Mean accepted RMS travel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian smoke |35.891|.0000345548|47.6%|.0938464|.00364197|
| Native100 |134.5813|.00154145|69.9%|.0286391|.00868497|
| Broad vector |6.5492|.104541|2.9167%|.481983|.0385138|

Every accepted sampled shape bound passes with maximum recorded violation0;
maximum accepted travel/radius ratio is at most1 in all three runs. Corrections
number851,12,653 and105 respectively. Maximum rounded first-order global
midpoint-mean derivative residual is1.291e−7 across all attempted corrections.
No whole ray is rejected, so this regression is not explained by a complete
zero-step stall. Native limits every ray; broad limits98.25%. The inherited `probe_calls` counter
counts check/replay callbacks but excludes the initial and gradient-building
shape probes; it is not a complete forward-pass or performance count. Shape is only
rarely active on the passing broad task, so that improvement cannot be assigned
solely to the new correction rather than the combined travel package.

The graph averaged nearly36 neighborhoods for one Gaussian; its assigned
variance capacity is about.0000346 while true Gaussian total variance is.25.
Real minibatch fragmentation and nearest-anchor selection can therefore impose
very small local response ceilings on a unimodal law. The saved native graph
had133 inferred neighborhoods for100 modes; actual native batches average134.6.
These are data-resolution estimates, not ground-truth mode identifiers. Native
first-order widths contract far below target while population coverage fails.
This supports overconstraint as a competing explanation. It does not isolate
its causal effect from travel rescaling, wrong anchors, useful adversarial
motion suppression or nonlinear finite mean drift. Global first-order mean
protection alone supplies no correct mode transport or mass-allocation guarantee.

Recommendation: **retain the exact winner/default and stop this exact local
graph-capacity package as a global repair**. Preserve broad-vector PASS as scoped
evidence and preserve all archived v1/v2 results. Before a separately admitted
substantive successor, inspect saved population/center transport and whether a
training-derived capacity respects a unimodal law and necessary initialization
expansion. Keep complete density lower bounds, coverage and distribution gates;
a small Jacobian upper bound can be satisfied by severe contraction. No tuning,
seed repeat, extra arm, continuation or adoption is authorized by this readout.

## Source, accounting and artifacts

Both arms executed commit `96199857598db700eb21d70991b952166ae5838b` with scientific digest
`83c7e1f14ec5409d0b258a0bda18dffa721f5ddecb314d79cb28cebf0af650a7`. All1202 frozen scientific files
match this publication checkout. Later changes add only reports, saved-state
diagnostics and the failed-prerequisite publication classification; no trainer
fix is silently claimed as measured. Publication GitHub head is recorded in the
parent handoff receipt separately from this trained source.

Public worker wall cost is **875.543026674seconds**,
against the frozen14,400-second ceiling. Declared per-arm allowances total12,840;
unsupported two-pole300 and failed-prerequisite stability600 spend nothing,
leaving11,940 full launched allowances. Eight workers finish their complete
protocols, with **zero retries and zero remaining reservations**. There are
23,480 new updates:9,200 candidate and14,280 control (including80explicit fixture
updates); control stability restores its1,000-update prefix and adds5,000.
Read-only pre/post diagnostics take3.635486seconds
separately; conservatively adding them gives879.178513seconds,
still below the ceiling. Software/media work is distinct. Contended wall cost
supplies accounting, not an optimizer-speed comparison.

All three matched pairs verify identical task declarations, effective recipe,
prior, public initialization, host, actual data and every consumed non-evaluation
stream, including constructor, data, training-jitter and prior-index streams.
Evaluation confirmation uses its original task rule and isolated checkpointed
stream; failed acquisition does not trigger synthetic confirmations. Full
checkpoints contain every consumed stream. Separate two-pole fixture and
software/initial-growth controls supply no MoG training qualification.

[Eight actual-training GIF receipts](media/index.json) reuse certified saved
observations and add no training or sampling. Candidate GIFs:
[smoke](media/candidate/gaussian1d_smoke.gif), [native](media/candidate/grid100.gif),
[broad](media/candidate/vector_two_broad.gif). Control GIFs:
[two-pole](media/control/two_pole.gif), [smoke](media/control/gaussian1d_smoke.gif),
[stability](media/control/gaussian1d_stability.gif), [native](media/control/grid100.gif),
[broad](media/control/vector_two_broad.gif). Numerical gates establish outcomes;
the GIFs illustrate actual training, including failure.

Raw stdout, checkpoints, event streams and saved tensors remain locally under
`/mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/hydraulic/queue`.
Compact metrics/provenance include original attempt IDs, artifact paths, byte
hashes and state hashes. No raw log/state dump is committed. Both subscriptions
are concluded, the bounded public drain has exited, and no worker/watcher remains.
Summaries-only compilation preserves historical qualifications, telemetry and
the single parent-owned leaderboard. This is a mechanism diagnostic on a
provisional profile, with no ordinary Tier2 qualification or default change.
