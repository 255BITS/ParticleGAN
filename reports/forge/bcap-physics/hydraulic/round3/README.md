# Hydraulic round three: finite local shape and mean transport

Research in progress on [PR360](https://github.com/255BITS/ParticleGAN/pull/360).
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
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/hydraulic/logs/drain.log
```

Pre-freeze software validation:159 checks pass across the finite-shape rule,
public Forge API, vector scorers, trainer integration and recipe defaults. Eight
unchanged hydraulic-v1/v2 checks also pass (167 distinct checks total). The new
checks exercise active CPU/CUDA finite correction, exact travel/progress bounds,
rounded first-order mean residual, legitimate full-proposal initial growth,
public checkpoint replay, consumed-stream parity and pre-mutation refusals.
An initial test harness used a config-only loader for the winner; it was fixed
to use the public declaration loader before source freeze. No training change
followed a scientific result. Final grades, costs, exact source identities, GIFs
and the stop/proceed recommendation follow after the complete admitted runs.
