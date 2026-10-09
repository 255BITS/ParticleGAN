# Round3: multiscale pairwise distribution witness

This is a preregistered bounded mechanism diagnostic, with one candidate and
the exact winning BCAP recipe as its sole primary matched control. Results are
pending. It supplies no ordinary Tier2 qualification, calibrated screen or
default-adoption credit. The parent owns the single current goal leaderboard.

## Hypothesis and public mechanism

Add a full pairwise raw-output Cauchy-kernel mean discrepancy to the existing
non-saturating generator objective, at one global weight1. Keep the critic,
BCAP cap1/coefficient1/every update, DualNorm smoothing .001/momentum0/per-offset,
constant G/E .012, D .018, prior .030, floors1, zero additive noise and EMA.
The control resolves the saved `bcap-dualnorm--5b1ef16597377d87cbc5a4cc4a152d207884e3d3c3b7ca48968f98c77a11fa36`
declaration; bare preset defaults do not define this comparison.

For an n-row real minibatch P, define squared global scale
S=mean_i ||x_i-mean(P)||². Let H be the median sqrt(n)-th other-neighbor squared
distance, clamped to [S/(16 n²),S]. An exactly atomic batch uses S=1 explicitly.
Freeze the scales C=(H,sqrt(H S),S), and use

    k(x,y) = (1/3) sum_c c/(c+||x-y||²).
    L = mean_QQ k - 2 mean_PQ k + mean_PP k.
    L_G = L_GAN + L.

Every pair and self diagonal participates in this empirical V-statistic. Its
generated-pair term repels distinct outputs; real/generated terms attract.
Backpropagation reaches the existing generator and learned prior, while the
real batch, scales and real/real term are detached. No target labels, centers,
mixture moments, component weights, reservoir, extra data draw or RNG is used.
There is no finite feature frame or finite real-anchor residual. Public entry
points are `Recipe.kernel_witness_weight`, `Recipe.kernel_witness_loss` and
`GANTrainer.step`. Default zero preserves the original path and checkpoints.

[Gretton et al.](https://jmlr.org/papers/v13/gretton12a.html) describe kernel mean
discrepancies. [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558),
section4, use this characteristic Cauchy/inverse-multiquadratic kernel and
describe its heavier-tail gradient advantage over RBF. For fixed positive
scales it is a positive mixture of Gaussian kernels with full spectral support;
the positive three-kernel mixture remains characteristic. These are established
methods, not an invention claim. Minibatch-dependent bandwidths and the biased
V-statistic do **not** provide an unbiased fixed-kernel population gradient.
[MMD Gradient Flow](https://arxiv.org/abs/1906.04370) studies convergence under
additional conditions; those guarantees do not apply to this shared-network,
normalized GAN update. The attraction/repulsion analogy is explanatory only.

## Prior evidence and force audit

The archived [MMD preflight](https://github.com/255BITS/ParticleGAN/blob/0d52b2c8b4e985a7859ef7ac7f2f0c00b510379b/reports/toy100/lrfree-search/mmd-witness-preflight/README.md)
tested a single narrow Gaussian kernel with analytically integrated served
output noise on a different native100 source/cohort. Its paired full-versus-self
advantage was unresolved on rotated/staggered tasks; it did not authorize an
online candidate. Preserve that identity and refusal. This experiment changes
the kernel, scale support, actual clean MoG law, source and tasks explicitly.

[Alchemy](https://github.com/255BITS/ParticleGAN/blob/966711e0efa4b16c663ae8040390b05e34d72a7e/reports/forge/bcap-physics/alchemy/README.md)
improves covariance while losing two width modes and the passing broad guardrail.
[Transport-local](https://github.com/255BITS/ParticleGAN/blob/9507a9e5c471a1d3f80f49e895acafac730dbf21/reports/forge/bcap-physics/kinetic_transport/round2/README.md)
supplies a sustained unequal-mass repair while width and scalar stability fail.
Neither archived source is a third causal arm in this study.

[Saved evidence](saved-evidence.json) binds exact original winner and alchemy
checkpoints before candidate admission. It replays the actual final real batch
to its saved named-stream state and uses separately labelled deterministic
antithetic MoG cubature over every saved row for parameter derivatives. On the
original winner's width state, generator witness/critic raw norm is5.532,
attraction norm .742 versus repulsion .272; witness/critic cosine is−.170.
Prior ratio is2.953 and cosine−.345. Thus the long-range attraction is present
in parameter space; agreement with the critic is not guaranteed. This is a
CPU float64 diagnostic, not the exact last G sample, applied DualNorm proposal,
heldout causal benefit or qualification. No optimizer update was added.

The competing explanation is that adaptive batch scales, rare observations,
repulsion, shared-network compensation or normalized critic conflict can defeat
the useful target signal. Characteristic population identification alone does
not establish finite-budget allocation or local covariance correctness.

## Frozen forecasts, protocol and reservation

Primary prediction: final unequal-width `component_covariance_error`≤.85;
>.85 is its falsifier. Additional forecasts: unequal-mass final `min_mass_ratio`
≥.5, all target components represented, broad-vector full sustained PASS retained,
Gaussian independently confirmed smoke PASS, final stability `cdf_ks`≤.05.
Actual receipt keys and all original full gates stay fixed. A forecast does not
replace full covariance/eigenvalue/mass/quality/CDF or temporal checks.

Six unchanged tasks: two-pole, Gaussian smoke and its own-state stability,
unequal mass, unequal width and passing two-broad. Three vector allowances
1800s, Gaussian120+600s and two-pole300s give6420s per arm /12840s campaign.
The fresh14400s ceiling leaves1560s for diagnostics and real execution repairs.
Native100 cannot fit alongside this full subset and is unmeasured. No sweep,
second substantive candidate, seed repeat or automatic continuation is admitted.
Two-pole retains its explicit identity/zero/stored-weight cohort: the candidate's
new sample-space signal is genuinely unsupported in its frozen components host
and is BLOCKED before reservation; the winner control executes that original
fixture. No target/objective substitution is made.

Both arms use seed0, public deterministic initialization, unchanged architectures,
data laws, actual seen batch sequences, prior width/weights/components, sampling,
update budget and evaluation cadence. Constructor/data/training-noise/evaluation
streams stay isolated and checkpointed. The frozen vector/Gaussian adapter
actually reuses one real tensor for D/G each update; that law is retained.
Clean/live scoring is authoritative and distinct from noisy/EMA cohorts.
Original scorer controls are reused only with byte-verified scorer sources.

Ready [candidate](../../../../../configs/forge/ideas/kernel_witness_r3_v1.json),
[candidate study](../../../../../configs/forge/studies/kernel_witness_r3_candidate_v1.json),
[control study](../../../../../configs/forge/studies/kernel_witness_r3_control_v1.json),
and [diagnostic view](../../../../../configs/forge/views/kernel_witness_r3_diagnostic.json)
declare this exact scope. Both arms will be frozen from the same pushed source
before the bounded public Queue/drain runner executes one shared GPU worker.

Tail raw logs outside Git:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kernel_witness/queue/events.jsonl
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kernel_witness/logs/drain.log
```

The runner uses `on_completion=None`, `allow_sharing=True`, `watch=False`.
Publication will retain compact metrics, provenance and actual-training GIFs;
raw traces/checkpoints stay local. Summaries-only reporting preserves archived
qualification. Any later scientific software change is unmeasured unless
separately admitted under this round's remaining ceiling.
