# Round3: multiscale pairwise distribution witness

**Reject this exact global weight1 revision.** It preserves the broad-vector
guardrail and improves scalar retention and mass allocation, but adds no sustained
task PASS and worsens unequal-width full covariance **6.287559→13.431491**.
Candidate outcomes are **2 PASS / 3 FAIL / 1 BLOCKED**, versus the exact winning
BCAP primary control's **3 PASS / 3 FAIL**. On five mutually executable tasks,
both pass2. Eleven full-budget workers finish once for **324.642767 paid seconds**,
with zero retries, incomplete/invalid workers or remaining reservations.

This completed round3 mechanism diagnostic supplies no ordinary Tier2
qualification, calibrated screen or default-adoption credit. The parent owns
the single current goal leaderboard; the following table is a task readout.
The original winner's archived7/21 Tier2 identity remains unchanged.

## Complete unchanged task gates

| Task | Exact winner primary control | Kernel witness candidate | Measured outcome |
| --- | --- | --- | --- |
| Two-pole fixed fixture | PASS,17/24 checks | BLOCKED before spend | Frozen public-components host has no sample-space witness consumer; no fixture substitution |
| Gaussian smoke | PASS,3/24 confirmed states | PASS,9/24 confirmed states | First confirmation375→125; both endpoint KS values fail, .071842→.074019, while the original any-confirmed-state smoke gate passes |
| Gaussian own-state stability | FAIL: stationary2/72, shifted hold0/24, deadline reacquisition FAIL | FAIL: stationary40/72, shifted hold17/24, deadline reacquisition PASS | Final KS .320623→.070755 still exceeds .05; full temporal retention remains FAIL |
| Unequal mass | FAIL,0/24 checks, suffix0 | FAIL,0/24 checks, suffix0 | Minimum mass ratio .207520→.969460; full covariance3.691653→3.890876 still fails |
| Unequal width | FAIL,0/24 checks, suffix0 | FAIL,0/24 checks, suffix0 | Full covariance6.287559→13.431491; narrowest component44.654877, despite all four populated modes |
| Two broad guardrail | PASS,22/24, suffix22 | PASS,23/24, suffix17 | Covariance .385581→.135281, mass TV .062988→.003418; full guardrail retained |

[Final metrics and temporal failures](results.json),
[source/recipe/stream receipts](provenance.json),
[unchanged scorer controls](scorer-controls.json),
[saved-center and actual batch replay](saved-state-diagnostics.json), and
[11 actual-training GIF receipts](media.json) bind every task. The
[unequal-width training GIF](candidate-vector_unequal_width.gif) illustrates the
numerically measured failure. All saved Gaussian/vector metric sets were rescored
exactly before rendering; media adds no optimizer update or sampling draw.

### Mass and central shape do not bound spill

The candidate's width counts are915/824/1073/1284 and mass TV .075439, with
minimum full eigen ratio .695481. The four-sigma core covariance average improves
.444178→.259211, yet the narrowest component's spill fraction rises0→.109290.
Its full covariance error rises .630566→44.654877. Thus central shape and populated
modes coexist with a severely wrong complete narrow-component law; the original
full gates reject it at every observation. Core summaries do not replace them.

Unequal-mass rare served count improves17→113 out of4096, against target mass .02.
All components are represented and mass TV falls .070654→.016797. Its third
component still has full covariance error8.927802 and spill .099631. None of the
24 complete mass-task observations passes. This is allocation improvement, not
transport-local's previously demonstrated sustained rare-density repair.

### The endpoint witness is present and can agree with the critic

[Final parameter-force diagnostics](final-witness-diagnostics.json) reuse the
reconstructed actual last real batch and explicitly separate all-row antithetic
MoG cubature. Width empirical MMD falls .087147→.029458, while full covariance
worsens. Candidate generator witness/critic cosine is .983449, prior .722043;
the witness/critic norms are1.194 and1.238. Attraction remains stronger than
repulsion in both groups. Persistent endpoint critic opposition or absent
attraction is therefore unsupported by this probe. These are raw float64
parameter derivatives, not actual last G samples, applied DualNorm proposals,
population MMD or a causal trajectory decomposition.

A characteristic bounded kernel identifies an exact population equality but does
not bound relative second-moment error at a finite nonzero discrepancy. For a
fixed real-derived frame, Q=(1−epsilon)P+epsilon*delta_R gives MMD²≤4 epsilon²
because k(x,x)=1, while an extreme R can make variance error grow as epsilon R².
This mathematical counterexample explains why the full spill/covariance gates
remain necessary; it does not identify the precise training cause. Adaptive
scales, minibatch rare evidence, shared-network deformation and normalized
finite motion remain competing explanations. This experiment does not isolate
the kernel family, each scale, weight, adversarial interaction or optimizer.

The primary covariance forecast is falsified. Rare-mass≥.5, populated modes,
broad full PASS and confirmed Gaussian smoke forecasts are observed; final
stability KS≤.05 is falsified. Candidate aggregate study decision stays
`incomplete` solely because the declared two-pole host is unsupported, with its
observed numerical falsifier true. Every runnable worker is complete. The control
study's same primary signature is falsified; it does not regrade archived winner
qualification. Frozen declarations and forecasts are preserved.

**Recommendation: stop this exact global weight1 candidate and retain the winner.**
Keep the opt-in implementation and negative evidence. Allocation and scalar
response are useful observations, but no task-specific improvement supplies a
new global configuration. Any future proposal needs a separately bounded scope
and saved-state evidence about tail-sensitive applied forces or network/prior
deformation; no sweep, seed study, tuning, continuation or adoption follows here.
Native100, images, conditional identity tasks and the full Tier2 suite are
unmeasured; the old native/noisy MMD preflight remains a different cohort.

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
declare this exact scope. [PR365](https://github.com/255BITS/ParticleGAN/pull/365)
was opened as a draft after scientific commit
`73200f450b10e428dc74ee70842587013f6f13fd` was pushed, before both arms were
enqueued. Both executed source digest
`576340bea8c650ecdf2ff0bbc0702a8ce8601c6c0ca177b3894aaeaa97751398`;
all1201 measured scientific source files match publication bytes. Candidate
revision is `690dad0a2bf4c00ef53fe9cf82934d305ead00a9d36ede8486d250b7d0fc1b5b`,
control revision `df1f0ed1772a9d01b3e39216ac746529d93263965682be987a5fb230e104a348`.
Later commits contain reporting/reproduction artifacts only; no later scientific
fix is presented as trained.

The consumed recipe delta is exactly `kernel_witness_weight:0→1` on each matched
host. Initial-model receipts, priors, actual data sequences and final named-stream
states match across both arms. All six control endpoint metrics, observation
counts, passing suffixes and grades exactly match the separately archived alchemy
winner control; this is parity, not an independent seed replication or third arm.
Archived receipts and original source identities remain intact.

The campaign declares12840s; actual full executed allowances total12540s because
the unsupported candidate fixture spends nothing. Paid worker cost324.642767s
is below the fresh14400s ceiling. Read-only preflight/final force diagnostics add
15.840310/13.047172 CPU seconds separately from the worker ledger; these fit within
the1560s diagnostic reserve even when conservatively added to full allowances.
No capacity probe, infrastructure retry or additional training was launched.
GPU0 sharing was authorized with one active scientific worker; contended wall
cost is accounting, not an optimizer-speed ranking. The bounded drain stopped,
both request subscriptions are concluded by normal readout, and no watcher or
active reservation remains.

159 focused public trainer, derivative, null/destructive, equivariance, checkpoint,
stream, legacy-recipe, mechanism-boundary and Forge study checks pass. Declaration,
media and source verification pass; Forge memory is CURRENT and the history catalog
has valid coverage. [Verification receipt](verification.json) records these checks.
GitHub reports no configured checks on this research branch. Logs and focused-check
output stay outside Git.

Tail raw logs outside Git:

```sh
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kernel_witness/queue/events.jsonl
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-physics-round3-20261009/kernel_witness/logs/drain.log
```

The runner used `on_completion=None`, `allow_sharing=True`, `watch=False`.
Publication retains compact metrics, provenance and actual-training GIFs;
raw traces/checkpoints stay local. Summaries-only reporting preserves archived
qualification. Reproduce reporting after restoring the exact bulk receipt paths
and original linked archive metadata:

```sh
PYTHONPATH=$PWD /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/kernel_witness/round3/publish.py
PYTHONPATH=$PWD /home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/bcap-physics/kernel_witness/round3/diagnose_final.py
```

These commands train nothing. The concluded studies must not be reset for an
unchanged scientific rerun. The pushed preregistration preserves the original
forecasts, exact global recipe, applicability and reservation decision.
