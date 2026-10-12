# Full balanced transport: feasible software, no ready paid recipe

**Full output-space OT is a distinct hypothesis worth keeping; a blind Sinkhorn
replacement is not ready for registration.** Dense matrices fit the unchanged
hosts, but current evidence fixes neither a universal entropy/blur scale nor a
safe loss normalization. This review selects no coefficient, implementation or
paid pilot. It follows [compiled memory](../../EXPERIMENT_MEMORY.md), the closed
[objective comparison](../README.md), [saved density](../density-readout.md), and
[prior transport review](../../r1r2-direction-floor-repair/transport/global-sliced-review.md).

The original source `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`
and execution `2b0ad9ec7c5b2c1e91ab0d70751da0b7e36a7b42` retain their failures.
Global sliced weight 1 gave Ring covariance 2.352042, HQ .760498 and no covariance
passes; words acquired once, then lost endpoint quality. Local weight .1 also
failed Ring. These reject those particular recipes, not every transport loss.

## Mechanism and two different biases

For existing fake rows x and detached real rows y, preserve uniform empirical
masses and solve the complete cost matrix C_ij=||x_i-y_j||², with P1=a and
Pᵀ1=b. Full OT retains simultaneous coordinate geometry that 32 separate sorted
projections discard. No component IDs, target centers, evaluation samples,
conditioning coordinates or learned critic features enter the cost.

If entropy is used, the relevant loss is the **debiased Sinkhorn divergence**
Sε(x,y)=OTε(x,y)-½OTε(x,x)-½OTε(y,y), not raw OTε. Raw entropic OT biases fitted
support toward shrinkage; the self terms correct that entropic bias. They add
no draws, and fake/fake derivatives must include both arguments. This does not
make the finite-batch estimate equal to population W2 or eliminate independent
batch sampling effects. [Feydy et al., 2019, §§1.2–1.3](https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf).

A useful inference from the unchanged Ring law is that correct independent
128-row, 16-mode uniform panels still have expected empirical mode-mass TV
**.1917003214**. Balanced coupling must send at least that mismatch across mode
identities on average. For correct 256-row/five-word panels it is **.0704504420**.
These are analytic batch-count calculations, not observed training forces or
lower bounds on any final gate. Formula: k/(2n) Σ_iΣ_j |i-j| Bin(n,1/k)_i
Bin(n,1/k)_j. No labels enter proposed training. Full-batch here means the
existing training batch, not the population. Minibatch OT remains a different
objective even after entropic debiasing. [Fatras et al., 2020, §3](https://proceedings.mlr.press/v108/fatras20a/fatras20a.pdf).

## Actual hosts and ownership

Batches are **not uniformly 256**. Frozen source/configurations give:

| Host | Rows × output coordinates | Updates / timeout | New-loss ownership and constraint |
| --- | --- | --- | --- |
| Gaussian | 128×1 | 1,000 /120s | G and actually sampled learned-prior rows. Exact 1D OT already equals sorted squared matching; no new geometry here. |
| Ring16 | 128×2 | 1,600 /300s | G and sampled prior; only about 8 target rows per component. Full coordinate coupling is the distinct hypothesis. |
| Words | 256×168 | 20,001 /900s | Whole probability-output rows, G and sampled prior. E retains the original joint/inverse objective; latent coordinates excluded. Softmax saturation can still suppress any output force. |
| AE hold | 64×2 | 250 /300s | Existing generated decoder branch, G/prior; original E reconstruction and paired correctness retained. |
| Direct pole | 12×1 | 80 /300s | Original generator-owned coordinate parameter; no sampled prior. Fixed zero initialization and symmetric real panel remain explicit. |
| Token hold | 8×2 | 200 /300s | Repeated original trained-concept output only; unused slot and its original hold objective excluded from transport. No sampled prior. |
| Clock | 8×2 | 24 /300s | Original G/prior batches; solver must not depend on step labels, horizon or evaluation cadence. |

Sources: [task cards](../../../../configs/forge/tasks),
[word fixture](../../../../benchmarks/toy_audit/api_images.py),
[component consumers](../../../../experiments/forge/behavior_adapters.py) and
[clock runner](../../../../experiments/forge/clockfree.py). Target tensors and any
target-derived scale stay detached. Initial prior distribution/capacity, actual
ordered batches, sampling, updates, initialization and gates remain unchanged.
Positive paired R1/R2 penalties continue every update. Changing G/prior gradients
is an explicit recipe objective delta, not a task or prior-distribution change.

## Scale, symmetry and finite compute remain unresolved

GeomLoss distinguishes blur from optimization tolerance: for its half-squared
cost ε=blur². Its .05 default assumes a unit square/cube, not these mixed hosts.
There is no evidence that this default resolves Ring component width. A
batch-spacing rule also needs a declared response to repeated points: token
has zero target variance; words have many exact duplicates. Reusing the current
variance-normalized sliced denominator with an arbitrary entropy floor would
hide a new scale choice. Log-domain solves, marginal residuals and gradient
accuracy must be checked; truncating an unconverged plan does not establish the
intended loss. [GeomLoss API](https://www.kernel-operations.io/geomloss/api/pytorch-api.html)
and [epsilon-scaling guide](https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_epsilon_scaling.html).

There is a distinct pole risk: a smooth balanced Sinkhorn loss gives zero
transport gradient when every fake coordinate is identically 0 and real targets
are symmetric. The unchanged adversarial objective can still move them, but
transport does not initiate separation. Sorted 1D/exact assignment can select
nonzero directions at that nonsmooth tie; deterministic tie handling becomes
part of the recipe and must not substitute a different initializer. Token's
repeated-point loss instead reduces to squared displacement; its zero-variance
scale case needs a generic mathematical treatment.

A 256² float32 matrix is 256 KiB; cross plus two self matrices are 768 KiB before
workspace/autograd. Form costs without materializing an N×N×D broadcast. Memory
therefore looks practical; this is an operation-count inference, not a runtime
measurement. The saved sliced word run costs 132.53455s against 900s, leaving
about 38.37ms of average extra work per generator update as a rough ceiling.
Solver iteration count, small-entropy conditioning, backward memory and CPU/GPU
synchronization could consume it. No timeout or update allowance may be reduced.
Generative Sinkhorn research explicitly finds accuracy/runtime dependent on
entropy and iterations; published successes do not certify this budget or
R1/R2 game. [Genevay et al., 2018, §4](https://proceedings.mlr.press/v84/genevay18a/genevay18a.pdf).

## Recommendation

Do not admit a Sinkhorn arm with borrowed entropy/blur defaults. **Exact
unregularized balanced OT is the clean entropy-free alternative**: uniform
same-sized batches permit assignment, with a deterministic selected-plan
cost derivative almost everywhere. It still has batch bias, nonsmooth ties,
normalization/weight questions and runtime risk. POT's exact solver demonstrates
cost-matrix gradients but uses CPU network simplex and copies GPU inputs;
that supplies feasibility, not a timing guarantee. [POT implementation](https://pythonot.github.io/_modules/ot/lp.html#emd2),
[POT API](https://pythonot.github.io/all.html#ot.emd2).

Before a paid hypothesis, require one explicit global loss/scale law, duplicate
and pole-tie semantics, numerical transport/gradient checks and resource evidence
within the original allowances. Pure synthetic solver checks are software
prerequisites, not task qualification or permission to run a pilot. Preserve
original adversarial/reconstruction/hold objectives and all consumed RNG streams.
A reusable opt-in public mechanism is required: replacing the existing sliced
method in place would rewrite its meaning. New scientific source must be frozen
and held across a future comparison; old source-bound certificates cannot be
silently credited as new-source results. This note creates no candidate,
registration, queue change, optimizer update, model construction or random draw.
