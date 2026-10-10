# ParticleGAN optimizer types and update rules

ParticleGAN's BCAP experiments compared raw gradient steps, normalization at
different scales, Adam adaptation, and spectral DualNorm steps. The main
difference is **how each method turns a loss gradient into a parameter
displacement**. Full DualNorm with zero momentum is the current public BCAP
default; its selected pacing recipe records 4/6 required Tier 1 passes. The
Gaussian distribution-shape and ring component-covariance gates still fail,
so that API default does not establish calibrated scientific qualification.

Implementation snapshot: `develop` at
[`7183d65d`](https://github.com/255BITS/ParticleGAN/tree/7183d65db57c1ab3b8fc6d0833d8c8553e1e3b4b),
2026-10-06. The public choices and factories are in
[`Recipe`](../../particlegan/recipes.py); the seven normalized/hybrid choices
are implemented by
[`NormalizedOptimizer`](../../particlegan/optim/dualnorm.py).

## Losses and the common training loop

**BCAP is a critic-loss penalty.** Its squared, one-sided cap discourages
large critic gradients with respect to real and generated **inputs**. The
optimizer then acts on gradients with respect to model **parameters**.
Changing `optimizer_family` changes this second operation while preserving
the declared BCAP loss and task objectives. The canonical pure-BCAP recipe
uses the paired relativistic logistic objective and coefficient/cap 1.
See the [BCAP formulation](families/bcap-pure.md) and
[penalty implementation](../../particlegan/grad_regularizers.py).

`D` denotes the critic, `G` the generator, `E` an optional encoder, and `Z`
the learned prior-location table. `g` is the current parameter-loss gradient;
`eta` is the role's scheduled step size; `eps` is its numerical floor
(BCAP default `1e-8`). `norm` means the Euclidean norm after flattening a
tensor, or the Frobenius norm for a matrix. Both players minimize their
respective public losses; the adversarial sign is already encoded there.

```text
for each task-owned update:
    apply the declared schedules to role step sizes
    draw the task's critic batch and prior samples
    backpropagate critic_loss + BCAP using detached generator samples
    update D with the selected optimizer rule

    draw the task's generator batch and differentiable prior samples
    recompute scores through the updated D
    backpropagate generator/encoder_loss + task-owned auxiliary terms
    record actual generator-side sampled prior row IDs for row normalization
    update G, optional E, and the learned prior with the selected rule
    evaluate at the task's frozen cadence and serving law
```

This sketch follows the [public trainer](../../particlegan/training.py).
The `sgda` name does not change its alternating order: the generator uses the
updated critic. Joint hosts retain their explicit generator/encoder objective.

## Optimizer comparison table

The helper operations below define the table's pseudocode. Rates have the
units of their own algorithm; equal numeric rates across rows do not imply
equal displacements. All table entries descend the already computed loss.

| Optimizer and public selection | Where the rule applies | Update pseudocode | How it works and what it remembers |
| --- | --- | --- | --- |
| **Adam** `optimizer_family="adam"` | D, G/E and learned prior | `W -= eta * adam_direction(g)` | Smooths each coordinate's squared gradient and divides by its estimated scale; an optional first moment smooths direction. Stores first/second moments and a parameter step counter. BCAP's `(beta1, beta2)=(0, .999)` gives no first-moment smoothing. |
| **SGDA** `"sgda"` | All players | `W -= eta * g` | Raw gradient descent on each player's loss. Large gradients cause large steps. No normalization, momentum or Adam moment history. |
| **Global nSGDA** `"nsgda_global"` | One norm for D; one for G+E; one for the prior | `q_P = sqrt(sum(norm(g_j)^2 for j in player P)); W_j -= eta_j * g_j / (q_P + eps)` | Preserves the relative gradient sizes within a player while removing that player's overall magnitude. G and E share the denominator, even in separate groups; the prior is separate. No moment history. |
| **Layer nSGDA** `"nsgda_layer"` | Every parameter tensor independently | `W -= eta * g / (norm(g) + eps)` | Gives each tensor its own approximately fixed Euclidean step budget. A weight matrix and its bias are separate tensors. A prior table is normalized as one tensor, rather than row by row. No moment history. |
| **Adam-magnitude nSGDA graft** `"ada_nsgda"` | Every parameter tensor independently | `a = adam_direction(g, beta1=0); W -= eta * norm(a) * g / (norm(g) + eps)` | Uses the raw gradient's direction and the norm of Adam's unit-rate direction as its magnitude. Retains Adam's scale adaptation without its coordinatewise direction. Stores the raw-gradient second moment and step counter; requires `beta1=0`. |
| **Full DualNorm** `"dualnorm"`, momentum 0 | Network matrices/vectors; sampled prior rows | `matrix -= eta * c * polar(g); vector -= eta * g / (norm(g)+eps); row_step(Z, g, sampled_ids)` | Replaces matrix singular values by one, with shape factor `c`; normalizes biases/vectors; normalizes each sampled prior row independently. No Adam moments. |
| **DualNorm with momentum** `"dualnorm"`, `optimizer_momentum=.5` or `.9` | Network groups use momentum; prior rows keep momentum 0 | `M = mu*M + g; matrix -= eta*c*polar(M); vector -= eta*M/(norm(M)+eps); row_step(Z, g, sampled_ids)` | Accumulates network gradients before normalization, so history changes the direction. Stores one buffer per network tensor. The buffer is not bias-corrected and is distinct from Adam's `beta1`. Matrix zero guards still apply. |
| **Critic-only DualNorm** `"dualnorm_D_only"` | DualNorm on D; native Adam on G/E and prior | `D: dualnorm_network_step(W,g); G/E/Z: W -= eta*adam_direction(g)` | Isolates the critic update change. Optional DualNorm momentum applies to D. The other roles retain native Adam moments and their separately specified Adam rates. |
| **Prior-row-only normalization** `"particle_rownorm_only"` | Row normalization on learned prior; native Adam on D and G/E | `row_step(Z, g, sampled_ids); D/G/E: W -= eta*adam_direction(g)` | Isolates row magnitude and sampled-row ownership. Prior rows have no momentum or Adam moments; networks retain native Adam. Requires a learned 2-D table in its own group. |

The eight distinct `optimizer_family` values in this table consist of Adam
plus the seven values in `NORMALIZED_FAMILIES`; DualNorm momentum is a
configuration of the same optimizer. `optimizer_family="formulation"`
selects the additional Adam wrappers described below.

### Adam and the magnitude graft

For the unregularized dense Adam law used by the BCAP control:

```text
adam_direction(g):                 # compute direction at learning rate 1
    t += 1                        # per parameter with a gradient
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*g*g    # coordinatewise square
    v_used = v
    if amsgrad:
        v_max = maximum(v_max, v)  # coordinatewise running maximum
        v_used = v_max
    m_hat = m / (1-beta1^t)
    v_hat = v_used / (1-beta2^t)
    return m_hat / (sqrt(v_hat) + eps)

ada_nsgda_step(W, g):
    a = adam_direction(g)          # beta1 must be 0; m_hat = g
    W -= eta * norm(a) * g / (norm(g) + eps)
```

Native Adam uses PyTorch's implementation through the
[plain-Adam factory](../../particlegan/recipe_schedules.py).
The bias correction and optional AMSGrad maximum follow the
[PyTorch Adam algorithm](https://docs.pytorch.org/docs/main/generated/torch.optim.Adam.html).
The graft's scheduled rate is applied **once**, after taking the norm of the
unit-rate Adam direction. It does not normalize an Adam-preconditioned
direction, and its second moment still accumulates the raw `g*g`.
The [adaptive-method dissection paper](https://arxiv.org/abs/2210.04319)
provides motivation for separating direction and magnitude; the repository
screen has its own recipes, hosts and qualification scope.

### DualNorm matrix and prior-row steps

```text
dualnorm_network_step(W, g):
    M = g                         # with momentum 0
    if mu > 0:
        buffer = mu*buffer + g    # buffer starts at zero
        M = buffer
    if W is a matrix with shape (fan_out, fan_in):
        if norm(g) < eps or norm(M) < eps: return
        U, S, Vt = reduced_svd(M)
        c = sqrt(max(1, fan_out/fan_in))
        W -= eta * c * (U @ Vt)
    else:                         # vector or scalar
        W -= eta * M / (norm(M) + eps)

row_step(Z, g, sampled_ids):
    for i in unique(sampled_ids):
        Z[i] -= eta_prior * g[i] / (norm(g[i]) + eps)
    clear pending sampled_ids
```

For matrices, `U @ Vt` is the polar factor: it keeps the gradient's singular
vectors while giving every reduced-SVD singular direction unit magnitude.
This is a steepest descent direction under a spectral-norm step constraint,
before the shape correction. It does not constrain the weight matrix's
norm or impose a Lipschitz bound on the entire critic. For a nonzero
rank-deficient matrix, the reduced SVD includes completed unit directions;
the explicit zero guards avoid updating a near-zero matrix gradient.

The [polar-factor implementation](../../particlegan/optim/dualnorm.py)
uses reduced SVD when the largest matrix dimension is at most 1,024.
Larger matrices try 30 Newton–Schulz iterations, then fall back to SVD if
the orthogonality residual is nonfinite or exceeds `1e-3`. The network rule
accepts matrix weights and vector/scalar parameters; tensors with more than
two dimensions are rejected by DualNorm groups.

Prior ownership comes from actual generator-side draws through
`set_sampled_rows`, rather than inference from nonzero gradients. Multiple
draws are unioned and duplicate IDs update once. Unsampled rows stay fixed,
even if standardization or an auxiliary loss creates a dense gradient.
Zero-gradient sampled rows also stay fixed. The ordinary `direct_particles`
fixture is assigned to the **generator** player; it does not silently become
a sampled latent-table cohort. A deliberately direct table group may instead
explicitly disable the sampled-row requirement.

Only full DualNorm and `particle_rownorm_only` use this row rule. SGDA,
global/layer nSGDA and the magnitude graft apply their ordinary tensor rules
to the prior; `dualnorm_D_only` uses native Adam there.

## Adam variants and formulation wrappers

These paths exist elsewhere in the public API. Their extra mechanisms have
different scientific scopes from the optimizer-only BCAP sweep.

| API path | Step pseudocode | Description and scope |
| --- | --- | --- |
| **K3P critic Adam** `optimizer_family="formulation"`, effective K3P formulation | `apply_tensor_spike_guard(g); Adam.step(); update_active_anchor_EMA_and_step_record()` | Before Adam, an eligible critic tensor's gradient RMS is capped against its historical bias-corrected second-moment RMS. Canonical guard: ratio 5 after 200 previous parameter steps. The EMA anchor and loss penalty are formulation state, not a different Adam denominator. [Source](../../particlegan/k3p.py). |
| **K3P generator Adam** same formulation path | `prepare_eligible_A2_and_direct_response(); Adam.step(); restore_base_settings()` | Eligible sparse prior rows multiply Adam's response by `rho=.75+.25*cos(g,previous_g)`, in `[.5,1]`, with `rho=1` without history. A2 requires existing Adam state, some zero-gradient rows and cumulative observed-row fraction below the recipe cutoff. Raw second moments are preserved. Direct generated-particle groups may use dedicated moments and a coherence gain up to 2; ordinary networks use Adam. [Source](../../particlegan/k3p.py). |
| **KA2 critic Adam** `optimizer_family="formulation"`, effective KA2 formulation | `apply_tensor_spike_guard(g); Adam.step(); observe_moment_surprise(); update_controller_and_anchor()` | Shares the critic Adam/guard step, then tracks moment surprise and adaptive EMA/reseed state for the KA2 penalty controller. Its generator uses the shared K3P generator wrapper. KA2 is a formulation/controller choice. [Source](../../particlegan/ka2.py). |
| **TensorFlow-v1-style dense Adam** `optimizer_family="adam", adam_variant="tensorflow_v1"` | `m,v = moment_updates(g); alpha=eta*sqrt(1-beta2^t)/(1-beta1^t); W -= alpha*m/(sqrt(v)+eps)` | Adds epsilon before second-moment bias correction, so it differs from native PyTorch Adam when epsilon matters. Each optimizer has an application clock shared by its groups, advancing even when a parameter has no gradient. Used by the historical Halloween preset; it was not a BCAP optimizer-screen arm. [Source](../../particlegan/tensorflow_adam.py). |

Pure BCAP disables the spike guard, anchor, A2 damping and direct-particle
gain, as well as additive training noise and EMA serving. **BCAP with K3P**
retains K3P training machinery and is a separate formulation. Likewise,
E22/Atlas control training and serving through policies; their names do not
denote additional normalized optimizer rules. See the existing
[family descriptions](technique-inventory.md).

## Recorded BCAP experiment results

The initial screen froze protocol seed 0, public deterministic initialization,
one complete global configuration across tasks, matched architectures/data
laws/batch sequences/priors/sampling, update budgets and evaluation cadence.
Explicit fixed task fixtures retained their separate initialization contracts.
The trainer delta was the optimizer rule and its declared role rates.
Hybrid arms pinned native-Adam components to the baseline rates.
Clean/live scoring and six required tasks stayed fixed; the clock audit was
a separate diagnostic.

The following table summarizes the **historical arm selections** in the
[41-configuration screen](dualnorm-tier1/README.md), executed at source
`15eb7cb0911905e401bdfcd7e264945a7ea64d97`.
It preserves the recorded selections rather than combining task-specific
winners. The single current goal leaderboard remains
[`technique-inventory.md`](technique-inventory.md).

| Optimizer arm | Whole configurations | Selected required passes | Selected G / D / prior rates | Gaussian ms/step (GPU) | Ring16 ms/step (GPU) |
| --- | ---: | ---: | --- | ---: | ---: |
| [Adam](configuration-search/bcap-optim-adam-tier1-v1.json) | 5 | 3/6 | `.016 / .016 / .032` | 7.497 (0) | 8.575 (1) |
| [SGDA](configuration-search/bcap-optim-sgda-tier1-v1.json) | 5 | 2/6 | `.1 / .1 / .2` | 6.991 (1) | 8.018 (1) |
| [Global nSGDA](configuration-search/bcap-optim-nsgda-global-tier1-v1.json) | 4 | 3/6 | `.1 / .15 / .03` | 7.743 (1) | 8.822 (1) |
| [Layer nSGDA](configuration-search/bcap-optim-nsgda-layer-tier1-v1.json) | 4 | 3/6 | `.1 / .15 / .03` | 7.725 (0) | 8.190 (0) |
| [Adam-magnitude graft](configuration-search/bcap-optim-ada-nsgda-tier1-v1.json) | 4 | 2/6 | `.016 / .016 / .032` | 8.228 (0) | 9.329 (0) |
| [DualNorm, momentum 0](configuration-search/bcap-optim-dualnorm-zero-tier1-v1.json) | 4 | 3/6 | `.03 / .045 / .03` | 9.751 (0) | 14.096 (0) |
| [DualNorm, momentum .5/.9](configuration-search/bcap-optim-dualnorm-momentum-tier1-v1.json) | 8 | 2/6 | `.01 / .015 / .03`, selected `mu=.9` | 9.583 (0) | 13.584 (0) |
| [Critic-only DualNorm](configuration-search/bcap-optim-dualnorm-d-only-tier1-v1.json) | 4 | 2/6 | `.00425 / .15 / .0085`, D `mu=.5`; G/prior use Adam | 8.428 (1) | 11.229 (1) |
| [Prior-row-only normalization](configuration-search/bcap-optim-particle-rownorm-only-tier1-v1.json) | 3 | 3/6 | `.00425 / .00425 / .01`; G/D use Adam | 7.674 (1) | 8.645 (1) |

Saved receipts mark FLOPs as `unavailable`, so the cost columns use recorded
**mean milliseconds per full training step**. For each task in the selected
trial, read `cost.phase_timing.phases.training_updates` and compute
`ms_per_step = 1000 * seconds / calls`. Gaussian has 1,000 measured calls;
Ring16 has 400. Each call completes one D update and one G/prior update,
including forward/backward passes and the BCAP penalty. The
[synchronized phase timer](../../experiments/forge/telemetry.py) surrounds
the [public training call](../../experiments/forge/adapters.py), including
its statistics and attached diagnostic hooks; it excludes real-input
preparation, evaluation, evaluation sampling and process startup.

The linked search records bind the exact selected candidates, task attempts
and timing totals. Their runtime cohort records NVIDIA RTX A6000 CUDA
devices, PyTorch 2.14.0, one Torch thread, deterministic execution and TF32
disabled; parentheses give each task's GPU index. These are averages from
the original instrumented executions, without repeated timing trials or
uncertainty estimates. The campaigns used two GPU workers, and their
protocol treats wall time as cost evidence without a speed ranking. Keep
the source cohorts separate when interpreting the measurements.

None of the new optimizer arms exceeded Adam's 3/6 required-pass count in
that source cohort. Five SGDA numerical failures remained `INCOMPLETE`,
with no pass credit. The source-bound analysis is in
[`dualnorm-tier1/analysis.json`](dualnorm-tier1/analysis.json).
The unchanged incumbent Adam control used `.00425 / .00425 / .0085` and
also passed 3/6; its Gaussian/Ring16 costs were **7.752 / 9.257 ms/step**,
both on GPU 0. The table shows the search-selected Adam configuration.

The owner chose a different tied zero-momentum DualNorm recipe,
`.01 / .015 / .03`, as the next experimental starter because it retained
word acquisition and improved ring HQ, despite failing the complete ring
gate. That choice did not overwrite the initial search's selected `.03`
configuration. The later
[25-configuration pacing study](dualnorm-pacing-v2/README.md) ran at source
`a0f7e70e50427e0d3221d1d7f4cb4aac6e18b1be`. Its matched control was this
DualNorm starter, rather than a fresh Adam comparison.

The later selected recipe uses **G/E `.012`, D `.018`, prior `.03`, and
momentum 0**, and improves from 3/6 to **4/6**. Two-pole gains a sustained
pass; unused-token hold, AE hold and joint-word acquisition retain theirs.
The remaining failures are Gaussian CDF KS **`.11428 > .05`** and ring full
assigned-component covariance error **`9.61552 > .85`**. Ring still has
16/16 modes and HQ `.94385`; those metrics cannot replace its failed shape
gate. Required passes need at least five consecutive terminal passing
evaluations, rather than a passing endpoint alone. These measured results
remain attached to their original recipe, task and source bindings.

The later pacing cohort's recorded step costs are below. These rows use the
same timing definition and task budgets, and the exact control/winner trials
in [`dualnorm-pacing-v2/results.json`](dualnorm-pacing-v2/results.json).

| Pacing recipe | G/E / D / prior steps | Gaussian ms/step (GPU) | Ring16 ms/step (GPU) |
| --- | --- | ---: | ---: |
| Matched DualNorm starter, momentum 0 | `.01 / .015 / .03` | 9.431 (0) | 13.299 (0) |
| Selected DualNorm winner, momentum 0 | `.012 / .018 / .03` | 9.497 (1) | 13.122 (1) |

## Choosing and configuring an optimizer

Use the current BCAP preset to recover the selected zero-momentum DualNorm
settings, or the explicit Adam preset for the earlier pure-BCAP control:

```python
from particlegan import get_recipe

dualnorm = get_recipe("bcap")       # G/E=.012, D=.018, prior=.03, mu=0
adam = get_recipe("bcap_adam")      # G/D=.00425, prior=.0085, betas=(0,.999)

# Resolve another update rule while retaining the pure-BCAP loss/settings.
# Explicit rates below illustrate configuration; they do not claim a result.
sgda = get_recipe("bcap", optimizer_family="sgda", lr=.01,
                  d_lr_mult=1., prior_lr_mult=2., optimizer_momentum=0.)
```

For full normalized methods, `lr` sets the G/E step,
`lr*d_lr_mult` the D step, and `lr*prior_lr_mult` the prior step before the
schedule multiplier. In hybrid arms, `optimizer_adam_lr` supplies the base
rate for native Adam groups, using the same role multipliers; `lr` supplies
the normalized groups' base step. Declare both bases explicitly when isolating
a role. Use `recipe.make_optimizers(...)` and the public trainer to preserve
role ownership, sampled IDs and checkpoint semantics.

Full DualNorm at the selected pacing is the supported starting point for
current BCAP work under the owner's
[default-selection decision](dualnorm-pacing-v2/DEFAULT_SELECTION.md).
Use SGDA to understand raw gradient magnitude, global/layer nSGDA to
separate player and tensor scale effects, and the hybrids to identify which
role's update rule matters. The current evidence does not support positive
shared momentum or the magnitude graft as an improvement over the whole
baseline recipe.

Before spending on another comparison, inspect the saved Gaussian CDF
errors and ring assigned tails. The existing failures concern distribution
shape even where location, coverage or HQ passes. Restore archived runs
from their saved resolved `Recipe` fields and original source; today's
`get_recipe("bcap")` default does not redefine their Adam or DualNorm
initialization, prior, budget or serving contracts. Scientific calibration,
independent confirmation and width/depth transfer remain separate work.
