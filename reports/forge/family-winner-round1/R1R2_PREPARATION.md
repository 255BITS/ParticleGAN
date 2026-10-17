# R1/R2 Modern GAN: round-one preparation

Eight new shared configurations are prepared; **no training has started**. The
[grid](../../../configs/forge/searches/r1r2-modern-family-round1-v1.json) crosses
cosine decay starts 0.2/0.6, matching network/prior floors 0.10/0.25, and critic
LR multipliers 0.5/1.0. LR stays 0.0085. Every configuration retains plain Adam,
beta1 zero, beta2 0.9→0.99 over the first 20%, and R1/R2 gamma 1→0.1 over the
same interval. All eight resolve through public Recipe factories and differ
from every configuration in the prior four-trial grid.

The numerical [preparation index](r1r2-grid-preparation.json) lists complete
content-addressed cards, settings and resolved-recipe hashes in config-hash order.
Each card is shared across the entire fixed view: 3 smoke, 19 quality and 2
endurance requirements. Each toy retains its own weights, resources, frozen
objective and declared prior exception. No architecture, initialization, seed,
measurement, horizon or serving changes enter this grid.

## What the historical failure supports

The [prior readout](../R1R2_CONFIGURATION_SEARCH_READOUT.md) records four
constant-rate trials. Three stop at `two_pole`: insufficient movement at
LR0.00425/gamma1→0.1, a short passing suffix at LR0.00425/gamma0.1→0.01, and
excess slope at LR0.0085/gamma0.1→0.01. LR0.0085/gamma1→0.1 passes all three
smoke tasks, then fails `trajectory` after 400 updates: identity MSE
**0.23911647498607635 > 0.02**, with zero passing observations out of 24.
Eighteen further quality tasks and both endurance tasks remain unmeasured.

The failed trajectory optimizes a conditional relativistic pair objective plus
set coverage, fixed latent L2 and spread. Set coverage alone cannot establish
row identity. The old retained aggregate metric does not reveal whether the
model swapped identities, ignored conditioning, oscillated, or failed for another
reason. The archived original envelope named by the
[manifest](../r1r2-modern-toy-archive.json) is currently unavailable locally;
the historical behavioral runner exposes no model checkpoint in its result.
These gaps prevent a saved-output causal diagnosis. No unchanged failed recipe
is rerun to recover it.

Late rate decay is a new, bounded optimization hypothesis: retain the movement
rate needed by smoke, then reduce later parameter steps. Half-rate critics test
a role imbalance independently of the decay schedule. Neither factor is a
demonstrated historical cause. The full-budget terminal and sustained predicates
will reject configs that merely acquire a transient good state or freeze too soon.

## Q1: exact trajectory representation

The actual host takes 16 slow coordinates and 4 latent coordinates, uses two
64-wide LeakyReLU(0.2) hidden layers, and returns 16 fast coordinates. Each
frame's target is a rotation of its slow frame by `(2.2 − 0.45) * t`. Put the
positive and negative input coordinates in 32 hidden channels, pass those
channels through the second layer, and combine them with the rotation matrix
divided by `1 + 0.2²`. The identity
`lrelu(lrelu(x)) − lrelu(lrelu(−x)) = (1 + 0.2²) * x` realizes that linear map.
All latent-input weights are zero. The original 12-row cloud and public
scheduled-serving wrapper are retained; this grid declares zero output noise.

The [numeric certificate](r1r2-trajectory-representation.json) executes the actual
host class and public component binder with **zero optimizer or fitting updates**.
All 12 required identities at all 24 schedule labels have MSE
**6.007302081045005e-15**, maximum coordinate error **4.6193599700927734e-07**.
The unchanged-slow control fails at MSE0.244148015976; swapped identities fail at
1.063333392143; collapse fails at 0.265833348036. Model, prior and optimizer states
remain unchanged during evaluation, and unintended RNG deviations are zero.

This supports representation at the exact finite task tolerance. The 24 checks
evaluate one immutable analytic state; they are explicitly **not training
checkpoints, ordinary qualification, acquisition timing or a learned solution**.
The certificate binds host/task/evaluator/native bytes and the actual public
recipe/prior/sampling law. Reuse across trials requires those identities and the
zero-noise serving law to remain compatible.

```sh
python reports/forge/family-winner-round1/r1r2_representation.py \
  --output /tmp/r1r2-trajectory-representation.json
```

The complete shared-suite representation status remains **UNRESOLVED** until
source-compatible witnesses for the other required hosts are joined. Historical
transpose12 supervised controls support the direction of investigation for
stripes, bars and intensity; the historical blobs GAN positive supports its
direction. Their parameter/source compatibility has not yet been certified.
Residual16 successes cannot fill original transpose12 representation cells.
Stagewise supported witnesses may admit the exact supported next task; missing
proof stays visible and does not become an exclusion or pass.

## Execution boundary and budgets

The immutable task reservation ceiling is **44,100 seconds per config**,
**352,800 seconds for eight configs**. This is the full-task worst-case reservation,
not measured cost. Execution also requires the frozen shared round allowance,
stagewise representation readiness and root-owned GPU slot. There is no automatic
launch from this preparation. Ordinary non-passes stop that config, and every
survivor advances through the same later requirements while the next complete
allowance fits. No mixture of per-toy or per-tier configs can qualify a family.

Common source and timing/selection behavior are still being frozen. The final
execution cohort must bind that source, task/protocol identities, actual backend,
resource policy and serving law. The current screen is provisional. A passing
search cannot promote public defaults without accepted calibration and separately
registered confirmation/robustness. No historical CPU result is relabeled as new
CUDA qualification, and no speed claim follows from an endpoint-only receipt.
