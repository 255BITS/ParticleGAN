# Energy-U public API binding plan

**A separate opt-in output objective can cover the seven unchanged hosts. It is
not implemented or admitted. After reviewing the software prerequisite, root
selected raw coefficient 1 for one future comparison against a fresh-source
gamma .1 non-saturating AMSGrad m110 control.**
This maps the law in [next-density-mechanism.md](next-density-mechanism.md), after
reading [compiled memory](../../EXPERIMENT_MEMORY.md) and the closed
[weaker-penalty readout](../README.md). A possible future parent is the measured
gamma **.1** non-saturating AMSGrad m110 recipe, not the failed gamma .01 arm.
Code references below describe scientific source
`25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`, inspected at
commit `af839b06668db7617bd460262bf38860fd3a9671`.

## Minimal public surface

Propose one field, **`Recipe.energy_distance_weight: float = 0.0`**, and one
weighted method, **`Recipe.energy_distance_loss(fake, real)`**. Follow the
existing [output-loss field](../../../../particlegan/recipes.py#L115),
[finite/nonnegative validation](../../../../particlegan/recipes.py#L447) and
[method](../../../../particlegan/recipes.py#L661) patterns; reject Boolean,
negative and nonfinite weights. Add a small stateless `energy_distance_u`
primitive in a dedicated public-package module. Do not overload the meaning of
either existing kinetic loss. No bandwidth, projections, schedule, normalizer,
feature encoder or task-specific switch is needed.

The primitive flattens output coordinates per row, detaches real targets and
implements the **raw Euclidean**, signed U-statistic in the research note. It
excludes diagonal indices, retains duplicate-valued off-diagonal rows and
differentiates both fake arguments. Its prescribed derivative is zero at exact
collisions; no squared norm, negative-value clamp, jitter or denominator floor.
Require floating, finite panels with matching feature shape/device/dtype and
at least two rows in each panel. A reusable primitive may accept unequal n,m;
these hosts retain their original equal-cardinality panels. Avoid an n×m×d
broadcast. The [reviewed derivative/resource probe](energy-distance/README.md)
uses direct-distance `cdist` and explicitly detaches zero-distance contributions;
software changes still require separate authorization.

Zero weight must skip the helper, consumer construction and added diagnostic
keys entirely. Positive weight adds `weight * U` to the existing G-phase total
exactly once; it does not replace GAN/joint/reconstruction/hold losses or become
a protected objective. The selected future contrast is **raw weight 1 versus
0**, globally, with no normalization or grid; raw output units differ across
tasks and this selection establishes no trained repair.
The sole recipe delta is energy weight 0→1: paired R1/R2 remains gamma .1 every
update, with the parent's G/E objectives, AMSGrad roles, learning rates, prior
sampling/update policy and prior regularization held fixed.

## Consumers and ownership

| Unchanged host | Existing insertion point; energy binding |
| --- | --- |
| Gaussian 128×1; Ring 128×2 | [GANTrainer._step](../../../../particlegan/training.py#L383), lines 383–410: use the existing `fake_g` and resolved `real_g`, alongside the optional output losses. The existing G/prior sample and target callback are consumed once. Gradients reach G and its sampled learned prior; D remains frozen. |
| Direct pole 12×1 | [two_pole.py](../../../../benchmarks/locked_shared/two_pole.py#L142), lines 142–157: existing `add_transport_loss(g_loss, generated, real)` dispatches energy on direct sample coordinates. They remain generator-owned, without a latent sampler; no target labels enter the helper. The exact symmetric zero-origin panel has zero auxiliary force, so this adds no pole-separation claim. |
| Token 8×2 | [unused_token_hold.py](../../../../benchmarks/locked_shared/hosts/unused_token_hold.py#L274), lines 274–284: use only the repeated trained-concept `fake_g` and `real` panel. Preserve the unused-slot hold and original adversarial/feature objectives; no prior sampling is introduced. |
| AE 64×2 | [ae_gan_hold.py](../../../../benchmarks/locked_shared/hosts/ae_gan_hold.py#L251), lines 251–268: use `generated, data`, already forwarded for the original generated branch. Preserve reconstruction, encoder and cover losses; do not substitute reconstructed outputs. Energy follows the generated decoder/prior graph. |
| Words 256×168 | [WordFixture.step](../../../../benchmarks/toy_audit/api_images.py#L1159), lines 1159–1168: add energy on `fake_words.flatten(1), real_words.flatten(1)`. Keep actual softmax probabilities and duplicate-word frequencies; exclude the latent/joint coordinates. G and learned prior receive the auxiliary gradient; E receives only its original joint objective. Preserve the zero-width prior exception, both samples and original D penalty. |
| Own clock 8×2 | [run_clockfree](../../../../experiments/forge/clockfree.py#L108), lines 108–118: it already builds GANTrainer and calls `trainer.step(real)`. The same enabled objective must execute there. No step label, horizon, cadence, clock or extra stream is an input to energy. |

The component insertion points already converge on
[BehaviorComponents.add_transport_loss](../../../../experiments/forge/behavior_adapters.py#L449),
lines 449–458. Keep that existing host hook and target receipt; dispatch energy
to a **separate signed consumer** rather than changing archived transport loss
semantics. The other existing component hooks also reuse this dispatcher:
[residual_student:282](../../../../benchmarks/locked_shared/hosts/residual_student.py#L282),
[unipolar:325](../../../../benchmarks/locked_shared/hosts/unipolar.py#L325),
[cover_leftover:512](../../../../benchmarks/locked_shared/hosts/cover_leftover.py#L512)
and [mid_scale_identity:517](../../../../benchmarks/locked_shared/hosts/mid_scale_identity.py#L517).
Their conditioning is checked but excluded from distances; original paired
objectives remain responsible for correspondence. A host without an exercised
output hook remains BLOCKED when energy is requested; an allowlist name alone
does not establish support.

## Eligibility, receipts and checkpoints

The four ordinary component tasks already declare `output_marginal_v1` and an
output-space/ownership/no-new-draw contract, rather than a particular estimator:
[pole:32](../../../../configs/forge/tasks/two_pole.json#L32),
[token:29](../../../../configs/forge/tasks/unused_token_hold.json#L29),
[AE:27](../../../../configs/forge/tasks/ae_gan_hold.json#L27),
[words:59](../../../../configs/forge/tasks/five_word_joint_smoke.json#L59).
Keep these task files byte-identical. Explicitly extend the **fresh source's**
capability checks in [task_policy_blockers:196](../../../../experiments/forge/api.py#L196)
and [FormulationContext:380](../../../../experiments/forge/api.py#L380) to require
this existing declared output contract for positive energy weight; do not
silently enable undeclared component hosts. Update
[WordFixture construction:1067](../../../../benchmarks/toy_audit/api_images.py#L1067)
separately: it does not use BehaviorComponents' dispatcher.

Declare the weight as recipe/technique-owned alongside
[the existing output weights](../../../../experiments/forge/boundaries.py#L34),
not a task field or automatically searchable grid knob. The registry rejects
unowned new fields at [boundaries:65](../../../../experiments/forge/boundaries.py#L65).
Its positive value then enters the existing
[technique signature](../../../../experiments/forge/techniques.py#L23); activation
is a structural idea rather than a same-technique hyperparameter trial.

Introduce a distinct `output_energy_u_v1` consumer receipt under optional
`component_energy`, with explicit weight, nonnegative call counters and
**finite signed** loss sum/extrema. The existing [OutputMarginalTransport loader:46](../../../../particlegan/conditional_transport.py#L46)
rejects negative loss sums; leave its schema and validator untouched. An active
energy consumer must reject missing/wrong-version/wrong-weight/corrupt state;
inactive fixtures omit it and reject an active receipt. Validate it before
mutating restored model, optimizer or RNG state. Preserve counters across
restore without making them inputs to the training law. Component guards at
[behavior_adapters:484](../../../../experiments/forge/behavior_adapters.py#L484)
must independently require one energy call per G update and retain separate
transport evidence. Add energy state to existing component
[receipt/provenance:546](../../../../experiments/forge/behavior_adapters.py#L546).

Words needs the same separate optional field in
[state/restore:1224](../../../../benchmarks/toy_audit/api_images.py#L1224),
[ordinary evidence/guards:174](../../../../experiments/forge/word_adapter.py#L174)
and [acquisition/hold restore:214](../../../../experiments/forge/word_tasks.py#L214)
and [evidence:341](../../../../experiments/forge/word_tasks.py#L341). Retain the
full 20,001-update question; adding an objective does not authorize a
continuation task or acquisition-only grading.

GANTrainer can call the stateless weighted method directly and return an
enabled-only energy scalar beside its existing
[output diagnostics:437](../../../../particlegan/training.py#L437). It needs no
new RNG or mathematical checkpoint state. Add weight zero to
[SEARCH_RECIPE_DEFAULTS](../../../../particlegan/recipe_compat.py#L3):
[Recipe.to_dict:562](../../../../particlegan/recipes.py#L562) then omits it for
inactive archived packets. Both [GANTrainer recipe checking:577](../../../../particlegan/training.py#L577)
and [UpdatePolicy's existing strict comparison:1249](../../../../particlegan/policy.py#L1249)
can retain their current behavior: enabled weights serialize explicitly, so
zero/positive/different-weight checkpoints remain incompatible. Do not loosen
the policy equality check. Add the new numerical helper to the explicitly
reviewed [clock source audit:58](../../../../experiments/forge/clockfree.py#L58)
and retain all four parity branches and existing named-stream checkpoints.

## Boundary before any paid comparison

The reviewed [prerequisite receipt](energy-distance/diagnosis.json) passes
derivative, collision, duplicate, signed-value and native-size resource checks;
its saved-output/explicit-fixture limits remain explicit. Word input-backward
peak workspace is about 46 MB, and synthetic array timing does not establish
integrated training allowance compliance. Meaningful software checks would
then verify disabled behavior, actual call/gradient
ownership, unchanged forward/draw/update counts, signed-state restoration and
fresh-source clock parity; none has been run by this plan.

Any implementation changes the [scientific source snapshot](../../../../experiments/forge/sources.py#L66).
A subsequent finite registration needs legitimate **new-source matched controls**
on the complete original six ordinary tasks plus each recipe's own clock;
old-source cached grades remain historical. Preserve seed 0/public initializer,
all task/model/prior/data/sampling/cadence/gate laws and the full 2,520-second
reservation for each recipe. The
aggregate paid cap remains 8,000 seconds, with 6,586.162437112915 already charged;
control reruns and any treatment must fit the remaining allowance. This document
adds no scientific code, candidate, model call, draw, update, test run, queue
mutation, paid reservation or qualification.
