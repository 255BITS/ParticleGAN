Atlas and E22 share the reference density backend on the small-population
questions, but they remain different policies. Five matched C3 configuration /
case pairs have identical retained clouds and learned endpoint states. C4 adds
an observed optimizer-reopening difference and diverging outputs. The
[compact proof](policy-failure-diagnosis.json) binds the exact raw receipts,
clouds, checkpoints, source lines and zero-update capacity states. This audit
read retained evidence only: no models, draws, rescoring, training or ranking.

The [public presets](../../../particlegan/recipes.py#L663) define Atlas as E22
plus automatic 128-cell birth/death selection and the settled reopening guard.
The [selection rule](../../../particlegan/feature_policy.py#L109) waits for the
first real output shape, respects caller-owned/routed representation, then checks
population resolution before enabling feature cells. Its
[exact rational feasibility rule](../../../particlegan/feature_cells.py#L28) is
`ceil(N / [.05 * (floor(N/2)+1)]) <= floor(.05*N)`.

| Original API host scope | Actual resource / raw width | Atlas route and reason |
| --- | --- | --- |
| Intensity2 and bars4, exact transpose12 | 32 prior rows / 64 output coordinates | Reference kNN: 38 minimum flags exceed the one-row guard limit. The 64-coordinate output also exceeds the auto complete-moment bound of eight. |
| Two-broad, unequal-mass and anisotropic vectors | 256 rows / 2 coordinates | Reference kNN: 40 minimum flags exceed the 12-row guard limit. |
| Grid100, rotated100 and staggered100 | 20,000 rows / 2 coordinates | Feature cells: 40 minimum flags fit the 1,000-row guard limit; the full raw moment frame fits the rank bound of eight and no caller representation callback owns the host. |

The first quality host to exercise the distinct Atlas backend is **api-grid100**;
the other two original native quality hosts do as well. Their
[declared budget](../../../benchmarks/toy_audit/api_vectors.py#L289) is 7,000
updates, batch 2,048, 20,000 evaluation points and 20,000 prior rows. Saved
capacity checkpoints confirm all three feature selections at zero optimizer
updates. C3 and C4 did not reach any of them, so there is no ordinary native
feature-cell quality result in these cohorts. Changing the particle count to
activate that backend would change a task resource, not merely tune an existing
configuration. Even an explicit feature-cell request retains the population
resolution check; increasing image population alone would still leave the
automatic raw-width limit.

On the feature route, the [public auto calibration](../../../particlegan/feature_policy.py#L125)
multiplies generator and learned-noise rate ceilings by **0.25**, retaining the
table and critic ceilings. It also installs the feature reaction/sampling law
and the population sequential table-settlement test. The
[served averaging decision](../../../particlegan/policy.py#L1047) then uses the
feature backend's paired-average certificate. The small-host fallback keeps the
existing reference birth/death, settlement and DV12 sampler, with rate factor
one. Output-noise exclusion does not disable
[latent perturbation](../../../particlegan/policy.py#L975).

Both shared search knobs are effective. `lr` supplies the
[generator and prior rates](../../../particlegan/recipes.py#L617), the
[critic rate](../../../particlegan/recipes.py#L528) and the
[learned-noise rate](../../../particlegan/policy.py#L340).
`prior_lr_mult` multiplies the trainable prior group's base rate. All eight
declared hosts own a trainable particle table. Actual rates subsequently receive
[role-specific stationarity scales](../../../particlegan/policy.py#L647); a
declared multiplier is not a constant applied displacement. Atlas's native
0.25 factor means the same common `lr` is not the same native generator/noise
ceiling in the two families.

[PR252's ownership registry](../../../experiments/forge/boundaries.py#L23)
classifies these rates as tunable hyperparameters, resources/prior/initialization
as task contracts, and backend/reopening/serving mechanisms as technique
fields. Its [host delegation](../../../experiments/forge/taskrecipes.py#L10)
does not delegate `lr` or `prior_lr_mult`. In this separate API study the
[actual provider adaptations](../../../benchmarks/toy_audit/api_family_search.py#L72)
are bound explicitly: vector `d_lr_mult=1.5`, `prior_reg=0.05`, betas `(0,.99)`
and EMA reference `.995`, while image/native critic multipliers are one.
Explicit shared rate overrides are applied last. A common configuration means
the same preset and searched knobs on those frozen hosts; complete effective
Recipes also contain their different task resources/adaptations. Forge's own
inactive-provenance receipts must not be substituted for this API binding.

Some serialized fields are unsuitable search axes here. Atlas/E22 have
`total_steps=null` and stationarity rate control; their cosine annealing/floor
branch is unused, consistent with
[PR252's activity checks](../../../experiments/forge/techniques.py#L94).
Feature-cell cell/rank/chunk settings cannot exercise the unconstructed feature
backend on these small hosts and are outside the finite search whitelist.
`standardize` is removed by
[the particle-cloud constructor](../../../particlegan/recipes.py#L439), so its
reference does not describe standardized cloud reads. `ema_decay` remains a
real [adaptive-average fallback](../../../particlegan/policy.py#L1065), not a
globally ineffective setting. `d_lr_mult` is active and tunable, but is fixed
in the present grid rather than a searched axis.

C3 source `eb2d77fbd776eab0d2ce5e2550ab9cdfdb6c1162` produced four paired
intensity runs and one paired two-broad run. Every retained target/sample array
matches bitwise between same-knob Atlas/E22 runs: 50 arrays per image pair and
150 for the vector pair. Learned models, optimizers, output noise, reference
controller, settlement, birth/death, row evidence and named-stream endpoint
states also match. Complete public states do not: recipe/selection/guard fields
differ, and some optimizer-surprise detector references differ. Both families'
monotone `surprise.fires` counters are zero in all ten runs, so no optimizer
reopen acted. Atlas's vector guard records one KA2 loss-epoch rebase; E22 has
none. Identical observed outputs grant no general policy equivalence.

C4 source `0335ecf024ea8a6b337a216345f444abd41e3a6c` makes that distinction
visible. At `lr=.002125 / prior=2`, E22's saved optimizer-surprise log is
**`[[360, 2.445]]`**, with one actual reopen; Atlas records zero. Both saved
clouds match bitwise at every captured update 0–350. The first retained
difference is update **375**, maximum pixel difference **0.58725548**.
E22 ends original PASS with HQ **0.98730469**, but confirms acquisition only
at update600 and has zero subsequent hold checks, leaving study INCOMPLETE.
Atlas ends original FAIL with HQ **0.625**. The other three C4 knob pairs have
identical clouds at all 25 captured boundaries, with zero reopens in both.

The source exposes the operative distinction:
[the surprise detector](../../../particlegan/continuous.py#L1037) lets the
guard change its slow reference and sustained-rise permission; the
[settled guard](../../../particlegan/continuous.py#L1079) needs a contracted
network witness for an excursion, whereas E22 has no guard. An accepted event
[restarts stationarity and rescales optimizer second moments](../../../particlegan/policy.py#L628).
The event, output boundary and implementation strongly locate this divergence
to reopening rather than feature cells. Full owners immediately before/after
updates359/360 were not saved, so this is not a restored counterfactual or a
quantitative causal-effect estimate.

The lower-rate grid's exact intensity shortfalls are:

| Rate / prior | Original gate / extra study gate | Retained failure or missing evidence |
| --- | --- | --- |
| .002125 / 1, both families | PASS / INCOMPLETE | Confirmation525 leaves only three passing hold checks under the full600 budget. |
| .002125 / 2, Atlas | FAIL / FAIL | Endpoint HQ .625, rejected mass .375, finite-template TV .4306641, modes1; no five-check acquisition. |
| .002125 / 2, E22 | PASS / INCOMPLETE | Actual reopen360; confirmation600 leaves zero hold checks. |
| .00425 / 1, both families | PASS / FAIL | After confirmation375, update475 has HQ .4101563, rejected mass / template TV .5898438 and modes1; eight of nine hold checks pass. |
| .00425 / 2, both families | PASS / FAIL | After confirmation375, update425 has HQ .7910156, rejected mass .2089844 and template TV .2099609; eight of nine hold checks pass. |

Both lower family grids have concluded without a study survivor. One separately
frozen `.0031875 / .0053125 × prior1 / prior2` grid is a bounded empirical
acquisition–hold hypothesis: the half rate acquired late, the preset rate
acquired earlier but lost template support, and `.006375/prior2` previously held
intensity. The measured outcomes are nonmonotonic, so interpolation promises
neither improvement nor a winner. A critic-rate grid would also use an effective
knob, but no retained critic-response defect currently makes it better justified.
The proposed rates are new numerical strengths within the same technique;
ownership, gates, seeds, serving laws, horizons, eight-case denominators and the
original 10,800-second total cap remain fixed. After the concluded cohorts the
remaining paid allowance is **10,336.011929200264 seconds**. No failed hold is
extended and no original PASS/FAIL is replaced. Root owns the separate
declaration, software admission and any execution.
