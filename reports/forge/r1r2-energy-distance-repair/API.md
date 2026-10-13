# Raw energy-U public integration

The selected reusable `Recipe.energy_distance_weight` adds raw Euclidean signed
energy-U with one global weight 1 versus a fresh-source weight-0 control on
exact gamma .1 non-saturating AMSGrad m110. There is no trained result here.
The [original API proposal](../r1r2-weak-penalty-repair/research/energy-api-plan.md)
and [numerical/resource prerequisite](../r1r2-weak-penalty-repair/research/energy-distance/README.md)
retain their original source identities. The
[software receipt](software-validation.json) records the integration checks.


After the prerequisite commit `238919ba69f86f862dd7ac5aee0549ea9ac6e826`, root
authorized the selected weight-1 implementation. The earlier proposals to edit
native component/word fixture files are superseded: **all 19 evaluation source
pins, native task/evaluator/host files and task cards remain byte-identical**.
In particular, `behavior_adapters.py`, `word_adapter.py`, `word_tasks.py`,
`api_images.py`, `policy.py`, `conditional_transport.py` and
`kinetic_transport.py` retain their bytes. No task revision or hash bypass is
used.

The unpinned [external binding](../../../experiments/forge/energy_adapters.py#L1)
inherits the original constructors, training loops and scorers.
`EnergyBehaviorComponents.add_transport_loss` delegates the existing hook,
then applies separate signed energy to the same output panel.
`EnergyWordFixture.step` temporarily bridges the existing output-loss consumer
and calls the exact inherited step; its `finally` block restores the native
transport object. Original objectives, target receipts, sampling, forwards,
draws, updates and optimizer ownership stay in the inherited code. The unpinned
[dispatch](../../../experiments/forge/adapters.py) selects this binding only
for positive energy weight; zero weight uses the exact original functions and
classes without consuming RNG or calling the new objective.

Scoped factory installation uses a reentrant lock and context-local selection.
Binding sessions serialize installation, restore every module export on
exception, and leave constructors in unrelated threads on the original class.
The receipt names the inherited native loop and pins both the binding and
objective source bytes. This external binding and the new numerical helper are
part of the fresh scientific source, not hidden task metadata.

Separate `component_energy` schema 1 contains consumer
`output_energy_u_v1`, weight, equal nonnegative `calls`/`active_calls` and finite
signed `loss_sum`/`minimum_loss`/`maximum_loss`. Enabled ordinary component
receipts, their provenance state and word fixture checkpoints retain it;
inactive states omit it. Word acquisition invokes the overridden `state_dict`
before saving. Hold validates signed state, exact weight and acquisition call
count **before** native restoration mutates models, optimizers or streams; the
native final fixture/stream digest checks remain enforced. The original
nonnegative `component_transport` schema and strict UpdatePolicy recipe
comparison are unchanged. Result/adapter receipt updates lie outside certified
checkpoint directories and preserve their artifact manifests.

Bounded software tests exercise the direct signed law, detached targets,
gradients, zero-weight compatibility, actual inherited steps, no added model
calls/draws, original source pins, signed corruption rejection, artifact
certification and all four own-clock parities. The hold restoration test is an
explicit synthetic prerequisite/counter fixture: **one real prefix update and
two real continuation updates**, with synthetic 834-step counters/metrics; its
original continuation grade is INCOMPLETE. It supplies no acquisition or
quality evidence. Fresh matched controls and complete original task budgets
are still required for any paid comparison.
