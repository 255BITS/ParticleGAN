# Prospective deployed-output observation

Status: SOURCE_ONLY_NOT_EXECUTED. This helper declares one sampling law and
returns samples and a receipt. It constructs no learner and assigns no grade.

The caller passes an existing ready GANTrainer or unconditional caller-owned
UpdatePolicy, the task's declared sample count and observation clock, a separate
evaluation generator, and a source-bound semantic state fingerprint reader.
`measure_served_samples` calls the owner's public `served_model()` once and
calls the returned `ServedModel.sample(n, generator=dedicated_eval_rng,
output_noise=True)` once. The public policy intrinsically chooses fast or
averaged weights. No score, clean/noisy comparison, checkpoint or forced EMA
choice enters this helper. The receipt records `served.source`, its actual
learned/floored `output_sigma`, clock and output shape.
The public API defaults to `output_noise=False`; this prospective protocol
explicitly enables it under the authenticated617/618 intended-serving request.
Existing declared clean/live observation contracts retain their original scope.

The existing prior, network, Recipe and public DV12/backend perturbation remain
in force. Output noise is enabled at its actual selected policy scale, including
an actual zero scale if the policy reports it. Samples are returned unchanged;
the task's existing metrics, thresholds and horizon stay with its evaluator.

This initial interface supports uniform independent-row unconditional GAN
sampling only. It refuses actual encoder, router and custom generation owners,
nonindependent/routed snapshots and explicitly declared contextual or enumerated
sampling laws. It does not infer a task's sampling law from its image shape.
The caller must bind the uniform law in the prospective task/protocol source.
An enumerated image task needs a separately declared public generation law over
its exact selected rows; random `sample(32)` cannot replace its fixed centers.

The dedicated generator must differ from every declared trainer/policy stream
and every additional caller data/training generator. The caller binds its named
evaluation stream to the scientific repeat and actual snapshot device. The
public `ServedModel` validates the real generator type/device. The helper does
not construct, seed or reseed any stream.

The fingerprint callback must cover semantic live models, optimizer state,
policy/controllers/averages and Recipe, global and owner RNGs, callback state
and data cursors. It excludes only the dedicated evaluation generator. Ready
phase and completed clock are checked independently. Equal before/after
fingerprints are required even when public sampling raises. Fingerprints must
not hash tensor mutation-version counters: the existing public serving snapshot
can temporarily release/reapply its compatibility swap while preserving values.
Checks also reject a served table or G/D/prior object that aliases the live
policy's actual `table`, `G`, `D` or `prior` attributes. The source-bound public
API supplies independent frozen inference copies; these guards do not replace
source ownership or storage checks during ROOT's runtime integration.

One receipt supplies no sustained or retrospective verdict. Existing exports
retain one final complete state, not selected model/noise states at every past
clock. Final-state samples can support a separately labelled terminal diagnostic
only. Full protocol credit needs fresh correctly bound observations at every
original clock, final-five and holdout requirement, under unchanged gates and
within a ROOT-authorized finite envelope. Old clean/noisy results and their
accepted grades remain immutable.

Before runtime use, ROOT must commit and copy the module with the actual public
policy/trainer, whole Recipe, task and repeat source identities; independently
review its caller integration; validate the source contract without construction;
and bind the actual dedicated stream and semantic reader. ROOT's paid runtime
must verify selector/noise evidence, owner state purity and every task observation
without choosing a better serving branch. This source handoff activates no
budget, preparation, model, scorer, queue or measurement operation.
