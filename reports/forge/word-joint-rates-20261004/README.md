# Prospective N11 word half-base binding

This is an unexecuted rate contrast. It adds one fixed candidate to a separate
cohort and reuses the existing public word fixture. The software controls
construct no training model, optimizer or policy and make no forward, sampler
or learned-capacity claim. Earlier quarter-base and slower-prior proposals are
unexecuted history and are excluded from this executable whitelist.

The only accepted profile is `half_base`, candidate
`word-min11-half_base-rates-v1`: `lr=.00265625`, `prior_lr_mult=1.5`,
`d_lr_mult=1`. Its cohort is `word_joint_policy_min11_rates_v1`, family
`atlas_word_joint_min11_rates`, and task
`five_word_joint_acquisition_word_joint_policy_min11_rates_v1`.

Relative to the separate C6 N11 word declaration, the base rate is halved.
Nominal G/E/noise and D rates are `.00265625`; the nominal prior rate is
`.003984375`. Public stationarity, latent damping, payoff damping, row evidence,
birth/death, surprise, reopening and selected averaging remain active. Their
applied rates and actions can differ during learning; nominal rates do not
describe a recorded trajectory or prove a cause of the previous failure.

The original question remains five equally likely canonical words plus exact
paired reconstruction of every known input. Resources remain eleven real 2D
ParticlePrior rows and batch 256, with a free continuous encoder. The original
joint RpGAN objective has no added reconstruction loss. Training uses the same
effective DV12 code in G and the joint critic, and learned output noise affects
only the 168 word coordinates. Evaluation uses selected G/E/prior together,
with output noise disabled and the actual public DV12 latent policy active.
The original N5 task remains BLOCKED; neither the old C6 result nor historical
capacity/qualification transfers to this candidate.

The unchanged full protocol is seed 0 and 20,001 updates, with
`Recipe.total_steps=None` and an explicit caller limit. There are exactly 24
post-update reads at `ceil(i*20001/24)`, each with 1,024 generated samples and
all five paired reconstructions. The final five reads are
16,668/17,501/18,335/19,168/20,001. Bounds remain sample count ≥1,024,
quality ≥.95, modes =5, mass TV ≤.1, exact paired reconstruction =1 and minimum
true reconstruction-token probability ≥.9. A nine-frame goal movie can use
the retained observation indices 0/3/6/9/12/14/17/20/23 without extra draws.

The [pure declaration helper](../../../experiments/forge/word_joint_rate_policy_contracts.py)
exports `candidate('half_base')`, `rates('half_base')`, `make_variant(root)`,
`validate_request(request, task, root=...)`, `resolved_recipe(candidate, task)`,
`binding_receipt(recipe, profile)` and
`validate_result_binding(request, task, raw_result)`. Candidates explicitly
declare the original raw ParticlePrior law. The top-level reference context
uses `public_trainer`; the source-pinned active task and its actual producer
use `public_components`. The model-free full 26-slot planner control checks this
distinction and preserves the 5/19/2 denominator. Only the word slot is proposed
for execution; the other 25 slots earn no credit from this contrast.

Actual execution remains `experiments.forge.adapters.run_adapter` through the
existing `WordJointPolicyFixture`/`run_word`; there is no second training loop.
The checkpoint has kind `forge_word_joint_policy_min11_rates_v1` and contains
its complete Recipe and `word_rate_binding`. The same binding appears at
`raw.evidence.policy_controls.word_rate_binding` and
`raw.applied.policy_lifecycle.controls.word_rate_binding`.
`raw.applied.recipe` carries the complete actual Recipe. The independent
`experiments.forge.evaluate.evaluate` process checks request profile, fixed tuple
ID, declared prior, actual resources, applied Recipe and both receipts before
accepting a numerical PASS or FAIL. Unknown profiles, retired tuples, hidden
extensions, altered initialization/resources/laws and mismatched receipts fail
closed. Original C6 dispatch and checkpoint fields retain their default behavior.

The new task JSON binds the byte identity of its original parent and every
declared implementation source. A future root driver must use the final
integrated source commit and regenerate this new source binding if integration
changes a pinned file. It must retain the full planner/import/config closure,
the screening seed-zero protocol, all original parent task/view JSONs and the
new variant JSON, and check imported module identities against that snapshot.
The isolated branch's pins are not evidence for a later source. Existing C6
declarations were not edited or repinned; a future current-source run of them
would separately need its own source-only repin. Earlier frozen rows still
use their original source.

Root owns the prospective single 900-second inclusive GPU1 allowance, inherited
leases, source freeze and any actual attempt. Independent grading, rendering
and final attestation must fit that same allowance. The earlier named costs
remain charged once; this implementation reserves no resources and resets no
budget. No passing-case rerun, ordinary/default or speed credit is declared.

The retained word failure has at least two plausible limits: unknown learning
dynamics and a joint-law mismatch between deterministic real E codes and
perturbed fake codes plus fake-only word noise at the saved states. This rate
contrast tests one dynamics hypothesis. It establishes no matched-code capacity
certificate, no causal explanation, and no promise of reconstruction recovery.

Validation: `tests/test_forge_word_joint_rates.py` contains model-free metadata,
JSON transport, full-recipe, source/gate drift, actual planner dispatch, optimizer
owner, retired-tuple, independent-evaluator substitution and synthetic checkpoint
round-trip controls. Fake metric curves and checkpoint payloads are labeled
software inputs and never represent scientific results.

The final CPU-only run passed 307 controls in 12.58 seconds across the new
86-control file and the existing named-policy grading/publication guard suites:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -p no:cacheprovider -q tests/test_forge_word_joint_rates.py \
  tests/test_forge_named_policy_grading.py tests/test_forge_policy_snapshot_publication.py
```
