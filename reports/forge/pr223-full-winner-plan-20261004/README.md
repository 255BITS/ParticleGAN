# Retest the full PR223 Atlas winner

**Fresh execution is pending.** The [original replay](../continuous-baseline-20261003/README.md)
passed all nineteen original questions. The [current score table](../technique-inventory.md)
preserves those source-bound passes as its first row. This plan retests the complete
winning recipe on current develop, then considers explicitly compatible new questions.
It grants no new gate, default or speed credit before execution.

The authoritative full configuration is
[`configs/100gaussians/atlas.json`](../../../configs/100gaussians/atlas.json), SHA256
`a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4`.
Preserve all its fields, including LR .00425, prior multiplier 2, D multiplier 1,
learned output kernel initialized at .029, public policy-selected serving,
averaging 4 and EMA .995. Portability uses seed0; moving/static hosts use seed1234.
The later C6 configuration and noise-OFF observations answer a different experiment.
They cannot establish a regression of the full original winner.

## Source review and scope correction

The [frozen design](design/DESIGN.md) binds nineteen complete definitions, sixteen
applied nonmoving Recipes, original numerical criteria, observation clocks and
historical cost evidence. Its inspection used the feature PR273 worktree. The
policy/Recipe differences it describes belong to that feature branch.

The actual retest uses develop. The independent
[package parity receipt](source-parity/develop-package-parity.json) verifies that
develop `d4cb48d36477b5622148251892b9dc761f55573f` and the successful replay source
`a0d6d89fb470f551b3f790016a237c40a377e1e8` have the same complete thirty-file
`particlegan` tree, `d85210289a35967493fc3345441b043a8d07860b`, and the same config.
The [adapter/function parity receipt](source-parity/scientific-adapter-function-parity.json)
also verifies the two RA15 scientific adapter files are unchanged. All twenty-three
other baseline functions match exact source bytes; the complete module AST matches
after excluding only its offline metric renderer. Its scientific child, host
definitions and numerical gates are unchanged.

These comparisons are source evidence. The new supervisor, pure media capture,
external harness/data, canonical imports and complete copied execution source still
need their own frozen identities and actual metadata preflight before GPU admission.
The copied design and mapping remain byte-original; this wrapper corrects their
inspection scope without replacing their evidence.

## One finite replay

1. Freeze the reviewed helper and complete current execution source. Verify actual
   copied-source imports, full JSON-wire request, all definitions and resolved Recipes
   with CUDA hidden and zero models, draws, scorers or training. Keep original
   initialization, named streams, backend admission, public perturbation and serving.
2. Run all nineteen original protocols through the existing public `GANTrainer`
   scientific path: thirteen portability, three moving and three static native,
   totaling **48,800 updates**. Preserve every original gate and observation clock;
   static native requires coverage AND accuracy, final-five 20k checks and independent
   100k holdout. Original noisy grades and clean diagnostics stay separate.
3. Capture existing observed generated/reference values for goal GIFs. Add no draws,
   forwards, scoring checks or training-stream movement for the illustration. Moving
   keeps its four actual clocks; other hosts may select existing checks. Construction,
   training, reads, checkpointing, scoring, media and final attestation all finish
   inside each inclusive allowance, with **zero grace and zero retries**.
4. Publish verified terminal metrics, actual goal GIFs, complete source/protocol/Recipe
   identities and paid/reserved cost. A completed numeric FAIL remains FAIL. Source,
   runtime or checkpoint faults halt as invalid; unfinished deadlines receive no
   accepted scientific grade. Preserve measured overruns and conservative interruption
   reservations. Missing work stays pending or incomplete.

The nineteen declared allowances in [design.json](design/design.json) sum to
**9,810 seconds**, each computed as
`ceil((1.5 * historical_case_paid_seconds + 90) / 30) * 30`.
Bounded shared metadata/publication reserves **180 seconds**, making **9,990 seconds**
under a separate **10,800-second ceiling**. No next full allowance is admitted unless
it fits. These are budgets, not guaranteed duration or convergence-time results.
The earlier named campaign remains **1,466.669318475 / 10,500 seconds**; its costs are
not reset or pooled into this replay. Historical replay cost is reference only.

Use the existing shared Forge queue and durable study/attempt leases. Admit one
physical GPU1 as logical `cuda:0`, one CPU/BLAS thread, memory fraction .2, at least
12,288 MiB free and at most 82 C. Recheck telemetry before each case. A busy/hot
device records a wait rather than shortening a protocol or starting a polling loop.
Other owners' work, pauses, credentials and wake settings remain under their control.

## Added questions and eventual winner

The [complete mapping](mapping/README.md) binds all twenty-six current tasks and eight
public-API questions to the original nineteen. Sixteen current slots relate to
original questions; ten are new. It asserts **zero automatic whole-protocol aliases**.
Matching a target or architecture does not establish the same recipe, initializer,
seed, streams, serving law, sample counts or gates.

After the full replay, audit its actual artifacts. The first small additional scope
proposed by the mapping is one separately declared **400-update ring16 acquisition**
using the complete winner recipe, N256/z4/b128, original bounds, explicit noisy
selected serving and its original **300-second inclusive allowance**. It needs its
own source-bound protocol and admission. It is not queued by this plan. Dense ring
retention, vector KS and routed/conditional/AE/word questions require their own
prospectively compatible measurements and owners. Their passes cannot be pooled
as an independent Atlas winner.

Family capacity asks whether a family can represent and learn each declared question.
Shipping defaults asks for one unchanged shared configuration that clears the full
compatible ladder, hold, calibration and robustness. Hyperparameter search selects
defaults within a family; compare convergence speed only among fully eligible,
matched finalists. The governing process remains [PR247](https://github.com/255BITS/ParticleGAN/pull/247).
Further word/NLL search is parked while this full winner is retested.

The [copy manifest](copy-manifest.json) records the byte-original review outputs.
Inspection indexes contain local source/artifact paths; bulk states and traces remain
local. These metadata reviews launch no preparation, model, training, scoring or GPU
job. Fresh execution receipts will bind the final source rather than borrowing the
inspection worktree or the historical nineteen passes.
