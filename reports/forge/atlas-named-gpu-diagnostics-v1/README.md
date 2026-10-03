# Eight named Atlas GPU host diagnostics

This is one corrected-source successor (`atlas-named-hosts-current-gpu-diagnostics-v3`), with the fixed shared Atlas pair
`lr=.0053125`, `prior_lr_mult=1.5` and seed `0`. It preserves the original
targets, objectives, gates and full horizons. Each actual named family retains
its own complete 26-question view with Tier 1/2/3 denominators **5/19/2**.
Only its adapted questions are executed here; every other slot is `NOT_RUN`.
Results cannot fill another family, the original source, or an ordinary
qualification, calibration, public default or speed comparison.

| Physical lane | Actual family | Adapted questions | Updates | Full allowance |
|---|---|---|---|---:|
| GPU 0 | `atlas_conditional` | trajectory, residual student, unipolar, mid-scale identity | 400/400/400/800 | 7200 s |
| GPU 0 | `atlas_ae_routed` | AE reconstruction and anchor hold | 250 | 300 s |
| GPU 1 | `atlas_routed` | unused slot hold and used concept edit | 200 | 300 s |
| GPU 1 | `atlas_multibank` | leftover/content preservation and pole edit | 800 | 1800 s |
| GPU 1 | `atlas_word_joint_min11` | five-word generation and paired inverse | 20001 | 900 s |

The lanes are disjoint and each runs serially. Their ceilings are 7500 and
3000 seconds, totalling **10500 seconds**. A full original allowance must fit
before every admission. Grading and media share that allowance; export grace
is zero. No automatic retries, extra seeds, shorter budgets, CPU fallback,
disabled controls or extra sweep are permitted. Completed numerical `FAIL`
continues the diagnostic lane. Invalid source/contract/runtime/checkpoint/media
or interrupted execution halts its lane and retains `INVALID`/`INCOMPLETE`
and all unreached `NOT_RUN` cells.

These are **inclusive lane ceilings**. The two original pre-update INVALID
attempts remain unchanged and debit GPU0 `12.873334385920316` seconds and GPU1
`12.449620655039325` seconds, with zero reserve. The exact
[original failure summary](../atlas-named-gpu-diagnostics-invalid-20261003/summary.json)
(`7fee64a4…`) and its old source and durable terminal proofs are frozen inputs.
Initial remaining paid capacity is GPU0 `7487.12666561408` and GPU1
`2987.5503793449607` seconds; together `10474.67704495904` seconds.
The unadmitted native-v2 preparation supplies no cost or scientific credit.

Before **each job**, historical lane debit + verified completed predecessor
families' charged costs + current family charged spend + the unchanged full
job allowance must fit that physical lane. The separate family cap must also
fit. Family caps stay 7200/300/300/1800/900; a debit is not subtracted from an
unrelated 300-second family cap. Another lane cannot supply spare capacity.
Predecessor study/source/grade/media and durable cost pins must verify in the
exact lane order. Missing, interrupted or unfinished predecessor records
refuse downstream admission. Current paid, interruption reserve, historical
debit and inclusive lane charge remain separate in state and human readouts.
Measured teardown overruns are durably retained as `BUDGET_EXCEEDED` and halt;
they never reset the ceiling or admit a later job. No old failure supplies a
current outcome. This authorization permits one successor and no retry sweep.

The generic planner uses the inherited Atlas `public_trainer` reference
context to resolve **metadata only**. Every adapted task overrides execution
to `public_components` and binds its actual public `UpdatePolicy`, optimizer
groups, complete callbacks, streams, selected models and checkpoint. The
request's `trainer_family` is the actual named family. Common candidate
requirements do not falsely require a cloud for AE; its exact task requires
the fixed positive-width MoG, hard AE and complete routed ownership. All
effective task Recipes and compiler annotations survive JSON transport and
are reconstructed against the frozen source before execution.

Each inline diagnostic declaration uses the supported schema-v1 metadata
resolver. It retains the unchanged ordinary v2 contract and its source hash
as an inactive `ordinary_decision_reference`. The exact immutable-legacy
admission refusal remains recorded as `ordinary_admission: BLOCKED`.
Only that exact refusal is advisory inside this fixed diagnostic dispatcher;
every source/API/capability blocker is preserved. The ordinary declaration,
decision guard and promotion gates remain unchanged and grant no admission.

The new laws are explicit. Conditional/routed/multibank rows are complete
conditional parameter banks, not independent full-function particles. AE
retains fixed MoG width `.025`, the original encoder/decoder/critic and
reconstruction/cover/particle-L2/GAN objective. Its selected clean
reconstruction and anchor-hold observations are distinct from DV12 training.
Word min11 retains five target words and the free continuous encoder, with
eleven actual prior rows, same-effective-code DV12 generation, and words-only
168-coordinate output noise. It adds no reconstruction training loss.
Original N5 remains `BLOCKED` under unchanged full-owner eligibility; min11
has no old-source or resource-equivalence credit.

Each host retains **24 post-update observations** and its original five
terminal passing-check gate. This is not a first-window/all-later hold or a
convergence-speed test. Goal GIFs use nine genuine retained NPZ boundaries:
paired temporal inputs/targets/predictions, protected slots/content, AE paired
reconstruction and anchors, or exact decoded word probabilities and padding.
Rendering performs no model call, draw or rescore. The original numerical
grade and media byte/source pins are retained together in each family archive.

The structural readiness card records actual preflight blockers and declares
`model_capacity_proved=false`, `learned_quality_proved=false`, and
`qualification_input=false`. It is not a trained or capacity certificate.
Preparation constructs no model, executes no update and initializes no queue.
Actual shared `PolicyCoordinator` admission is through the existing queue;
the coordinator's first argument is that queue, not the source checkout.

After root freezes a clean committed source and repins all eight declarations,
prepare a fresh external output. This captures the complete source, all parent
and variant JSONs, actual Recipes, runtime, five views, compiler annotations,
and the source-bound delegated immutable pin/lease/charge utilities:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python reports/forge/atlas-named-gpu-diagnostics-v1/run_diagnostics.py \
  --prepare-only --output /path/to/new-named-diagnostic-output
```

Root may then execute either disjoint lane or both in separate processes:

```sh
python reports/forge/atlas-named-gpu-diagnostics-v1/run_diagnostics.py \
  --output /path/to/new-named-diagnostic-output --gpus 0 \
  --queue-root /ml2/hypergan/ParticleGAN-single-recipe/runs/forge
python reports/forge/atlas-named-gpu-diagnostics-v1/run_diagnostics.py \
  --output /path/to/new-named-diagnostic-output --gpus 1 \
  --queue-root /ml2/hypergan/ParticleGAN-single-recipe/runs/forge
```

Only the declared physical lane is exposed as logical `cuda:0`. Before any
reservation/child, telemetry must report the exact RTX A6000, at least
12288 MiB free and temperature at most 82 C. Children inherit admitted lease
descriptors, CPU/BLAS threads 1, deterministic cuBLAS settings and a `.2`
process memory fraction. Training uses unchanged `runtime.execute`; grading
uses unchanged `evaluate.evaluate` in a fresh CPU-only subprocess. Durable
supervisor paid time is retained separately from any conservative interruption
reserve, whose charged total matches central admission. Neither wall time nor
FLOPs establish fastest configuration here.

Tail an actual child's `<output>/<family>/attempts/<task>/run.log`.
Each family `study.json` and `README.md` retains its full denominator and every
grade/GIF. This declaration and its CPU tests alone contain no scientific run.
