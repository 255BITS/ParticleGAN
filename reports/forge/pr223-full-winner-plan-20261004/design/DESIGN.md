# Faithful PR223 Atlas winner retest

The original winner is the complete config at `configs/100gaussians/atlas.json`, SHA256 `a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4`: **learning rate 0.00425, prior multiplier 2, discriminator multiplier 1, learned output kernel initialized at 0.029, and actual policy-selected serving**. This design declares one fresh full 19-case replay. It does not declare a search, NLL objective, changed seed, or C6 replay. No science has been prepared or executed by this review.

The authoritative machine-readable protocol is [design.json](design.json); [input-index.json](input-index.json) pins 88 inspected metadata/source files. [derive_design.py](derive_design.py) reproduces it using only standard-library JSON/file reads and read-only Git commands. These inspection pins are not a certificate for the next execution source.

## Winner identity and historical proof

Root's primary [PR223](https://github.com/255BITS/ParticleGAN/pull/223) packet identifies merged head `bc9d9aec23618e84b6b3cc8f5169638b0925671b`, merge `437c7554235a6de0c6777aca9d8993b2c3e62c67`, and body scientific source `bdf05d1be0f68cfdb0c71e81e7e0d3cce477572f`. The three Git config blobs are byte-identical. The selected package/config/two RA15 adapter files have no bdf-to-head diff. The API file list covers only the first 100 of 4,379 changed files; it is incomplete and cannot establish source closure.

The already verified [original19 publication](/ml2/hypergan/ParticleGAN-atlas-forge-unblock-20261003/reports/forge/continuous-baseline-20261003/README.md) records **19 original PASS / 19 completed, 48,800 updates**, with source `a0d6d89fb470f551b3f790016a237c40a377e1e8`, derived digest `e2b6c0e675c5da26663ac7a6181025df1d964dfe43e9a70591374ee94688d985`, and paid **5,210.63884702418 seconds**. The 19 raw result hashes and complete 16 nonmoving result Recipes were rechecked as metadata. Checkpoint tensors, bulk raw arrays and the archive were not deserialized or rescored by this review. The raw archive is LOCAL_ONLY; remote archival was not performed.

Those 16 Recipes differ only in `num_particles`, `z_dim`, and `batch_size`. The original `get_recipe(**full_config)` call records `Recipe.name="ka2"`; the full config and active Atlas mechanisms identify the winner. This label supplies no ordinary KA2 result credit. For moving cases the executed wrapper source, config and resource adaptation establish the expected Recipe; checkpoint fields were not independently loaded.

The inspected current package differs from the a0d reference at `particlegan/policy.py` and `particlegan/recipes.py`. Root must freeze and bind its final clean source, inspect the changes and pass parity controls before the fresh replay. Historical grades stay on their historical source; they are not silently attached to a changed source.

## Preserve the full law

- Use the entire pinned config and original resource-only adaptations. In particular preserve AMSGrad, betas `[0,0.999]`, regularization, DV12, birth/death, row evidence, settled reopening, serving, and all default Recipe fields. Do not substitute current `get_recipe("atlas")` defaults or C6 host exceptions.
- Preserve the raw learned ParticlePrior. `standardize=True` in this Recipe does not standardize ParticlePrior reads. It is not a fixed-width MoG. Keep `total_steps=None` and the external original horizons; do not inject a new cosine schedule or a horizon-dependent factory adaptation.
- Primary scoring uses the current learned output kernel and the original policy-selected weights. Selection may be fast or averaged. Preserve `serve_average=4` and `ema_decay=.995`, including source-defined adaptive averaging; do not force EMA. Clean output and forced EMA remain separate diagnostics.
- Keep automatic backend admission and its calibration. Small hosts use the reference path when feature cells are ineligible. Feature-cell admission retains the original 0.25 generator/noise nominal-rate factor; discriminator/prior are unchanged. Record actual backend, selected state, kernel and endogenous rates at observations instead of treating nominal rates as displacement.
- Correct an earlier prose overstatement without changing old evidence: `image_prior_perturb=False` skips only the host's extra `prior.perturb` call. Its indexed `trainer._generate` still calls the public policy and retains actual DV12 perturbation. “Ordered rows, no extra prior perturb, public policy perturbation retained” is the faithful description.

The reference-path nominal rates are G/noise 0.00425, D 0.00425 and prior 0.0085. An admitted feature-cell path uses G/noise 0.0010625 with D/prior unchanged, before endogenous controls. These are configured rate mappings, not a claim that measured movement has a fixed ratio.

## Exact original matrix and inclusive caps

Portability uses seed **0**; all moving/static cases use seed **1234**. The image generator is the original **residual-upsample width16** host, not C6 transpose12. Each row's full original definition, target, discriminator card, thresholds, observations, Recipe, raw result pin and cost is in `design.json`.

| Original case | Updates | N / z / batch | Observation protocol | Prior paid s | Proposed inclusive s |
|---|---:|---|---|---:|---:|
| intensity2 | 600 | 32 / 8 / 32 | 24 reads, every 25 | 28.750 | 150 |
| mode hold | 1,200 | 12 / 4 / 128 | 24 reads, every 50 | 51.111 | 180 |
| blobs4 | 600 | 32 / 8 / 32 | 24 reads, every 25 | 28.690 | 150 |
| bars4 | 600 | 32 / 8 / 32 | 24 reads, every 25 | 28.302 | 150 |
| stripes2 | 600 | 32 / 8 / 32 | 24 reads, every 25 | 27.536 | 150 |
| broad mixture | 1,200 | 256 / 4 / 128 | 24 reads, every 50 | 42.261 | 180 |
| unequal mass | 1,200 | 256 / 4 / 128 | 24 reads, every 50 | 51.719 | 180 |
| unequal width | 1,200 | 256 / 4 / 128 | 24 reads, every 50 | 46.389 | 180 |
| anisotropy | 1,200 | 256 / 4 / 128 | 24 reads, every 50 | 52.486 | 180 |
| overlap | 1,200 | 256 / 4 / 128 | 24 reads, every 50 | 45.266 | 180 |
| spiral | 1,600 | 256 / 4 / 128 | 24 original ceil-spaced reads | 55.692 | 180 |
| stationary ring | 7,500 | 20,000 / 2 / 2,048 | 750 reads, every 10 | 918.056 | 1,470 |
| shifted ring | 4,600 | 20,000 / 2 / 2,048 | 460 reads, shift after 2,400 | 570.518 | 960 |
| moving grid100 | 1,500 | 20,000 / 2 / 2,048 | 500 / 1,000 / 1,500 gates | 193.613 | 390 |
| moving rotated100 | 1,500 | 20,000 / 2 / 2,048 | same | 209.265 | 420 |
| moving staggered100 | 1,500 | 20,000 / 2 / 2,048 | same | 206.848 | 420 |
| static grid100 | 7,000 | 20,000 / 2 / 2,048 | 34 reads, final five + 100k holdout | 910.207 | 1,470 |
| static rotated100 | 7,000 | 20,000 / 2 / 2,048 | same | 891.651 | 1,440 |
| static staggered100 | 7,000 | 20,000 / 2 / 2,048 | same | 852.279 | 1,380 |

Each proposed allowance is `ceil((1.5 * historical_paid + 90) / 30) * 30`. They sum to **9,810 seconds**. Reserving **180 seconds** for bounded shared metadata/final publication gives **9,990 seconds**, within a separate proposed **10,800-second** campaign ceiling, with 810 seconds unallocated. Root owns the final envelope. All start, construction, updates, original reads/scoring, retained-array media and final attestation fit inside each case cap, with **zero grace and zero automatic retries**. The old named 10,500-second campaign and its debits remain unchanged; historical 5,210.639 seconds is reference cost, not newly spent retest time.

For a completed terminal, retain measured paid time even for numeric FAIL. Noncompleted/error/cancelled/missing terminals conservatively charge `max(allowance, measured)`, preserving paid versus interruption reserve. Record an actual overrun and halt rather than clipping it. Before every full job verify the ledger plus its full allowance fits. Busy/hot devices or insufficient remaining funds register a precise wait/shortfall; they do not shorten a test or trigger a polling/retry loop.

## Gates and observer distinctions

Images require complete mode count and HQ >= .9 over the original five-check terminal suffix, after all 24 reads. The nearest-template HQ RMSE cutoff is **.06 for intensity2 and .10 for blobs4/bars4/stripes2**; the source's per-mode fraction classification cutoffs also remain. TV is diagnostic, not an added image convergence gate.

The four identifiable mixture questions retain normalized SW1 <= .18, mass TV <= .15, HQ >= .85, component covariance error <= .85 and minimum eigenvalue ratio >= .15. Unequal mass additionally requires minimum mass ratio >= .25. Overlap and spiral instead retain normalized SW1 <= .18, mean error <= .15 and covariance error <= .45; they do not acquire identifiable-component mass/mode gates. Every vector finishes its original 24 reads and five-check terminal suffix.

Mode hold retains eight modes/HQ >= .9 with its five-check terminal suffix. Stationary/shifted rings use the original 20,000-row host and every-10-update observation schedule, without an initial zero observation. Each required segment retains its original five-check suffix; ring shift keeps the source-defined turn after update 2,400 and no extra frozen-control branch.

Moving tests rotate 30 degrees after updates 500 and 1,000. Both later periods require >= 95 modes and HQ >= .9 times the **same run's observed update-500 HQ**. Their original 20k gate draws remain separate from four 4,096-point movie states at updates 0/500/1,000/1,500, with target angles 0/0/30/60.

Static native tests require **coverage AND accuracy**, not the runner's reported accuracy-only status. Coverage retains 100 modes, precision >= .97, minimum HQ-mode mass >= .005, mass TV <= .1, maximum mode mass <= .02, covariance eigenvalue ratios .4 to 1.7 and radial-median ratios .65 to 1.4. Accuracy retains mass TV <= .06, center RMS in sigma units <= .2, absolute covariance-trace bias <= .1 and radial KS <= .04. Every final 20k state at updates 6,000/6,250/6,500/6,750/7,000 and the independent 100k holdout must pass the original fidelity gate. Keep all 34 original read clocks and legitimate unavailable early accuracy fields. Historically all three primary-noisy static cases pass and their clean diagnostics fail; these are separate laws.

## Maintained execution path and remaining implementation

Reuse `reports/forge/continuous-baseline-20261003/run_atlas_baseline.py` and its `original_inputs`, `task_definition`, `prepare`, `child` and `certify` path. Its scientific owner is public `particlegan.GANTrainer.step` using bound RA15 fixtures. Native and moving consumers must use the immutable relocated harness/initializer/native-root/scorer/rotation source. A copied but unused external tree does not certify a mutable absolute consumer path. Do not copy a training loop or substitute the C6 API wrapper.

The new envelope needs only:

1. New explicit protocol/source/case/runtime identity; the exact 19 source-bound definitions and resource-only Recipes above; proposed inclusive caps instead of legacy 1,800/2,400 plus 60-second grace.
2. A complete clean source/config/external-data snapshot, exact import guards and a real copied-source CUDA-hidden metadata preflight. It must prove canonical namespace locations, lossless full JSON-wire request identity, all definitions and resources, zero models/draws/scorers/CUDA and RNG purity before registration/admission.
3. Maintained `PolicyCoordinator` on `/ml2/hypergan/ParticleGAN-single-recipe/runs/forge`, inherited study/attempt leases, durable fencing/recovery and one deadline covering every stage through media and final attestation.
4. An optional pure observation capture at existing evaluation sites. Save already-computed image/vector/ring arrays for goal media; never add sampler draws, forwards, reads or a new RNG cursor. Moving/native arrays already have source-defined capture. Preserve the original metrics and isolate any additional illustrative export receipt.
5. Full owner clocks/state/checkpoint/RNG proof and numeric JSON status, paired with source/deadline/media completeness. Source/model/harness faults halt as INVALID; valid numeric FAIL may continue the other independent predeclared cases. Missing finalization or timeout supplies no accepted numerical qualification.

The proposed envelope interface is `prepare(output, final_source, protocol, queue_root)`, `metadata_preflight(output)` and `run(output, max_new_attempts=None)`, with the existing admitted scientific child delegated unchanged. These are implementation requirements, not a claim that a new executable already exists. Root has assigned the envelope to `/root/failure_diagnosis`; root alone freezes, prepares, verifies and launches it.

Admission is one physical GPU1 exposed as logical `cuda:0`, CPU1, memory fraction .2, free memory >= 12,288 MiB and temperature <= 82 C. Preserve source-defined deterministic initialization, TF32-off/highest float32 precision, serial backward and original named streams. Do not modify other owners' work. Earlier telemetry is not an admission certificate. This design makes no fair-speed claim.

The fresh 19-case row remains source-scoped, preserving the original historical 19/19 row. Additional current-26/API gates are mapped separately by `/root/merge_readiness`; original19 does not automatically fill their changed hosts/observers/resources. In particular current API native24+initial without an independent100k holdout differs from ordinary Forge native34+100k. Ordinary MoG KA2/K3P 4/5 results also remain separate. No partial history or C6 noise-off result can supply a shipping/default winner.

## Reproduce this inspection

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/ml2/hypergan/.venvs/particlegan-develop-integration/bin/python \
/ml2/hypergan/pg-pr223-faithful-winner-retest-design-20261004/derive_design.py \
--output /tmp/pr223-faithful-design-check
```

This command reads metadata/source and writes only a derived design into the requested output directory. It does not prepare an execution snapshot, import scientific code, build models, restore tensors, draw, score, train, register/admit a queue, or query a GPU.
