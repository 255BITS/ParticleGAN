# CB64-RA learned CUDA retest

Valid completed learned runs: 4/4. Exact semantic/loss replay checks: 4/4. These results cover the learned lane. Original frozen portability/native acceptance is reported by root.

## Nonlinear toy at 2000 updates

| Variant | Evidence | Precision | Modes /25 | Mass TV | Toy gate | CUDA train s | Updates/s | Ordinary / isolation moves |
|---|---|---:|---:|---:|---|---:|---:|---:|
| E22 | COMPLETE | 0.7155 | 25 | 0.2845 | FAIL | 72.34 | 27.65 | 1873 / 0 |
| CB64-RA | COMPLETE | 0.6233 | 24 | 0.3780 | FAIL | 80.79 | 24.75 | 1958 / 0 |

Independent toy gate: precision>=.9, 25 supported modes, massTV<=.1. Clean particle-centre metrics and complete mass vectors are in results.json.

## MNIST at 2000 updates

| Variant | Evidence | Raw FD | Active FD | Active precision | Active recall | Class TV | Confident classes /10 | CUDA train s | Ordinary / isolation moves |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E22 | COMPLETE | 6.8522 | 0.5445 | 0.8691 | 0.8472 | 0.0364 | 10 | 90.96 | 332 / 83 |
| CB64-RA | COMPLETE | 19.1878 | 1.4888 | 0.8525 | 0.3228 | 0.1244 | 10 | 103.26 | 5182 / 41 |

The classifier/embeddings are trained only on real MNIST; raw64 and real-training-active standardized features use the declared first5000-image rule. FD is learned embedding Frechet distance. No numerical image quality gate was declared. Confidence, clipping, class masses, normalization hashes and heldout controls are retained in results.json.

CB64-RA minus E22 at equal updates: active FD +0.9443, active precision -0.0166, active recall -0.5244, classTV +0.0880. Lower FD/TV and higher precision/recall are favorable; assess quality and diversity together.

## Common contemporary CUDA training time

| Problem | Variant | Common budget s | Selected update | Used train s | Unused s | Precision / active FD | Coverage / active recall | Mass TV / class TV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| toy | E22 | 72.34 | 2000 | 72.34 | 0.00 | 0.7155 | 25.0000 | 0.2845 |
| toy | CB64-RA | 72.34 | 1750 | 70.73 | 1.61 | 0.6409 | 23.0000 | 0.3697 |
| mnist | E22 | 90.96 | 2000 | 90.96 | 0.00 | 0.5445 | 0.8472 | 0.0364 |
| mnist | CB64-RA | 90.96 | 1750 | 88.43 | 2.53 | 1.0358 | 0.2446 | 0.0996 |

Timers synchronize around complete trainer updates/controllers and exclude batch staging, evaluation and checkpoint I/O. Checkpoint selection uses the minimum final training duration; unused budget comes from checkpoint granularity. GPU0 may share other work. Archived timings are noncontemporary.

## State continuation

| Problem | Variant | Status | All loss bits | Every semantic section | Exact restoration |
|---|---|---|---|---|---|
| toy | E22 | PASS | True | True | True |
| toy | CB64-RA | PASS | True | True | True |
| mnist | E22 | PASS | True | True | True |
| mnist | CB64-RA | PASS | True | True | True |

Each check comprises two saved CUDA checkpoint1000 continuations through1010. Every returned loss tensor and every semantic state section are compared after every update; global CPU/CUDA RNG, trainer streams, all controllers and optimizer states are included. Only birth_death.last.eval_seconds is excluded. Full endpoints and loss tensors are retained; CPU RNG buffers remain on CPU.

## Activity and resources

Candidate ordinary moves observed in learned runs: 7140. Full ordinary reaction/isolation/parent selection diagnostics are retained at every checkpoint. The required ordinary activity across declared training or frozen tasks is combined by root. Row-evidence asymptotic effective n99 remains below required384; observed fractions/counters are retained.

| Problem | Variant | Allocated peak MiB | Reserved peak MiB | CPU RSS MiB | Whole run s | Process s |
|---|---|---:|---:|---:|---:|---:|
| toy | E22 | 91.6 | 112.0 | 1658.3 | 74.82 | 75.32 |
| toy | CB64-RA | 57.2 | 74.0 | 1744.0 | 81.66 | 82.22 |
| mnist | E22 | 237.5 | 324.0 | 1909.7 | 93.39 | 94.68 |
| mnist | CB64-RA | 155.0 | 224.0 | 1995.2 | 104.91 | 105.83 |

CUDA memory peaks include setup/evaluator/checkpoint state across the process. PhysicalGPU0 only, fraction.2, CPU threads2, deterministic algorithms, serialized backward and TF32 disabled.

Frozen inputs and local source hashes verified against this lane freeze at reporting time. Source/data/config/checkpoint/endpoint receipts are in results.json and artifact-manifest.json. The earlier CPU study remains archived with its device scope; its quality failures supply no canonical CUDA verdict.
