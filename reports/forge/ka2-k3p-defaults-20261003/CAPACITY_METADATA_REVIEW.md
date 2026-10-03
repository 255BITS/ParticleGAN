# KA2/K3P capacity metadata review

The immutable packet records all 16 CPU snapshots as SUPPORTED under the new family-owned law. This read-only metadata review found no capacity/law inconsistency. It did not load a model, restore state, read array values, invoke a sampler or scorer, initialize CUDA, or use Git. Root’s source-bound replay remains the sampler certifier.

Frozen source: `26ff278c3796d775969391adc0bde52e3af11149`. Capacity SHA-256: `f291f8dc1d5c8557f7047580de52e4df3f0d7f5a86ea0b4e9f344486b557f923`. Snapshot digest: `48bc63ed6c48e7fdb0028313b950ebb0946598fd100f60a612c4ecec16bc990e`. Root preflight reports 3,404 snapshot files, 177 definitions, the eight original case hashes and all sixteen resolved Recipes with zero new models, draws, updates or CUDA.

| Family | Required case | Capacity | Captured updates / full horizon | Eval samples | Recorded HQ | Recorded TV |
|---|---|---|---|---|---|---|
| ka2 | `image-develop-img_intensity2-source-transpose12` | SUPPORTED | 0 / 600 | 1,024 | 1.000000 | 0.002930 |
| ka2 | `api-vector-two-broad` | SUPPORTED | 0 / 1,200 | 4,096 | 0.981201 | 0.006592 |
| ka2 | `api-grid100` | SUPPORTED | 0 / 7,000 | 20,000 | 0.990050 | 0.026400 |
| ka2 | `api-rotated100` | SUPPORTED | 0 / 7,000 | 20,000 | 0.990050 | 0.026400 |
| ka2 | `api-staggered100` | SUPPORTED | 0 / 7,000 | 20,000 | 0.990050 | 0.026400 |
| ka2 | `api-vector-unequal-mass` | SUPPORTED | 0 / 1,200 | 4,096 | 0.985840 | 0.009209 |
| ka2 | `api-vector-anisotropic` | SUPPORTED | 0 / 1,200 | 4,096 | 0.989990 | 0.007731 |
| ka2 | `image-develop-img_bars4-source-transpose12` | SUPPORTED | 0 / 600 | 1,024 | 1.000000 | 0.014648 |
| k3p | `image-develop-img_intensity2-source-transpose12` | SUPPORTED | 0 / 600 | 1,024 | 1.000000 | 0.002930 |
| k3p | `api-vector-two-broad` | SUPPORTED | 0 / 1,200 | 4,096 | 0.981201 | 0.006592 |
| k3p | `api-grid100` | SUPPORTED | 0 / 7,000 | 20,000 | 0.990050 | 0.026400 |
| k3p | `api-rotated100` | SUPPORTED | 0 / 7,000 | 20,000 | 0.990050 | 0.026400 |
| k3p | `api-staggered100` | SUPPORTED | 0 / 7,000 | 20,000 | 0.990050 | 0.026400 |
| k3p | `api-vector-unequal-mass` | SUPPORTED | 0 / 1,200 | 4,096 | 0.985840 | 0.009209 |
| k3p | `api-vector-anisotropic` | SUPPORTED | 0 / 1,200 | 4,096 | 0.989990 | 0.007731 |
| k3p | `image-develop-img_bars4-source-transpose12` | SUPPORTED | 0 / 600 | 1,024 | 1.000000 | 0.014648 |

All cells use the exact shared tuple LR 0.006375 / prior multiplier 1 / critic multiplier 1, seed 24002 and evaluation seed 34002. Fast-only serving, no DV12 perturbation, AMSGrad false, fixed output-noise warmup and the original full 600 / 1,200 / 7,000-update horizons are recorded. The serialized Recipe standardize flag is true, while the actual ParticlePrior row read is not standardized.

Every capture is at zero completed updates after one original begin-step/abort-step prelude, with empty optimizers and recorded state/global-RNG purity. All sixteen reported primary observations pass their recorded bounds; the metrics above are copied from the packet, not recomputed. The 32 declared state/sample paths and byte sizes were checked, but their tensor or array contents were not loaded or replayed by this review.

Native captures request noisy primary samples, but effective sigma is zero at clock zero. Their radial row construction passes the recorded cold snapshot gate; it does not prove that later fixed sigma 0.029 training or sampling will converge. Original Atlas image parameters and analytic vector/native rows are construction inputs, without old policy, optimizer, clock or grade credit. Matching KA2/K3P snapshot metrics are expected from the shared constructions and fixed evaluation seed; their complete state fingerprints differ, and no training or algorithm equivalence follows.

Learning is UNEXECUTED at this review boundary: all sixteen scientific cells remain UNKNOWN. Q1 grants no terminal-noise, persistence, default-adoption or speed qualification. The packet’s 10.171206 CPU seconds (13.077 seconds reported process time) are separate diagnostic work, not ordinary learning credit. Prior campaign cost 113.99425188452005 seconds remains debited once from the original 15,360-second cap.

Future goal-GIF QA will use only root-announced immutable evidence. It must pair the original grade with the added first-five acquisition and at least five later all-passing checks; show actual targets and outputs; identify fast-only / no DV12; and keep noisy native primary separate from output-noise-off diagnostics from the same selected fast policy. No forced EMA branch or independent 100k native gate is declared here. The earlier shared score index remains immutable at its own earlier boundary.

Full checks, copied metrics, artifact metadata and future media requirements are in [CAPACITY_METADATA_REVIEW.json](CAPACITY_METADATA_REVIEW.json).
