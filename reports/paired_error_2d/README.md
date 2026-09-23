# Paired 2D transport: MSE-free cap/schedule transfer

**The extraction reproduces all 12 matching model-glue runs exactly.** There is no observed numerical discrepancy in the tested 2D cases. The earlier application headline of 8.2% worse EMA error pooled three seeds and included a 3D task; this extraction holds seed 0 fixed and includes only its two 2D tasks. Those different aggregates must not be compared as if they were the same experiment.

The standalone host uses this PR's public ParticleGAN losses, gradient cap, VIC and cosine schedule. It imports neither model-glue nor particle-sliders. The routed architecture, known target maps, training-only normalization and random streams are extracted with pinned provenance. This adds a transfer case without changing the nine-host behavioral ranking or production defaults.

[Task and commands](../../benchmarks/paired_error_2d/README.md) · [Source provenance](../../benchmarks/paired_error_2d/SOURCE.md) · [Full measurements](results.json) · [Exact artifact comparison](reference-audit.json) · [Runtime/source hashes](protocol.json)

Seed 0, 6,000 updates, batch 64, 1,024 training points, 1,024 validation points. Each row uses its validation-selected EMA checkpoint; live is scored at that same step. Historical test points (4,096) are reused for reproduction, not claimed as a fresh holdout.

| Task | Recipe | Cloud | EMA test NMSE ↓ | Live test NMSE ↓ | EMA p95 ↓ | Stable live |
| --- | --- | --- | ---: | ---: | ---: | --- |
| affine2 | baseline | fixed | 1.97680856e-05 | 0.000295227248 | 0.01119229 | True |
| affine2 | baseline | movable | 1.89389048e-05 | 0.000267451862 | 0.01089154 | True |
| affine2 | cap-cosine | fixed | 1.64690791e-05 | 2.11970691e-05 | 0.01024519 | True |
| affine2 | cap-cosine | movable | 1.67789949e-05 | 1.99767437e-05 | 0.01038984 | True |
| affine2 | cap-cosine-vic005 | fixed | 1.64690791e-05 | 2.11970691e-05 | 0.01024519 | True |
| affine2 | cap-cosine-vic005 | movable | 1.75206133e-05 | 2.20082584e-05 | 0.01037537 | True |
| swirl2 | baseline | fixed | 0.00224647718 | 0.00572161516 | 0.08047713 | True |
| swirl2 | baseline | movable | 0.0021647315 | 0.00492737163 | 0.08058671 | True |
| swirl2 | cap-cosine | fixed | 0.00245497725 | 0.00256467308 | 0.07799378 | True |
| swirl2 | cap-cosine | movable | 0.00218096073 | 0.00224257563 | 0.07634751 | True |
| swirl2 | cap-cosine-vic005 | fixed | 0.00245497725 | 0.00256467308 | 0.07799378 | True |
| swirl2 | cap-cosine-vic005 | movable | 0.00242614676 | 0.00250734505 | 0.07644982 | True |

## Matched recipe comparisons

Geometric mean candidate/baseline EMA error ratio over the two tasks. One seed, descriptive only; do not use this to claim a universal winner.

- movable, cap-cosine: **0.944773×**; task ratios [0.8859538140141934, 1.0074971086161142].
- movable, cap-cosine-vic005: **1.018248×**; task ratios [0.9251122743455894, 1.1207610559066135].
- fixed, cap-cosine: **0.954169×**; task ratios [0.8331145181741725, 1.092812015406259].
- fixed, cap-cosine-vic005: **0.954169×**; task ratios [0.8331145181741725, 1.092812015406259].

## Application-artifact reproduction

All 12 extracted runs exact: **True**. Comparison includes full final G/D/EMA/Adam/sampler/global-RNG state, all recorded validation metrics, checkpoint selections, and historical live/EMA test scores. See `reference-audit.json` for every mismatch; no tolerance turns a mismatch into PASS.
## Interpretation and recommendation

For the movable cloud at seed 0, cap + cosine reduces affine EMA NMSE by 11.4% and increases swirl EMA NMSE by 0.75%. Its geometric mean ratio across these two tasks is 0.944773 (5.52% lower). Reducing VIC to .05 increases swirl error by 12.1%; its two-task ratio is 1.018248. These reproduce the corresponding original rows. They do not reproduce the **aggregate** 8.2% regression because that number includes a different task/seed set.

Live errors improve with the scheduled candidates, while all twelve 2D runs already pass the live and EMA sustained thresholds. Lowering VIC has no effect in the fixed control: its final generator/critic/optimizer/EMA states match the other scheduled fixed arm. Particle movement remains a matched control, not an established general advantage.

The evidence supports keeping this as an explicit transfer benchmark, retaining current defaults, and evaluating new formulations on paired fidelity and live/EMA behavior separately. There is no evidence here of an application loss-sign, normalization, target-order, lazy-penalty or optimizer-state bug. This is not a proof about untested high-dimensional or pretrained-model paths. Merely placing the same problem in ParticleGAN does not change its results.

The historical test set is reused strictly to audit reproduction. No checkpoint or parameter was tuned against it, and no fresh-held-out or seed-robustness claim is made. The earlier three-seed application results remain historical context; this PR adds no seed experiment.

## Verification and execution

- **49 focused tests passed**, including seven new task/runner checks and existing public primitive/regularizer/convergence tests. The existing regularizer test emitted one tensor-to-scalar warning.
- The early independent 16-update float32 comparison matched G, D and EMA exactly for all three recipes. The full run then matched all final G/D/EMA/Adam/sampling/global-RNG state, all 13 validation observations per run, selected steps and all live/EMA historical test metrics. Every comparison uses exact equality, without a tolerance hiding differences.
- The public cosine helper differs from the application's algebraic expression by at most 4.44e-16 in its scalar multiplier across the budget; final learned tensors, optimizer states and measurements still match exactly in FP32.
- All **12 runs / 72,000 updates** completed in **81.40 seconds wall time** with four CPU workers. Timing is one shared-host observation, not a speed benchmark.
- The first new launcher attempt failed before any training because multiprocessing could not import a worker defined in a package `__main__`. Moving workers to `run.py` fixed it; the new spawn-import regression test passes. That packaging failure does not affect the original application experiment. The failed attempt was retained locally rather than counted as a training result.
- Ruff and staged whitespace checks passed. Runtime source hashes and the full comparison output are retained above. Generated checkpoints and point samples stay under ignored artifacts.
