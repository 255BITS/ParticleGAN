# Actual RA12 auto: original Toy25/MNIST and CUDA continuation

The actual candidate fails the unchanged original learned Toy25 gate and
regresses against corrected E22 on original MNIST. Both original native vs
CPU-mapped CUDA continuations pass. This candidate has not met the shared
generalization objective.

| Original 2000-update result | Precision | Modes / recall | TV / active FD |
|---|---:|---:|---:|
| Frozen RA11 Toy25 | 0.965332 | 25 | 0.052114 |
| Actual RA12 Toy25 **FAIL** | 0.668579 | 22 | 0.336016 |
| Corrected E22 MNIST | 0.869141 | 0.847168 | 0.544488 |
| Actual RA12 MNIST | 0.767578 | 0.719238 | 1.880969 |

Toy25 requires precision >= .9, all 25 modes with supported mass >= .01,
and mass TV <= .1. Its gate AST is identical to the frozen original. MNIST
has no original numerical pass threshold; its original comparative metrics
and all earlier failures remain preserved. Image evaluator JSON is exactly
corrected E22, including 39 active dimensions. No scorer/gate was changed.

## Source and fixture validity

Source-aware validity passes for all 20 original checkpoints. Root-reviewed
package is `52cf04c3b629f054f81c9f9efdc8648ce71124c70e5d12d245a5c4876c5611b7`;
shared config is `9ae50f9fd0e903b7516ac04a72318ba79a35e52076393317511ec29ae19744b4`.
Only original N1024/z128/batch128 override that one shared configuration.
Current public init reproduces every original G/D/prior hash without RNG
consumption. Original data/seed314159/streams/two-real-batches-per-update,
serial backward, 2000 updates and ten checkpoints are unchanged. Frozen
draw/scorer code receives explicit noisy primary samples through a fixture
proxy. No extra begin_step, training forward or synthetic update was inserted.

Checkpoint zero retains pending auto selection. Toy selects feature_cells
from raw shape [2], finite-resolution certificate (minimum flags40 <=51)
and complete raw moment frame. Only generator/noise bases receive factor .25:
G/noise .0010625, table .0085 and D .00425. MNIST selects KNN from raw shape
[1,28,28]/width784 and retains G/noise .00425, table .0085 and D .00425.
Every recorded selection/rate certificate matches saved reaction state.

Toy metrics and applied LR match old RA11 exactly at 100/250/500/750, before
R1 fires at820, ratio2.096. MNIST metrics at0/100 match corrected E22 exactly,
before R1 fires at202, ratio3.984. Neither case has an anchor-release event.
These observations motivate investigation of endogenous detector fires;
they do not identify the driving optimizer group without per-step traces.

## Original curves

For Toy the last columns are mode count / mass TV; for MNIST recall / active FD.

| Fixture | Step | Precision | Modes / recall | TV / active FD |
|---|---:|---:|---:|---:|
| toy | 0 | 0.159302 | 1.000000 | 0.958779 |
| toy | 100 | 0.147095 | 2.000000 | 0.852905 |
| toy | 250 | 0.206543 | 8.000000 | 0.793457 |
| toy | 500 | 0.355469 | 20.000000 | 0.644531 |
| toy | 750 | 0.671265 | 25.000000 | 0.328735 |
| toy | 1000 | 0.337646 | 17.000000 | 0.662354 |
| toy | 1250 | 0.504395 | 22.000000 | 0.495605 |
| toy | 1500 | 0.647339 | 24.000000 | 0.352700 |
| toy | 1750 | 0.329102 | 14.000000 | 0.670898 |
| toy | 2000 | 0.668579 | 22.000000 | 0.336016 |
| mnist | 0 | 1.000000 | 0.000000 | 56.842754 |
| mnist | 100 | 0.000000 | 0.000000 | 199.791500 |
| mnist | 250 | 0.689941 | 0.004395 | 33.037647 |
| mnist | 500 | 0.684082 | 0.155273 | 15.847996 |
| mnist | 750 | 0.456055 | 0.149414 | 18.511378 |
| mnist | 1000 | 0.587402 | 0.210938 | 14.393982 |
| mnist | 1250 | 0.579102 | 0.130371 | 10.071581 |
| mnist | 1500 | 0.670898 | 0.510254 | 4.671100 |
| mnist | 1750 | 0.724609 | 0.639160 | 3.425155 |
| mnist | 2000 | 0.767578 | 0.719238 | 1.880969 |

## Native vs CPU-map replay

Both fixtures pass two independent ten-update continuations of saved step1000.
Branch0 loads native placement; branch1 maps the same checkpoint to CPU then
uses the original public loader to restore CUDA0. Full semantic restoration,
all per-update loss/state fingerprints and endpoint state are bit-identical.
Original primary sample bytes (Toy8192/MNIST4096, seed314259) also match, and
sampling leaves semantic training state unchanged. Only the original
observational `birth_death.last.eval_seconds` is excluded. Every tensor's
input/restored device, dtype and shape is recorded. CPU RNG/stream buffers
remain CPU uint8; R1 pending tensors restore onto CUDA0 in both branches.

Toy train time 148.676s; MNIST
93.547s. Launcher walls are175.078s/101.495s,
with replay31.099s. All phases signaled no process, held the existing shared
serial-phase lock inside the owned outer GPU0 slot and preserved all seals.
The lane was released at20:55:17UTC. `comparison.json` includes exact final
and minimum-final-training-time comparisons without changing any quality law.
