# Fixed initialization: public control runtime audit

All three completed results are valid failures of this frozen 1,200-update mode-hold screen. No coverage observation passed. The high-quality fraction alone does not establish coverage of all eight modes.

| Configuration | Passing observations | Final modes / 8 | Final HQ | Gate |
|---|---:|---:|---:|---|
| public-k3p-new-init | 0/24 | 6 | 0.970947266 | FAIL |
| public-ka2-new-init | 0/24 | 6 | 0.999267578 | FAIL |
| public-ka2-constant-new-init | 0/24 | 4 | 1.000000000 | FAIL |

Independent standard-library inspection verified every archived artifact hash, the complete package source and reviewed harness seal, all 1,200 frozen data/index/cursor receipt rows, all 24 observations and the final-five rule, and all 1,200 applied LR/noise rows. Runtime imports match declared isolated packages; FP32/deterministic/serial/default-CPU metadata matches the declared execution. The separate dry CUDA sampling preflight completed before any update and retained the training stream.

Both raw checkpoints for each control were read through a restricted storage-only parser without Torch. Initial complete non-RNG state matches the initial receipt, and all initial model tensors plus every independently listed parameter/buffer match the CPU repeatability preflight. Global, private and caller initial RNG states match the frozen host, while prior/network tensors use the new initializer rather than archived random weights. Final data RNG matches the frozen final cursor. Native Adam state is absent at construction; all 17 final raw counters are CPU scalars at 1,200 with CUDA moments, matching separate step1/1200 device receipts. Source/execution identities agree across checkpoints.

Scheduled K3P and KA2 retain declared horizon1,200 and noise milestones120/240. Historical constant-rate KA2 retains horizon4,600 and noise milestones360/720; its nominal G/D/prior rates remain .00425/.00425/.0085 through the screen. This is a comparison of the declared configurations, not an LR-only ablation. These failures do not transfer any old quality evidence or prove failure under every horizon.

[Full audit and source/checkpoint hashes](public3-runtime-audit.json). No new training, Torch import, GPU work, or replay was performed by this audit.
