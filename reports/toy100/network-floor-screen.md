# Shared network-floor transfer screen

The single declared recipe with a network-only learning-rate floor of 0.005
passed **8 of 9** frozen transfer bottlenecks. It does not establish a
19-toy or 22-toy pass. The full 19-host replay was withheld under the
predeclared rule requiring 9/9 in this screen.

This run used the source at commit `a3be165e2ab47290d35ed98426be77d148f04320`
and [`accuracy_network_floor.json`](../../configs/toy100/accuracy_network_floor.json),
with one CPU thread, seed 0, the original host budgets, data, architectures,
and thresholds. Relative to the preceding κ=1.0 shared recipe, its sole
control change is the global `network_lr_floor=0.005`: generator and critic
rates use that floor, while the prior retains the recipe floor of 0.05.
The policy declares a 1600-step network horizon across tasks. It names the
affine-square generator for native toy100 tasks; transfer host architectures
remain frozen. All nine episodes used the same output noise 0.029
with 20% warmup and input noise 0.5 ending at 10% of each host budget.

| Frozen bottleneck | Strict live gate | Terminal passing suffix |
| --- | --- | ---: |
| Trajectory | PASS | 9 |
| Mode hold | PASS | 5 |
| Unequal mass | PASS | 20 |
| Unequal width | PASS | 22 |
| Anisotropic | PASS | 12 |
| Overlap | PASS | 7 |
| Two stripes | PASS | 8 |
| Four bars | **FAIL** | 4 |
| Four blobs | PASS | 19 |

The network floor repaired the preceding κ=1.0 misses on overlap and stripes.
Four bars missed exactly one sustained checkpoint: at step 500, its live HQ
was 0.875 against the frozen 0.9 threshold, although all four modes were
present. Steps 525–600 pass, including final HQ 0.90625, but the gate
requires five consecutive passing observations and has only four. Its EMA
readout remained at three modes in the terminal checks.

The original run and a relocated copy both passed independent source,
configuration, schedule, noise, and receipt integrity checks. Strict regrading
returned `FAIL 8/9` in both places. Raw local evidence is at
`artifacts/toy100-accuracy/compatibility/network-floor005-nine/` and retained
in RAM at `/dev/shm/particlegan-toy-floor-nine-a3be165/`. The durable copy's
20 original files match the SHA-256 values in `files_sha256.json`; its
source archive SHA-256 is
`71c64f39e3af1aa4c9c694212fa3011304a4e884afaf1af5678147256ca40dbf`,
and config SHA-256 is
`544fca1b89fe39950e4be0b37dd07f4787e4f8e2433e84c0ad1c4cf478f2e455`.
The `artifacts/` paths are local workspace evidence, not GitHub links.
