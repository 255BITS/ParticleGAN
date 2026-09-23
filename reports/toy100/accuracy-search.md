# Toy100 accuracy search: fixed-seed grid screen

These full 7,000-update CPU trials used seed 1234, the frozen 20,000-draw
live-weight gate, and the independently declared
[`toy100-accuracy-v1`](accuracy-diagnostics.md) diagnostics. The base is the
current low-learning-rate recommendation with its staggered batch override
removed; each trial changed only the fields named below. Complete configs,
events, scored final draws, logs, and source hashes are retained under
[`artifacts/toy100-accuracy/search-agent`](../../artifacts/toy100-accuracy/search-agent).
The one interrupted duplicate attempt is retained under `grid-screen` with
partial events; its completed rerun is the batch-1024 row below.

`Streak` counts consecutive passing original-gate evaluations at the end of
training; 5 are required. The new accuracy gate additionally requires mass TV
≤0.06, center RMS ≤0.20σ, absolute conditional covariance trace bias ≤0.10,
and radial KS ≤0.04. A lower accuracy score is better, but a score alone is
not a gate verdict.

| Grid trial | Streak | HQ | Mass TV | Center RMS / σ | Abs. cov bias | Radial KS | Accuracy score | Final accuracy | Original run gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Recommended: σout .026, batch 512, floor .05 | 5/5 | .98120 | .08345 | .6638 | .1794 | .0258 | 1.787 | FAIL | PASS |
| σout .029, batch 512 | 3/5 | .98400 | .07195 | .1541 | .0464 | .0252 | .766 | FAIL | FAIL |
| σout .030, batch 512 | 3/5 | .97990 | .07225 | .1616 | .0999 | .0489 | 1.058 | FAIL | FAIL |
| σout .029, floor .02, batch 512 | 4/5 | .98705 | .06815 | .1298 | .0002 | .0071 | .491 | FAIL | FAIL |
| σout .029, anneal start .30, batch 512 | 4/5 | .98445 | .07375 | .1515 | .0428 | .0236 | .751 | FAIL | FAIL |
| σout .029, batch 1024 | 4/5 | .98810 | .06120 | .1298 | .0121 | .0113 | .518 | FAIL | FAIL |
| σout .029, batch 2048 | **5/5** | .98955 | **.05250** | .1333 | .0159 | **.0052** | .457 | **PASS** | **PASS** |
| σout .029, cap 2, floor .02, batch 512 | 4/5 | .98850 | .05445 | **.1111** | .0009 | .0063 | **.408** | PASS | FAIL |
| σout .029, cap 2, floor .02, batch 2048 | 6/5 | .99065 | .06535 | .1245 | .0390 | .0141 | .613 | FAIL | PASS |

The shared batch-2048, σout .029, floor-.05 trial is the only tested
configuration that passes both grid gates. Its final precision is close to the
2D Gaussian oracle's 3σ probability .98889, while its mass TV and per-mode
moments are substantially closer to the target than the current recommendation.
It costs roughly four times the batch-512 compute and has **not** been promoted
to rotated or staggered: the low-learning-rate core failed several of the
other 19 toy tasks in the wider shared-recipe transfer check. The cap-2,
batch-512 trial has the best final score but reaches original gate quality at
step 6250, leaving only four of the five required terminal checks. Those
failures remain part of the selection evidence.

## Shared-v3-core residual generator branch

The separate scratch [probe source](accuracy_architecture_probe.py) tests a
generic data-space architecture under the established v3 optimizer core:
`G(z) = z + α·MLP(z)`, with a 2D learnable prior initialized uniformly in a
radius-6.5 disk and a Fourier-3 discriminator. All use batch 1024, 7,000
updates, seed 1234, and σout .029. The disk and architecture see no target
centers or labels. Each [run directory](../../artifacts/toy100-accuracy/search-agent/residual-architecture)
contains the exact copied probe source, declared configuration, model options,
source hash in provenance, events, final draws, and independently rescored
gate/accuracy JSON.

| α | Global cap | Input-noise end | LR anneal start | Final modes | HQ | Mass TV | Min in-radius count | Original gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| .01 | 6 | .5 | .6 | 68 | .8573 | .25550 | 0 | FAIL |
| .10 | 6 | .5 | .6 | 88 | .9714 | .16470 | 6 | FAIL |
| .30 | 6 | .1 | .6 | 80 | .9413 | .17635 | 0 | FAIL |
| .50 | 6 | .1 | .6 | 88 | .9585 | .16835 | 1 | FAIL |
| .30 | 2 | .1 | .6 | **94** | **.97935** | **.11155** | 33 | FAIL |
| .30 | 6 | .1 | .4 | 84 | .97655 | .18330 | 0 | FAIL |

The final row uses the exact v3 cap-6, output/input-noise, and annealing core
that passed the separate six-vector transfer screen. It is far from balanced
100-mode coverage. The cap-2 row is the best of these residual probes, but it
also fails the original gate and changes a global optimizer field. No residual
candidate was promoted to rotated or staggered. The branch stopped after
these full-budget failures; high early mode counts did not persist reliably.

The low-LR grid pass and the v3 residual failures are research evidence until
a single common optimizer/noise recipe and a single toy100 architecture pass
all 22 cases.
