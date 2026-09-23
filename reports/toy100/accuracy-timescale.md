# Shared generator-timescale probe

This is one declared, target-agnostic [recipe](../../configs/toy100/accuracy_timescale.json)
for the frozen transfer suite and toy100. Relative to the public v3 core, the
generator learning rate is .0015, while discriminator and learned-prior rates
remain near their public absolute values at .0045 and .009 through global
multipliers 3 and 6. Adam β=(0, .99), cap coefficient 6, κ=1.25, prior
regularization .05, LR anneal start .6/floor .05, output noise peak .029 with
20% warmup, and input noise peak .5 ending at 20% are common across hosts.
Each host keeps its frozen architecture, batch, particle count, and budget;
toy100 would use one batch size 2048 for all three geometries. No target centers
or task-specific optimizer overrides enter training.

The hypothesis is that the original low-LR toy100 success required a slower
generator, while its failures on the other 19 tasks may have come from slowing
the discriminator and prior at the same time or changing the cap/betas. The
eight-task bottleneck screen (local evidence: `artifacts/toy100-accuracy/compatibility/g-timescale-bottleneck8`)
checks two-pole, trajectory, mode-hold, three difficult vectors, and two image
cases with frozen gates and complete replay receipts. The saved protocol,
source snapshot, compressed episodes, index, summary, and log make every
verdict independently reviewable.

| Frozen host | Verdict | Terminal pass streak | Decisive final readout |
| --- | --- | ---: | --- |
| two_pole | PASS | 12 | Mean absolute separation .730 |
| trajectory | FAIL | 0 | Identity MSE .1288 |
| mode_hold | FAIL | 4 | Final 8/8 modes, one terminal check short |
| vector_unequal_mass | FAIL | 3 | Mass TV .0794, minimum mass ratio .537 |
| vector_unequal_width | FAIL | 3 | Minimum component covariance eigen-ratio .166 |
| vector_overlap | PASS | 8 | Final normalized sliced Wasserstein .0712 |
| img_stripes2 | PASS | 20 | 2 modes, HQ 1.0 |
| img_bars4 | PASS | 14 | 4 modes, HQ 1.0 |

The frozen subset passes 4/8. Its summary correctly marks the scope
`INCOMPLETE` because only eight of nineteen hosts ran, but the four failed
frozen gates already reject this as a common recipe. The branch stops here;
there is no full-19 or toy100 grid promotion. Slowing only G while keeping D
and prior rates near v3 does not reproduce the successful toy100 behavior in
a recipe that transfers across the older tasks.
