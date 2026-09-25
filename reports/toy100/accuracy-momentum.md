# Shared first-moment probe

This [candidate](../../configs/toy100/accuracy_momentum.json) changes one
optimizer mechanism in the strongest current transfer core: G and D use Adam
β=(.5, .999), while the learned particle prior keeps β=(0, .999). The prior
override is explicit because its sparse particle visits have different
gradient history from the dense network weights. All other global fields are
the β2=.999, prior-LR-multiplier-3 transfer recipe: G/D LR .00425, prior LR
.01275, cap coefficient 6 at κ=1.25, prior regularization .05, anneal from
60% to floor .05, output noise peak .029 warmed over 20%, and input noise .5
reduced to zero by 20%. Toy100 would use one batch size 2048 for all three
geometries; transfer hosts retain their frozen resource sizes and budgets.

The eight-task bottleneck screen (local evidence: `artifacts/toy100-accuracy/compatibility/beta1-half-bottleneck8`)
checks trajectory, mode hold, unequal mass, unequal width, vector overlap,
stripes, bars, and blobs using unchanged live-weight gates. It retains the
config, source snapshot, protocol, compressed episodes, and numerical
readouts.

| Frozen host | Verdict | Terminal pass streak | Decisive final readout |
| --- | --- | ---: | --- |
| trajectory | FAIL | 0 | Identity MSE .4072 |
| mode_hold | FAIL | 0 | 7/8 modes, HQ .827 |
| vector_unequal_mass | PASS | 15 | Mass TV .0291, minimum mass ratio .745 |
| vector_unequal_width | FAIL | 0 | Minimum component covariance eigen-ratio .091 |
| vector_overlap | FAIL | 1 | Covariance error .427 |
| img_stripes2 | PASS | 9 | 2 modes, HQ 1.0 |
| img_bars4 | PASS | 7 | 4 modes, HQ 1.0 |
| img_blobs4 | FAIL | 0 | 3/4 modes, HQ .969 |

Only 3/8 frozen bottlenecks pass. The aggregate summary is marked
`INCOMPLETE` by design because it is a subset of the nineteen-case suite, but
five failed gates already reject this as a common recipe. Global G/D first
momentum improves unequal-mass allocation here while substantially damaging
trajectory identity, mode retention, covariance, and blobs. The branch stops
without a full-19 or toy100 grid promotion.
