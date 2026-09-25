# Final frozen LayerNorm Softplus beta interpolation

The LayerNorm-all width-96 three-layer Softplus(beta=5) critic remains this
study's closest unequal-mass architecture. Its final component minimum eigen
ratio is .11905 against the .15 gate, with additional failures earlier in the
five-checkpoint final window. Width changes to 64/128/160, beta 2/10, and a
zero-initialized raw score skip did not pass. Separately, a related raw
Softplus(beta=8) width-128 critic achieved a sustained unequal-width PASS in a
parallel study. This motivates a last narrow check of activation curvature in
the LayerNorm rare-mass lead; it does not change the fixed recipe or gates.

The [four-card plan](plans/shared_pointnorm_beta_interpolation_rare.json) is
frozen before training: LayerNorm after every hidden linear layer, raw 2D
input, width 96, depth three, and Softplus beta 3, 4, 6, or 8. All four run
the unequal-mass case only. There are no data-dependent features, batch
statistics, feedback from evaluation during training, or seed sweeps. All
source and optimizer settings are archived; every attempt keeps its 24 live
and EMA observations. A PASS still requires five consecutive final live
observations meeting every original bound. This partial screen alone cannot
support a full 19-test score.
