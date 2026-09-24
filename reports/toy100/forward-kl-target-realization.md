# First likelihood targets can be realized by the actual neural generator

Both isolated copied-model checks pass with the existing pre-start joint
G/prior Gauss–Newton budget of20 iterations and12 halvings. This removes one
cheap implementation blocker, but is not native GAN training or a hold test.

| Original pre-state | GN iterations | GH9 before | GH9 target / actual fitted | Late-noise grade |
| --- | ---: | ---: | --- | --- |
| Cold1 |7|4486.952029|1.718853542 /1.718853584|8 modes, HQ .999756|
| Warm1324 |5|2.217382|.200632113 /.200632112|8 modes, HQ .999512|

The fixed targets are the first accepted outputs of the source-bound
[finite-GH9 screen](round8-forward-kl-gh9-stress.md). Each original pre-G neural
model and learned prior is copied, then fit from its original parameters.
The actual post-fit finite-GH9 objective is reevaluated. Both source real banks
and original pre-fit objectives match their free-output fixture exactly.
Caller RNG and complete saved input states remain unchanged. No Adam step,
critic update, clock advance, parameter-gain selection or quality-based
acceptance occurs. The evaluator alone uses true means.

The cold objective uses its actual output sigma0 plus the fixed smoothing
bandwidth; the table's sample-quality grade uses the separate standard late
output noise .029. They are not the same emitted law. The numerical target
fit tolerances and all source hashes are in the
[portable archive](continuous-evidence/round9-forward-kl-target-realization/manifest.json),
including copied fitted models, inputs, targets and every solver receipt.

The next neural experiment must use actual finite-GH9 acceptance, preserve
complete learner history and account for the live native updates. The
remembered-data rescue before rest addresses the independent current-bank
symmetry trap. Neither this successful fit nor the pure-output screen proves
long-term parameter stability or production distribution fidelity.
