# RMS-overlap group memory: four cheap association checks

The first [half-separation memory](sample-group-memory-filter.md) protected an already observed mode through an omitted minibatch, but can absorb a new mode when the cache is incomplete: exact cached centers `{0, 4}` give matching radius `2`, so a new center at `1` joins the old group at `0`. That suppresses acquisition even with noiseless data. Its frozen result remains a useful counterexample, not a general solution.

The revised [data-only memory](sample_group_dispersion_memory.py) stores each group's ordered float64 sample sum, positive integer count, and sum of squared sample norms. For a current MST group, it computes the centroid distance to the nearest **pre-bank** cached group. It matches only if that distance is no greater than both (a) the sum of the two groups' empirical RMS radii and (b), when at least two groups are cached, half the minimum cached centroid separation. Otherwise it appends the new group. An absent cached group's statistics remain unchanged. This is a geometric overlap rule based on observed real samples, not a Gaussian confidence interval or an anytime guarantee; it has no target labels, configured group count, gain or time-dependent optimizer rate.

The [predeclared free-output filter](sample_group_dispersion_filter.py) passes four distinct checks:

| Check | Result |
| --- | --- |
| Full eight-group bank, then archived component-0-absent bank | Current MST has seven groups, cache retains eight, absent centroid is bitwise unchanged, one output MM step remains 8 modes/HQ 1.0. Current-bank-only MM had removed the mode. |
| Start from archived incomplete seven-group bank, then full bank | Cache appends exactly one group. From the resulting seven-mode support, the first frozen-bank MM step remains 7 modes/HQ .91577; the second reaches 8 modes/HQ 1.0. |
| Exact cached Dirac groups `{0,4}`, current Dirac groups `{0,1}` | Old half-separation rule retains only two groups; revised rule has three, correctly detecting the new center at `1`. |
| One cached Dirac group `{0}`, current Dirac groups `{0,1}` | Revised rule retains the known group and adds the new one; no inter-cached separation is needed. |

The helper's strict state round-trip reproduces the cached centroids, rejects a Boolean count, and leaves the global PyTorch RNG untouched. Exact cached-centroid outputs are a fixed point of the free-output MM map to double precision. Source hashes, original conditioned-bank bytes, result, and deterministic gzip hashes are in the [manifest](continuous-evidence/round6-sample-group-dispersion/manifest.json). No neural update or additional data draw was run.

This still needs a host gate. Present-mode association requires the actual centroid difference to fit inside both its two empirical RMS disks and the cached separation cap. A novel group is found only if its current centroid exceeds the allowed radius to the nearest cached centroid. Finite Gaussian batches can violate either condition; singletons have RMS radius zero and may falsely split. Multiple current groups can match one cached group, while multiple groups far from the pre-bank cache can append separately. The proposed memory must be checkpointed with G, D, prior, optimizer moments, noise and RNG across continuation. It supplies no evidence that a truly removed target mode should be forgotten, and it does not solve mode-mass or full distribution fidelity.
