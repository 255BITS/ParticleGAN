# Frozen minibatch discriminator feature screen

The earlier pointwise discriminator trials did not sustain rare-component
spread. This [eight-card plan](plans/shared_batch_feature_rare.json) tests
whether a discriminator can see within-batch concentration directly, using
generic current-batch features. No card uses labels, target centers, target
moments, evaluation metrics, an auxiliary loss, or a detached teacher.

The cards are frozen in this order before training:

| Card | Trunk | Current-batch feature and placement |
| --- | --- | --- |
| `batchfeat_center6_std_scalar` | Fixed centered 96×3 Softplus β6 | Mean feature standard deviation, head |
| `batchfeat_center6_std_vector` | Same | Full 96-channel feature standard deviation, head |
| `batchfeat_layer4_std_scalar` | LayerNorm 96×3 Softplus β4 | Mean feature standard deviation, head |
| `batchfeat_center6_density_head` | Fixed centered 96×3 Softplus β6 | Four local Gaussian densities, head |
| `batchfeat_center6_distance_head` | Same | Four local weighted square distances, head |
| `batchfeat_center6_density_input` | Same | Four local Gaussian densities, first hidden input |
| `batchfeat_layer4_density_head` | LayerNorm 96×3 Softplus β4 | Four local Gaussian densities, head |
| `batchfeat_center6_density_std_head` | Fixed centered 96×3 Softplus β6 | Four local densities plus mean feature standard deviation, head |

The fixed kernel widths are [.1, .25, .5, 1] in raw coordinate units. Each
sample's density or distance excludes itself and uses all other samples in
that same discriminator call. A real call and a fake call compute independent
current-batch features. Batch feature outputs are permutation-equivariant;
the feature mean/std path is differentiable, and the squared-distance kernels
have smooth first and second input derivatives. Every path to the score stays
attached to autograd. These cards are distinct from earlier pointwise radial
features anchored at fixed centers.

This is an explicitly **batch-dependent discriminator**. The unchanged native
`b_cap` implementation differentiates the sum of the batch logits with
respect to each input sample. For these cards that gradient includes the
cross-sample feature dependence; no separate or modified penalty is used.

All cards retain the original generator, prior, seed, data, 256 particles,
batch 128, 1200-step budget, 24 live/EMA observations, and behavioral gates.
The unchanged recipe is `shared_c6`: Rp logistic, b_cap 6 and κ1.25,
spread .05, no particle L2, Adam (0, .99), G/D LR .00425 and particle LR
.0085, 60% hold then cosine to a 5% floor. Stop at the first sustained live
PASS; otherwise retain all eight failures. A rare-only PASS still requires
separate main-branch replay and is not by itself a 19-case profile.

```sh
python -u -m benchmarks.transfer_suite.shared_batch_feature_search \
  --plan benchmarks/transfer_suite/plans/shared_batch_feature_rare.json \
  --output /tmp/shared-batch-feature-rare > /tmp/shared-batch-feature-rare.log 2>&1
```
