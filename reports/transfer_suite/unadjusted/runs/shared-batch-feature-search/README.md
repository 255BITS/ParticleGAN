# Shared-c6 current-minibatch discriminator screen

**The fifth frozen card passes `vector_unequal_mass` with seven consecutive
final live observations.** The runner stopped immediately, as declared in the
[eight-card plan](../../../../../benchmarks/transfer_suite/plans/shared_batch_feature_rare.json).
Five cards were trained; the three later cards remain declared and unrun.
The first four failures, winner, complete 24-point live/EMA curves, actual
optimizer receipts, original specs, D cards, and exact source archive are
retained in the [screen index](screen/index.json) and [log](screen/run.log).

| Frozen order | Architecture | Live status | Final passing streak | Final minimum eigen ratio |
| ---: | --- | --- | ---: | ---: |
| 1 | `batchfeat_center6_std_scalar` | FAIL | 0 | .2347 |
| 2 | `batchfeat_center6_std_vector` | FAIL | 0 | .0166 |
| 3 | `batchfeat_layer4_std_scalar` | FAIL | 0 | .1459 |
| 4 | `batchfeat_center6_density_head` | FAIL | 3 | .1560 |
| 5 | `batchfeat_center6_distance_head` | **PASS** | **7** | **.6329** |
| 6–8 | Declared in plan | Unrun after first PASS | — | — |

The winner is a fixed centered, width-96 three-hidden-layer Softplus β6
discriminator with a four-dimensional smooth current-batch feature appended
to its score head. It has **19,013 trainable parameters**. For each input
sample `i` and fixed scale `s` in `[.1, .25, .5, 1]`, the feature is

`sum(j != i, k(i,j,s) * ||x_i-x_j||²) / (s² * (sum(j != i, k(i,j,s)) + 1e-5))`,
where `k(i,j,s) = exp(-||x_i-x_j||² / (2s²))`.

The feature uses only the current discriminator input batch. It is smooth,
permutation-equivariant, and attached to autograd. Real and fake calls
compute their own features. Training and evaluation mode use the same rule;
there are no running statistics. Behavioral evaluation samples the generator
and prior without calling the discriminator. The unmodified native `b_cap`
implementation differentiates the sum of batch logits, so its input gradient
includes cross-sample paths when this batch-dependent D is used.

| Live metric | Worst of final five | Required |
| --- | ---: | ---: |
| Minimum normalized component eigen ratio | .42385 | ≥ .15 |
| Component covariance error | .36954 | ≤ .85 |
| High-quality fraction | .97119 | ≥ .85 |
| Mixture mass error | .04260 | ≤ .15 |
| Normalized sliced distance | .04979 | ≤ .18 |
| Minimum mass ratio | .67233 | ≥ .25 |

The [preflight](preflight.json) confirms permutation equivariance, nonzero
cross-sample input gradients, and finite first/second input derivatives and
native `b_cap` parameter gradients for all eight declared cards. The
[primary importer check](import_check.json) accepted the five trained
episodes with existing shared-c6 evidence, including source, optimizer,
setup, duplicate-architecture, and behavioral validation. It assembled
19/19 supported tests; this screen itself is only one test and does not
replace an independent replay or fresh full-profile verification. The
winner's EMA curve separately fails the sustained gate.
