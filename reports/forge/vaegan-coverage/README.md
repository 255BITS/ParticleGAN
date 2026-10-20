# VAEGAN prior coverage with BISPF

This bounded experiment compares the inventory-selected
`bispf-optimized-quintic-v1` against one global transport repair on the existing
hard and categorical VAEGAN toys. Both train a shared decoder/generator with
adversarial loss and a genuine Gaussian negative ELBO including KL. The original
CPU hard FAIL and categorical PASS remain under their original source and
contract; the present matched GPU0 cohort supplies its own baseline.

The [saved-state diagnosis](../vae-failure-investigation/README.md) found all183
hard decoder-mean misses on low-posterior-usage prior rows. The passing categorical
control has the same concentration with26 misses, so usage alone does not prove
causation. Hard ELBO gradients reach4/16 rows at its saved endpoint; independent
prior sampling remains uniform over16 rows. Intermediate failure states were
unavailable, motivating complete scheduled checkpoints in this study.

[Larsen et al.](https://proceedings.mlr.press/v48/larsen16.pdf) combine variational
inference and adversarial decoder training; their D0 ablation uses pixel-space
reconstruction. Our vector Gaussian observation model follows that scoped
variant, with discrete MoG inference rather than their image architecture.
[Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) motivates examining
aggregate posterior/prior mismatch. Our data-space auxiliary is an inference
from that diagnosis, rather than a reproduction of WAE.

The [frozen protocol](protocol.json) allows matched baseline and candidate on
both tasks:4 runs,1200 updates each,1200 reserved seconds. If either task fails,
inspect saved states before up to two separately frozen substantive repairs,
600 seconds/two runs each, at most2400 reserved seconds total. No seed study,
unchanged repetition, gate changes, outcome-selected checkpoints or automatic
promotion follows. Every failed revision stays in the current comparison.

Round1 activates the existing public `Recipe.kinetic_transport_loss` with
weight1 and32 deterministic directions on independent-prior generated batches.
It matches projected empirical quantiles, normalized by detached target variance;
gradients reach generator and sampled prior rows, while the target is detached.
No new training samples, labels, target anchors, RNG stream or persistent
controller are introduced. Adversarial loss, KL/reconstruction, Born-Infeld
critic, optimizer, thermal policy, learned prior updates and SBNP weight.01 remain
unchanged. Architecture, initial prior/capacity, actual sampling, seed0 public
initializer, data batches,1200 updates and24/five sustained checks are task-owned
and fixed across recipes. This tests coverage, not calibrated mode variance.

The [diagnostic view](../../../configs/forge/views/vaegan_coverage.json) owns only
these two tests and confers no ordinary qualification. The [source migration](source-migration.json)
retains original task blobs/hashes after adding optional read-only observation
checkpoint recording; archived grades are unchanged. Full model, optimizer,
policy, constructor and named-stream states are saved at every actual check.

Bulk artifacts and tail-friendly logs are outside Git:
`/nvme/dev/particlegan/artifacts/vaegan-bispf-20261019/stdout.log`.
