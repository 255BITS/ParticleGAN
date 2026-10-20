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

<!-- current-comparison -->

| Global recipe | Hard VAEGAN | Categorical VAEGAN | Task passes |
| --- | --- | --- | ---: |
| `bispf-optimized-quintic-v1` | [FAIL](media/bispf-optimized-quintic-v1-model_vae_hard.gif) (2/5 terminal; suffix 1) | [FAIL](media/bispf-optimized-quintic-v1-model_vae_categorical.gif) (3/5 terminal; suffix 2) | 0/2 |
| `bispf-vaegan-transport-v1` | [FAIL](media/bispf-vaegan-transport-v1-model_vae_hard.gif) (0/5 terminal; suffix 0) | [FAIL](media/bispf-vaegan-transport-v1-model_vae_categorical.gif) (0/5 terminal; suffix 0) | 0/2 |

All measured arms complete 1,200 updates and24 checks; the last five determine
PASS. This table is the single current comparison for `vaegan_coverage`.
Every revision and failure remains in [compact results](results.json).
GPU0 was shared under load; recorded costs supply no speed ranking.

<!-- /current-comparison -->

Round1 completed all four full-budget runs and all failed the sustained gate.
The matched baseline passes2/5 hard and3/5 categorical terminal observations,
although both endpoints pass. Transport passes0/5 for both, with final likelihood
HQ.8191/.6858 and decoder-mean HQ.8372/.6948 (hard/categorical). Good reconstruction
remains insufficient. These GPU measurements do not replace the archived CPU
categorical PASS; this study has not isolated the source of that cohort difference.

The [saved terminal-state diagnosis](round1-diagnosis.json) uses zero updates and
an explicitly separate CPU row-stratified panel. Baseline hard and categorical
misses remain entirely concentrated on rows below1% expected posterior use at
all five states. Transport also creates errors among higher-usage rows. Hard
transport has13–14 low-usage rows versus11–12 for baseline; categorical has8–9
versus7. The auxiliary did not repair posterior coverage. All state files remain
byte-exact; the detailed row arrays stay in the artifact archive.

Round2 keeps transport and activates the existing cosine schedule from update0,
with network/prior floors.05, over the same1200-update task horizon. This is a
schedule-only repair with one global recipe; it explicitly gives up the baseline's
constant-rate claim. Saved update1000 groups still use G/E.012, prior.03 and
D.018. Lowering these together tests whether finite normalized updates keep
otherwise recoverable rows fluctuating. The cosine schedule is related to the
annealing used in [SGDR](https://arxiv.org/abs/1608.03983); our single decay has
no restarts and makes no general GAN convergence claim. The complete numerical
gates decide success. This comparison cannot establish that transport is necessary.
