# Independent read-only review of the rare-case batch-feature winner

**No blocking bug found.** The winning `batchfeat_center6_distance_head` card
uses only coordinates in its current discriminator call. Real and fake batches
are passed separately, and the card contains no target labels, centers,
evaluation metrics, running statistics or feedback. Its fixed widths
`[.1,.25,.5,1]` and centered Softplus β6 trunk are declared in the D card.
The generator, particle prior, shared-c6 optimizer recipe, data, budget and
gates remain the original host settings.

The [static verifier](verify_static.py) independently checks the archived
source manifest and episode hash, exact D-only spec, recipe and optimizer
receipts, 24 observations, 1200 G/D updates and recomputed live verdict. It
also differentiates a fixed six-point input through the critic and native
`b_cap` without fitting. The score is permutation-equivariant; first input
gradients and second-order cap parameter gradients are finite and nonzero.
The measured sum of off-diagonal score-input gradient magnitudes is **13.88**
for that static input. Use `--checkout` to point the script at a checkout with
the winning module, and `--output /tmp/shared-batch-feature-review.json` to
retain its raw result locally.

The verifier requires the checkout's numerical source to match the selected
run's archive exactly. After the main-branch full-profile run, verify its
winner with:

```sh
python -m reports.transfer_suite.unadjusted.runs.shared-batch-feature-review.verify_static \
  --stage reports/transfer_suite/unadjusted/runs/rare-profile-replays/winner \
  --output /tmp/shared-batch-feature-review.json
```

Run outputs must exist locally. A source-version mismatch fails explicitly;
use an exact replay to validate a result after changing numerical code.

The meaningful limitation is **batch-context dependence**. For pointwise
critics, native `b_cap`'s `grad(sum_i D_i(X), X_j)` equals
`grad_{X_j} D_j(X_j)`. Here each score uses its neighbors, so the cap includes
cross-sample derivatives, and the generator receives cross-fake gradients too.
The six-point verifier finds an L1 difference of **9.21** between the cap's
summed-score input gradient and the vector of each logit's own-input gradient.
This is the unmodified native cap applied to a declared D architecture, and
is valid under the benchmark's D-only variant rules; it represents a
batch-contextual adversarial game, so it should not be described as a
pointwise critic with only added independent features. Scores also change
with batch composition and size, limiting transfer claims to other batch
settings. The benchmark deploys only G at evaluation.

The kernel tensor has `B² × 4` entries: at the original training batch 128,
that is 65,536 float32 values, or 256 KiB for this tensor alone. Pairwise
differences and autograd buffers add memory and work; the true cost is
quadratic in batch size. The 4096-sample behavioral scoring path calls G and
the metric scorer, not D, so it never constructs that larger kernel tensor.

The archived live curve has seven passing final checks. EMA has only a
two-check final passing suffix and remains a separate failure. This review
does not replace the main-branch exact replay or complete 19-case profile.
