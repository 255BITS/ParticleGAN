# Continuation: R1/R2, BCap and release 0.7 comparison

The user requests these comparison arms after compaction. This expands the next
study beyond the noise-removal ablation; authorization and scope persist. This
note pins identities and launches no training. [Structured pins](FORMULATION_COMPARISON_CONTINUATION.json)
bind the release sources and observed default fields.

- **R1/R2:** historical `a_r1r2`, the zero-centered squared-L2 real/fake gradient
  penalty. It differs from K3P's early real R1 plus one-sided fake cap.
- **BCap:** historical `b_cap`, a one-sided squared gradient cap. Preregister its
  coefficient, norm and cap; do not run a tuning sweep.
- **0.7 release:** tag `v0.7.0`, commit
  `180d18f400335fb295611d624b48a4e072ae3bae`, whose default `Recipe` is **GAN v3**:
  `b_cap`, coefficient 6, cap 1.25, Adam betas (0,.99), prior spread .05,
  z-dimension 4, batch 256 and its original full-budget schedule. This is a full
  released formulation, not just another BCap coefficient. The separately
  documented `LOCKED_SHARED` demo stamp is not the same released recipe.
- **K3P control:** retain current and historical evidence under exact identities.

The current 0.8 public API removed the historical arm selector and legacy
penalties. Frozen copies remain under `benchmarks/legacy/`; they are source
references, not permission to silently bypass Forge's public API. First inspect
and implement the smallest shared public formulation/penalty extension with
parity checks and truthful mechanism observations. Do not copy a training loop
for each arm. This changes scientific source and requires a new resolved cohort;
older diagnostic receipts cannot automatically fill it.

Before reserving training, distinguish two questions: faithful behavior of the
released package under its original recipe, prior and sampling law, versus a
matched-host mechanism comparison. The released GAN v3 and mechanism-only BCap
control are distinct unless their full resolved formulations are identical;
identical requests must reuse evidence. Pin original particle/cloud exceptions
and current learned-MoG variants separately. Keep the standard MoG sigma .025,
uniform masses and no standardization for current comparisons.

Declare historical noisy and current clean scoring before measurements, keeping
live/EMA distinct. Report both without picking whichever passes. Shared training
can produce paired diagnostic sampling outputs where the supported contracts
allow it; they must not count as independent reference tasks or grant duplicate
qualification. Preserve all 16 reference purposes and unchanged criteria; no
failed family, endurance requirement or terminal check may be dropped.

Use seed 0 with named isolated streams and matched initialization on shared
parameters; no seed-only runs. Resolve current GPU ownership, queue root,
preflight, task runtime identities and resources afresh. Freeze only the next
small informative task selection with explicit candidate/campaign budgets, then
execute through Forge and publish tail commands, metrics and measured costs.
This continuation note has **zero training allowance**; it is not a registered
campaign or authorization to reuse the old study's unused reservation.

The preceding study is complete at `41e34ade` on
`codex/tiered-experiment-qualification`, pushed to PR #221. Source remains
`5c9c9298…`. Its sole noise-removal native run failed for 93.746687201 seconds,
and the new roster was infeasible; the exact candidate is abandoned. Seven older
receipts were reused once, not rerun. Memory has 233 records and no pending
readouts. [Completed readout and leaderboard](../SCIENTIFIC_CALIBRATION_20260930_READOUT.md),
[promotion blockers](../SCIENTIFIC_CALIBRATION_PROMOTION_READINESS.json), and
[saved geometry](NO_OUTPUT_NOISE_SAVED_GEOMETRY.md) preserve what passed, failed
and remains unknown. Do not revive its matrix or extend the failed parent.

The user's earlier promotion recollection is correct: K3P became the package
default after historical 22/22 passes. PR #209 later changed sampling/scoring to
omit training output noise; Forge also uses a different MoG/initialization
protocol. Current failures do not establish spotty identical repeats or an
implementation regression. This comparison should resolve those differences
before another open-ended formulation search.
