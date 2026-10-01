# RA16 portability closure

RA16 is cloned from immutable RA15. Five modules differ: seven CPU planning
factory calls now explicitly allocate on CPU, and backend restore validates
the exact FIFO sample shape before creating/installing controls or mutating
models, optimizers, RNG streams, parameter versions, or live controls.
An unresolved output shape cannot carry an initialized FIFO. Resolved output
shapes may still carry an untouched FIFO, and legacy KNN shape metadata stays
compatible. Checkpoint schemas, config bytes, arithmetic, ordering, budgets,
R1 trigger, serving guard, and quality thresholds are unchanged.

## Executed CPU evidence

- 94 contracts pass: 58 existing source contracts, 17 portable partial
  recovery contracts, and 19 new portability contracts. CUDA is visible and
  remains uninitialized.
- `meta` default contexts exercise the actual streamed projection, complete
  and partial frozen moments/odd queries, absent-moment proposals, an accepted
  paired packet including real chart queries/epoch guards/hash, feature
  control installation, and empty/nonempty population scalar queries.
- Trainer and policy restore reject product-preserving shape mismatches
  atomically in feature and KNN fallback routes. Unresolved initialized-FIFO
  cases are rejected, while valid pending/resolved untouched-FIFO states load.
- The original RA15 mixed-device projection failure and accepted malformed
  shape followed by generated-observation failure are preserved in the CPU
  bridge receipt.
- Default CPU RA15/RA16 loss and complete checkpoint state match at every
  update for 18 feature updates through two reactions (2048 mean-forward
  rows) and two KNN fallback updates. Samples, bidirectional valid restore,
  and the next update match exactly. Only the diagnostic elapsed clock is
  shared. AST normalization proves the only source changes are the seven CPU
  keyword additions and the pre-mutation shape validator.

## Prepared GPU contract

`test_cuda_default_preserves_feature_reactions_and_checkpoint_replay` runs the
same seed1234 and explicit GPU models/table/data under CPU and CUDA defaults,
through 18 updates, valid checkpoint restore, exact next update, and samples.
It skips only if CUDA is unavailable. The root agent's serial GPU queue owns
execution; it is not part of this completed CPU count. The CPU fixture permits
an already initialized CUDA runtime and rejects any new lazy initialization.

Old RA15 packages, lanes, failure logs, and qualification remain unchanged.
This source bridge preserves the ordinary CPU-default helper allocation law;
it does not claim that the deferred actual CUDA-default test has passed.
