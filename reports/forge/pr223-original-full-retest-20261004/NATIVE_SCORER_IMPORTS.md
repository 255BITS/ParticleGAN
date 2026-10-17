# Prospective native scorer import repair

This patch changes future scorer bootstrap and copied-source metadata preflight.
It changes no scientific call, package implementation, Recipe, observation,
model, stream, gate, budget, or prior evidence. It does not resume or certify the
stopped run.

Root reports that the frozen `2068a661` / `00cadbfd` run stopped with 16 accepted
PASS, native grid100 INVALID, and two NOT_RUN. Its native child retained 7,000
updates and 34 observations, but produced ERROR and no final attestation. The
reported stack is `native-score-observed-wrapper.py: main -> _retest_check ->
guard_imports`, ending with `ValueError: ambiguous/missing protected namespace
lib`. These are reported execution facts; this source-only change reads no
actual numerical output or checkpoint and supplies no numerical credit.

The original `native100_score.py` explicitly uses a separate process to import
the frozen native package. Its `main` inserts the relocated native root before
importing `benchmarks.toy100.gate` and `accuracy_gate`; their train dependency
imports `lib.toy_models`. The new envelope prefix first loads `run_retest.py`,
which makes the envelope snapshot root searchable. Both roots contain a PEP420
`lib` directory. `guard_imports` already canonicalizes aliases, so alias removal
alone cannot collapse these two distinct directories. A synthetic reproduction
of the exact generated prefix and original scorer import sequence triggers the
same error. The actual failure log did not record `lib.__path__`; the synthetic
two-root count is a source-derived reproduction, not a measurement from that
run.

The prospective wrapper now calls `normalize_native_scorer_paths` immediately
after the original native-root insertion, before importing either scorer. It
removes only the exact envelope snapshot root and canonical aliases of that
root, retains every other distinct path, and keeps the original native root
first. The already-loaded envelope control modules retain their pinned source
paths. A stdlib `PathFinder` check, without executing source modules, requires
one pinned native `lib` namespace and the original native owners of
`lib.toy_models`, `particlegan`, `benchmarks`, the toy100 package, both scorers,
and train. Foreign/ambiguous namespaces, regular-package substitution, missing
source, stale hashes, a different native root, or a preloaded candidate package
are refused. The existing strict imported-source guard, inherited descriptor
verification, process-group fencing, and deadline checks remain unchanged.

The predecessor copied preflight compiled all generated wrappers and checked
only modules loaded by the envelope. It did not resolve the separate native
scorer's import sequence. Future copied preflight now performs the same
metadata-only native-owner resolution and records its source-bound proof. A
missing or substituted native proof is rejected before registration/admission.
This adds no original scorer call, model import, forward, draw, observation,
update, or score. Parent work remains inside the existing metadata ledger.

`test_native_scorer_imports.py` uses synthetic packages and the original scorer
SOURCE ONLY. Its subprocesses load the exact generated prefix and stop after
import validation, before either synthetic score function can run. They prove
the predecessor collision, the corrected single native owner, unchanged strict
foreign refusal, and zero Torch/model/scorer/sampler/queue calls. They also check
stale/missing source and preflight proof substitution. These controls establish
bootstrap behavior, not a scientific outcome.

Any future use requires a new committed helper/import closure, derived-wrapper
hashes, prepared source identity, copied preflight receipt, and explicit
root-approved execution/accounting scope. The immutable `2068a661` / `00cadbfd`
snapshot, its preflight, all 16 accepted rows, native INVALID, and two NOT_RUN
remain unchanged. This patch grants no retry, regrade, capacity, default, speed,
or fresh full19 credit.
