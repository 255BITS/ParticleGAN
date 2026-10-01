# Declared indexed API metadata validation

This directory declares one validation expectation correction for the exact
already-frozen RA4 package: `collect.expected_options.evaluation_generate`
changes from `plain` to `indexed`. The original canonical collector/monitor,
validation lane, harness, package, task inputs, metrics and quality gates remain
unchanged on disk. The collector is compiled in memory through an import hook;
reverting this one constant reconstructs its entire original AST exactly.

The correction follows the pre-frozen AXIS API repair. The original harness
resolves `indexed` because `GANTrainer._generate` has a positional `indices`
parameter, then supplies sampled row IDs as the fifth positional argument.
The collector's old `plain` expectation described the previous API. No
numerical run is repeated: these RA4 jobs already executed the frozen indexed
sampling law, with their original data/init/stream checks and quality scorers.

## Outputs and provenance

The new live review output is:
`integration/review/ra4-indexed-api-monitor/`.

The original strict output remains:
`integration/review/ra4-validation-monitor/`.

Every adapted acceptance receipt includes its original strict status/reasons,
the exact declared API difference, original result/receipt hashes, and complete
AST/source guard proof. A `strict-plain-acceptance-receipt.json` is saved beside
each adapted receipt. Original strict ERROR receipts are preserved. The
inherited monitor's REPORT text still describes its original collector; this
README and the explicit adapter annotation in every JSON record identify the
declared exception. The new CHECKER-IDENTITY marks that exception explicitly.

Source gates pin the full RA4 package, root READY, source freezes, API method,
canonical harness, original collector and monitor, AXIS declaration, and prior
composition audit. Source/hash mismatches abort validation. CPU mechanism
reads use `map_location='cpu'` with hidden CUDA; no GPU context is opened.

## Pilot and independent review

`PILOT.json` is VALID. The first three affected screens and completed ring
screen replay all collector checks as VALID/PASS; their original primary
verdicts are unchanged. Five memory-only controls reject wrong options,
package, stream deviations and plain API mode, and preserve quality FAIL.
Source/task bytes and the original strict ERROR receipts are unchanged.

`READY.json` declares the exact hash-bound adapter; `PILOT-FROZEN.json` freezes
its sources and pilot. The first failed pilot, caused by omitting the exact
host's two unset budget-option removals, remains in `failed-attempt1/` with its
source/declaration/log. The correction matches frozen `screen.main` lines
1245–1250 and leaves budget/step checks intact.

Independent collector pilot:
`performance/training-regression/count-review/INDEXED-COLLECTOR-PILOT-FROZEN.json`.
The direct adapter review is performed by `state_review`; root will link its
frozen receipt alongside this declaration. The independent RA4 composition
receipt is `cpu-plan-review/composition-review/COMPOSITION-FROZEN.json`.

## Limits

This is metadata validation for this exact frozen RA4 API. It does not make
indexed RA4 numerically equivalent to an older plain sampling law. It changes
no score, threshold, streak, data/init fixture, stream requirement, step budget,
native terminal/holdout check, or warning semantics. All other discrepancies
remain INVALID/ERROR, quality FAIL remains FAIL, and unfinished tasks remain
PENDING. The three native 7000-update jobs still require their original full
schedules, terminal clouds, independent 100k holdout, initial parameter/range
receipts, and official source/gate checks before an adapted verdict is valid.
