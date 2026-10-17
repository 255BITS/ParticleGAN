# Check calibration feasibility before spending

`calibration-preflight` checks whether a frozen current-cohort profile can still
meet its declared adoption criteria. It preserves the exactly bound published
matrix's original validation semantics and checks its original receipt hashes.
An unpublished current profile uses the existing original-receipt reducer.
Neither path writes a report, registration, queue, or changed qualification, or
launches training. The published matrix is advisory and grants no qualification.

```sh
python -m experiments.forge calibration-preflight --profile host-profile-transfer-v1
python -m experiments.forge calibration-preflight --profile no-output-noise-reference-v1 --require-feasible
```

`INFEASIBLE` means no completion of the remaining UNKNOWN decisions can meet
the reference minima and error limits, or retained costs already violate a
frozen ceiling. For example, if every lineage has smoke FAIL, the requirement
for one reference-positive and zero false rejections cannot both be met.
The original results, unknowns, costs and criteria remain unchanged.

`POSSIBLE` is a necessary logical condition, not evidence that the screen works.
Hypothetical completion counts are labelled separately from observed decisions.
Missing cost vectors remain unavailable; a possible completion does not pass a
cost test. Missing or changed originals produce explicit receipt issues and
BLOCKED readiness even when an archived matrix retains its original decisions.
Later compatible attempts, retries and selected import history also block a
stale publication until its complete receipt and cost coverage is explicitly
refreshed. New unreadable request identities block because their relevance and
retry costs cannot be verified; unrelated readable scientific cohorts remain
excluded. This coverage check does not regrade the archived decisions.
Newer host schema requirements do not retrospectively turn the original failed
diagnostics into UNKNOWN. A published compact summary cannot grant a pass.
`--require-feasible` exits nonzero for INFEASIBLE or unresolved receipt issues,
and exits zero for a logically possible profile. A zero exit does not authorize
training or certify adoption.

New calibration-lane registrations reject an INFEASIBLE profile or unresolved
original receipts before source freezing or reservation. Existing immutable registrations and evidence are not
rewritten. A successor must retain its own version, scientific cohort, explicit
independent reference scope, criteria hashes, selected measurements and complete
task/candidate/campaign budgets. Diagnostic-only investigation of an infeasible
profile also requires a separately justified successor registration; it cannot
use a new lane to bypass this admission check. Do not fill the two failed profiles solely for
adoption, weaken their thresholds, or repeat a seed.

An accepted calibration still requires the existing `calibrate` report and
`verify_calibration` bindings. Independent references must be numerically
validated for the same actual recipe, task, prior, initialization, budget,
sampling and runtime cohort. Separate noisy served-policy outcomes from clean
MoG outcomes. Complete positive and negative references and cost vectors count;
unknown cells do not become failures, and diagnostics confer no qualification.

No new scientific screen or positive reference is fabricated by this tooling.
Preparation can be completed before a winning GAN exists. Default adoption and
registered robustness still depend on real, compatible qualification and
calibration evidence. Observed calibration errors on a bounded lineage set do
not establish population-level error rates.
