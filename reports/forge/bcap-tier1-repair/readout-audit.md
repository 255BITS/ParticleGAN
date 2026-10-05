# Independent final readout verification

**PASS for evidence integrity; no fully passing BCAP recipe.** The read-only
[checker](audit_readout.py) verifies the final
[aggregate](results.json) and both configuration-search reports against all paid
receipts. Its compact [proof](readout-audit.json) retains the exact report hashes.
The historical selected incumbent remains unchanged; this readout creates no
additional leaderboard or qualification.

The matrix is exactly **12 ordinary candidates × 7 tasks + 2 duration diagnostics
= 86 unique paid attempts**, represented by 13 candidate rows. Counts are
**40 PASS / 46 FAIL**, with no missing, retried or substituted task cell. Every
ordinary recipe has at least three failures; the maximum is four passes out of
seven. Paid cost is **2,687.373636 seconds**:

| Campaign | Paid seconds |
| --- | ---: |
| Eight rate recipes | 1,749.030615 |
| Four moment recipes | 884.242491 |
| Two duration diagnostics | 54.100531 |

The checker verifies canonical result/certificate identities, independently
hashed raw grading, candidate revision and actual recipe bindings, prior and
initialization receipts where recorded, task metrics and charged costs. It
checks all 2,336 files in the two frozen source snapshots and reconciles final
search-report task cells and receipt bindings. No campaign retains a reservation,
and each paid total stays within its declared cap.

All **1,786 temporal observations** have their bounds and final passing suffix
recomputed independently from the declared task cards. The 12 schedule audits are
checked separately against their declared tolerances and source hashes. Every
classification agrees with its certified grade. All **634 Gaussian/ring
observations** are rescored from the retained tensors; every gate metric matches
exactly. Word scored records match the numerical observations and independently
recomputed pass/fail decisions; this check does not recreate missing word logits
from the display views. Two complete checker executions produce byte-identical
proof JSON and preserve CPU RNG. Negative controls reject an endpoint-only pass,
a nonfinite metric and a missing observation.

## Scientific interpretation

The lower-rate round supplies three sustained Gaussian passes, but all eight
recipes fail direct-coordinate movement and ring acquisition. The four moment
recipes restore movement at global LR `0.00425`, while all four fail Gaussian,
ring and sustained word acquisition. Direct-particle betas remain `[0, 0.9]`;
movement recovery must not be credited to shorter candidate beta2 alone. That
round changes the global rate as well as moment memory relative to the slower
round.

Eight failed cells finish with passing endpoint bounds: three Gaussian, four
word and one AE hold. Their prior terminal observations fail, so their sustained
FAILs are correct under the declared question. A final image or final number
cannot replace that criterion. Of the four moment WORD runs, three endpoints
pass; beta2 `.99` / prior multiplier `1` still fails at the endpoint with three
modes, quality `.59375`, reconstruction failure and minimum token probability
`.009016`. The fourth endpoint-pass WORD cell belongs to the rate round.
Certified aggregate metrics represent final live metrics, not an earlier failed
row. Every one of the 60 ordinary late ring checks
fails full-component covariance; 41 also fail high-quality mass and 21 fail mode
count. These are substantial distribution failures, not evidence that the ring
gate should be weakened to core covariance.

The original [duration audit](duration-audit.md) independently proves exact
24-check prefixes and partial repair without a sustained pass. It supports no
ordinary budget increase at the tested horizons.

The gates remain **provisional scientific criteria**. Passing exact oracle
controls establishes scorer behavior, not calibrated false-rejection rates for
trained models. The [task audit](task-audit.md) shows that near-boundary Gaussian
laws can fail a five-check 4,096-sample test through measurement noise, while the
incumbent drift and broad ring tails are much larger effects. A future precision
revision should retain the numerical bounds, freeze a larger evaluation sample
count and validate positive/negative controls before new training. It must keep
its sampling identity separate from these results.

The shared-failure premise also needs its original scope: selected K3P ring
attempt `d6d907c833524c1ea23374d90c548cf6` passes all five original late checks.
Its recipe differs from the BCAP incumbent, so that receipt establishes an
achievable ring gate without isolating the penalty as the causal difference.

## Two bounded follow-up angles, after review

1. **Separate direct-coordinate speed from generator/prior speed.** The same
   global learning rate currently couples rapid 80/200-update movement to stable
   distribution learning. Declare one public direct-coordinate rate multiplier
   applicable to every relevant host, then test at most two whole recipes using
   a slow Gaussian-performing base rate and movement-compatible direct rates.
   First validate the actual public optimizer against numerical displacement
   controls. Keep beta2, sampling, budgets and gates fixed; complete all seven
   Tier 1 tasks for each candidate. Reject the proposal if it restores movement
   by sacrificing Gaussian, word or hold behavior. It does not itself resolve
   the independently observed ring-tail problem.

2. **Isolate BCAP penalty behavior against the passing K3P ring recipe.** Freeze
   a comparison with the selected K3P control's actual LR `0.006375`, D multiplier
   `1`, prior multiplier `1`, coefficient `1`, initialization and clean sampling.
   Change only the penalty arm for one bounded ring diagnostic; reuse the
   original K3P receipt only within that exact source/recipe comparison. Retain
   original gates and all 24 observations, and collect critic-gradient/cap
   activity using existing scored draws with an explicit state/RNG purity check.
   If BCAP also passes, earlier recipe choice is the leading explanation. If it
   fails while the matched control passes, inspect the measured gradient units
   and prior forces before declaring at most one targeted cap/penalty revision.
   This diagnostic supplies no whole-recipe qualification; any resulting
   candidate still owes all seven ordinary tasks.

Both are new scientific hypotheses, not authorizations for more work in the
concluded campaign. They preserve failed evidence and avoid task-by-task recipe
selection or repeated seeds.

## Reproduce without training

```sh
PYTHONPATH=. .venv/bin/python -u reports/forge/bcap-tier1-repair/audit_readout.py \
  --root . --output /tmp/bcap-readout-audit.json
cmp reports/forge/bcap-tier1-repair/readout-audit.json /tmp/bcap-readout-audit.json
```

The saved raw artifacts and frozen snapshots must be available at their recorded
queue paths. The checker adds zero training updates and zero model-sampling
draws. Its PASS certifies the stated readout verification scope, not scientific
calibration or BCAP qualification.
