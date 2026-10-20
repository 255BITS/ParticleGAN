# Generator geometry: scoped positives and preservation limits

The Oct 11 CPU research produced concrete improvements that remained local.
This summary publishes the most reusable comparisons while retaining their
original source, build, initialization, task and baseline contracts. It adds
no implementation, new run, GPU equivalence claim or qualification.

## Exact full-Frobenius reproduction

Study `task-e50af8b0-7461-4bec-b941-2418027347bc` uses frozen source
`d378734f40b09ce223a389e8f54a9783ec6a0c75`, original direction-blend view 8,
public named orthogonal initialization and separately recorded seeds 0 and 1.
The complete overlap horizon is 1200 updates with 24 original 4096-sample checks;
the original bounds remain SW1≤.18, mean≤.15, covariance≤.45 and terminal
suffix≥5. The recipe changes only full G matrix Frobenius movement relative
to its declared raw-bias/provenance control.

| Task / cohort | Control | Candidate | Original sustained verdict |
|---|---|---|---|
| Overlap seed 0 replication |14/24 passes, suffix3/5 |15/24, suffix 5/5 |FAIL→PASS |
| Overlap seed 1 generalization |14/24, suffix3/5 |16/24, suffix 5/5 |FAIL→PASS |
| Broad seed 0 preservation |20/24, suffix20/5 |18/24, suffix18/5 |PASS→PASS |

The seed 0 overlap endpoint mean and SW1 worsen despite the new full PASS.
Broad loses two early passing checks while improving endpoint mass/shape.
Those tradeoffs remain in the original comparison metrics. The broad control
is reused after exact prelaunch identity checks; it is not an additional fresh
control run. A later full-Frobenius conditional trajectory fails, so the exact
recipe is not a preserves-all-wins default.

## Fixed half-polar/half-Frobenius comparison

Study `task-f3fa0cca-aa40-477a-b203-a636fd1b989d` is one bounded seed 0
CPU 2.13 comparison. A global fixed half-sum changes only the G matrix proposal,
retaining source polar cutoff, smoothing, aspect gain/rates, joint direction
blend, raw G biases, unit path cap and unit provenance.

| Task | Control / candidate result | Original complete protocol |
|---|---|---|
| Conditional trajectory |FAIL 0/24→PASS 18/24; identity MSE .021320→.000101 |400 updates / 24 checks |
| Overlap |PASS 15/24→PASS 17/24; suffix 5 retained |1200 updates / 24 checks |

Overlap endpoint covariance worsens from .230126 to .402998 and mean/SW1 also
worsen while remaining inside original bounds. The complete original panels
were rescored in the archived experiment; this publication performs no regrade.
Wider preservation, current builds and archived CUDA equivalence remain
unmeasured. Later length-matched and combined extensions retain negative
transfer results in the study catalog.

## Why these results help

They isolate matrix movement from raw biases and joint direction constraints,
provide exact positive controls, and show that a geometric repair can lose a
conditional task even when it preserves broad. Reuse the specified matched
source/build/task control and original gates; do not infer a global recipe
from overlap endpoints or transplant a CPU result to a new CUDA cohort.

Full comparison fields, recorded decisions, limitations and local artifact
hashes are in [studies.json](studies.json), [the study table](STUDIES.md), and
[archive-pins.json](archive-pins.json). The original readout identities are
`archive://future-workspaces/2026-10-11/task-e50af8b0-7461-4bec-b941-2418027347bc/REPORT.md`
and `archive://future-workspaces/2026-10-11/task-f3fa0cca-aa40-477a-b203-a636fd1b989d/REPORT.md`.
Their exact original byte hashes are preserved; this prose is a publication
summary, not a rewritten original receipt.
