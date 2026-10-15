# ParticleGAN research attempt catalog

Use this catalog to find what Mikkel and the research team actually tried,
which original gates passed or failed, and what can be reused for the next
comparison. Positive and negative findings share the same source-bound trail.
Preparation, incomplete execution and analytical proposals remain visibly
different from complete trained results. This catalog grants no qualification.

Start with the [recent attempt catalog](ATTEMPTS.md), its [CSV register](run-register.csv),
or the [older study catalog](STUDIES.md). Exact hypotheses, declared changes,
controls, original statuses, source/revision/task hashes, seeds, budgets,
initial priors, serving/evaluation laws, archive pins and publication locations
are retained in [attempts.json](attempts.json) and [studies.json](studies.json).
The linked original publication receipts contain complete contracts and
metrics; the catalog summaries do not replace those original receipts.

## Most useful published entry points

| Finding / question | What the evidence says | Reuse / next implication | Publication |
|---|---|---|---|
| Confidence acquisition versus retention | The unchanged six-task screen passes **6/6**; own-word hold passes **25/25**. Original Gaussian retention fails | Preserve actual optimizer histories and test each recipe's own continuation; an acquisition confidence proxy is not a retention certificate | [PR405](https://github.com/255BITS/ParticleGAN/pull/405), [screen readout](https://github.com/255BITS/ParticleGAN/blob/8a29473c9e52a38df7711c25fa058da68189f247/reports/forge/generator-confidence-lr-bcap-direction/README.md) |
| Signed/positive-excess response | Both acquisition gates pass; full retention fails. Positive-excess final quality is inside bounds while stationary/deadline/shift requirements still fail | A passing endpoint cannot waive sustained failure; zero parameter motion and continuing moment evolution are different facts | [response variants](https://github.com/255BITS/ParticleGAN/blob/8a29473c9e52a38df7711c25fa058da68189f247/reports/forge/generator-confidence-lr-response-variants/README.md) |
| One extra critic reaction | Corrected task-law and D1→D2 analysis finds terminal falsifiers and an acquisition reversal; zero new experiments | Reject this exact stop-gradient proposal before spending. Local W2 improvement is insufficient without the fixed mean/variance and acquisition prerequisites | [analytical rejection](https://github.com/255BITS/ParticleGAN/blob/8a29473c9e52a38df7711c25fa058da68189f247/reports/forge/confidence-critic-reaction/README.md) |
| Generator matrix geometry | CPU full-Frobenius movement repairs overlap in two separately recorded seed cohorts but loses conditional trajectory; fixed half-polar/half-Frobenius repairs that trajectory and retains seed-0 overlap | Retain source/build-specific positives and regression controls; broader CUDA and whole-suite preservation do not follow | [geometry lessons and exact source pins](CPU-GEOMETRY.md), full study records in [STUDIES.md](STUDIES.md) |
| Cell-moment geometry | Forty-five full runs retain 35/36 aggregate passes and 5/9 unique control passes; metric regressions and Bures Ring failure remain | Use exact scoped positives as controls, then require original long retention and heterogeneous transfer | [merged PR402](https://github.com/255BITS/ParticleGAN/pull/402) |
| Gradient history and quota continuity | Five of twelve Gaussian retention recipes pass. Quota continuity repairs mode hold and preserves Gaussian stability but fails unequal-mass covariance | Gaussian success does not establish a global survivor; quota continuity does not alone solve heterogeneous shape | [204-run receipt](https://github.com/255BITS/ParticleGAN/blob/dacf96749a4effc0648a7a0945266d508804110e/reports/forge/test-performance-evidence/nonatlas-reconciled-204.json), [quota/Astra receipt](https://github.com/255BITS/ParticleGAN/blob/dacf96749a4effc0648a7a0945266d508804110e/reports/forge/test-performance-evidence/astra-cumulative-24.json) |
| Twelve rejected portfolio mechanisms | All twelve hypotheses fail full unequal-mass gates; three covariance endpoints pass while their terminal suffixes fail | Preserve exact negatives and avoid endpoint tuning or replay of the same failed revisions | [portfolio records](ATTEMPTS.md#astra-remaining-mechanisms) |
| Later local history/reference/banks | Twenty-one further non-Atlas runs contain nine passes and twelve failures, with complete source/revision identities | Crossing useful objectives and optimizer laws does not automatically preserve either component's result | [published PR399 receipt](https://github.com/255BITS/ParticleGAN/blob/e71cd6ea13f841092ed346b3217ecee0c4d53d29/reports/forge/test-performance-evidence/successor-nonatlas-21.json) |
| EMA hard-idle and device/replay repair | Control 5/6, candidate 4/6; dropped Gaussian/word histories remain INCOMPLETE, and the control word timed out | Keep device/replay fixes separate from adopting the unsuccessful idle formulation | [PR399](https://github.com/255BITS/ParticleGAN/pull/399) |
| Atlas historical and current compatibility | Restored-state compatibility, adapted shape gains and causal negatives stay in their own cohorts. Later Gaussian/Ring horizons ended in checkpoint-export ERROR; four questions are BLOCKED and clock audit FAILs | Complete-state export and real host compatibility are part of completion. A blocked task is not a numerical failure | [separately owned PR404](https://github.com/255BITS/ParticleGAN/pull/404), [ten Atlas lessons](ATLAS-LESSONS.md) |
| Test stage separation | Fixed-target quality, stationary retention and adaptation have distinct measurement contracts; the split adds no experiments | Keep old combined contracts and grades; missing submetrics stay UNMEASURED | [PR403](https://github.com/255BITS/ParticleGAN/pull/403) |

## What is counted

The frozen recent register has **285 distinct canonical attempt IDs**:
**142 PASS / 136 FAIL / 7 INCOMPLETE**. These are original source-specific
verdicts across different protocols, not a pooled performance score.
The reconciliation is **249 non-Atlas full runs ∪ 81 successor attempts**,
with **45 shared IDs**. Shared controls and copied cumulative entries count
once. The known non-Atlas update total is **509,400**; an exact all-attempt
update total remains unavailable because the timed-out word control has no
exact final count.

| Publication boundary for the frozen recent register | Unique IDs |
|---|---:|
| Merged into develop through PR402 | 228 |
| Additional evidence published on draft PR399 | 42 |
| Confidence follow-ups published on draft PR405 | 15 |
| Total, disjoint within this verified snapshot | 285 |

**PR404's three later Atlas starts are outside this 285-ID snapshot.** They
are linked through the owner's exact head and lesson index, not silently added
to the historical register. Its seven-question row retains four BLOCKED,
two INCOMPLETE/ERROR and one FAIL. The lesson index itself adds no experiment.

The older ledger retains **312 study identities and 406 comparison records**:
100 trained comparisons, 77 diagnostics, 129 full-horizon diagnostics and six
unrun records. It also retains [65 original arm records](original-arms.json).
These collections overlap source results, controls, reproductions and analyses;
they are not added to the 285 count. Older study/arm identities are preserved,
but missing canonical execution UUIDs are not invented. Incomplete/invalid and
unrun work receives no scientific-win credit. Reproduction strengthens the
specified original cohort; it is not a new mechanism or a current-source pass.

[Seventeen operational records](operational-records.json) retain twelve
pre-runtime startup ERRORs, three CPU nonfinite preflight rejections and two
64-update mechanism prefixes against an original 7000-update quality horizon.
They are outside the 285 attempt register and receive no scientific-win
credit. Their value is documenting admission, numerical-preparation and
completion limits. The prefix pair contributes 128 actual bounded updates,
not two full-budget quality experiments.

No exhaustive project-wide experiment total is claimed. Martyn's recipes and
results appear only as explicitly named controls or external reference
cohorts; unrelated Martyn PRs are not credited as Mikkel's output.

## Evidence and remaining gaps

[coverage.json](coverage.json) states the exact boundaries and exclusions.
[archive-pins.json](archive-pins.json) records available byte hashes and missing
files. `archive://future-workspaces/<date>/<task>/...` denotes the original
local task archive, and `archive://hypergan/...` denotes the original project
experiment archive. These locators are not public downloads. Public evidence
links use exact Git commits. Catalog publication does not copy tensor states,
raw stdout, per-update streams or checkpoint arrays into Git.

The audit checked all 285 recent logs, requests and terminal records, 251
successor byte pins including 73 tensor/checkpoint files and 29 GIFs, and the
147-member original Atlas closure. The original data remain recoverable in
that checked scope. Six historical transfer mode-hold runs lack saved sample
panels and retain explicitly logged-metric grades. The timed-out PR399 word
control lacks its final raw result, final scored archive and final state.
Four older Atlas dynamics attempts have no complete grade files. Some older
human reports, preparation timings and matched controls are unavailable; the
study records preserve these limitations and distinguish UNKNOWN from FAIL.

The source hashes in the catalog are retrieval and identity checks, not new
model decoding, scientific regrading, causal certification or qualification.
Later analysis cannot fill a missing trained gate. The never-started finite-cap
retention dependency and the rejected analytical critic-reaction proposal each
add **zero training attempts**.

## Maintain this entry point

Generate the tables and CSV with `python reports/forge/research-catalog/generate.py`;
use `--check` to verify deterministic freshness. This standard-library command
reads only committed catalog metadata and never trains, samples, changes a
scientific gate or updates the ordinary qualification/selection machinery.
Add future evidence with canonical IDs and original receipt pins, group shared
controls/reproductions, and update [coverage.json](coverage.json) explicitly
before changing any denominator. Keep the original result and source cohort
visible even when a later recipe or implementation supersedes it.
