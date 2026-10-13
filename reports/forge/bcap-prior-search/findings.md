# What the prior searches establish

Prior regularization can change a failing BCAP task into a pass: the initial
mean/covariance tether at global strength **0.01** repairs `mode_hold`. This
same recipe loses the leader's word-retention pass. A task repair therefore
does not establish a better replacement for the leader. The final aggregate
report records all fifteen recipes against the stricter registered criterion:
six Tier 1 passes, at least one Tier 2 repair, and every control Tier 2 pass
preserved.

## The repair changes the sustained window

Mode hold requires a terminal passing suffix of at least five observations.
The leader records **3 / ≥5**; the moment tether records **7 / ≥5**. At update
1,050, both cover **8 / ≥8** modes, but their high-quality fractions are
**0.869384765625 / ≥0.9** and **0.998046875 / ≥0.9**, respectively. Both final
endpoints pass, so an endpoint-only comparison would hide the repair.
[Original receipts and discriminating observations](covariance/repair-witness.json)
bind this comparison to the actual recipes, initialization, source, streams,
budget and sampling law.

Word retention exposes the tradeoff. The tether passes **23 / ==25** mandatory
checks in each primary and independent confirmation cohort; the leader passes
**25 / ==25**. At global update 5,167, the tether covers **4 / ==5** modes and
has mass TV **0.203515625 / ≤0.1** in the primary cohort and
**0.2046875 / ≤0.1** in confirmation. At update 6,167, minimum reconstruction
token probability is **0.7967675924301147 / ≥0.9** in both cohorts. Its endpoint
recovers, but the mandatory intervening failures remain failures.

The original word protocol restores each recipe's own earliest certified
acquisition prefix. These holds begin after 834 updates for the control and
4,167 for the tether, then execute the same 4,000 new updates. This comparison
grades that declared pipeline; it does not start from identical trained
weights. The [moment-tether readout](covariance/readout.md) retains this
exception explicitly.

## Global latent statistics do not guarantee local output density

The moment tether reduces word-hold full-table mean/covariance discrepancy
from **0.4674966931** for the control to **0.0001051523**, despite losing the
word gate. It restrains its intended latent statistic; that statistic is an
incomplete proxy for output retention. This saved-state diagnostic adds no
training or qualification credit and compares the declared own-prefix holds.

For the three 100-Gaussian tasks, the same tether's independent live holdout
accuracy precision is **0.29935**, **0.22570** and **0.34371**, each targeting **≥0.97**.
Coverage recomputed from those same draws gives high-quality mode counts
**17**, **10** and **17**, each targeting
**==100**. The targets also require component mass, covariance, centers and
radial density accuracy; covering a broad output footprint is insufficient.
Full-location moment discrepancy improves on grid and rotated layouts but
worsens on staggered. [Saved moment diagnostics](covariance/moment-diagnostics.md)
and the aggregate native-density table keep those measurements separate.

Coulomb repulsion at strengths **0.01** and **0.1** reaches Tier 2, but repairs
no failed task. The former loses word retention; the latter loses
`vector_two_broad`. For the latter, the endpoint satisfies its bounds, but
the terminal suffix is **3 / ≥5**. At update 1,050, normalized SW1
is **0.2445433295 / ≤0.18**, versus an endpoint of **0.1750254555 / ≤0.18**.
[The Coulomb readout](coulomb/readout.md) explains the original window and
saved geometry. This objective uses neighbors by supplied-row index, so this
search does not test spatial nearest-neighbor repulsion.

SIGReg and the proximal spring do not reach Tier 2 at any of their registered
strengths. All six recipes fail Ring16; some also fail another Tier 1 gate.
Their Tier 2 cells are blocked measurements, not observed formulation
failures. [SIGReg](sigreg/INTERPRETATION.md) and
[proximal spring](proximal/readout.md) list each score and target. Passing a
small public-API mechanism demonstration establishes execution and intended
local behavior; it does not substitute for ordinary task qualification.

The initial-distribution MMD tether at **0.1** clears Tier 1. Its Ring16
full-component covariance error is **0.54613911 / ≤0.85**, with a passing
suffix of **11 / ≥5**. It repairs none of the eighteen non-native Tier 2
tasks and loses word retention. That hold resumes its own certified prefix
at 1,667 updates, versus the control's 834, then adds the declared 4,000
updates. Both mandatory cohorts pass **24 / ==25** checks. At global update
2,001, output quality **1 / ≥0.95**, mode count **5 / ==5**, and mass TV
**0.03671875 / ≤0.1** primary and **0.02890625 / ≤0.1** confirmation all pass;
exact reconstruction is **0 / ==1** and minimum token probability is
**0.19466007 / ≥0.9** in both cohorts. The endpoint recovers. This is a
different word failure from the covariance tether's missing-mode observation:
preserving these marginal output checks did not preserve inverse
reconstruction. [The MMD readout](mmd/interpretation.md) retains the original
cohort and observation hashes. Strengths 1 and 10 are blocked by Tier 1,
including the AE10 reconstruction witness described in that readout.

MMD at 0.1 finishes at **8/21 Tier 2**, with no repairs. Its grid, rotated and
staggered independent accuracy precision is **0.24413**, **0.23907** and
**0.34425 / ≥0.97**; same-draw coverage mode counts are **11**, **12** and
**22 / ==100**. All three native tasks fail for each of the four positive
recipes that qualified to run Tier 2. The other eleven recipes never measured
those tasks. The prior hypotheses tested here therefore supply no demonstrated
100-Gaussian solution under the fixed architecture, law and budget.

These objectives operate through different prior policies. Above 1,024 rows,
the unchanged GANTrainer passes unique generator-sampled locations to its
regularizer. Native tasks have 20,000 locations. Coulomb's graph and MMD's
current panel therefore change with the supplied batch; fixed initial
references remain fixed. DualNorm updates only unique sampled rows and uses
smoothed, saturating normalization. A larger coefficient need not produce
proportionally larger displacement. The spring acts after that step and
tests a distinct policy.

## Recommendations

Keep the five features opt-in and retain the existing BCAP leader as the
default. The moment tether at 0.01 is the best demonstrated starting point
for studying the mode-retention repair, with word retention as a mandatory
regression control. The finite strength searches do not establish a smooth
response curve or a general solution for Tier 2.

For 100 Gaussians, the next useful investigation is whether the prior update
has enough control over **local output density**: measure the regularizer's
gradient contribution on update-owned rows and relate generator local
expansion to component width, center error and missing high-quality modes.
This is a proposed mechanism diagnostic, not a result or an additional run
in this search. Stronger global moment constraints alone have not justified
claiming those output errors will disappear.

Use the frozen numerical gates and temporal witnesses to select any future
experiment. The original grid oracle satisfies the density gates, showing
the scorer can accept the target law; it does not prove the trained recipe
will represent and reach it within the current architecture and budget.
