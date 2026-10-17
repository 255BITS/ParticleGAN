# Existing-control mode screen: stop this calibration profile

Both registered controls failed `mode_hold` for **39.241742148 paid seconds**.
Independent audit reproduced the frozen grades and verified the receipts.
All three lineages in `host-profile-transfer-v1` now fail its smoke predicate.
This exact profile cannot satisfy its unchanged adoption criteria; collecting
more reference cells solely to approve it cannot change that conclusion.

| Exact lineage | Mode-hold verdict | Modes | HQ | Paid seconds |
| --- | --- | ---: | ---: | ---: |
| K3P control, reused | FAIL | 5/8 | 0.998779 | 17.930311 |
| No critic penalty | FAIL | 0/8 | 0 | 21.671808 |
| Anchor off | FAIL | 5/8 | 0.992920 | 17.569934 |

Each completed 1,200 generator, discriminator and prior updates, with zero
passing observations across 24 checks. The frozen coverage gate requires all
eight modes and HQ >= .9 for five stable checks. No gate, scoring policy,
initialization policy, seed, prior or source changed. These are scientific
failures, with no infrastructure errors, retries or remaining reservations.

## What changed and what was verified

No-penalty changes only `reg_coeff: 1 -> 0`; the public mechanism consequently
disables both the critic penalty and its anchor contribution. Anchor-off changes
only `reg_anchor_weight: 1 -> 0`, retaining all 1,200 penalty applications.
Baseline anchor applications were 236; both ablations record zero. All three
record zero actual A2 applications on this host. Labeled synthetic mechanism
checks are not evidence of training activation.

The runs share learned MoG sigma .025, fixed width, uniform masses, clean public
sampling, seed zero and the same initializer declarations. The audit matched
all 14 named RNG bindings and 24 evaluation-isolation audits, each reporting
zero unintended deviations. **This host saves no initial tensors, model
checkpoint or final RNG states**, so these artifacts establish declaration and
audit parity; they cannot independently establish exact tensor/final-state
equality. Native experiments have separate stronger saved-state evidence.

Registration `host-profile-control-mode-v1`, SHA-256
`0c3688b09c50b8dbb5fc036180bbf82fc5940b84f0f45036540faa1f4eec7c51`,
was published at `1b2d8915` before execution. Both cells bind source
`5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`
and cohort `eca8051cdbaaf1e7b56ef6096a6d29bb403e581403b48d3b4f85796190eab08b`.
Exactly two new attempts ran serially on GPU 0; the existing K3P control was
reused without another launch. The 3,600-second ceiling was not spent out.
Measured wall time includes setup and evaluation; FLOPs remain unavailable.

- [No-penalty receipt](attempts/6d10bbf90f1b4233a73034391599ecdb/result.json),
  request `1ecb2cad5a5080af7160d599`, revision
  `5bbe11c14ddca9d5334589d6901e38838d627fdecee94d6bd6aa86e1f1e76125`.
- [Anchor-off receipt](attempts/eac540ccfe564a5ab396abdd545ab013/result.json),
  request `3bd0097419b99a5836ca77e6`, revision
  `d297bd4d9cf9c012b927c8e10af4c5145a4e73aee1680a675396f99aa7cf048f`.
- [Preparation and frozen contract](studies/CONTROL_MODE_SCREEN.md).
- [Independent audit](CONTROL_MODE_INDEPENDENT_AUDIT.md) and
  [machine-readable checks](CONTROL_MODE_INDEPENDENT_AUDIT.json).

## Calibration conclusion

The [unchanged 57-cell matrix](calibration/host-profile-transfer-v1.md) contains
**7 measured cells and 50 unknown**. K3P has one paired true rejection, because
its independently measured native grid reference also failed. The two controls'
independent references remain unknown. No full-reference positive is established;
the false-reject fraction is **unknown**, not zero or one. Missing reference and
complete control-smoke costs remain unavailable, not zero.

The frozen criteria require at least one full-reference positive and zero false
rejects. Smoke is the conjunction of the three required tasks; every declared
lineage has a measured failure in its first predicate. If any of those lineages
later proves reference-positive, it will be a false rejection. If none does,
the required positive count is not met. Therefore no completion of this exact
matrix can meet both criteria. This is a conditional feasibility conclusion,
not an empirical population error estimate or a universal algorithm rejection.

The seven cells cost **195.525508143 seconds** across the three registrations.
The separate A2-off native study and all older cohorts retain their own costs
and compatibility keys. No evidence is pooled across them.

Both control revisions are now concluded with exact-revision readouts:
[no penalty](records/readout-75d5cdc5a88167d59ceff1b2.json) and
[anchor off](records/readout-5470a57cf1a74abc896c8a55.json). Their scope is the
completed mode-only diagnostic, with downstream results explicitly unknown.
The compiled memory preserves the failed ideas and their comparison limitations.

## Recommendation and remaining work

Stop filling this profile solely for adoption. Retain its receipts and frozen
criteria; do not waive mode coverage, substitute EMA or run seed-only trials.
The failed native parent also supplies no reason to launch its 14k continuation.

Before another calibration campaign, declare a new justified screen/profile and
its independent full-reference cohort. Bind any recovered compatible positive
or missing #155 continuation/scoring evidence to exact source, fixture and
sampling identities. Register only the measurements that can resolve the new
study's question, with a bounded cost and explicit reuse. The
[reference-source audit](POSITIVE_REFERENCE_SOURCE_AUDIT.md) found no compatible
full-positive receipt in its inspected sources; older MoG envelope passes remain
searchable historical context. Inventing a positive, copying cloud/EMA credit
or blindly searching for a production formulation would not close this gap.

The engine and operational acceptance are implemented. Scientific calibration
and default adoption remain unfinished under the accepted plan.
