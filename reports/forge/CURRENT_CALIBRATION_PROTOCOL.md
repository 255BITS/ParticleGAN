# Current-cohort calibration protocol

The reducer now evaluates real current learned-MoG receipts. The completed
[v2 smoke batch](CURRENT_SMOKE_READOUT.md#corrected-current-source-v2) has nine
measured cheap cells; every control fails the screen. The
[CPU reference batch](CPU_REFERENCE_READOUT.md) adds three K3P task passes;
all full-reference decisions remain unknown. With every lineage failing smoke,
v2 cannot meet both the required positive reference and zero false rejections.
Preserve it and preregister a separate screen study instead of completing its
matrix solely for adoption. No current profile has accepted adoption. This document declares the remaining work;
it launches or authorizes no training.

## Freeze the comparison before spending

Create a profile under `configs/forge/calibration/` using
[`current-profile.schema.json`](../../configs/forge/calibration/current-profile.schema.json).
Keep the acceptance thresholds from `criteria-v1.json` and bind its file SHA256.
The profile records exact candidate revisions, one source/runtime/initializer/
prior/compute cohort, and the single screening protocol seed 0. Obtain the cohort
object with `calibration_cohort(request, smoke_tasks + reference_tasks)`; do not
construct or edit its digest by hand. Every planned lineage must produce that
same cohort object. Candidate mechanism knobs differ; prior, scientific fixtures,
initializer law, named RNG streams, runtime and device profile do not.

For the initial screen, preserve the three 530-update smoke tasks and the **16
independent reference tasks already frozen in `initial.json`**. The reference
excludes both competing smoke profiles. Do not reduce that denominator to a
single convenient native result or include a renamed smoke measurement. Use the
same reference if comparing a separately versioned alternate screen.

At least three distinct substantive lineages are needed: one reference-positive
and two reference-negative outcomes, measured on this exact cohort. Those are
observed outcomes, not trusted labels. Start from a justified public baseline and
two declared mechanism controls; do not create seed variants or relabel archived
cloud runs. Three substantive controls are now declared: `k3p`, the onboarding critic-anchor
ablation, and `forge-no-critic-penalty`, which removes the public critic penalty.
These are a baseline and two mechanism-ablation hypotheses; their reference
outcomes remain unmeasured and must not be assigned in advance. Existing CPU `two_pole` failures make the baseline's host
parity and current smoke behavior an immediate question; they do not establish
current independent-reference positivity.

## Spend in bounded stages

1. Resolve source/public-host parity and reuse exact existing current cells when
   they match the final frozen cohort. Source changes invalidate compatibility;
   historical cloud results remain context only.
2. Freeze the full lineage/reference profile and register a narrowly selected
   diagnostic lane for the missing cheap cells. Its separate campaign, task and
   candidate budgets authorize only that subset. A smoke failure retains its
   ordinary qualification veto. Diagnostic receipts never confer qualification.
3. Review this first batch. If a known useful reference is rejected, diagnose the
   mechanism/host difference before automatically spending on all deeper cells.
   A changed screen requires a new profile revision and a fresh declared question.
4. Register only the independently justified missing downstream cells. Complete
   all 16 reference cells per included lineage before requesting adoption;
   blocked, missing, cancelled and unmeasured-cost cells stay visible. Continue
   past a failure only within this explicitly registered diagnostic selection.
   Ordinary qualification remains fail-fast.
5. Run `python -m experiments.forge calibrate --profile <profile-id>` after receipts
   are durable. The reducer independently grades raw curves/artifacts, verifies
   diagnostic registrations and certificates, and publishes the matrix without
   launching work.

The current task declarations permit at most 900 smoke seconds and 34,200
reference seconds per lineage: **35,100 cumulative task seconds**, or 105,300 for
three fully empty lineages. These are reservation ceilings, not runtime estimates
or an authorization to launch the full matrix. A smaller registered subset must
carry its own summed ceiling; actual saved costs and compatible reuse determine
what remains. Cheap-task CPU workers and GPU reference workers have distinct
compute identities. Preserve the same per-task identities across all controls.

## Adoption and evidence rules

The frozen criteria require at least 3 paired lineages, 1 positive and 2 negative
references; paired fraction at least .9; false-reject fraction 0; false-accept
fraction at most .1; complete smoke/reference cost vectors; maximum smoke runtime
900 seconds; and maximum per-lineage smoke/reference runtime ratio .1. Current
adoption additionally requires every selected cell complete. Unknown outcomes
never become failures or disappear from the selected-lineage denominator.

Certified infrastructure repairs may replace an incomplete verdict. Their paid
runtime remains in the cost. A multi-task uninterrupted job is counted once per
screen/reference, not once per predicate. Conflicting scientific repeats cannot
be selected or averaged into a pass. Other seeds, promotion stages and different
source/prior/runtime cohorts never fill missing cells.

Only a passing current report may be bound into `view.calibration` with
`status: accepted` and the exact `report`, `report_sha256`, `profile_sha256`,
`criteria_sha256`, and `cohort_sha256`. `verify_calibration` replays the saved
evidence and checks these hashes, the current request cohort and its exact
required Tier 1 task set. Editing an accepted string is insufficient. Retiering
continues to reuse raw task evidence; changing the required screen requires a
matching calibrated profile before public-default promotion.

This matrix is a diagnostic study of selected lineages, not an unbiased estimate
of population false-accept probability. A passing report validates the declared
screen/cohort criteria; it does not bypass candidate qualification, prove a
production-ready GAN, choose an EMA scoring policy, or satisfy the separate real
multi-GPU pilot and migration requirements.
