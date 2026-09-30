# Separate quick-screen study

## Explicit image-profile transfer

The [residual16 diagnostic](IMAGE_PROFILE_TRANSFER_READOUT.md) passed under
source `79501bcf`, with the current shared K3P formulation and fixed protocol.
Its [profile](../../configs/forge/calibration/image-profile-transfer-v1.json)
changes the intensity host explicitly, retains all 16 independent references,
and imports no old-source results. One cell passed for 11.199 seconds; 56 remain
unknown. This is evidence to guide further host-profile work, not adoption.

## Corrected source cohort — v3

The [v3 profile](../../configs/forge/calibration/develop-20260929-quick-v3.json)
binds the image-data clamp, versioned sampling contracts and CUDA telemetry
repairs at `f35be792`. All 57 cells begin unknown; no prior-source results are
imported. The three substantive lineages, fixed seed/initializer, scientific
thresholds, 16 independent references and adoption criteria remain unchanged.

The [intensity registration](calibration-lanes/develop-20260929-intensity-repair-v3/registration.json)
selects only K3P `img_intensity2`: 600 updates, 24 observations, 1,800 seconds
maximum total. This measures the corrected real-data law before any wider
scientific expansion. Separately, the [physical pilot](MULTI_GPU_PILOT.md)
selects two vector cells and at most one deliberate-cancellation repair. Neither
registration confers ordinary qualification, and neither authorizes filling
the full calibration matrix.

The [corrected intensity cell](INTENSITY_REPAIR_READOUT.md) completed FAIL for
16.837 seconds. The [physical pilot](PHYSICAL_GPU_PILOT_READOUT.md) completed
two vector PASS cells and one deliberate cancellation/repair for 45.001 seconds.
The remaining 54 scientific cells are unknown; no lineage has a complete
independent reference label. Adoption remains blocked. Host-profile provenance
now needs explicit binding before a further positive-reference transfer probe.

## Merged-develop cohort — 2026-09-29

The [completed GPU baseline](DEVELOP_QUICK_SCREEN_READOUT.md) measured three
scientific FAILs for 38.915 seconds, with complete curves and zero reservations.
Duplicate submission reused the exact request/results. All 16 independent
baseline references and both ablation screens remain unknown; adoption is blocked.

[`develop-20260929-quick-v2`](../../configs/forge/calibration/develop-20260929-quick-v2.json)
pins commit `b4fae98a`, including develop `a8b9d397` and the legacy shared-RNG
compatibility repair, with scientific source
`44bbbe0db5639cc855874930bbc9f8a98f921d0d702a22a753b27c547bea4cfa`.
It preserves the same three lineages, three smoke tasks, 16 independent reference
tasks, seed 0 and acceptance criteria. Clean public sampling and retained MoG
vector gates have new evidence identities. **No old-source receipts are imported**;
all 57 cells started unknown. Three baseline smoke cells are now measured FAILs;
the remaining 54 cells are unknown and adoption remains blocked.

The new [baseline registration](calibration-lanes/develop-20260929-quick-baseline-v2/registration.json)
selects only K3P's three screen tasks: learned-MoG `mode_hold` (1,200 updates)
and explicit particle-cloud `img_bars4` / `img_intensity2` (600 each). It reserves
at most 1,800 seconds per task, 5,400 total, with no ordinary qualification reuse.
Registration and planning launch no training. Review the complete baseline result
before selecting ablations or reference work; a failing baseline does not justify
automatically filling the matrix.

Commands used at the pinned source before execution, with the shared queue:

```sh
python -m experiments.forge calibration-lane plan develop-20260929-quick-baseline-v2
# Enqueue freezes work; drain separately after selecting an available device.
python -m experiments.forge calibration-lane enqueue develop-20260929-quick-baseline-v2
```

Compatible reuse and duplicate submission were verified before closing the
lifecycle, followed by the exact-revision readout and calibration reduction.
Future source changes cannot be run under this old registration; use a new frozen
cohort for further scientific work. Do not repeat this completed baseline merely
to repair reporting.
The unexecuted [develop v1 profile](../../configs/forge/calibration/develop-20260929-quick-v1.json)
and both v1 registrations remain immutable but cannot run from the repaired
checkout. They consumed zero training time. The older study below also remains
frozen and separately reproducible.

## Preserved earlier source cohort

[`current-k3p-mog-quick-v1`](../../configs/forge/calibration/current-k3p-mog-quick-v1.json)
freezes the already-proposed `mode_hold`, `img_bars4`, and `img_intensity2` screen
against the same three candidate revisions and 16 independent reference tasks
as v2. Task thresholds, learned-MoG defaults, cloud exceptions, seed 0, named RNG
streams and adoption criteria are unchanged. Ordinary leaderboard policy is
unchanged; this profile alone authorizes **zero training**.

Current budgets are 1,200 updates for `mode_hold` and 600 each for the two image
tasks (2,400 total). The historical proposal listed 1,200 for `img_intensity2`;
this study freezes the existing current task's 600-update law and does not claim
an exact replay of historical budgets. All three native tasks remain in the
independent reference: the historical alternative admitted the PR217 QR native
adapter despite its native failures.

The question is whether this screen can retain a reference-positive formulation
and reject reference-negative controls within the frozen error and cost limits.
The historical quick-screen replay failed adoption, so acceptance is unknown.
The [original v2 profile](CPU_REFERENCE_READOUT.md) cannot meet its unchanged
criteria with every selected lineage failing smoke. Its receipts remain intact.

## Reused evidence and current limits

The [reduction](calibration/current-k3p-mog-quick-v1.json) explicitly imports the
three already-recorded K3P CPU reference passes. It preserves their original
attempts, certificates, task identities and **20.864916388 seconds** of paid cost;
the new reduction spends zero training time. These are the same measurements,
not additional replicates or a second cost charge. All nine quick-screen cells
and the other 45 reference cells remain unknown. No candidate has a complete
positive or negative reference, and adoption remains `BLOCKED`.

The import list was generated with:

```sh
python -m experiments.forge calibration-lane imports \
  current-k3p-mog-cpu-reference-a-v2 --lineage k3p \
  --tasks trajectory residual_student unipolar
python -m experiments.forge calibrate --profile current-k3p-mog-quick-v1
```

The profile contains exact source cohort `c673226cad226889b05269d71786100dcfe2122c8fbe880067d2d2ebd79ab32d`,
the software from commit `416d04ca`. Its cohort was derived from the original
registered request's complete catalog, not by relabelling newer source as old.
The subsequent orchestration fixes change the live checkout's source digest.
Any future execution for this study must therefore use the frozen source checkout
and matching declarations, or explicitly declare a different cohort. Other-source
or other-seed results cannot fill this profile.

## Registered next bounded execution

The [baseline lane](calibration-lanes/current-k3p-mog-quick-baseline-v1/registration.json)
is registered for those three tasks, with 1,800 seconds per task and 5,400 seconds
for the candidate and campaign. Its [contract](../../configs/forge/campaigns/current-k3p-mog-quick-baseline-v1.json)
has **zero submissions and launches**. A read-only plan verified the exact source
and candidate using the existing local checkout
`runs/forge/fresh-checkout-v2` at commit `416d04ca`. Only Forge declarations and
registration data were copied there; scientific source files were preserved.

Use the current coordinator CLI from the feature worktree with that explicit
execution root and the common queue:

```sh
python -m experiments.forge --root runs/forge/fresh-checkout-v2 \
  --queue-root /home/martyn/dev/ParticleGAN/runs/forge \
  calibration-lane plan current-k3p-mog-quick-baseline-v1
```

The path above records this machine's preparation, not a portable fresh-clone
location. Recreating it requires the pinned source and these same declarations.
Enqueue and drain remain separate actions pending actual GPU ownership.
Review the complete baseline screen before expanding to controls
or independent references; a baseline screen failure is a reason to diagnose,
not automatically launch the entire matrix. Keep the full reference denominator
and every result visible. A separate operational GPU pilot can proceed only
under its own registration and available capacity.

The CPU imports demonstrate compatible reuse after a policy change. They do not
establish that the alternative screen is useful, cheap enough, or calibrated.
