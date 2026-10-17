# Training output-noise removal: bounded reference-gap study

Preregistered 2026-09-30 before enqueueing or training. This study executes Phase 1
of [the handoff](../../scientific-calibration-2026-09-30.md), with one new measurement.
The prior profile remains infeasible; its seven original outcomes and costs are
explicitly imported rather than rerun. No accepted calibration or finished
production candidate exists.

The hypothesis is that adding output noise to training fakes while scoring the
clean public prior encourages a narrower clean distribution. The native target
has sigma .03; training output noise has sigma .029. A simple Gaussian matching
model would leave clean variance .03²−.029², but the saved control does not
converge to that model. This motivates a diagnostic, not a positive prediction.
See [candidate analysis](SCIENTIFIC_CALIBRATION_CANDIDATE_ANALYSIS.md), including
historical noiseless failures under different laws.

| Item | Frozen declaration |
| --- | --- |
| Candidate | `k3p-no-output-noise-diagnostic`, revision `0fa6918ca4d0456515de09dbcfaeedffba1d4cecaab00dee5028bcf3b0b34603` |
| Only resolved recipe change | `output_noise_std: .029 → 0.0`; remove output convolution in both training objectives |
| Mechanism | Structural objective ablation through an existing public scalar field; no noise-amplitude search |
| Source | `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7` |
| Cohort | `eca8051cdbaaf1e7b56ef6096a6d29bb403e581403b48d3b4f85796190eab08b`, derived with `calibration_cohort` for all four lineages |
| Profile | [no-output-noise-reference-v1](../../../configs/forge/calibration/no-output-noise-reference-v1.json) |
| Roster | Original K3P, anchor-off, no-critic-penalty, plus the new noise-removal hypothesis; no trusted positive labels |
| Required smoke | Unchanged `mode_hold`, `img_bars4_residual16`, `img_intensity2_residual16` |
| Independent reference | All original 16 host-transfer quality purposes, unchanged task/evaluator/resource identities |
| Selected new cell | `grid100_affine_square_named_v1` only, full 7,000 updates |
| Prior | Learned locations; fixed MoG sigma .025; uniform masses; no standardization |
| Initialization | Named seed 0, identity-affine G, Fourier3/Xavier D, uniform [-5,5] locations |
| Scoring | Clean live public sampling, retaining MoG kernel noise; EMA remains diagnostic |
| Decision | All five terminal checks and independent 100,000-sample holdout under unchanged coverage/accuracy gates |
| New allowance | Exactly one task; task/candidate/campaign ceiling 3,600 seconds; expected wall time about 90 seconds from the original control |
| Execution | Plain `python` (Python 3.14.7), GPU 0 NVIDIA RTX A6000, one exclusive worker; GPU 1 has unrelated training |

The new profile adds a substantive lineage while retaining every failed gate,
all independent reference families, full horizons and `criteria-v1.json`.
Changing the roster does not imply the new lineage is positive. Selection follows
observed contraction on one fixed-seed host; it is not a population error estimate
or robustness study. Explicit `diagnostic_imports` bind seven original attempts,
their registrations, hashes, exact revisions and costs. Imports confer no ordinary
qualification. The profile's `training_allowance_seconds: 0` remains unchanged;
only [the one-cell lane](../../../configs/forge/campaigns/no-output-noise-native-v1.json)
authorizes the selected reservation.

Removing noise intentionally stops consumption of the isolated generator-output
noise stream. Shared initial tensors, step-zero artifacts, other stream bindings,
MoG width and scoring law must match the control. A2 remains enabled after its
removal already failed. Neither a best checkpoint nor EMA can replace terminal
live failure. No failed parent is extended.

Stop this study on native FAIL and publish the remaining positive-reference gap.
On native PASS, review the complete metrics and independently graded artifacts
before separately registering the smallest informative smoke measurement. This
contract contains no automatic second batch or matrix expansion. A native PASS
alone cannot establish a full-reference positive, screen acceptance or promotion.

Read-only [preflight](no-output-noise-v1-preflight.json) verifies the exact one-field
recipe difference, all four cohort objects, seven verified imports and zero
candidate/task blockers. `origin/develop` is already an ancestor at `a8b9d397`;
no incoming E22/API change is present. Original criteria, source and receipts
retain their frozen bytes. The older project `.venv` has a different runtime;
plain `python` supplies the exact compatible cohort.

Commands from the Forge checkout:

```sh
python -m experiments.forge calibration-lane register --contract configs/forge/campaigns/no-output-noise-native-v1.json
python -m experiments.forge calibration-lane plan no-output-noise-native-v1
python -m experiments.forge calibration-lane enqueue no-output-noise-native-v1
python -m experiments.forge drain --gpus 0 --campaign calibration-no-output-noise-native-v1
python -m experiments.forge calibrate --profile no-output-noise-reference-v1
```

The shared queue is `/home/martyn/dev/ParticleGAN/runs/forge`. Tail its
`events.jsonl` and `calibration-no-output-noise-native-v1/progress.jsonl`; each
attempt has its own `run.log`. Launch communication supplies exact paths. Paid
wall time, hardware, updates, phase timings and allocator/RSS memory accompany
the verdict. FLOPs remain unavailable. Reused costs are reported once.
