# Scientific calibration protocol recommendation — 2026-09-30

Read-only protocol analysis; no training was launched. The coordinator owns declarations, registration, budgets and execution.

Retain `host-profile-transfer-v1` as infeasible for adoption: its three lineages all fail smoke, so completing their references cannot yield both one positive reference and zero false rejections. That does not estimate the empirical false-reject fraction; it remains unknown.

Declare `no-output-noise-reference-v1` with the **unchanged three smoke gates, unchanged 16 independent reference tasks, criteria-v1.json, live clean scoring and all three existing lineages**, adding `k3p-no-output-noise-diagnostic` as a fourth substantive lineage. Its single recipe change is `output_noise_std: 0`; learned-MoG sigma .025, uniform masses, named seed-zero streams, task-owned initialization, A2 and full 7k native gates remain fixed. This is a training-law ablation, not an evaluation-noise switch or a positive label. Declare `mechanism_class: structural` as removal of stochastic convolution in both training objectives, with the existing scalar endpoint disclosed; this is not a tuned-amplitude search. The question is whether removing training output noise resolves the current native clean-live shape bottleneck and eventually establishes a complete independent positive under the same screen.

The smallest first selection is **only the new lineage's `grid100_affine_square_named_v1`**, retaining all 7,000 updates, the final five observations and the independent 100k holdout. The task/candidate/campaign reservation ceilings are each **3,600 seconds**, matching the frozen task resource. No allowance comes from the profile's `training_allowance_seconds: 0` or historical whole-matrix ceilings. This deeper diagnostic is justified while smoke is unknown because the already-measured native failure is the immediate obstruction to a reference-positive lineage. It grants no ordinary qualification. A FAIL ends this reference-gap hypothesis; do not automatically spend on smoke, other references or a 14k continuation. A PASS resolves one bottleneck only; review it before a separate missing-smoke registration and then the remaining 15 references. All three smoke cells and all 16 reference purposes must still be accounted for before adoption.

Gate meanings are fixed: mode hold requires sustained 8/8 mode retention and HQ >= .9 (five stable observations), detecting early coverage collapse; bars4 requires 4/4 modes and HQ >= .9 under its RMSE-based image metric, detecting failure to preserve separated image components; intensity2 requires 2/2 modes and HQ >= .9, detecting intensity separation/quality failure. The image gates retain their explicit enumerated finite-cloud exception. These are candidate predictors of later independent quality, whose predictive accuracy is still uncalibrated. Native qualification jointly requires coverage, all four accuracy metrics, the terminal window and holdout: 100 modes, mass TV <= .06, centre RMS <= .20 sigma, absolute covariance-trace bias <= .10 and radial KS <= .04. High HQ or an earlier good checkpoint cannot offset a failed shape gate.

Do not replace the screen with `unused_token_hold + ae_gan_hold`: K3P passes those cheap tasks but already fails the independent native reference, so that proposal knowingly false-accepts a selected negative. Do not delete mode hold to rescue the existing three-lineage profile.

Selection is deliberately informed by observed K3P late contraction and by the existing all-FAIL screen. Four chosen mechanism lineages are not a random population sample; report false accepts/rejects only as selected-cohort fractions. The added hypothesis does not establish a positive before all 16 references pass. Frozen acceptance remains >=3 paired lineages, >=1 positive, >=2 negatives, paired fraction >=.9, false-reject fraction 0, false-accept fraction <=.1, complete costs, maximum smoke time 900 seconds and maximum per-lineage smoke/reference ratio .1. Endurance, adaptation and promotion remain later distinct requirements.

## Resolver and compatible reuse

Use **plain `/usr/bin/python`**, Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0. The main-checkout `.venv` instead has Python 3.12.13/NumPy 2.5.2 and yields a different cohort. Plain Python verified all three existing candidate revisions and exact cohort `eca8051cdbaaf1e7b56ef6096a6d29bb403e581403b48d3b4f85796190eab08b`, source `5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`, and zero relevant preflight blockers. The subsequently declared fourth candidate resolves to revision `0fa6918ca4d0456515de09dbcfaeedffba1d4cecaab00dee5028bcf3b0b34603`, output noise zero, exactly the same original source/cohort, and zero global or 19 selected-task blockers; all four lineage cohorts therefore match.

After declaring the new idea, derive the cohort for **every** lineage; compare the complete resolver output, not just a manually copied digest:

```python
from pathlib import Path
from experiments.forge.planning import resolve_idea
from experiments.forge.calibration import calibration_cohort
root = Path.cwd()
request = resolve_idea(root, candidate_id, view_id="host_profile_transfer",
    through_tier=3, freeze_source=False, execution_backend="cuda",
    cuda_model="NVIDIA RTX A6000")
assert not request["preflight_blockers"]
assert not any(request["tasks"][name]["preflight_blockers"]
               for name in smoke_tasks + reference_tasks)
cohort = calibration_cohort(request, smoke_tasks + reference_tasks)
# Set each lineage revision from request["candidate_revision"].
# Set profile["cohort"] from this object; require all lineage objects equal.
```

Generate all seven verified old-cell import bindings before freezing the new profile:

```sh
python -m experiments.forge calibration-lane imports host-profile-first-cells-v1 --lineage k3p --tasks img_bars4_residual16 vector_unequal_mass_published grid100_affine_square_named_v1
python -m experiments.forge calibration-lane imports host-profile-smoke-pair-v1 --lineage k3p --tasks mode_hold img_intensity2_residual16
python -m experiments.forge calibration-lane imports host-profile-control-mode-v1 --lineage forge-onboarding-anchor-ablation --tasks mode_hold
python -m experiments.forge calibration-lane imports host-profile-control-mode-v1 --lineage forge-no-critic-penalty --tasks mode_hold
```

Each prints bindings without writing or training. Combine the four arrays in `diagnostic_imports`; these seven distinct attempts retain original request/result/evidence hashes, registration hashes, costs and nonqualification keys. Import verification succeeded read-only for all seven. Recipe knobs are excluded from the cohort while still entering exact candidate revision; adding a Forge idea/profile alone leaves scientific source unchanged. A source/runtime/task change requires deriving a new cohort and prevents incompatible old cells filling it.

After the coordinator writes and freezes the one-cell contract, the supported sequence is:

```sh
python -m experiments.forge validate
python -m experiments.forge calibration-lane register --contract configs/forge/campaigns/no-output-noise-native-v1.json
python -m experiments.forge calibration-lane plan no-output-noise-native-v1
python -m experiments.forge calibration-lane enqueue no-output-noise-native-v1
python -m experiments.forge drain --gpus 0 --workers-per-gpu 1 --campaign calibration-no-output-noise-native-v1
python -m experiments.forge calibrate --profile no-output-noise-reference-v1
```

The contract binds exact profile SHA, `view: host_profile_transfer`, the one selected task/lineage, all three 3600-second ceilings, CUDA/A6000 identity, `failure_policy: continue_registered_diagnostics` and `qualification_reuse: false`. Substitute the actual declared paths/IDs/device before execution; this recommendation is not an execution contract. Publish the resolved shared queue's `events.jsonl`, campaign `progress.jsonl` and attempt `run.log`, then conclude the exact candidate revision with metrics, hardware, optimizer updates, wall/phase times, allocator peaks and a comparison with the imported 92.037031-second native control. The old seven imported cells cost 195.525508143 seconds total and must not be charged again.
