# Registered formulation comparison v1

This bounded diagnostic compares the user's requested R1/R2, BCap, released
v0.7.0 GAN v3, and K3P. Freeze the profile and lane before enqueueing. No outcome
is assumed positive and no gate is relaxed.

| Arm | Training formulation | Host and prior |
| --- | --- | --- |
| K3P | Current public default, coefficient1/cap1 | Identity affine z2; Fourier3 critic; batch2048; 20k uniform-square initial locations; learned MoG sigma.025, uniform masses, no standardization |
| R1/R2 | Only reg_arm changes to a_r1r2; fixed squared L2 real/fake gradients, coefficient1 | Same host, initialization and prior as K3P |
| BCap | Only reg_arm changes to b_cap; fixed one-sided L2 real/fake caps, coefficient1/cap1 | Same host, initialization and prior as K3P |
| GAN v3, MoG adaptation | Released BCap6/cap1.25, betas(0,.99), spread.05, common7k cosine; disable all K3P-only hooks and input/output noise | Adapt release z4/batch256/cloud to the same z2/batch2048/MoG affine host |
| Released GAN v3 | Exact effective released public recipe including z4/batch256 | MLP128×3 G and Fourier2 MLP critic; named Xavier/zero-bias networks, normal(0,1) initial raw particle cloud |

The first four select grid100_affine_paired_laws_v1; the last selects the
separate diagnostic grid100_release07_cloud_named_v1. The first three isolate
penalty choice. The fourth compares complete training formulations on a matched
host. The last preserves released public components/defaults, but its different
architecture, batch, dimension and prior preclude a fair penalty or speed rank.
No coefficient/cap/width sweep or seed-only trial is authorized.

The release is v0.7.0, commit180d18f400335fb295611d624b48a4e072ae3bae.
Pinned source/gradient tests and full public-trainer parity verify losses,
active BCap/spread, optimizer/model/EMA states, sampling and schedule boundaries.
The shared public trainer remains the sole experimental update loop; archived
source in the test fixture is used only for software parity.

Release standardize=True is inert for ParticlePrior: its factory discards it.
The actual raw cloud law is explicitly standardize=False, sigma0 with learned
locations. Release input/output noise are both zero, so its recipe-noisy arrays
must equal clean arrays exactly and consume no extra noise draws. Caller-owned
networks use named Xavier draws and Gaussian locations here. This restores
neither a historical fixture nor its global draw order. Forge reuses the real
batch for G, matching the released public trainer's default call; the old toy100
runner's fresh generator-real callback is a distinct host choice.

All jobs execute7000 updates. Required scoring remains clean/live, with all34
declared observations, five final20k fidelity clouds and an independent100k
holdout. The same clean draws receive the public recipe's scheduled output noise
using separately named evaluation streams. Keep noisy/live, clean/EMA and
noisy/EMA outcomes. Noisy artifacts run the unchanged coverage and accuracy
evaluators; they cannot replace the required verdict or count as a reference.
Their transformations are timed within evaluation; the sampling phase measures
required public sampling calls only.

The new profile preserves all three host-transfer smoke predicates and all16
independent reference purposes, substituting the prospectively paired grid task.
Optional diagnostic_tasks binds the release cloud task and reports its outcomes
and costs separately with zero smoke/reference/qualification credit. Ordinary
calibration retains19 required cells per lineage. Missing/blocked cells remain
unknown. criteria-v1 bytes stay unchanged. Neither prior failed profile is filled;
full-matrix expansion and continuations require a distinct justified registration.

Five unique jobs are selected, each with task/candidate3600s and campaign18000s
maximum reservations. Expected measured runtime is several minutes; the reserved
five hours is a worst-case ceiling. Use one GPU0 worker and preserve unrelated
GPU1 work. Scientific failure cannot trigger retry, threshold changes, a seed
rerun or14k extension. Publish all outcomes/costs and conclude every exact revision.

The restored fields and paired evaluator change source identity: older receipts
remain context. Compare new K3P training tensors and primary arrays with the saved
control as a deterministic regression check, without importing qualification.
Exact source/runtime/task/candidate pins live in the profile, plan and registration.

```sh
tail -F /home/martyn/dev/ParticleGAN/runs/forge/calibration-formulation-comparison-v1/progress.jsonl
tail -F /home/martyn/dev/ParticleGAN/runs/forge/events.jsonl
```

No public-default change or robustness stage follows automatically.
