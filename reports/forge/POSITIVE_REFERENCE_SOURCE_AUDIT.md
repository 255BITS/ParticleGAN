# Positive-reference source audit

The subsequent [supplemental import](supplemental/local-mog-envelope-v1/README.md)
preserves the four selected receipts, shared source archive and complete study
table in the repo. Its historical context card is included in compiled memory.
The audit below describes the inspection before that import; it found no
compatible current full-reference positive and grants no qualification credit.

**No omitted, compatible full-reference-positive learned-MoG receipt was found.**
This was a read-only local audit on 2026-09-29/30: no fetching, training,
regrading of sample arrays, source edits, queue changes or imports. The dirty main
worktree was read and left unchanged. New hypotheses are outside this audit.

Compatibility target: `host-profile-transfer-v1`, source
`5c9c929877c141ccf7352c16987d3a5aadf1aff3a0fedbfa78e7d9b8fe06fdb7`,
seed 0/named streams, learned MoG with fixed absolute latent sigma .025 and no
standardization, declared hosts, clean live terminal checks/holdout, matching
runtime and all 16 independent reference tasks. A native-only positive would
still not establish that full reference.

## Searched scope

Enumerated 93 local worktrees and 375 local/remote-tracking refs; selected eight
plausible worktrees and 18 distinct relevant ref tips for report/source inspection.
This is not a claim to have exhaustively read every file in all worktrees.

| Worktree under `/home/martyn/dev/` | Revision | Inspected scopes |
|---|---|---|
| `ParticleGAN` | `4410d03d` (`experiment/100gaussians-pacgan8-no-reg`) | Dirty config; ignored PacGAN current/archive outputs; `results/mog` (81 summaries); native/batch-init/readme reports |
| `ParticleGAN-continuous-search` | `7e64df02` | Continuous/native reports and prior local-source audit |
| `ParticleGAN-explicit-sigma` | `c80243f0` | Report/result path inventory; MoG/AE/VAE scopes |
| `ParticleGAN-mog-autoencoder` | `f04b46cc` | Report/result path inventory; reconstruction/routing scopes |
| `ParticleGAN-sigma` | `47e504c8` | `reports/mog-sigma/{README,CONCLUSIONS}.md`, local result inventory |
| `ParticleGAN-sigma-develop` | `8101e250` | README native summaries and MoG report inventory |
| `ParticleGAN/.claude/worktrees/agent-a736c848eeea4e443` | `5852f380` | DV12-default report/result inventory and source-tip declaration |
| `ParticleGAN/.claude/worktrees/wf_c8e8433a-30f-15` | `2e3ca278` | Native-refactor report/result inventory |

Also consulted the tracked `reports/forge/calibration/local-followup-source-audit.json`
(previously inspected LR-control worktree `ceeae78f`) and exact #155 pins
`origin/pr-155=0d52b2c8…`, `origin/pr155-latest=bb9036e3…`.
The `/ml2/hypergan/lrfree-20260926` and gap-fill roots remain unavailable locally.

## Findings

- **Main PacGAN-8:** `results/100gaussians/pacgan8_no_reg/` and its
  `.run_grid_history` predecessor have no summary/completion/full-native gate
  receipt. Current log ends in `KeyboardInterrupt`; last logged step 6701 has
  14 modes/HQ .43455. Saved request uses atom prior, sigma_rel=0, seed1234,
  Fourier2/pack8/no_regularizer and LR .0002. The dirty TOML changes LR .00425
  to .0002; no change was made by this audit. This is neither a MoG positive
  nor evidence of current qualification.
- **Real MoG positives, different criterion:** four positive-noise runs under
  ignored main `results/mog/{component_scale,scale_longer}` pass their frozen
  `c0_observed_envelope_v1` rule. All use seed1, zdim4/batch256, LR .0006,
  final 200k **EMA** samples and calibrated relative width; three use 28k
  updates. The 7k positive's actual latent sigma is .00497839, not .025.
  All four retain `passed_strict=false`. These are original study outcomes,
  not Forge/native gate FAILs or PASSes. No rotated/staggered full-live terminal
  receipts or full independent reference matrix accompanies them.
- **Other apparent positives:** `reports/toy100/{shared22,simpler22,k3p-base}`
  are historical particle/cloud protocols; batch-feature-init 22/22 replay
  explicitly uses the frozen old runtime and disclaims qualification of the
  current public trainer. README native results are atom/EMA illustrations.
  Trainable-sigma studies are different 28k EMA objectives and report shape
  collapse. AE/VAE/denoising passes concern different tasks.
- **#155/default branches:** row-EM renewal has native 3/3 but a changed public
  sampling law and all22=10 PASS/4 FAIL/8 parity ERROR. Latest D-tracking has
  native 2/3, not the missing later 14k/EMA claims. KA2's candidate README
  explicitly says unqualified. A DV12 default-setting commit supplies no
  compatible full-reference receipt. These sources do not fill the gap.

## Small historical context import is feasible, not performed

Preserve the four runs below as **legacy C0-envelope/EMA-only context**, with
original `passed`/`passed_strict` fields, no normalized Forge qualification and
no inferred full-native/full-reference positive. Include the complete component
study table/CSV (including failures), so retaining four positive receipts does
not hide the study's denominator.

Verified for all four: completion receipt's summary SHA matches actual bytes;
receipt config and provenance equal the summary; all 41 source.zip members
match the saved provenance hashes. All four source archives are byte-identical.
Their original source calls recipe factories plus a separate prior Adam; it
does not use `GANTrainer`. Criteria in every saved config exactly match the
retained criteria file; its baseline CSV SHA matches actual bytes.

A compact bundle would retain each run's `summary.json`, `requested_config.yaml`,
`run_grid_complete.json`, `provenance.json`, `environment.json` (73,987 bytes
combined), plus criteria, baseline CSV, `COMPONENT_SCALE.md`,
`COMPONENT_SCALE_PLAN.md`, and `component_scale_results.csv` (56,880 bytes).
Then choose either the three original evaluator/driver files (35,585 bytes;
**166,452 bytes total**) or preferably the single complete original `source.zip`
(108,930 bytes; **239,797 bytes total**, roughly 234 KiB). No weights, sample
arrays, logs or duplicate sorted leaderboard are needed for context retention.

Without sample arrays this bundle cannot independently recompute metrics. The
legacy completion certificate attests runner completion/config/source consistency,
not Forge's task/sampling/terminal/holdout contract. The source/evaluator identities,
criteria, study-specific outcomes and all scope differences must stay explicit.
No new training or adoption is justified solely by this import.

## Exact candidate artifacts

Paths below are relative to `/home/martyn/dev/ParticleGAN/`; SHA values identify
original bytes. Complete retained size/provenance inventory is now in
[the durable manifest](supplemental/local-mog-envelope-v1/manifest.json).

`results/mog/component_scale/n20000_r1over40_shipped_std0_7k_s1` (source commit `a0ec34921c9dd3ebc5b789f60be8681900080f3e`):

- `summary.json`: `4ca6bf6da643dad10809d89b3faaee3fe58d230877e09f512602414034c6dc40` (7708 bytes).
- `requested_config.yaml`: `2a8d7924b03c7bc549b290dbc9f17ff294783abf2b7d95021c0e5beb30cea3c9` (802 bytes).
- `run_grid_complete.json`: `ed61594cc80c6cf245fd26c60ee3e4a72ba84d7c2628414eb50d69a1de0093d9` (5609 bytes).

`results/mog/scale_longer/n20000_r1over16_shipped_std0_28k_s1` (source commit `a0ec34921c9dd3ebc5b789f60be8681900080f3e`):

- `summary.json`: `7be1ac6f43095573645d988c4e00de63b6cb03aab08c4fd25a79561c72ff96ed` (7686 bytes).
- `requested_config.yaml`: `f7bf1352a7b3190b5e27ab81b0d07c6628cea3b72fff5bd50159a50303741030` (802 bytes).
- `run_grid_complete.json`: `7f5f747a987a24db5ffd07027c30f9e2c66008a77d3df99d01ecb59c864a4f50` (5609 bytes).

`results/mog/scale_longer/n20000_r1over40_shipped_std0_28k_s1` (source commit `a0ec34921c9dd3ebc5b789f60be8681900080f3e`):

- `summary.json`: `0ccdba398cfb5cd269499ca5ee2d0b41da32b12d17ab2ab548ab84331bb259da` (7708 bytes).
- `requested_config.yaml`: `1c835e5d957c425ff872eab15cd6948b73a6899e8a74c39d6938c5cf9f1d0b6d` (801 bytes).
- `run_grid_complete.json`: `aa875e4584955d3f83fe9c1c314abe4e9d0e3ac21be0dfab239a803816516c8b` (5608 bytes).

`results/mog/scale_longer/n400_r1over40_28k_s1` (source commit `a0ec34921c9dd3ebc5b789f60be8681900080f3e`):

- `summary.json`: `137ac7eaefd7147ca58ef443f58ea89bb2d17ea587f92a925e883fc818237fd6` (7675 bytes).
- `requested_config.yaml`: `d84d10eccc5476e4c1bc62e6f82c3148fd31e502fc4b9b3bc762484f22472f33` (784 bytes).
- `run_grid_complete.json`: `c4a385f0a34551eee0445201076bf455f765556001ad2e30ba25f2eb90328204` (5591 bytes).

Shared source archive: `b7db0a9deffabd771a6449d186c5a1ef293ea8196498a071e36d2e17754cd255`.

- `configs/mog/stage1_criteria.json`: `6eded9516b2e8215631ca7cfba753096dc4d30a705544b3832bb2a69abd319e3`.
- `results/mog/results.csv`: `57cc6dd57dd7fe4d60646a57b1e74464498ef6499554bf039febf5681f939db6`.
- `results/mog/component_scale_results.csv`: `c068c24dcf5ff668706720cf7944f4ae6dfd5602593c74949d4e8232edb31089`.

Extractable original evaluator/driver identities:

- `lib/mog_metrics.py`: `70389fe190357d9aa49b8dfe43c822c0eb167be3a2f3e39102c2a32fa8ff5f21` (7116 bytes).
- `experiments/train_100gaussians.py`: `1a8a0cc9bf2a4a36415ea5170a764feb5771cfe98165af574ff9c7605222028d` (5695 bytes).
- `examples/100gaussians.py`: `1a0155e6a245d878691a4c354e970adacdced24807278b9a55bd62eeac3736f9` (22774 bytes).

## Ref-tip audit and limits

`codex/k3p-continuous-search@7e64df02`; `codex/mog-trainable-sigma@47e504c8`; `experiment/100gaussians-pacgan8-no-reg@4410d03d`; `feat/mog-denoising@4c3989a2`; `feature/mog-autoencoder@f04b46cc`; `feature/mog-particle-prior@224142de`; `fix/mog-explicit-sigma@c80243f0`; `k3p-dv12-default@5852f380`; `merge/master-explicit-sigma-into-develop@8101e250`; `toy-refactor/toy100_native@2e3ca278`; `origin/codex/k3p-continuous-search@ef402576`; `origin/codex/ka2-default-candidate@1e040820`; `origin/cursor/gan-native-followup-f60e@d5ec127b`; `origin/particle-finetune/native-2d@c7e8a73f`; `origin/particle-finetune/native16-autopsy@a1ec801e`; `origin/pr-155@0d52b2c8`; `origin/pr155-latest@bb9036e3`; `origin/research/continuous-learning@db7bdea1`.

Ref enumeration and path/keyword inventories are retained locally at `/tmp/forge-positive-{ref-inventory,path-inventory,ref-scans,mog-results}.json`. Pinned report narratives, selected raw summaries/configs/certificates and archived evaluator sources were inspected; no blanket claim is made about unscanned ignored files, inaccessible machines, un-fetched remote refs, or the user’s still-unbound later receipts. Absence here does not disprove their existence. Exact evidence remains required before compatibility credit.
