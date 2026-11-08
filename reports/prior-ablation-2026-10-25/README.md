# Matched prior ablation and BISPF re-qualification (2026-10-25)

**Result.** At seed 0, on the 15 tests where all three prior arms produced a
grade, the learned particle table passes 2 more tests than either control:
**7/15 PASS learned, 5/15 frozen table, 5/15 fresh N(0, I)**. The difference is
exactly two tests, `ring16_acquisition` and `mode_hold` (plus the dependent
`ring_extension`, and `ae_gan_hold` over the frozen table, outside the matched
15). Both decisive learned PASSes sit at a +0.00 mode-count margin, and neither
test has a recorded noise floor (floors exist only for the `vector_*` gates), so
this is a one-draw ordering on 2 of 15 tests, not an established superiority.
No control passes a test that the learned table fails. The two controls
pass exactly the same tests. The advantage is sample **precision**: on the
16-mode ring the share of high-quality samples is 0.95 learned, 0.61 frozen and
0.53 fresh. On the 100-mode grids the controls cover modes *more evenly* (lower
mass TV) but put only 3–8% of samples near a mode, compared with 22–34% for the
learned table. Every arm still fails all three 100-mode grids.

The re-qualification of the default reproduces every one of its 29 recorded
PASS/FAIL verdicts (18 PASS, 11 FAIL) on the current develop source. It also adds
`clockfree_audit` PASS, and FAIL for `target_shift_recovery` and four
newly measured API model toys. Turning off the BISPF shape/step gates (A4) loses
4 graded passes (6 counting a five-word wall-clock INCOMPLETE and its dependent). Arm 5 (`ot_target_column_normalize`) cannot be measured: the option
exists only on the standalone `BISPFEngine`, which no recipe uses.

On CIFAR-10 (particle DDGAN host, 10k updates, FID50k at the final EMA
checkpoint) the final FIDs are **79.7 learned, 108.5 frozen, 82.7 fresh N(0, I)**.
The evaluation noise is small (std ≤ 0.21 FID). Run-to-run training noise is much
larger: FID5k moves by up to 36 points between successive checkpoints of one run, and
same-config reruns differ by up to 9.4 FID5k. The final ranking is therefore
**not resolved** at one seed: these are single point estimates, and this run
establishes neither that the learned table is better nor that it is not worse.
The frozen arm's 108.5 is most likely a bad final-checkpoint excursion (its FID5k
was 80.8 at step 7500).

These are one-seed diagnostic results (protocol seed 0, a single fixed draw).
They grant no qualification or default-adoption credit.

## Method

- **Tests.** The dated list
  [`technique-inventory-full-suite-2026-10-25`](../../configs/forge/leaderboards/technique-inventory-full-suite-2026-10-25.json)
  (49 tests) is the denominator for every arm. Each test counts once, whether it
  is held, removed or failing.
- **View.** Runs used the diagnostic view
  [`prior-ablation-full-suite-v1`](../../configs/forge/views/prior-ablation-full-suite-v1.json).
  All tasks sit in one tier, so independent tasks run without tier vetoes.
  Checkpoint and gate dependencies still apply.
  - `ring_hold` was added because `ring_extension` continues that run.
  - `grid100_mlp16` and `grid100_res16_tanh` were left out of the run view:
    their native model variants have no explicit profile, so study admission
    refuses them for every arm. They remain in the denominator as blocked.
- **Held fixed per task.** Architecture, data law, seen batch sequence, initial
  prior distribution and capacity, the deterministic initializer, sampling,
  update budget and evaluation cadence.
- **Protocol.** Seed 0 with named, isolated, checkpointed streams.
- **Hardware.** RTX A6000, CUDA, one worker per GPU on GPUs 0 and 1.
- **Studies.** One Forge study per arm, all in campaign `prior-ablation-20261025`.
- **Source.** Executed source `b4d7f2c42` (Forge source digest `4621fa12e6fd`).
  The A3 repair ran at `23d6f532a`, which differs only by the construction fix
  below and the CIFAR/report files.

| Arm | Candidate | Declared trainer delta vs `bispf-optimized-quintic-v1` |
| --- | --- | --- |
| A1 | `bispf-optimized-quintic-v1` | none. This re-qualifies the selected BISPF recipe after the review fixes. |
| A2 | [`bispf-prior-frozen-v1`](../../configs/forge/ideas/bispf-prior-frozen-v1.json) | `prior_update="frozen"`. Same initial table, locations never updated. Frozen tables make the SBNP regularizer and the prior optimizer group inactive. |
| A3 | [`bispf-prior-fresh-gaussian-v1`](../../configs/forge/ideas/bispf-prior-fresh-gaussian-v1.json) | `prior_update="frozen"` and `prior_code_law="fresh_gaussian"`. Every latent is a fresh `init_std·N(0, I)` code (init_std = 1 on every task here). |
| A4 | [`bispf-pure-global-v1`](../../configs/forge/ideas/bispf-pure-global-v1.json) | `bispf_shape_gates=False`: every declared BISPF shape/step gate is off. |
| A5 | not run | `ot_target_column_normalize=True` exists only as a `BISPFEngine(...)` argument. `BISPFEngine` and `BISPFLossAdapter` are not on the recipe path: the `bispf` critic formulation uses its own loss, so this flag cannot change any Forge run. |

**How A3 matches capacity.**
- Same latent dimension.
- Same number of latent draws per update and per evaluation.
- Same marginal scale as the task's initial table.
- The uniform row indices are still drawn from the latent-index stream.
- Codes come from the task's existing prior-noise stream (else the latent
  stream). For MoG tables this is the same stream position the kernel noise
  used, so the A3 code equals the A2 code minus its table centre, divided by σ
  (tested).
- Index draws after the first stay aligned with the table law only when the
  prior-noise stream is separate, or for a MoG table with σ > 0. Under the
  latent-stream fallback (a plain table, or a MoG table with σ = 0), the code
  draws shift every later index draw, so A3's index sequence then differs from
  A2's after the first batch. The index *law* (uniform rows) is unchanged.
- The only difference from A2 is unbounded support (fresh draws) instead of a
  finite table.
- A3 is defined only where every latent is drawn through `prior.sample` from a
  sampled MoG. Hosts that enumerate, encode against or directly optimize table
  rows are explicitly **blocked**: the behaviour hosts, images, five-word, AE/VAE
  and DDGAN model toys, and `ae_gan_hold`.

**Where "freezing the prior" is not a latent ablation.** On `two_pole`,
`unused_token_hold`, `unipolar`, `mid_scale_identity`, `trajectory`,
`residual_student`, `cover_leftover` and the five-word tasks, the "prior" table
*is* the trained sample, embedding or residual control. A2 on these hosts
removes the trained object. The tables mark them *table is the trained object*.

## Explanations

- **Precision versus coverage.**
  - Freezing the table, or replacing it with N(0, I), keeps the generator from
    concentrating latent mass onto modes.
  - Ring16: all arms find 16 modes, but high-quality fraction falls from 0.95
    (A1) to 0.61 (A2) and 0.53 (A3). Per-component covariance error rises from
    0.47 to 5.9 and 6.7 (gate 0.85).
  - `mode_hold`: A3's hq collapses to 0.50. (Its graded modes/hq sample
    `prior.sample`, i.e. the fresh law. The optional `support` diagnostic in
    `benchmarks/locked_shared/mode_hold.py` enumerates `generator(prior.z)`, so
    under A3 it describes the frozen table rows, not the sampled N(0, I) law.)
  - Native 100-mode grids: precision falls from 0.30/0.22/0.34 (learned, on
    grid/rotated/staggered) to 0.03–0.08 for both controls. Mass TV improves,
    for example rotated100 0.111 → 0.068 (A2) → 0.037 (A3).
  - The learned table is what lets one global BISPF recipe trade coverage
    evenness for sharp modes.
- **Frozen versus fresh N(0, I).**
  - Their verdicts are identical on all 15 matched tests. Finite versus
    unbounded support changes margins, not outcomes.
  - Mixed margins, for example:
    - A3 is worse on `vector_unequal_mass` (component covariance error 10.1 vs 2.6).
    - A3 is better on `vector_overlap` (0.31 vs 0.83; gate 0.45). It still fails there.
- **The 1-D Gaussian is a fair test for a Gaussian latent.** A3 ends
  `gaussian1d_smoke` with every gate passing (KS 0.035). A1 passes on an
  earlier confirmed state but ends at KS 0.135. All four arms fail
  `gaussian1d_stability`; A1 is closest (final KS 0.070 vs 0.28 and 0.34).
- **The shape/step gates matter.** A4 loses four of A1's graded passes, plus
  two five-word cells that are not graded losses:
  - `ring16_acquisition`: every endpoint gate passes (hq 0.958), but only 2 of
    the 5 required consecutive stable checks.
  - `mode_hold`: 7 of 8 modes.
  - `ring_extension`: not run, because its prerequisite failed.
  - `img_stripes2`.
  - `five_word_joint_smoke`: a 900-second wall-clock exhaustion, not a graded FAIL.
  - `five_word_joint_hold`: not run, because its prerequisite did not pass.
- **Failures shared by every arm** are not prior-related:
  - the three native 100-mode grids;
  - four vector tasks (unequal mass, unequal width, anisotropic, overlap);
  - three image tasks (bars4, blobs4, intensity2);
  - `gaussian1d_stability`;
  - the measured API model toys.

### Re-qualification details (A1)

- **Reproduced.** All 29 PASS/FAIL cells recorded for the selected BISPF row
  have the same verdict (18 PASS, 11 FAIL).
- **Newly measured.**
  - `clockfree_audit`: PASS.
  - `target_shift_recovery`: FAIL.
  - FAIL: `model_ae_ddgan_gaussian_chain`, `model_ddgan_gaussian_chain`,
    `model_vae_categorical`, `model_vae_hard`.
- **The RNG-contract fix has no Forge effect.** The review's `d_step`
  generator fix changes only the standalone `BISPFEngine`/`BISPFLossAdapter`,
  which no recipe uses. The `bispf` critic formulation runs through its own
  Sinkhorn loss, so A1's unchanged verdicts are expected.
- **Ordinary view.** An ordinary study
  ([`prior-ablation-a1-requal-ordinary`](../../configs/forge/studies/prior-ablation-a1-requal-ordinary.json),
  `discriminator_stability` revision 14, through Tier 3, same source digest)
  stops at Tier 1, so BISPF does **not** qualify under revision 14.
  - Tier 1 contains six model toys that BISPF fails or cannot run.
  - `zero_tanh_smoke` has no routed evaluator (BLOCKED).
  - `five_word_joint_smoke` hit its 900 s wall clock (INCOMPLETE). The same
    task passed in A1, which ran in 802 s.
  - The published inventory therefore keeps its revision-9 qualification; no
    new row was registered.
  - `regenerate_technique_inventory.py --source-commit b4d7f2c42` also cannot
    rebuild this study from the frozen commit, because the study file is newer
    than that commit.

### Execution defects found (not scientific failures)

- `model_ae_gan_multimodal` and `model_ae_reconstruction` crash in every arm.
  `benchmarks/toy_audit/api_model_ae.py::list_cases` reads
  `configs/forge/tasks/<id>.json` from the frozen source snapshot, which does not
  contain task cards (`FileNotFoundError`).
- `zero_tanh_smoke` is BLOCKED at run time ("unsupported evaluator").
  `zero_tanh_rail_escape` is refused at plan time.
- `misgan_block_*` and `ring16_bigan_continuous` fail with "unsupported vector
  target declaration". The two native model variants are refused at admission.
- Frozen tables crash `trajectory` and `residual_student` (A2): the behaviour
  host registers the table with its optimizer ("sampled table is not owned by
  this optimizer").
- The first A3 request crashed at prior construction in every task.
  `Recipe.__post_init__` rejected `fresh_gaussian` while hosts build the table
  via `recipe.replace(prior_update="learned")`.
  - Fixed in `fix(priors): check fresh_gaussian/frozen pairing…`, with a
    regression test.
  - The cancelled request's INCOMPLETE receipts remain in the queue. A3 values
    come from `prior-ablation-a3-fresh-gaussian-r2`.
- Five-word smoke runs within 10% of its 900 s wall clock (802 s, 895 s
  historically). Under two concurrent workers it can time out.

## Cost

- **Estimate before launch** (from recorded BISPF receipts): about 1.3 GPU-h for
  A1, 1.4 for A2, 0.6 for A3 and 1.3 for A4, about 5 GPU-h in total.
  - The upper bound was 11.8 GPU-h if every unrecorded task used half its
    timeout.
  - Under the 12 GPU-hour cap, A4 was run. A5 had no runnable delta.
- **Actual Forge spend:** A1 3,203 s, A2 2,557 s, A3 1,433 s (plus 51 s in the
  crashed first request), A4 2,744 s, ordinary re-qualification 1,301 s. That
  totals about **3.1 GPU-h**.
- **CIFAR:** about 15 GPU-minutes per arm (C1 926 s, C2 924 s, C3 785 s
  including the FID50k evaluation), plus about 8 GPU-minutes for the evaluation
  floors. The first C1/C2 attempts died at step 6000 when the host ran out of
  memory, and were restarted from step 0 (about 0.5 GPU-h lost). The budget was
  8 GPU-h per arm.

## Leaderboard

Statuses: PASS, **FAIL**, *blocked* (refused before training, no spend),
*incomplete* (crash or wall-clock exhaustion), and *not run* (a prerequisite was
not PASS). Per-cell receipts, with attempt ids, graded-result hashes, reasons and
oracle noise floors, are in [receipts.json](receipts.json). The blockers captured
before enqueue are in [plan-blockers.json](plan-blockers.json).

<!-- prior-ablation tables: start -->

### Leaderboard (fixed denominator: 49 tests)

| Arm | Declared delta | PASS | FAIL | Blocked | Not measured | Incomplete | GPU seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | BISPF default (learned table) | 19 | 16 | 7 | 5 | 2 | 3203 |
| A2 | Frozen table | 13 | 18 | 11 | 3 | 4 | 2557 |
| A3 | Fresh N(0, I) latent | 5 | 10 | 33 | 1 | 0 | 1433 |
| A4 | Shape/step gates off | 13 | 18 | 7 | 8 | 3 | 2744 |
| A5 | `ot_target_column_normalize=True` | 0 | 0 | 0 | 49 | 0 | 0 |

A5 is not measured on any test: the flag exists only on the standalone `BISPFEngine`, which no recipe uses (see Method).

### Matched sampled-latent subset (15 tests graded in A1, A2 and A3)

The only tests where all three prior arms produced a PASS/FAIL grade: `clockfree_audit`, `clockfree_audit_measurement_v1`, `gaussian1d_smoke`, `gaussian1d_stability`, `grid100`, `mode_hold`, `ring16_acquisition`, `rotated100`, `staggered100`, `vector_anisotropic`, `vector_overlap`, `vector_spiral`, `vector_two_broad`, `vector_unequal_mass`, `vector_unequal_width`.

| Arm | PASS | FAIL | Passing tests |
| --- | ---: | ---: | --- |
| A1 | 7 | 8 | `clockfree_audit`, `clockfree_audit_measurement_v1`, `gaussian1d_smoke`, `mode_hold`, `ring16_acquisition`, `vector_spiral`, `vector_two_broad` |
| A2 | 5 | 10 | `clockfree_audit`, `clockfree_audit_measurement_v1`, `gaussian1d_smoke`, `vector_spiral`, `vector_two_broad` |
| A3 | 5 | 10 | `clockfree_audit`, `clockfree_audit_measurement_v1`, `gaussian1d_smoke`, `vector_spiral`, `vector_two_broad` |
| A4 | 5 | 10 | `clockfree_audit`, `clockfree_audit_measurement_v1`, `gaussian1d_smoke`, `vector_spiral`, `vector_two_broad` |

### Per-test outcomes and final-state margins

Margin: the smallest relative signed distance to a declared threshold at the graded final state (negative = that gate fails there). Tasks graded on confirmed earlier passes can PASS with a negative final margin. σ: the vector oracle noise-floor margin of that gate in oracle-std units, where recorded. Only the `vector_*` gates record a noise floor; the other margins, including the +0.00 mode-count margins of `ring16_acquisition` and `mode_hold`, have none.

Ordinary view: the test's role in `discriminator_stability` revision 14. A *held* test is excluded from qualification totals there but still counts in this fixed denominator; a held FAIL is still a FAIL here.

| Test | Prior role | Ordinary view | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- | --- |
| `ae_gan_hold` | sampled latent | counted | PASS +0.93 `recon_mse` | **FAIL** -1.00 `recon_mse` | blocked | PASS +0.86 `recon_mse` |
| `clockfree_audit` | sampled latent | not in view | PASS | PASS | PASS | PASS |
| `clockfree_audit_measurement_v1` | sampled latent | diagnostic | PASS | PASS | PASS | PASS |
| `cover_leftover` | table is the trained object | counted | PASS +0.18 `u_kept` | PASS +0.16 `u_kept` | blocked | PASS +0.18 `content_kept` |
| `five_word_joint_hold` | table is the trained object | counted | PASS +0.00 `modes` | PASS +0.00 `modes` | blocked | not run (prereq.) |
| `five_word_joint_smoke` | table is the trained object | counted | PASS +0.00 `modes` | PASS +0.00 `modes` | blocked | incomplete |
| `gaussian1d_smoke` | sampled latent | counted | PASS -1.70 `cdf_ks` | PASS -1.19 `cdf_ks` | PASS +0.19 `std_ratio` | PASS -0.65 `cdf_ks` |
| `gaussian1d_stability` | sampled latent | counted | **FAIL** -0.41 `cdf_ks` | **FAIL** -4.63 `cdf_ks` | **FAIL** -5.81 `cdf_ks` | **FAIL** -0.97 `cdf_ks` |
| `grid100` | sampled latent | counted | **FAIL** -9.60 `radial_ks` | **FAIL** -0.95 `precision` | **FAIL** -0.95 `precision` | **FAIL** -1.13 `mass_tv` |
| `grid100_14k` | sampled latent | not in view | not run (prereq.) | blocked | blocked | not run (prereq.) |
| `grid100_mlp16` | sampled latent | held | blocked | blocked | blocked | blocked |
| `grid100_res16_tanh` | sampled latent | held | blocked | blocked | blocked | blocked |
| `img_bars4` | sampled latent | held | **FAIL** -0.50 `modes` | **FAIL** -0.50 `modes` | blocked | **FAIL** -0.50 `modes` |
| `img_blobs4` | sampled latent | held | **FAIL** -0.25 `modes` | **FAIL** -0.31 `hq` | blocked | **FAIL** -1.00 `modes` |
| `img_intensity2` | sampled latent | held | **FAIL** -0.34 `hq` | **FAIL** -0.65 `hq` | blocked | **FAIL** -1.00 `modes` |
| `img_stripes2` | sampled latent | counted | PASS +0.00 `modes` | PASS +0.00 `modes` | blocked | **FAIL** +0.00 `modes` |
| `mid_scale_identity` | table is the trained object | counted | PASS +0.14 `identity_at_mid` | PASS +0.14 `identity_at_mid` | blocked | PASS +0.16 `identity_at_mid` |
| `misgan_block_neural_imputer` | sampled latent | held | blocked | blocked | blocked | blocked |
| `misgan_block_recovery` | sampled latent | held | blocked | blocked | blocked | blocked |
| `mode_hold` | sampled latent | counted | PASS +0.00 `modes` | **FAIL** +0.00 `modes` | **FAIL** -0.45 `hq` | **FAIL** -0.12 `modes` |
| `model_ae_ddgan_bimodal_chain` | sampled latent | counted | not run (prereq.) | not run (prereq.) | blocked | not run (prereq.) |
| `model_ae_ddgan_gaussian_chain` | sampled latent | counted | **FAIL** -3.32 `t1_q0_normalized_mean` | **FAIL** -1.56 `t1_q0_normalized_mean` | blocked | **FAIL** -2.63 `t1_q0_normalized_mean` |
| `model_ae_gan_multimodal` | sampled latent | counted | incomplete | incomplete | blocked | incomplete |
| `model_ae_reconstruction` | sampled latent | counted | incomplete | incomplete | blocked | incomplete |
| `model_ddgan_bimodal_chain` | sampled latent | counted | not run (prereq.) | not run (prereq.) | blocked | not run (prereq.) |
| `model_ddgan_gaussian_chain` | sampled latent | counted | **FAIL** -6.85 `t1_q0_normalized_mean` | **FAIL** -2.85 `t1_q1_normalized_mean` | blocked | **FAIL** -4.90 `t1_q0_normalized_mean` |
| `model_vae_categorical` | sampled latent | counted | **FAIL** +0.00 `decoder_mean_modes` | **FAIL** +0.00 `decoder_mean_modes` | blocked | **FAIL** +0.00 `decoder_mean_modes` |
| `model_vae_hard` | sampled latent | counted | **FAIL** +0.00 `decoder_mean_modes` | **FAIL** +0.00 `decoder_mean_modes` | blocked | **FAIL** +0.00 `decoder_mean_modes` |
| `residual_student` | table is the trained object | counted | PASS +0.00 `success_rate` | incomplete | blocked | PASS +0.00 `success_rate` |
| `ring16_acquisition` | sampled latent | counted | PASS +0.00 `modes` | **FAIL** -5.96 `component_covariance_error` | **FAIL** -6.86 `component_covariance_error` | **FAIL** +0.00 `modes` |
| `ring16_bigan_continuous` | sampled latent | held | blocked | blocked | blocked | blocked |
| `ring_extension` | sampled latent | counted | PASS +0.00 `modes` | not run (prereq.) | not run (prereq.) | not run (prereq.) |
| `rotated100` | sampled latent | counted | **FAIL** -0.86 `mass_tv` | **FAIL** -0.97 `precision` | **FAIL** -0.92 `precision` | **FAIL** -0.87 `mass_tv` |
| `rotated100_14k` | sampled latent | not in view | not run (prereq.) | blocked | blocked | not run (prereq.) |
| `staggered100` | sampled latent | counted | **FAIL** -8.23 `radial_ks` | **FAIL** -1.91 `mass_tv` | **FAIL** -1.34 `mass_tv` | **FAIL** -8.32 `radial_ks` |
| `staggered100_14k` | sampled latent | not in view | not run (prereq.) | blocked | blocked | not run (prereq.) |
| `target_shift_recovery` | sampled latent | not in view | **FAIL** | blocked | blocked | not run (prereq.) |
| `trajectory` | table is the trained object | counted | PASS +0.99 `identity_mse` | incomplete | blocked | PASS +0.99 `identity_mse` |
| `two_pole` | table is the trained object | counted | PASS +0.12 `grad_med` | PASS +0.12 `grad_med` | blocked | PASS +0.12 `grad_med` |
| `unipolar` | table is the trained object | counted | PASS +0.16 `cover` | PASS +0.16 `cover` | blocked | PASS +0.17 `neu_hold` |
| `unused_token_hold` | table is the trained object | counted | PASS +0.14 `concept_move` | PASS +0.14 `concept_move` | blocked | PASS +0.07 `concept_move` |
| `vector_anisotropic` | sampled latent | held | **FAIL** -0.19 `sw1_normalized` (-5.0σ) | **FAIL** -0.14 `mass_tv` (-5.4σ) | **FAIL** -0.71 `component_covariance_error` (-61.4σ) | **FAIL** -1.19 `mass_tv` (-44.9σ) |
| `vector_overlap` | sampled latent | held | **FAIL** -0.16 `covariance_error` (-5.4σ) | **FAIL** -1.33 `mean_error` (-20.7σ) | **FAIL** +0.30 `covariance_error` (+10.1σ) | **FAIL** -1.01 `mean_error` (-15.7σ) |
| `vector_spiral` | sampled latent | counted | PASS +0.52 `mean_error` (+12.7σ) | PASS +0.46 `mean_error` (+11.2σ) | PASS +0.68 `mean_error` (+16.5σ) | PASS +0.66 `mean_error` (+15.9σ) |
| `vector_two_broad` | sampled latent | counted | PASS +0.17 `hq` (+76.3σ) | PASS +0.08 `sw1_normalized` (+2.3σ) | PASS +0.14 `hq` (+66.0σ) | PASS +0.18 `hq` (+81.2σ) |
| `vector_unequal_mass` | sampled latent | held | **FAIL** -1.08 `component_covariance_error` (-45.9σ) | **FAIL** -2.05 `component_covariance_error` (-87.4σ) | **FAIL** -10.89 `component_covariance_error` (-463.2σ) | **FAIL** -0.09 `component_covariance_error` (-3.8σ) |
| `vector_unequal_width` | sampled latent | held | **FAIL** -10.40 `component_covariance_error` (-808.5σ) | **FAIL** -0.55 `component_covariance_error` (-42.7σ) | **FAIL** -5.47 `component_covariance_error` (-425.1σ) | **FAIL** -35.88 `component_covariance_error` (-2790.7σ) |
| `zero_tanh_rail_escape` | sampled latent | held | blocked | blocked | blocked | blocked |
| `zero_tanh_smoke` | sampled latent | counted | blocked | blocked | blocked | blocked |

<!-- prior-ablation tables: end -->

## CIFAR-10 U-Net (DDGAN) prior ablation

- **Host and budget.** `experiments/train_cifar_particle_ddgan.py`, `model: ddgan`,
  `arm: gan` (no encoder), at its existing budget: 10k updates, batch 64,
  1024-row table with z_dim 64. Configs are
  [`configs/prior_ablation_cifar/`](../../configs/prior_ablation_cifar/).
  Executed at source `76b8179f3` (identical training sources in all three arms).
- **Held fixed.** Protocol seed 0, the deterministic orthogonal initializer,
  the data/time/corruption/reverse/prior streams, and the update budget.
  The initial table is identical across arms (checked: same values; the
  `initialization_sha256` differs only because a frozen table stores `z` as a
  buffer, which changes the state-dict key order).
- **Declared deltas** (the same recipe fields as A2/A3):
  - C2: `prior_update: frozen`;
  - C3: `prior_update: frozen`, `prior_code_law: fresh_gaussian`.
- **Metric.** FID50k of the final EMA generator and prior, using the protocol in
  `lib.cifar_metrics.PROTOCOL` (torch-fidelity Inception-v3 2048 against the
  CIFAR-10 train set). There is no best-checkpoint selection. FID5k is shown at
  each 2.5k evaluation.
- **Noise floors.**
  - *Evaluation floor:* the final weights re-scored with two more
    evaluation-stream seeds
    ([`prior_ablation_cifar_fid_floor.py`](../../experiments/prior_ablation_cifar_fid_floor.py)).
  - *Training-nondeterminism floor:* the first C1/C2 attempts were killed by a
    host out-of-memory at step 6000. They used the same config and seed. These
    were not seed experiments; cuDNN autotuning makes GPU training
    non-bitwise.

<!-- prior-ablation cifar: start -->

| Arm | Prior | FID50k ↓ | Eval floor (std, n) | FID5k @2500 / 5000 / 7500 / 10000 | Feature var. ratio | Prior RMS move | Train s |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| C1 | Learned table (host default) | 79.70 | 0.09 (n=3) | 82.3 / 81.4 / 86.6 / 84.1 | 0.813 | 0.2418 | 662 |
| C2 | Frozen table | 108.50 | 0.21 (n=3) | 71.9 / 98.1 / 80.8 / 112.7 | 0.771 | 0.0000 | 654 |
| C3 | Fresh N(0, I) latent | 82.70 | 0.09 (n=3) | 115.8 / 79.9 / 85.5 / 86.5 | 0.884 | 0.0000 | 667 |

Training-nondeterminism floor: interrupted same-config attempts differ from the final runs by 1.5–9.4 FID5k at matched checkpoints (n=4).

<!-- prior-ablation cifar: end -->

**Reading the CIFAR result.**
- The final FID50k ranks learned (79.7) < fresh N(0, I) (82.7) < frozen (108.5).
  Every gap is far above the evaluation floor (≤ 0.21).
- The training noise is as large as the gaps.
  - The frozen arm's FID5k moves 71.9 → 98.1 → 80.8 → 112.7 across checkpoints.
  - At step 7500 it is better than the learned arm (80.8 vs 86.6).
  - Same-config reruns differ by up to 9.4 FID5k.
- **Conclusion.** The ordering is not established in either direction. These
  are single point estimates at one seed. The 3-point gap between learned and
  N(0, I) is inside the rerun noise, so this run shows neither that the learned
  table is better nor that it is not worse. The frozen arm's last checkpoint is
  probably a bad excursion, not a stable deficit.
- **Separate cohort.** All three arms are far from the archived DDGAN result in
  [`reports/cifar-particle-ddgan`](../cifar-particle-ddgan/LEADERBOARD.md)
  (FID50k 49.5). That run used seed 24002, the pre-deterministic-initializer
  source and a different stream contract. Do not compare it to this table.

## Recommendations

1. **Keep the learned table as the default, but state the claim precisely.**
   - It buys sample precision on multimodal targets (ring16, mode_hold,
     ae_gan_hold). It does not buy even mode coverage: the controls have lower
     mass TV on the 100-mode grids.
   - README now cites this controlled table instead of the uncited
     `100gaussians_no_particle_prior.py`.
2. **Next experiment: combine precision with coverage.**
   - The learned table loses mode-mass evenness on the 100-mode grids while the
     frozen and N(0, I) controls keep it.
   - Register a structural idea that keeps a frozen quasi-Gaussian *anchor*
     table with a learned offset, or a learned table with an explicit mass
     equalizer. Run it on the native grids and the vector mass/width tasks.
   - Prediction: mass TV at A2's level with precision at A1's level.
3. **Repair the Forge execution defects above** before the next full-suite
   measurement:
   - AE model-toy snapshot reads;
   - zero_tanh routing;
   - MisGAN/BiGAN vector target declarations;
   - native profiles for grid100_mlp16/res16;
   - five-word wall-clock headroom.

   Ten of the 49 cells are blocked or incomplete for infrastructure reasons in
   every arm.
4. **Revision 14's Tier 1 cannot be passed by the selected BISPF recipe.** The
   API model toys added to Tier 1 fail or crash. Either hold them until a
   recipe passes them, or treat BISPF's current qualification as revision-9
   only, as the inventory does now.
5. **CIFAR: do not claim a prior effect from this table.** If the image-scale
   prior question matters, use a longer horizon or an averaged FID over the last
   checkpoints. Do not add training seeds. The current host's ~36-point FID5k
   excursions dominate any prior effect at 10k updates.
6. **`ot_target_column_normalize`.** Either wire it to a recipe field, so it
   can be measured, or remove it from the review's candidate list. As it
   stands it cannot affect any qualified recipe.

## Reproduce

Commands run from the repository root. Logs stay outside Git; tail them while
the runs are going.

```bash
python -m experiments.forge plan bispf-prior-frozen-v1 --study prior-ablation-a2-frozen
python -m experiments.forge enqueue bispf-optimized-quintic-v1 --study prior-ablation-a1-learned
python -m experiments.forge enqueue bispf-prior-frozen-v1 --study prior-ablation-a2-frozen
python -m experiments.forge enqueue bispf-prior-fresh-gaussian-v1 --study prior-ablation-a3-fresh-gaussian-r2
python -m experiments.forge enqueue bispf-pure-global-v1 --study prior-ablation-a4-pure-global
python -m experiments.forge drain --gpus 0,1 --workers-per-gpu 1 --campaign prior-ablation-20261025
tail -F runs/forge/events.jsonl
tail -F runs/forge/prior-ablation-20261025/progress.jsonl
# CIFAR arms (one per GPU; logs are easy to tail), then the FID evaluation floor
python experiments/train_cifar_particle_ddgan.py --config configs/prior_ablation_cifar/ddgan_learned.yaml > ddgan_learned.log 2>&1
python experiments/train_cifar_particle_ddgan.py --config configs/prior_ablation_cifar/ddgan_frozen.yaml > ddgan_frozen.log 2>&1
python experiments/train_cifar_particle_ddgan.py --config configs/prior_ablation_cifar/ddgan_fresh_gaussian.yaml > ddgan_fresh_gaussian.log 2>&1
tail -F ddgan_*.log
python experiments/prior_ablation_cifar_fid_floor.py --run /nvme/dev/particlegan/prior-ablation-runs/cifar/ddgan_learned --draws 2
python -m experiments.prior_ablation_report --queue-root runs/forge --cifar-root /nvme/dev/particlegan/prior-ablation-runs/cifar
```
