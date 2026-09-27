# LR-free continuous training: knob map for the public-API packages

Read-only code and evidence audit, 2026-09-26. No runs were launched. Every claim cites code or a log.

Path abbreviations:
- `KA2/` = `/ml2/hypergan/ParticleGAN-ka2-default/particlegan/` (public default candidate, commit 1e040820)
- `PS/` = `/ml2/hypergan/ParticleGAN-k3p-continuous-search/reports/toy100/deterministic-init-retest/port-source/`
- `RP12/`, `RP14/`, `RP15/`, `RP5/`, `DV12/`, `C13R1/` = `PS/api-<name>/package/particlegan/`
- `EV/` = `.../deterministic-init-retest/evidence/<cand>-new-init/`; `FE/` = `.../deterministic-init-retest/followup-evidence/`

## 0. Summary

1. **Every passing API config reduces its own LR, 10–100x, before or during its passing window.** The reduction is state-driven, not clock-driven. RP12, RP14 and RP15 switch G/D to 1% and the prior to 5% at updates 551, 857 and 788. DV12 slides continuously to about 6%. Every config that stayed at a constant full LR on mode_hold failed: RP5, RP11, RP13, C1, C13-R1 and ka2-constant (§4.2). No reviewed config passes without adjusting its LR.
2. **The four survivors already configure no horizon.** They set `total_steps=None`, which RP packages allow only with `continuous_precision` and DV packages only with `continuous_policy`. They use constant noise and no LR anneal. `lr_anneal_start`, `lr_floor`, `network_lr_*`, `input_noise_anneal_end` and `output_noise_warmup` still appear in their declarations but have no effect. Two timers remain in the config:
   - `d_guard_min_steps=200`, a startup delay that config can remove;
   - KA2's hard-coded `WARMUP_CALLS=800` penalty phase, which needs a code change.
3. **Base LR is not tuned per task.** The image and vector screens reuse the declared `recipe_overrides` and change only `num_particles`, `z_dim` and `batch_size` (`PS/new-init-image-screen/image_screen.py:86-88`, `PS/new-init-vector-screen/vector_screen.py:55-57`). `lr=.00425`, `d_lr_mult=1` and `prior_lr_mult=2` are package defaults in every survivor.
4. **The public KA2 package cannot run indefinitely at a constant LR through config alone.** `total_steps` must be a positive int (`KA2/recipes.py:109-112`), and the trainer raises at the budget (`KA2/training.py:170-171`). A huge `total_steps` with floors of 1.0 is the only workaround. The ka2-constant control used horizon-derived noise (360/720 = fractions × 4600) and a 4600-step stop.
5. **One package family gives constant LR and no budget stop from config alone: `continuous=True`.** This is the C family: C1–C13-R1. Its noise still uses fixed startup counts, which config can neutralize. It has no quick-screen pass (C13-R1: 7/8 modes).

## 1. Candidate packages vs the public KA2 default

All RP, DV and C packages keep `ka2.py`/`k3p.py`/`grad_regularizers.py` from KA2, except where listed. That means all of them keep `WARMUP_CALLS=800`.

| Package | Files added / changed vs `KA2/` | Main new switches (default → declared) | New-init mode_hold | Broader gate |
|---|---|---|---|---|
| **RP12** `PS/api-rp12/package` | +`precision.py` +`game_update.py`; changed `gan_loss.py`, `recipes.py`, `training.py` | `continuous_precision` None→rp5, `game_update` ordinary→secant_resolvent, `noise_policy` initialization→constant, `relativistic_pairing` row→all_pairs, `particle_loss_weighting` frequency→uniform_sampled, `adam_eager_state` F→T | **PASS 19/24**, arrival 300 | img_bars4 0/24 (blobs/intensity/stripes pass) |
| **RP15** `PS/api-rp15/package` | +`precision.py` +`game_update.py` +`generator_metric.py` +`generator_support.py` | RP10 base + `generator_update`→shared_tangent_support | **PASS 14/24**, 550 | img_bars4 0/24 |
| **RP14** `PS/api-rp14/package` | same files as RP15 | RP10 base + `generator_update`→tangent_support | **PASS 12/24**, 600 | img_bars4 0/24 |
| **DV12** `PS/api-dv12/package` | +`continuous.py`; changed `__init__.py`, `ka2.py` (DV gating block), `recipes.py`, `training.py` | `continuous_policy` None→dv12 | **PASS 12/24**, 650 | vector_unequal_mass 0/24 |
| RP13 | RP10 + `generator_metric.py` | `generator_update`→activation_metric | FAIL 6/24 | — |
| RP10 / RP11 | RP5 + `noise_policy` (RP10); + `relativistic_pairing` in `gan_loss.py` (RP11) | constant noise; all-pairs loss | FAIL 0/24 (7/8, 5/8) | — |
| RP5 (`RP4`–`RP9` similar) | +`precision.py` +`game_update.py`; **no `noise_policy`** | rp5 + secant; noise uses 360/720 init-step clocks | FAIL 0/24 (5/8) | 10 broader passes on the old init |
| DV16 (newest DV) | DV + changed `particle_prior.py` (learned widths, rank-one correlation) | dv16 | FAIL 0/24 (7/8); old init passed 11/24 | — |
| C13-R1 (newest C) | +`game_update.py` +`checked_game.py` +`nonlinear_game.py` +`update_limit.py`; changed `k3p.py`, `ka2.py` (Moving/Fresh records) | `continuous`→True, `critic_memory`→fresh, `game_update`→nonlinear_residual | FAIL 0/24 (7/8) | intensity 11/24 on the old init |
| public ka2 / ka2-constant | `KA2/` itself | ka2-constant: `lr_floor=network_lr_floor=1.0`, `total_steps=4600` | FAIL 0/24 (6/8; 4/8) | — |

RP lineage: RP1–3 add precision → RP4+ add `game_update` → RP10 adds `noise_policy` → RP11 adds all-pairs → RP12 adds uniform particle weighting. RP13, RP14 and RP15 branch from RP10 and do not have the RP11/RP12 loss fields. This comes from the field lists in each `PS/api-rp*/package/particlegan/recipes.py:30-50` and each `candidate-declaration.json`.

## 2. Knob table

Line numbers refer to `KA2/recipes.py`. In RP12 add 8, in RP14/RP15 add 7, in DV12 add 1.

"Clock?" values:
- **H** = depends on the horizon or `total_steps`;
- **T** = fixed step-count timer;
- **S** = state-driven or stationary;
- **—** = no effect on GANTrainer.

### 2a. LR and optimizer

| Field (line) | Default | Meaning | Clock? | Clock-free value |
|---|---|---|---|---|
| `lr` (36) | .00425 | Base Adam LR for G/E. D uses `lr*d_lr_mult` (`KA2/recipes.py:275`); the prior uses `lr*prior_lr_mult` (`:353`). | S | keep .00425; no survivor changes it |
| `d_lr_mult` (37) | 1.0 | Static critic multiplier | S | 1.0 |
| `prior_lr_mult` (38) | 2.0 | Static prior-table multiplier | S | 2.0 |
| `betas` (39) | (0, .999) | Adam betas for all groups. β1=0 is required for A2 on the prior (`KA2/k3p.py:494-498`). | S (Adam bias correction is intrinsic) | (0, .999) |
| `prior_betas` (40) | None (uses `betas`) | Prior group betas (`:354`) | S | None |
| `direct_particle_gain` (56), `direct_particle_betas` (63) | True, (0, .9) | Up to 2x LR for a direct-particle group (`KA2/k3p.py:274-283`) | **— inert in GANTrainer**: `make_optimizers` never passes `direct_particles` (`KA2/recipes.py:357`) | any |
| `latent_damping_max_rate` (61) | .5 | A2 row damping. Acts only when some prior row got no gradient **and** the lifetime observed-row rate is below `max_rate` (`KA2/k3p.py:177-191`). The rate is cumulative since step 0, so it is age-weighted. | S/age-weighted; inactive when every row is sampled each step (mode_hold) | .5, or 0 to remove A2 |
| `adam_eager_state` (RP only, `RP12/recipes.py:38`) | False | Allocates Adam state and device step counters at construction (`RP12/recipes.py:339-356`). Changes KA2 telemetry arithmetic. | S | as declared (True in survivors) |

### 2b. Horizon and schedule (all disqualifying unless neutralized)

| Field (line) | Default | Meaning | Clock? | Clock-free value |
|---|---|---|---|---|
| `total_steps` (35) | 7000 | (1) LR horizon for prior and network (`KA2/recipes.py:449-460`). (2) Noise horizon (`KA2/training.py:14,23`). (3) **Hard stop** (`KA2/training.py:170-171`). KA2: must be an int >0 (`:109-112`). RP: `None` only with `continuous_precision` (`RP12/recipes.py:133-136`). DV: `None` iff `continuous_policy` (`DV12/recipes.py:83-84`). C: ignored when `continuous=True` (`C13R1/recipes.py:473-476`, `C13R1/training.py:202`). | **H** | `None` (RP/DV with a controller). Otherwise a huge int such as `2**62`; the budget stop remains. |
| `lr_anneal_start` (46) | .6 | Fraction of the horizon at full LR before the cosine (`KA2/recipes.py:398`) | H | inert if floors are 1.0 or a controller is used |
| `lr_floor` (47) | .05 | Prior cosine floor (`:460`) | H | **1.0**, which gives scale ≡1 (`:399`) |
| `network_lr_floor` (51) | .01 | G/D cosine floor; None means use `lr_floor` (`:297-299`) | H | **1.0** |
| `network_lr_horizon_cap` (52) | 1600 | G/D anneal over `min(total, cap)` (`:451`), a fixed 1600-step anneal whatever the budget | **T/H** | inert when `network_lr_floor=1.0`; None also allowed |
| `NetworkLRTransition` (`KA2/recipes.py:402-437`) | — | Caller marks a plateau, then G/D decay | caller-managed phase | do not use |

### 2c. Noise

| Field | Default | Meaning | Clock? | Clock-free value |
|---|---|---|---|---|
| `input_noise_std` (`KA2/recipes.py:67`) | .5 | Critic input-noise peak, linear to 0 at `input_noise_anneal_end*total_steps` (`KA2/training.py:12-15`). RP: at `input_noise_init_steps` when `continuous_precision` is set; constant when `noise_policy=constant` (`RP12/training.py:12-18`). DV: **forced to 0** when `continuous_policy` is set (`DV12/training.py:14-15`). C: over `startup_input_steps` (`C13R1/training.py:14`). | H or T | **0.0** (all survivors), or RP10+ `noise_policy=constant` with any std |
| `input_noise_anneal_end` (68) | .1 | Anneal end as a fraction of the horizon | H | inert when std=0 or noise is constant |
| `output_noise_std` (69) | .029 | Gaussian added to G output, in training and in `sample()` (`KA2/training.py:260`) | S if constant | .029 |
| `output_noise_warmup` (70) | .2 | Linear warmup over `warmup*total_steps`. **0 gives a constant std** (`KA2/training.py:20-21`). | H | **0.0** (KA2), or `noise_policy=constant` (RP10+); DV is always constant (`DV12/training.py:22-23`) |
| `input_noise_init_steps` / `output_noise_init_steps` (RP, `RP12/recipes.py:39-40`) | 360 / 720 | Absolute noise-schedule counts under `continuous_precision` (`RP12/training.py:16,25-26`) | **T** | inert with `noise_policy=constant`. In RP5 (no `noise_policy`) use std 0 and `output_noise_init_steps=1`. |
| `noise_policy` (RP10+, `RP12/recipes.py:41`) | "initialization" | "constant" returns the stds unchanged (`RP12/training.py:14-15,23-24`) | T/H → S | **"constant"** |
| `startup_input_steps` / `startup_output_steps` (C, `C13R1/recipes.py:43-44`) | 360 / 720 | Same timers under `continuous=True` (`C13R1/training.py:14,22`) | **T** | input std 0; `startup_output_steps=1` (one-update ramp) or output std 0 |

### 2d. Regularization and internal clocks

| Field / constant | Default | Meaning | Clock? | Clock-free value |
|---|---|---|---|---|
| `reg_coeff`, `reg_kappa` (41-42) | 1.0, 1.0 | KA2 penalty strength and grad-norm cap | S | as is |
| `reg_every` (43) | 1 | Lazy penalty applied every k steps with coefficient ×k (`KA2/grad_regularizers.py:180-185`) | periodic when k>1 | 1 |
| `reg_anchor_weight` (55) | 1.0 | Scales the EMA-critic proximity term; 0 removes it | S | as is |
| `reg_anchor_min_decay` (53) | .90 | EMA-critic decay = `1-alpha*(1-.9)` (`KA2/ka2.py:139`) | S | as is |
| `ema_decay` (45) | .995 | EMA copies of G and prior, used for sampling only (`KA2/training.py:228-233`) | S | as is |
| `prior_reg` (44) | 0 | Weight of the particle regularizer | S | 0 |
| `d_guard_ratio` (58) | 5.0 | Clip critic grad RMS above 5·sqrt(v̂) (`KA2/k3p.py:111-113`) | S | 5.0 |
| **`d_guard_min_steps`** (59) | 200 | Guard is off until each tensor has ≥200 Adam steps (`KA2/k3p.py:112`) | **T** (startup only) | **0**; untested behaviour change |
| **KA2 `WARMUP_CALLS=800`** (`KA2/ka2.py:23,222-229`) | code constant | Penalty calls 1–799 are pure-A. From call 800 the penalty is .5A+.5(B+w·prox), and the EMA anchor starts (`:240-243`). | **T, not configurable** | code change |
| **KA2 `sur_base`** (`KA2/ka2.py:104-105`; `GATE_MIN_SAMPLES=25`, `BASE_WINDOW=24`) | code | Median of the first 24 blended surprise samples, frozen for the whole run. It is the reference for release, reseed and `alpha`, and for DV12's `game_trust`. First ratio at update 824 in every log. | birth-time reference, not configurable | code change (e.g. a rolling baseline) |
| KA2 `RESEED_STREAK=60`, `K_ATK=1/60`, `K_REL=.5`, `REL_HI/LO=3/1.75`, `HIST_CAP=400` (`KA2/ka2.py:28-35`) | code | Dwell counts and gains | S (fixed time constants) | acceptable |
| K3P legacy blend `s = f(lr_last/lr_max)` (`KA2/grad_regularizers.py:136-142`) | — | Only the public-k3p control uses it. The penalty handover is **driven by the LR schedule**, so at constant LR `s≡1` forever (pure A). | LR-coupled | K3P cannot be horizon-free without changing its penalty |

### 2e. Candidate-specific switches

| Switch | Values | What it does to LR, step size or time | Clock? |
|---|---|---|---|
| `continuous_precision` (`RP12/recipes.py:36`) | None, rp1, rp4, rp5 | Replaces `learning_rate_scales` with `ReversiblePrecision.scales()` (`RP12/training.py:217-221`). See §3.1. rp1 and rp4 run identical code in these packages; only `'rp5'` branches (`RP12/precision.py:20,24,71`). | S |
| `game_update` (`RP12/recipes.py:37`) | ordinary, secant_resolvent, two_direction_resolvent, minimum_residual_resolvent, secant_particle_exploration | Preview/corrector transaction (`RP12/game_update.py:243-301`), 2–3 field evaluations per update. Scales the joint displacement (§3.2). | S |
| `generator_update` (RP13–15) | ordinary, activation_metric, tangent_support, shared_tangent_support | Changes gradients only, before Adam (`RP15/training.py:252-286`; `RP15/generator_support.py:5-98`; `RP15/generator_metric.py`). No LR change, no state, no clock. | S |
| `relativistic_pairing`, `particle_loss_weighting` (RP11/RP12) | row/all_pairs; frequency/uniform_sampled | Loss shaping (`RP12/gan_loss.py`, `RP12/recipes.py:259-274`). Per-batch weights only. | S |
| `continuous_policy` (DV, `DV12/recipes.py:36`) | None, dv1…dv12 | Replaces LR scales with `DataDriftController` (§3.3). Forces input noise to 0 and keeps output noise constant. | S (see hidden timers) |
| `continuous` (C, `C13R1/recipes.py:41`) | False/True | LR scales ≡(1,1) (`C13R1/recipes.py:473-476`); no budget stop (`C13R1/training.py:202`) | clock-free LR; noise T |
| `critic_memory` (C) | ka2, moving, fresh | Moving: `alpha≡.1`, no release or reseed (`C13R1/ka2.py:188-197`). Fresh: anchor copied every step (`:216`). The anchor still starts at call 800. | S, plus WARMUP |

## 3. What the adaptive mechanisms do to the applied LR

### 3.1 RP `ReversiblePrecision` (RP12/14/15: `continuous_precision=rp5`)

- **Output.** A binary switch (`RP12/precision.py:23-25`):
  - open: G/D ×1.00, prior ×1.00 (×2 `prior_lr_mult`);
  - closed: G/D ×0.01, prior ×0.05.
  - rp1/rp4 open: .208 / .24.
- **No continuous LR value, and no dependence on step count.** The `updates` counter is telemetry only.
- **Inputs, read after each accepted update** (`RP12/game_update.py:296-299`, `RP12/precision.py:40-58`):
  - `gap` = mean-square difference between the live critic's input gradient on the real batch and a reference critic. The reference critic is a parameter EMA with rate .001, about 1000 updates (`:55`).
  - `activity` = RMS of the generator/prior parameter displacement **divided by the group LR** (`:32-38`), so it is LR-invariant.
- **Smoothing.** `gap_s` EMA .99; velocity EMA .95 (rp5 only); activity EMA .9; `activity_peak` decays ×.9997 per update while open (`:66-76`).
- **Close** (`:77-87`) requires all of the following:
  - ≥50 consecutive updates with smoothed gap velocity <0;
  - current contraction <25% of its peak;
  - activity <50% of its decaying peak;
  - all of the above for 25 consecutive updates.
- **Reopen** (`:88-101`) requires 5 consecutive updates with gap_s >4× the quiet reference, activity >2× the quiet reference, and velocity >0. The quiet references track slowly (.01) while inside 2×.
- **Bounds.** LR is always one of the two levels; there is no floor or cap in between.

### 3.2 `game_update=secant_resolvent`

The step fits a local game Jacobian on the preview direction `u` and applies `delta = ((1-a⁻)u + r)/((1-a⁻)²+b²)` (`RP12/game_update.py:90-111`). Here a⁻ = min(a,0), so the radial term is at least 1.

The effect is an extra, per-update, state-driven step multiplier `norm_ratio = 1/sqrt(den)` in (0,1], plus a rotation. It is never above 1 and has no clock. Measured on mode_hold (`EV/…/learning-rates.jsonl.gz`, `game_stats.secant.norm_ratio`):

| Run | Median | p10 | Median after update 600 |
|---|---:|---:|---:|
| RP12 | .974 | .318 | .986 |
| RP14 | .382 | .254 | .991 |
| RP15 | .489 | .284 | .991 |

Adam moments come from the base gradients; the preview state is discarded (`:78-87`).

### 3.3 DV12 `DataDriftController`

Applied scales (`DV12/continuous.py:175-178,199-201`, `DV12/training.py:213-223`):
- G = (.01+.99·m)·gt
- prior = (.05+.95·m)·gt·2
- D = (.01+.99·m)·gt / (1+pe²)

Terms:
- **m (mobility).** EMA toward `max(data_drive, min(1, pe²))`, rate .05 up and .005 down (`:160-166`).
- **data_drive.** A z-score of the fast (.1) versus slow (.01) mean of random Fourier features of real batches. Location, scale and projection are fixed at the **first real batch** (`:121-156`).
- **pe (payoff error).** EMA .02 of `max(0,(loss_g − loss_d_adv)/ln2)` (`:215-218`).
- **gt (game trust).** `1/(1+((ratio−1)⁺·(1−data_drive))²)`, where ratio = KA2 surprise / `sur_base` (`:203-212`). gt is 1 until KA2 has a `sur_base` (update ≥824).

Bounds: m ∈ [0,1]; gt ∈ (0,1]. There is no hard lower bound, because gt can go below the .01 floor. There is no step-count input, but gt is gated by the KA2 warmup.

In DV12 the KA2 critic `alpha` is additionally multiplied by gt, and anchor release is blocked unless data_drive ≥ .1 (`DV12/ka2.py:255-267`). Latent perturbation uses the prior-geometry bandwidth EMA .01, clipped to half the nearest-particle distance (`:47-88`).

### 3.4 Other mechanisms

- KA2 moment-surprise changes the **penalty and EMA-critic decay only, never the LR** (`KA2/ka2.py:95-149`).
- A2 changes prior-row momentum on sparse steps only (§2a).
- `DirectParticleResponse` is inert in GANTrainer.

## 4. Survivors and control: exact overrides and time dependence

### 4.1 Non-default overrides

Computed by diffing each `EV/<cand>-new-init/declaration.json` `recipe_overrides` against its own package's `Recipe` defaults. The full declarations are in those files.

| Candidate | Non-default `recipe_overrides` |
|---|---|
| RP12 | `total_steps=None, continuous_precision="rp5", game_update="secant_resolvent", noise_policy="constant", input_noise_std=0.0, relativistic_pairing="all_pairs", particle_loss_weighting="uniform_sampled", adam_eager_state=True` + task `num_particles=12, z_dim=4, batch_size=128`; `serial_backward=True` |
| RP15 | same as RP12 minus pairing/weighting, + `generator_update="shared_tangent_support"` |
| RP14 | same as RP12 minus pairing/weighting, + `generator_update="tangent_support"` |
| DV12 | `total_steps=None, continuous_policy="dv12", input_noise_std=0.0, output_noise_warmup=0.0` + task fields; `serial_backward=True` |
| public ka2-constant | `total_steps=4600, lr_floor=1.0, network_lr_floor=1.0, input_noise_anneal_end=0.07826 (→360), output_noise_warmup=0.15652 (→720)` + task fields |

All five keep the package defaults `lr=.00425, d_lr_mult=1, prior_lr_mult=2, betas=(0,.999), d_guard_min_steps=200, d_guard_ratio=5, latent_damping_max_rate=.5, reg_*=1`. The survivors also still carry `lr_anneal_start=.6, lr_floor=.05, network_lr_floor=.01, network_lr_horizon_cap=1600, input_noise_init_steps=360, output_noise_init_steps=720`, but these are **inert**: the controller path bypasses `learning_rate_scales`, and noise is constant.

### 4.2 Measured applied rates on mode_hold

From `EV/*/learning-rates.jsonl.gz`. Values are G LR ÷ .00425.

| Run | G/lr0 start → final | LR switch point | Noise in / out | Result |
|---|---|---|---|---|
| RP12 | 1.00 → 0.01 (prior 2.0→0.1) | precision closes at **551**, no reopen | 0 / .029 constant | PASS 19/24, arrival 300 (6 checks at full LR before closing) |
| RP15 | 1.00 → 0.01 | closes at 788 | constant | PASS 14/24, arrival 550 |
| RP14 | 1.00 → 0.01 | closes at 857 | constant | PASS 12/24, arrival 600 |
| DV12 | .995 → .066 (min .023) | m<.5 at 648; gt<1 only after 827 | constant | PASS 12/24, arrival 650 |
| RP10 | 1.00 → 0.01 | closes at 819 | constant | FAIL 7/8 |
| RP5 / RP11 / RP13 | 1.00 constant (never closed) | — | RP5 timed; others constant | FAIL (5/8, 5/8, 7/8; RP13 6/24 transient) |
| C1 / C13-R1 | 1.00 constant (`continuous=True`) | — | 0.5→0 over 360; 0→.029 over 720 | FAIL (4/8, 7/8) |
| ka2-constant | 1.00 constant | — | 0.5→0 over 360; 0→.029 over 720 | FAIL 4/8 |
| public ka2 | 1.00 → 0.01 (cosine 720→1200) | horizon | timed | FAIL 6/8 |

Image follow-ups (`FE/api-rp1{2,4,5}-img-*`):
- In **img_bars4**, RP12, RP14 and RP15 all stay open at **full LR for all 600 updates** and fail on coverage. RP12 plateaus at 2/4 modes with HQ .97; RP15 at 3/4 with HQ .72. The bars failure is not caused by a premature LR cut.
- Blobs closes at 309–522 and passes.

DV12 on unequal_mass cuts G below .5 at 323 and ends at .043. It fails covariance (.879 > .85) and min mass (.232 < .25) (`dv12-vector-runtime-audit.json`).

### 4.3 Time dependence and verdict

| Candidate | Remaining time-dependent elements | "No LR adjustment" under the user goal? |
|---|---|---|
| RP12 | KA2 `WARMUP_CALLS=800` penalty phase and frozen `sur_base` (code; not an LR). `d_guard_min_steps=200` guard delay (config). No horizon: `total_steps=None`, so it runs indefinitely. LR is a state-driven binary rp5 switch (1.0 ↔ .01/.05) with fixed EMA time constants and no age input. | **Yes, state-driven.** The caveat is that it performs an automatic 100x LR drop. |
| RP15 | Same as RP12 | **Yes**, same caveat |
| RP14 | Same as RP12 | **Yes**, same caveat |
| DV12 | KA2 warmup gates the LR input `game_trust` (inactive before about update 824; reference frozen from updates 800–824). Frozen first-batch feature normalization. `d_guard_min_steps=200`. No horizon; input noise forced to 0; output noise constant. | **Yes, state-driven, continuous.** The caveat is the warmup-gated `gt` term; it slid to about 6% LR on mode_hold. |
| public ka2-constant | `total_steps=4600` hard stop. Input-noise anneal (360) and output-noise warmup (720), both derived as fractions × `total_steps`, which is horizon-based noise. KA2 warmup; guard delay. | **No as declared.** LR is literally constant, but horizon-derived noise and the budget stop fail the goal. Template T0 below fixes it through config. |

## 5. Clock-free templates

Each template gives constant or purely state-driven LRs with no horizon. Task fields `num_particles`, `z_dim` and `batch_size` are per task. Fields not listed stay at package defaults.

| ID | Package | Overrides | LR behaviour | Residual timers (not removable by config) |
|---|---|---|---|---|
| **T0** | `KA2/` public | `total_steps=2**62, lr_floor=1.0, network_lr_floor=1.0, input_noise_std=0.0, output_noise_warmup=0.0, d_guard_min_steps=0`. `lr_anneal_start`, `network_lr_horizon_cap` and `input_noise_anneal_end` become inert. | Constant G/D .00425, prior .0085 | Budget stop at `total_steps` (`training.py:170`); KA2 WARMUP 800 and `sur_base` |
| **T1** | RP12 (state-driven) | RP12 as declared + `d_guard_min_steps=0` | Binary precision switch | KA2 WARMUP 800 / `sur_base` |
| **T1c** | RP12 (constant LR, **untested**) | `continuous_precision=None, total_steps=2**62, lr_floor=1.0, network_lr_floor=1.0, noise_policy="constant", input_noise_std=0.0, game_update="secant_resolvent", relativistic_pairing="all_pairs", particle_loss_weighting="uniform_sampled", adam_eager_state=True, d_guard_min_steps=0` | Constant, with secant step shrink | Budget stop, because `total_steps=None` requires a precision policy (`RP12/recipes.py:133-136`); KA2 WARMUP |
| **T2** | RP14 / RP15 | T1 or T1c with `generator_update="tangent_support"` or `"shared_tangent_support"`, without the RP12 loss fields | As T1 / T1c | As T1 / T1c |
| **T3** | DV12 | DV12 as declared + `d_guard_min_steps=0`. `input_noise_*` and `output_noise_warmup` are ignored under `continuous_policy`. | Continuous mobility | KA2 WARMUP gates `gt`; first-batch data normalization |
| **T4** | C13-R1 / C-family | `continuous=True, input_noise_std=0.0, startup_output_steps=1 (or output_noise_std=0), d_guard_min_steps=0`, plus the family's `critic_memory` / `game_update` | **Constant LR, no budget stop** | One-update output-noise ramp (step 0 has no noise); KA2 WARMUP. Moving/fresh anchors still start at call 800. |
| **T5** | RP5–RP9 (no `noise_policy`) | `continuous_precision="rp5", total_steps=None, input_noise_std=0.0, output_noise_init_steps=1` | Binary precision | One-update output ramp; KA2 WARMUP |

## 6. Hidden timers that need a code change

1. **KA2 `WARMUP_CALLS=800`** (`KA2/ka2.py:23,222`). The penalty form switches and the anchor starts at call 800. This is in every package, and in DV12 it also gates the LR term `game_trust`.
2. **KA2 `sur_base` is frozen** after the first 24 blended calls (`KA2/ka2.py:104-105`). Release, reseed, EMA `alpha` and DV12 `gt` are all measured against a birth-time baseline for the whole run.
3. **Budget stop and `total_steps` validation.**
   - KA2 cannot express an unbounded constant-LR run (`KA2/recipes.py:109-112`, `KA2/training.py:170-171`).
   - RP packages allow `total_steps=None` only with a precision controller (`RP12/recipes.py:133-136`).
   - DV packages allow it only with a controller (`DV12/recipes.py:83-84`).
   - Only C's `continuous=True` removes the stop.
4. **Timed noise.** Packages without `noise_policy` (RP1–RP9) and the C family (`startup_*`) have no constant-noise switch. Config can only reduce this to a one-update ramp or zero noise.
5. **DV input noise is hard-forced to 0** under any `continuous_policy` (`DV12/training.py:14-15`). This is not a timer, but `input_noise_std` cannot be used there.
6. **The rp5 precision close cannot be disabled or bounded by config.** There is no knob for its levels (.01/.05) or its dwell counts (50/25/5) (`RP12/precision.py:23-25,77-101`).
7. **DV12 first-batch feature normalization** is frozen at construction (`DV12/continuous.py:124-130`), a birth-time data reference.
8. **A2 uses a lifetime observed-row rate** (`KA2/k3p.py:177-179`); config can only disable A2 entirely with `latent_damping_max_rate=0`.
9. **K3P's penalty handover is LR-schedule-coupled** (`KA2/grad_regularizers.py:136-142`); at constant LR it never leaves phase A.

## 7. Implications

- The evidence does not separate two explanations for the survivors' passes. Either they need the state-driven LR cut, or they need only the secant step and loss shaping. RP12 held 8/8 at full LR for updates 300–550 before closing. **Template T1c** (RP12 at constant LR with no controller) is the one config that isolates this. It is a new configuration, not a seed repeat.
- The bars4 failure happens at full LR with the controller open. A fix for bars4 is therefore a coverage mechanism, not an LR one.
