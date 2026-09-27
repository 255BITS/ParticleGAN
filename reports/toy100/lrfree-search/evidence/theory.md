# LR-free / schedule-free methods for ParticleGAN: theory, evidence, implementation map

Scope: which learning-rate-free or schedule-free mechanisms could make the public `GANTrainer` converge **and hold**
with fixed nominal rates, under the benchmark rules (live `trainer.G`/`trainer.prior` scored; no clock/horizon LR
or noise; one setting for all tasks; indefinite running). Code references are to the newest reviewed package
`.../deterministic-init-retest/port-source/api-rp12/package/particlegan` (called **RP12-pkg** below). The public
default checkout `/ml2/hypergan/ParticleGAN-ka2-default/particlegan` has the same optimizer classes but not the
precision, secant or particle-weighting code. Both checkouts are read-only. Any change goes into a copied package
under `/ml2/hypergan/lrfree-20260926`.

Every number below comes from archived evidence and was recomputed for this report, unless it is marked
*derived* (linear analysis) or *hypothesis*. No new training was run.

---

## 1. Bottom line (ranked)

| # | Method | Target failure | Code | Cost / run | Priority |
|---|---|---|---|---|---|
| 1 | **AMSGrad (optionally leaky) at constant nominal LR**. It stops Adam's denominator from renormalizing to the shrinking gradients. | Late hold loss and bursts at constant LR | 1 recipe field (`amsgrad`), about 4 lines | about 0 | **High** |
| 2 | **State-driven instance noise** (ADA-style: the critic's real-vs-fake AUC drives the noise level). It replaces the disqualified timed input-noise anneal. Zero-code precursor: constant `input_noise_std`. | Early mode capture (bars4 2/4 by step 50), coverage traps | about 50 lines in `training.py` (the precursor needs 0) | about 0 | **High** (this is the blocker for RP12) |
| 3 | **Lookahead-minmax** (joint G+prior+D, k=7, α=0.5). The slow weights are the live parameters after each sync. | Rotation and jitter at constant LR | about 60 lines (new module + trainer hook + checkpoint key) | 1 parameter copy, <1% compute | Medium |
| 4 | **Existing precision controller** (`continuous_precision="rp5"`), kept as the state-driven hold. Possible refinement: close on "critic cannot discriminate" (AUC from #2) instead of on gap contraction alone. | Hold (already achieved on mode_hold) | 0 now; about 30 lines to refine | 0 | Medium (baseline) |
| 5 | **Zero-centered R1+R2 penalty** (Mescheder / R3GAN local convergence), which also removes KA2's timed 800-call switch | Damping at the equilibrium; removes a clock | about 40 lines in `ka2.py`/`grad_regularizers.py` | 0 | Medium-low |
| 6 | Adam β₂ → 0.9999 (keep β₁=0) | Delays renormalization (period ×10) but does not remove it | 0 (override) | 0 | Low-medium |
| 7 | TTUR (`d_lr_mult` 2–4, `prior_lr_mult` 1) | Rotation (weak) | 0 (override) | 0 | Low |
| 8 | Schedule-free AdamW with x kept in the live parameters (constant-c variant) | Jitter only | about 50 lines (wrapper around `_step`) | 2 parameter copies | Low |
| 9 | Extragradient / optimistic Adam | Rotation. Already superseded by `secant_resolvent`; 79 + C9 tested, no pass | about 60 lines | 2× gradients | Low |
| 10 | Consensus optimization | Rotation. Attracts bad stationary points | about 80 lines, double backprop through G and D | 2–3× | Very low |
| 11 | Critic gradient normalization / clipping (beyond the spike guard) | Nothing relevant: Adam is scale-invariant | — | — | Very low |
| 12 | Prodigy / D-Adaptation / DoG | Picks the LR magnitude only; has monotone or implicit-time step decay | about 150 lines, breaks KA2's Adam-moment contract | 0 | Very low |
| ✗ | Post-hoc Polyak/EMA copy as the scored model | Not allowed; also 0/24 wherever the live model fails | — | — | Excluded |

Short version: the constant-LR failure is not a missing "right LR". **Adam with any fixed LR drives its effective
step to the game's stability edge.** Hold therefore needs a step bound that does not renormalize (#1, #4) or
damping/averaging that lives inside the loop (#3). The best candidate so far, RP12, fails bars4 from **early
capture with zero input noise**. That failure is not an LR problem; #2 addresses it.

---

## 2. Diagnosis from existing evidence

### D1. Constant-LR "quality oscillation" is Adam renormalization at the edge of stability (measured)

In C6/C7 (ring, secant resolvent, KA2, **constant** lr 0.00425, betas (0, 0.999), 7,500 updates), the
normalized generator step ‖ΔG‖/lr, averaged over 500-update windows, creeps up steadily and then bursts. Each
burst coincides with an observed hold failure:

```
C7: 3.7 → 4.4 → 6.2 → 7.3 → [60.4, b²=5.0, fails 2740–2960] → 2.1 → 2.4 → 3.0 → 4.0 → 5.4 → 6.7 → [40.1, fails 6440–6720]
C6: 3.6 → 4.0 → 5.6 → 6.8 → 7.3 → [48.2, b²=7.1, fails 3390–3680] → 2.1 → 2.7 → 3.6 → 4.7 → 6.0 → [69.6, fails 6780–7010]
```

Source: `continuous-api-search/evidence/api-c{6,7}-stationary/learning-rates.jsonl.gz`; b² is the secant
rotation estimate.

The mechanism:
1. Near equilibrium the gradients shrink.
2. Adam's v forgets the older, larger gradients with time constant 1/(1−β₂) = 1,000 updates.
3. The effective step lr/√v̂ therefore grows until it crosses the game's local stability limit.
4. The dynamics burst (rotation b² jumps from about 1 to 5–7) and quality collapses.
5. The large gradients refill v, the step shrinks, and the cycle repeats with a period of about
   3.4–3.7 k updates ≈ 3.5/(1−β₂).

This is the game analogue of "Adam at the edge of stability" (Cohen et al. 2022). **Any constant nominal LR
reaches the same edge**, because the effective step is set by gradient decay, not by lr. A lower lr only
postpones it.

Corollaries:
- **The 1,200-update screen is partly blind to this.** For t ≲ 1/(1−β₂), bias-corrected v̂ is close to the
  uniform mean of the whole gradient history, so Adam behaves like AdaGrad and self-anneals. RP12's normalized
  G step fell from 0.58 (updates 1–100) to 0.016 (updates 501–600) at full LR, before precision closed at
  update 551. Hold claims at constant LR need ≥ 4–6 k-update continuations.
- The only research-host constant-rate run that nearly held, `bg016` (4/5 terminal checks), used **AMSGrad**.
  Its G movement fell 18× (1.4e-3 → 7.7e-5 RMS) at fixed nominal rates (`constant-game-screen.md`). This is
  consistent with D1: a max-denominator cannot renormalize.
- Gradient noise bounds v̂ from below (v̂ ≥ σ²), so the effective step is at most lr/σ. Low-noise problems
  (the ring at batch 2,048) burst sooner. Instance noise (#2) also acts as a denominator floor (*hypothesis*).

### D2. The dominant new-init mode_hold failure is a coverage trap, not jitter (measured)

Most of the 49 API runs and the research runs end at **7/8 or 6/8 modes with HQ ≈ 1.0**: all particles are on
modes and one mode is empty. With 12 equal-mass particles on 8 modes, an exact equilibrium is impossible (mass
k/12 against 1/8). The critic's mass-imbalance force is local, and nothing pulls a particle 2.3 units to the
empty mode. The RP chain isolates the lever:
- RP10 (precision + secant, constant noise) ends at 7/8, and precision closes at 819, **freezing the wrong state**.
- RP11 (+ `relativistic_pairing="all_pairs"`) ends at 5/8.
- **RP12** (+ `particle_loss_weighting="uniform_sampled"`) passes, 19/24.

`uniform_sampled` removes the multinomial count noise in the per-particle force. That reduces gradient
variance, which is exactly the σ in the Adam jitter floor below. RP11's normalized G step keeps bursting
(0.04–0.37) with rotation b² of 3–8 until the end. RP12's step settles to 0.02–0.03 with b² falling from 4.0 to
0.28. This is the landscape argument of Sun et al. (RpGAN pairing has no bad particle basins) implemented
faithfully. **Keep both RP12 loss settings in every candidate.**

### D3. The bars4 failure of RP12/14/15 is early capture with zero input noise (measured; cause confounded)

RP12 on bars4 has 2 modes (0.47/0.53/0/0) at update 50 and never recovers. Precision stays open for the whole
run (`image-runtime-review/api-rp12-img_bars4-runtime-audit.json`). The same initialization with the public
recipe **passes** bars4 at the same lr 0.00425: modes 1→4 by 175, final 12-check suffix. That run had input
noise 0.5 annealed over 700 updates, so noise ≥ 0.07 for the whole 600-update task
(`/ml2/hypergan/pr194-init-search-20260926/batch-feature-full-suite/runs/toy-img_bars4`). All RP candidates use
`input_noise_std=0` (noise_policy `constant`). Timed noise is disqualified, so a **state-driven or constant**
noise level is needed.

### D4. Averaged copies never rescue the live model (measured)

Across all 49 new-init API runs, the EMA copy (decay 0.995) passes 0/24 whenever the live model fails. In the
passing runs it lags (RP12 7/24 against 19/24 live; DV12 4 against 12). Averaging G weights of a moving,
nonlinear generator (and hopping particles) gives bad samples. Averaging therefore only helps when it is inside
the dynamics **and** the live model has already stopped moving much.

### D5. Latent timed phases in the "horizon-free" candidates (flag)

- `ka2.py:23 WARMUP_CALLS = 800`: the penalty is pure A for 799 calls, then the blend. It is a clock-driven
  formulation switch (at update 800 on mode_hold).
- `k3p.py CriticSpikeGuard(min_steps=200)`.
- `input_noise_init_steps` / `output_noise_init_steps` when `continuous_precision` is set and
  `noise_policy="initialization"`. RP12 avoids these by using `constant`.

None of these is horizon-dependent, but if the PR rule means "no timed phases of any kind", #5 removes the KA2
switch.

---

## 3. Theory summary (*derived*; linear models, used only to rank)

**Dirac-GAN with R1 (Mescheder et al. 2018).** The Jacobian eigenvalues are λ = −γ/2 ± √(γ²/4 − f′(0)²), with
f′(0) = ½ for the logistic loss. Simultaneous GD with step h is stable iff h < 2·Re(−λ)/|λ|².

| γ | eigenvalues | max stable h |
|---|---|---|
| 0.1 | −0.05 ± 0.50i | 0.4 |
| 0.5 | −0.25 ± 0.43i | 2.0 |
| 1.0 (critical, 2\|f′\|) | −0.5 (double) | 4.0 |
| 2.0 | −1.87, −0.13 | 1.07 |

- The optimum γ is about 2|f′(0)|. A larger γ stiffens the critic and slows the generator (λ ≈ −f′²/γ).
- Only zero-centered terms damp at the equilibrium. One-sided caps relu(‖∇D‖−κ)² are inactive near equilibrium,
  like WGAN-GP, which is not locally convergent.
- In this repo the A-term is `coeff/2·E‖∇D(real)‖²/d`, so γ_eff = reg_coeff/d: 0.5 on the ring (d=2) and
  0.016 on 8×8 images (d=64). One `reg_coeff` gives dimension-dependent damping. After call 800 the zero-centered
  weight halves (S_FIX=0.5), and the fake side is only ever capped.
- **With Adam, h_eff = lr/√v̂ grows until it meets this boundary (D1).** R1 decides *where* the burst happens,
  not *whether* it happens.

**Spectral radius on a pure rotation (λ = iω), simultaneous updates:**

| method | hω=0.05 | hω=0.1 | hω=0.3 |
|---|---|---|---|
| GDA | 1.00125 | 1.00499 | 1.04403 |
| heavy ball +0.5 | 1.0138 | 1.0457 | 1.2063 |
| schedule-free, constant c (β∈{0,.5,.9,1}, c∈{.01,.1}) | 1.0013–1.0135 | 1.005–1.030 | 1.02–1.08 |
| **Lookahead k=5, α=.5** | **0.9953** | **0.9813** | **0.8396** |
| **Lookahead k=10, α=.5** | **0.9751** | **0.9009** | **0.3038** |
| extragradient | 0.99875 | 0.99504 | 0.95818 |
| OGDA | 0.99875 | 0.99494 | 0.94868 |

- Constant-c schedule-free does not contract rotation. At β=1 it *is* heavy-ball momentum with μ = 1−c (primal
  averaging ≡ momentum), which is the wrong sign for games (Gidel et al. 2019). Lookahead-minmax contracts at no
  extra gradient cost.
- The trainer actually alternates D then G, which keeps bilinear orbits bounded (|μ|=1). LA then contracts them
  by |1−α+αμᵏ| every k steps.

**Adam jitter floor (OU approximation).** Per-coordinate stationary spread ≈ √(lr·σ_g/(2a)), with σ_g the gradient
noise and a the restoring curvature. At fixed lr, the hold improves by lowering σ_g (all_pairs, uniform particle
weighting: RP12's lever), raising a (sharper equilibrium), or averaging inside the loop. Momentum β₁>0 does
**not** reduce the long-run diffusion (Σmₜ ≈ Σgₜ). AdaBelief/SNR-type denominators do not either.

---

## 4. Is averaging legitimate for the live model?

Benchmark rule: the scored model is `trainer.G`/`trainer.prior`, and EMA copies never rescue it.

- **Post-hoc EMA/Polyak (`ema_G`)**: a passive readout that never enters the gradients. It is excluded, and D4
  shows it would not help anyway.
- **Schedule-free (Defazio et al. 2024).** Gradients are taken at y = (1−β)z + βx, the base optimizer updates z,
  and the model to evaluate is x = (1−c)x + c·z. To make x the live model, GANTrainer would keep x in the
  parameters between steps and swap y in only inside `step()`.
  - It is legitimate **only for β > 0** (β ≈ 0.9 recommended): then x shapes every gradient point and is part of
    the dynamics. At β=0, x is exactly an EMA copy with a different name.
  - Caveat: D trains against G(y), not G(x). The gap ‖y−x‖ = (1−β)‖z−x‖ is small but nonzero.
  - **The canonical c = 1/(t+1) is disqualifying for this goal.** x's per-step motion then scales as γ/t, which
    is an implicit 1/t schedule on a clock. The model freezes and cannot follow a target change. The
    horizon-free variant (constant c, or c reset by a state event) loses the schedule-free guarantee and, per §3,
    does not damp rotation.
- **Lookahead-minmax (Chavdarova et al., ICLR 2021).** Every k steps the slow weights move halfway to the fast
  weights, and the fast weights are **reset to the slow weights; training continues from them**. The live
  parameters are the averaged point, not a side copy, so it is legitimate. Between syncs the live model is the
  fast weights; D was trained against those, which is also consistent.
  - To avoid any "evaluation lands on sync" objection, **use k coprime with the observation cadences (25/50):
    k=7**. The scored checks then sample all 7 phases of the cycle.
  - The paper fixes α=0.5, keeps the base-optimizer state, and uses k=5 for alternating GAN updates.

---

## 5. Method cards (RP12-pkg paths)

### #1 AMSGrad, optionally leaky: fixed nominal LR, no renormalization
- **Effect on the observed failure.** It removes D1's growth of lr/√v̂: max-v never decreases, so the effective
  step is non-increasing while gradients shrink. Constant rates can then settle instead of cycling. It does not
  help coverage (D2/D3).
- **Implementation.**
  - `recipes.py`: append field `amsgrad: bool = False` after `initialization` (fields are appended to keep
    positional meaning), and add a bool check.
  - In `make_critic_optimizer` (line ~316) and `make_generator_optimizer` (~332), set
    `options = {"lr": ..., "betas": ..., "amsgrad": self.amsgrad, **adam_kwargs}`.
  - `_initialize_adam` already creates `max_exp_avg_sq`. The harnesses pass only `recipe_overrides`, so a recipe
    field is required; `optimizer_options` is frozen to foreach/fused.
  - Compatibility: KA2 surprise (`ka2.py:_surprise_of`) and the spike guard read `exp_avg_sq`, so they are
    unchanged. A2 `LatentRowDamping` edits only `exp_avg`/betas; its docstring formula then uses max-v (benign).
  - Leaky variant, for reversibility after target shifts: field `amsgrad_leak: float = 1.0`. After `_adam_step`
    in `K3PGeneratorAdam.step` (k3p.py:517) and `K3PCriticAdam.step` (k3p.py:436), for each state run
    `max_exp_avg_sq.mul_(leak).clamp_min_(exp_avg_sq)`. Use leak 1 − (1−β₂)/10 for a 10k-update memory. That is
    a fixed time constant, not a horizon.
  - The state-driven alternative to a leak: reset `max_exp_avg_sq ← exp_avg_sq` on a precision `reopen` event.
- **Risk.** Irreversible when pure: after a target change with smaller gradients than the historical max,
  re-acquisition is slow. PyTorch takes the max over *uncorrected* v, so the startup spike is diluted by the bias
  factor. Untested with the new init. bg016 (old init) came within 1 check of passing.
- **Priority.** High. This is the most direct fix for "hold with fixed rates".

### #2 State-driven instance noise (replaces the timed anneal)
- **Effect.**
  - Early: while the critic separates real and fake easily, noise rises. This keeps critic gradients informative
    across modes and prevents early capture (D3).
  - Near equilibrium, AUC → ~0.5 and noise decays toward 0, so fine quality is retained. After a target shift,
    AUC rises and noise returns, giving re-exploration.
  - It also floors Adam's v, bounding h_eff (D1 corollary).
  - The optimum is unchanged for realizable targets, because the same noise is applied to real and fake
    (Sønderby 2016; Arjovsky & Bottou 2017).
- **Zero-code precursor.** RP12 overrides plus `input_noise_std: 0.15` (policy is already `constant`). 0.15 is
  below the ring HQ radius of 0.21 but large relative to the σ=0.07 modes. It may be too weak for bars4, whose
  passing run had 0.5 → 0.07.
- **Implementation** (`training.py`):
  - Add `"adaptive"` to the allowed `noise_policy` values (recipes.py:92). New fields `input_noise_target=0.6`
    (as ADA) and `input_noise_rate=0.02` (fraction of the cap per update). `input_noise_std` becomes the cap.
  - In `GANTrainer.__init__`, set `self.noise_state = {"std": cap, "auc_ema": None}`, starting at the cap.
  - In `_ordinary_step` (line ~222), use `sigma_in = self.noise_state["std"]`. Keep the logits built at line 233
    and compute `auc = (real_logits[:,None] > fake_logits[None,:]).float().mean()` under `no_grad`. It is
    pairwise because RpGAN logits have no absolute level. Store it in `self._last_auc`.
  - Apply the controller update **once per accepted update** in `_step` (line 200) after `joint_step`/ordinary
    returns: `std = clip(std + rate·cap·sign(auc_ema − target), 0, cap)`. The preview fields of `joint_step`
    must not advance it, because `restore_preview_state` does not restore trainer attributes.
  - Add optional checkpoint key `"noise_state"` in `state_dict`/`load_state_dict` (same pattern as `precision`).
  - Log `std` and `auc` per update.
- **Risk.**
  - Interaction with precision: noise changes D's gradient field, which may delay close or trigger reopen.
  - If the target is unrealizable (12 particles on 8 modes), the noise settles above 0 and blurs fine structure;
    watch vector covariance metrics.
  - `target` and `rate` are fixed controller constants, not per-task values.
- **Priority.** High for the user's goal: it is the measured blocker of the best candidate.

### #3 Lookahead-minmax
- **Effect.** It contracts rotational and limit-cycle modes at zero gradient cost (§3). For random-walk noise,
  the slow-weight variance falls by about α (std ×0.71 at α=0.5). It slows directed motion by about α, so
  arrival is likely later. It delays but does not remove D1 bursts, because the fast weights still use Adam.
- **Implementation.**
  - New `particlegan/lookahead.py`: `JointLookahead(params of G, prior, D; k, alpha)` holds `slow` clones and a
    counter. `after_step()`: every k accepted updates run `slow.lerp_(p, alpha); p.copy_(slow)` under
    `no_grad`, then `state_dict`/`load_state_dict`.
  - Hook it in `GANTrainer._step` after `joint_step`/`_ordinary_step` returns. It must not go inside
    `_ordinary_step`, which `joint_step` calls 2–3 times.
  - Recipe fields `lookahead_k: int = 0` (off) and `lookahead_alpha: float = 0.5`. Optional checkpoint key
    `"lookahead"`.
  - Keep the Adam, KA2 anchor and precision reference states as they are (paper default).
- **Risk.** Slower acquisition. Precision gap statistics see the sync jumps. Low implementation risk.
- **Priority.** Medium. Use it if #1 alone still shows rotation bursts (b² rising) in long holds.

### #4 Precision controller (in repo)
- **Status.** `precision.py:ReversiblePrecision` gives a state-driven 100× (network) / 20× (prior) LR cut on
  "contraction finished + low activity" and reopens on a shock. RP12 closes at 551 after arriving at 300 and
  passes. RP2's 30 k run retained 537/537, 145/145, 1879/1879 and 269/269 across three changes. It effectively
  resets the D1 cycle by cutting lr whenever calm, and reopening covers bursts.
- **Known failures.** It freezes wrong states (RP10 closed at 7/8). It never closes on bars4 (2/4 stuck at full
  LR, since activity is not calm).
- **Refinement (optional).** Require "critic cannot discriminate" (#2's AUC_ema < target + δ) as an additional
  close condition in `advance()`, so a coverage-deficient state (AUC high) never closes.
- **Priority.** Medium: keep it as the reference hold. #1 is the fixed-nominal-rate alternative.

### #5 Zero-centered R1+R2, no timed switch
- **Effect.** Damping at the equilibrium (§3), R3GAN's locally convergent RpGAN + R1 + R2 formulation, and it
  removes the `WARMUP_CALLS` clock. Weakness: γ_eff = coeff/d is dimension-dependent (§3).
- **Implementation.** In `ka2.py:KA2GradientPenalty._k3p_penalty` (line 206), add a
  `reg_formulation="r1r2"` branch: `coeff/2·(E‖∇D(real)‖² + E‖∇D(fake)‖²)/d` for every call. Either keep the
  anchor prox (critic proximal term) or not, but decide explicitly.
- **Evidence.** Research `c01_r1r2` reached 8/8 at 500 but ended at 0.81 HQ (FAIL 8/24). Not sufficient alone.
- **Priority.** Medium-low. Do it only if D5's clock is ruled disqualifying.

### #6 Adam β choices
- Keep β₁=0. Positive momentum worsens rotation (§3), and negative momentum ≈ optimistic, which failed.
- β₂=0.99 (StyleGAN/R3GAN) renormalizes 10× faster than 0.999: worse for hold, faster early.
- β₂=0.9999 stretches the D1 period by about 10× (to ~35 k updates) without removing it. The 96-row
  constant-LR sweep (β₂ up to 0.9998, old init) found no pass.
- Override: `betas: [0.0, 0.9999]`. It also changes the KA2 surprise statistics.

### #7 TTUR
- Heusel's convergence theorem needs *decaying* two-timescale steps, which is a schedule. At constant rates it
  only separates speeds, and Adam normalizes the magnitudes anyway.
- Sweeps with D ×0.5–2.5 and prior ×0.5–5 found no pass. A cheap override if needed: `d_lr_mult: 2.0`,
  `prior_lr_mult: 1.0`, since particles currently move fastest (×2), and they are what hops.

### #8 Schedule-free AdamW (x live)
- **Implementation.** Wrap `GANTrainer._step`:
  1. Before the step, compute y = (1−β)z + βx and write it into the parameters of G, prior and D.
  2. Run the existing step. The base Adam updates y by Δ.
  3. Set z += Δ, x = (1−c)x + c·z, and write x back.
  This works unchanged with `joint_step`, because the whole joint step runs inside the bracket.
- Costs: 2 parameter copies and a checkpoint key.
- Use a constant c; 1/t is disqualified (§4). §3 shows no rotation damping, and D4 makes a lagging x risky.
  **Low.**

### #9 Extragradient / optimistic
- 79 constant-rate optimistic/AMSGrad candidates and API-C9 were measured without a pass. The existing
  `secant_resolvent` (`game_update.py`) is an implicit (backward-Euler) step fitted from the same two field
  evaluations. It is A-stable in the fitted plane, which is stronger than EG. **Low.**

### #10–12
- **Consensus** needs ∇‖v‖² through both networks (2–3× cost). It also makes any stationary point more
  attractive, including the 7/8 trap (D2).
- **Critic gradient normalization/clipping**: Adam is invariant to gradient scale. The spike guard already
  handles relative spikes.
- **Prodigy / D-Adaptation**:
  - They estimate a monotone non-decreasing d, and their Adam variants keep EMA denominators, so D1 still
    applies. Prodigy is used with cosine annealing in its paper.
  - Distance-to-solution estimates are ill-posed in rotating games.
  - They also break the Adam-moment contracts that KA2, the spike guard and A2 read.
- **DoG / AdaGrad-norm**: Σ‖g‖² grows linearly under persistent noise, so the step goes as 1/√t. That is an
  implicit clock and irreversible.
- These methods only answer "which LR magnitude", and the public recipe's 0.00425 already transfers (bars4
  passes with noise).

---

## 6. Proposed experiment ladder (single runs, no seeds; frozen harnesses)

Base = RP12 `recipe_overrides` (`evidence/api-rp12-new-init/declaration.json`). Timings from evidence:
mode_hold about 110 s, bars4 about 30 s. Log per update: the normalized step (`game_stats.{g,d}.applied_l2`/lr),
secant `rotation_squared`, AUC and noise std.

| ID | Overrides on RP12 | Code | Question | Tasks, in order |
|---|---|---|---|---|
| E0 `RP12-CONST` | `continuous_precision: null, total_steps: 1000000000` (integer budget guard only; nothing depends on it once both floors are 1), `lr_floor: 1.0, network_lr_floor: 1.0` | none | Does RP12 hold without the LR cut? | mode_hold, then a **6,000-update continuation** (D1 needs more than 3/(1−β₂)) |
| E1 `RP12-CONST-AMS` | E0 + `amsgrad: true` | #1 | Does a non-renormalizing denominator hold at fixed rates? | mode_hold + 6 k continuation |
| E2 `RP12-NOISE015` | `input_noise_std: 0.15` | none | Does constant noise fix bars4 without breaking mode_hold? | bars4 → mode_hold |
| E3 `RP12-ADAPTNOISE` | `noise_policy: "adaptive", input_noise_std: 0.5, input_noise_target: 0.6` | #2 | Same, with state-driven noise | bars4 → mode_hold → other images/vectors |
| E4 `RP12-CONST-LA7` | E0 + `lookahead_k: 7, lookahead_alpha: 0.5` | #3 | Is rotation damping needed beyond #1? | only if E1 shows b² bursts |

The final candidate is the E1/E3 combination (constant nominal rates, AMSGrad, adaptive noise, RP12 loss and
secant, with or without precision). It must pass all of: mode_hold, all 4 images, 6 vectors including
unequal_mass, and the ring shift/long hold.

**Stop rules.** If E0's normalized G step shows a sustained rise over more than 1 k updates in the continuation
(the D1 signature), constant Adam is confirmed insufficient and E1 becomes mandatory. If E2 and E3 still capture
early on bars4, the coverage lever is latent smoothing (DV16 passes bars 19/24) rather than noise.

---

## 7. Sources

These were fetched this session: Schedule-Free ([arXiv 2405.15682](https://arxiv.org/abs/2405.15682)): update
equations, β≈0.9, c=1/(t+1) uniform averaging, β=0 Polyak ↔ β=1 primal averaging. Lookahead-Minmax
([arXiv 2006.14567](https://arxiv.org/abs/2006.14567)): α=0.5, k=5 for alternating GANs, joint backtracking,
local convergence given a convergent base. Also a search on R3GAN ([arXiv 2501.05441](https://arxiv.org/abs/2501.05441)),
confirming RpGAN + R1/R2 local convergence.

The rest are cited from memory and were not re-fetched:
- Mescheder et al. 2018, [1801.04406](https://arxiv.org/abs/1801.04406)
- TTUR, [1706.08500](https://arxiv.org/abs/1706.08500)
- Optimistic Adam, [1711.00141](https://arxiv.org/abs/1711.00141)
- Extragradient VI, [1802.10551](https://arxiv.org/abs/1802.10551)
- Consensus, [1705.10461](https://arxiv.org/abs/1705.10461)
- Negative momentum, [1807.04740](https://arxiv.org/abs/1807.04740)
- AMSGrad, [1904.09237](https://arxiv.org/abs/1904.09237)
- Prodigy, [2306.06101](https://arxiv.org/abs/2306.06101)
- D-Adaptation, [2301.07733](https://arxiv.org/abs/2301.07733)
- DoG, [2302.12022](https://arxiv.org/abs/2302.12022)
- Adam at the edge of stability, [2207.14484](https://arxiv.org/abs/2207.14484)
- RpGAN landscape (Sun et al. 2020), [2011.04926](https://arxiv.org/abs/2011.04926)
- StyleGAN2-ADA, [2006.06676](https://arxiv.org/abs/2006.06676)
- Instance noise, [1610.04490](https://arxiv.org/abs/1610.04490)
- Arjovsky & Bottou 2017, [1701.04862](https://arxiv.org/abs/1701.04862)
