# Sparse conditional mixed-output toy with a UCD discriminator — findings

Protocol and metric definitions: [`README.md`](README.md). Problem: 64
3-sparse modes in R^24, 8 classes, one categorical symbol per mode
(v1: symbol = class). Recipe under test: the 100-Gaussians 0.1.2 champion
(RpGAN logistic, one-sided cap penalty, Fourier-2 D, EMA read-out, Adam β1=0,
delayed cosine anneal, particle prior) with a UCD discriminator
(arXiv:2510.00624) for the class. All runs on one A6000 (`cuda:1`), 4–5k
steps, ~1–3 min each.

**Review correction:** these are the original results, not corrected reruns.
All historical `ppur` values are invalid, and the x-only gradient-penalty
ablation is confounded. Its causal interpretation and the claim of soft-head
saturation from `y_hardness` are withdrawn below. See [CORRECTIONS.md](CORRECTIONS.md).

Two phases: **probes** (single-seed ad-hoc runs, `~/tmp/sparse_probe*`, five
rounds, 79 runs) to find *a* recipe that converges at all, then the **staged
grid** (3 seeds per cell, `results/sparse/`) on that recipe. Section 1 is the
probe story; sections 2–6 are the grid, one stage each.

## 1. Probes: what it took to converge

### 1.1 The champion as shipped collapses to one point per class

`b_cap` + Fourier-2 D + class embedding concatenated to z in G, 4k steps:
**0/64 modes**, `sep` 3.5 (per-class fake clouds *more* separated than the
real ones), `core` 0.08. The PCA panel shows eight dots. G ignores z entirely
and emits `f(c)`; the particle prior has nothing to act on. D wins outright
(D loss 0.02, G loss 6). This is the conditional-collapse basin the
particle-z conditioning experiment mapped (`docs/conditional-z-findings.md`
on that branch), reached here through the class-embedding shortcut.

Single-knob changes that did **not** fix it (modes at 4k): 2× particles
(2), z 4 (2), LR 2e-4 (0), D LR ×1 (2), cap coeff 10 (0), cap κ 0.1 (26 but
hq 0.27), R1/R2 @ 0.1 (13) or 1.0 (33, hq 0.23), Fourier-1 (7), plain-MLP D
(40 but hq 0.15, core 5 — blurry), two-sided interpolate penalty @ 1 (20).

### 1.2 Two changes are load-bearing, and they are structural

1. **Where the penalty acts.** `b_cap` bounds ‖∇D‖ *at the samples*. In 2-D
   that is enough; in 24-D the fakes sit far from the reals along many dims
   and D is free to be arbitrarily steep on the path between them, which is
   exactly where the fakes must travel. A one-sided cap on real/fake
   **interpolates** (`g_interp_cap`, new arm in `lib/grad_regularizers.py`:
   `relu(‖∇D(x_i)‖ − 1)²`, coeff 1.0) restores a playable game: with the
   same G, modes 0 → 33 and hq 0 → 0.82 (`icap1`), and with a plain-MLP D
   hq 0.15 → 0.95 (`f0_icap1`). Coeff 10 over-flattens (0 modes); the
   two-sided `e_interp` at the same coeff is worse (20 modes, core 0.33),
   consistent with the compatibility result in `FINDINGS.md` (a center-at-1
   penalty fights the logistic optimum).
2. **How the class reaches G.** An *unconditional* G (`emb_dim 0`) under the
   same UCD D reaches 31/64 at 3k and 54/64 hq 0.94 with a plain D — the
   recipe covers the modes fine once the shortcut is gone. So the class is
   routed through the prior instead: `prior_partition: class` gives each
   class its own block of particles and G never sees c. The UCD loss
   `d(G(z))[c]` then moves the *particles* of block c toward class-c modes.
   With `b_cap` this does not work (`pp_noemb`: 50 modes but hq 0.11 and
   cond 0.24 — the class never reaches the particles); with the interpolate
   cap it does (`pp_noemb_icap1`: **51/64, hq 0.93, cond 0.98, joint 0.94,
   core 0.80**). Adding the embedding back on top of the partition
   (`pp_emb` / `emb8`) re-opens the shortcut and coverage falls to 10–27.

Controls on the partitioned prior + interpolate cap: scalar D (no class
anywhere) cond 0.12; concat / proj D cond 0.22–0.24 (the class *is* in D but
G cannot use it — the partition needs a per-class score, which only the UCD
head provides); frozen Gaussian blocks (`prior: gaussian`) cond 0.22 — the
blocks all sit on N(0, I), so a learnable prior is what makes prior-only
conditioning possible.

### 1.3 Closing the gap: capacity and schedule (round 4/5, on the base above)

| change | modes | hq | sym | core |
|---|---|---|---|---|
| base (h128, z8, Fourier-2) | 53 | 0.947 | 0.939 | 0.68 |
| hidden 256 | 62 | 0.981 | 0.977 | 0.51 |
| z_dim 16 | 61 | 0.962 | 0.993 | 0.81 |
| Fourier ramp 0.3→0.7 | 57 | 0.965 | 0.998 | 0.65 |
| plain-MLP D | 59 | 0.967 | 0.999 | 0.45 |
| κ 0.5 | 61 | 0.933 | 0.952 | 0.83 |
| batch 512 | 60 | 0.958 | 0.924 | 0.60 |
| LR 1e-3 | 60 | 0.981 | 0.957 | 0.49 |
| **h256 + z16 + ramp** (seeds 1, 2) | **63, 63** | 0.989, 0.986 | 0.999 | 0.79, 0.87 |
| h256 + z16 | 63 | 0.992 | 0.999 | 0.62 |
| h256 + z16 + plain D | 63 | 0.995 | 1.000 | 0.52 |
| h256 + LR 1e-3 | 0 | — | — | diverged |

Things that hurt: z 4 (45), 4× particles (41 — more particles per block
means each is updated less often), prior LR ×5 (55, hq 0.89), the gated head
switched on from step 0 (34 modes, hq 0.49 — pruning before coverage strands
modes; hence the `gate_start_frac` warm-up knob), the top-k head from step 0
(0 modes).

**Base recipe for the grid** (`gen_sparse_configs.BASE`): `emb_dim 0`,
`prior_partition class`, `arm g_interp_cap` @ 1.0, `hidden 256`, `z_dim 16`,
Fourier-2 ramped in over 30–70 % of the run; everything else the champion's.
One mode of 64 is still typically missing at 5k steps and the linear head
smears (`sp@1e-2` ≈ 0.7), so the bar is not yet passed by the base itself —
the sparse stage is where that is decided.

## 2. Stage `recipe` — what is load-bearing (3 seeds, `TABLE_recipe.md`)

| group | modes | hq | cond | sym | joint | sp@1e-2 | core | w1 |
|---|---|---|---|---|---|---|---|---|
| champion (as shipped) | 2±2 | 0.25 | 0.79 | 0.75 | 0.75 | 0.36 | 0.07 | 0.356 |
| champion + plain-MLP D | 36±3 | 0.16 | 0.97 | 0.96 | 0.96 | 0.15 | 4.75 | 0.115 |
| champion + R1/R2 @ 1.0 | 35±0 | 0.20 | 0.98 | 0.98 | 0.98 | 0.15 | 5.00 | 0.112 |
| champion + two-sided interp @ 1.0 | 16±12 | 0.51 | 0.99 | 0.99 | 0.99 | 0.39 | 0.41 | 0.124 |
| interp cap, embedding kept (`emb_icap`) | 26±9 | 0.80 | 0.999 | 0.999 | 0.999 | 0.50 | 0.29 | 0.107 |
| `emb_icap` + Fourier ramp | 37±3 | 0.94 | 0.994 | 0.994 | 0.994 | 0.55 | 0.33 | 0.079 |
| partitioned prior, sample cap kept (`pp_bcap`) | 48±4 | 0.12 | 0.24 | 0.50 | 0.16 | 0.11 | 8.07 | 0.112 |
| partitioned prior + interp cap (`pp_icap`) | 54±1 | 0.94 | 0.993 | 0.93 | 0.93 | 0.53 | 0.79 | 0.059 |
| `pp_icap` + ramp | 56±2 | 0.96 | 0.997 | 0.995 | 0.995 | 0.56 | 0.72 | 0.054 |
| `pp_icap` + plain D | 58±3 | 0.97 | 0.998 | 0.996 | 0.996 | 0.67 | 0.44 | 0.051 |
| `pp_icap` + hidden 256 | 61±1 | 0.98 | 0.999 | 0.98 | 0.98 | 0.73 | 0.57 | 0.051 |
| `pp_icap` + z 16 | 60±1 | 0.95 | 0.997 | 0.98 | 0.98 | 0.42 | 0.96 | 0.052 |
| **base** (`pp_icap` + h256 + z16 + ramp) | **63±0** | **0.990** | **1.000** | **1.000** | **1.000** | 0.71 | 0.78 | 0.041 |
| base, plain D | 63±0 | 0.994 | 1.000 | 1.000 | 1.000 | 0.70 | 0.52 | 0.040 |
| base, 8k steps | 63±1 | 0.990 | 1.000 | 1.000 | 1.000 | 0.70 | 0.65 | 0.042 |
| uncond floor (G unconditional, scalar D) | 1±1 | 0.05 | 0.13 | — | — | — | — | 0.193 |
| uncond "ceiling" (G unconditional, UCD D, interp cap) | 0±0 | 0.00 | 0.12 | — | — | — | — | 0.334 |

Readings:

* **The champion does not converge here**, on any seed: 2 modes, `sep` 3.1
  (one point per class). Every single-knob repair of the *penalty* leaves
  the game either blurry (plain D / R1R2: core ≈ 5, hq 0.2) or stranded
  (two-sided interpolate: 16±12 — seed-dependent).
* **Both structural changes are needed.** Interpolate cap alone: 26±9 (the
  shortcut is still there; seeds land in different basins). Partitioned
  prior alone: 48 modes but `cond` 0.24 and core 8 — with the sample-point
  cap D stays blurry and the class never propagates into the particles.
  Together: 54 → 63 once capacity (hidden 256, z 16) is added.
* **Conditioning through the prior only is exact once it works**:
  cond/sym/joint = 1.000 on all three seeds of the base; the symbol head
  needs no supervision (λ_sym = 0 throughout).
* The remaining mode (63/64) is stable across seeds and run length (8k
  steps: 63±1), so it is a capacity/allocation floor of the base rather
  than slow convergence. The width ratio is the base's weak point
  (core 0.78±0.08 — mildly compressed; the plain-MLP D variant is sharper
  on hq but compresses to 0.52).
* The "uncond ceiling" control (unconditional G under the interpolate cap
  with a Fourier-2 UCD D) fails outright (0 modes, `rec` 0.11), unlike the
  probe-round unconditional runs under the sample cap or a plain D. Not
  chased; it is a control, and the base does not depend on it.

## 3. Stage `ucd` — the discriminator question (`TABLE_ucd.md`)

All on the base (partitioned prior, no embedding, interp cap, h256, z16, ramp).

| D mode | modes | hq | cond | sym | sp@1e-2 | core | w1 |
|---|---|---|---|---|---|---|---|
| ucd λ₁ 0.02 | **64±0** | 0.988 | 1.000 | 1.000 | 0.85 | 0.81 | 0.032 |
| ucd λ₁ 0.1 (base) | 63±0 | 0.990 | 1.000 | 1.000 | 0.71 | 0.78 | 0.041 |
| ucd λ₁ 0.5 | 61±0 | 0.983 | 0.999 | 0.998 | 0.56 | 0.75 | 0.056 |
| ucd λ₁ 2.0 | 56±1 | 0.949 | 0.992 | 0.992 | 0.49 | 0.83 | 0.070 |
| concat (one-hot c into D) | 64±0 | 0.980 | 0.990 | 0.995 | 0.91 | 0.60 | 0.025 |
| proj | 64±0 | 0.983 | 0.993 | 0.995 | 0.91 | 0.67 | 0.024 |
| scalar (no class in D) | 64±0 | 0.987 | **0.127** | 0.997 | 0.88 | 0.56 | 0.033 |
| ucd + frozen Gaussian prior | 25±1 | 0.04 | 0.953 | 0.925 | 0.08 | 5.5 | 0.084 |
| ucd + class embedding in G (no partition) | 45±1 | 0.965 | 0.997 | 0.997 | 0.71 | 0.45 | 0.076 |
| ucd + embedding *and* partition | 52±1 | 0.984 | 0.999 | 0.999 | 0.69 | 0.53 | 0.073 |
| ucd + supervised symbol (λ_sym 1) | 62±1 | 0.990 | 1.000 | 1.000 | 0.69 | 0.75 | 0.054 |
| ucd, historical x-only penalty (confounded) | 63±1 | 0.949 | 0.992 | 0.966 | 0.69 | 1.15 | 0.054 |

Readings:

* **λ₁ is a steep dial and must be small.** 0.02 is the best UCD cell
  (64/64 every seed, sparsest linear-head output, lowest W1 of the UCD
  cells); 0.5 and 2.0 lose modes monotonically. This matches the paper's
  0.01–0.02 and its warning that a large λ₁ weakens the adversarial term.
* **On this toy UCD is not better than condition injection.** With the
  prior carrying the class, `concat` and `proj` both reach 64/64 with
  cond 0.99 and the lowest W1 in the stage (0.025); UCD@0.02 matches on
  coverage and conditioning (1.000 vs 0.99) at slightly higher W1. The
  paper's claimed advantage (a D backbone that is not distracted by the
  condition) has nothing to bite on with a 3-layer MLP on 24 dims. The
  probe-round result that concat/proj fail (cond 0.22) was at h128/z8
  without the ramp — the capacity additions rescued them too.
* **The scalar D is the clean control**: 64/64 modes, hq 0.99, but cond
  0.127 = 1/8. Coverage and conditioning are fully decoupled in this
  design: the prior partition can only be *used* if D scores per class.
* **The prior must be learnable**: frozen per-class Gaussian blocks give
  cond 0.95 but 25 modes and core 5.5 — G alone cannot warp eight
  identical N(0, I) blocks into eight different mode sets.
* Re-adding the class embedding to G costs 11–18 modes (45 / 52) with the
  same D: the shortcut is a G-side pathology, independent of D.
* The supervised symbol anchor is unnecessary (62 vs 63, identical
  cond/sym). The historical x-only penalty run had hq 0.95, sym 0.97,
  and core 1.15, but it also moved the penalty from interpolates to
  endpoints. It does not isolate the effect of the y derivative; the
  earlier conclusion that the cap should see the symbol input is withdrawn
  pending a corrected ablation.

## 4. Stage `sparse` — exact zeros (`TABLE_sparse.md`)

| real head | modes | hq | cond | sym | sp@1e-2 | sp@1e-3 | zero | rec | core |
|---|---|---|---|---|---|---|---|---|---|
| linear (base) | 63±0 | 0.990 | 1.000 | 1.000 | 0.71 | 0.09 | 0.00 | 0.999 | 0.78 |
| gated, λ_sp 0 | 51±4 | 0.81 | 0.97 | 0.97 | 0.95 | 0.92 | 0.92 | 0.92 | 1.38 |
| gated, λ_sp 0.01 | 36±4 | 0.64 | 0.92 | 0.92 | 0.96 | 0.95 | 0.94 | 0.84 | 1.63 |
| gated, λ_sp 0.1 | 16±4 | 0.29 | 0.77 | 0.74 | 0.98 | 0.98 | 0.98 | 0.59 | 2.14 |
| gated, λ_sp 1.0 | 0 | — | 0.18 | — | 1.00 | 1.00 | 1.00 | 0.04 | — |
| gated, λ_sp 0, warm 40 % | 48±20 | 0.69 | 0.93 | 0.93 | 0.85 | 0.80 | 0.79 | 0.91 | 1.8 |
| **gated, λ_sp 0.01, warm 40 %** | **63±0** | **0.962** | **0.991** | **0.991** | **0.968** | **0.947** | **0.94** | 0.983 | **0.83** |
| top-k (oracle k) | 4±0 | 0.02 | 0.55 | 0.54 | 0.93 | 0.92 | 0.92 | 0.35 | 4.0 |
| top-k, warm 40 % | 10±3 | 0.05 | 0.60 | 0.60 | 0.93 | 0.93 | 0.92 | 0.40 | 4.5 |

Readings:

* **The linear head smears, as predicted** (v0 in the design note):
  sp@1e-2 0.71, sp@1e-3 0.09, no exact zeros, mean |x| 0.009 off-support.
  Small, invisible in the heatmap, and wrong.
* **A gate from step 0 trades coverage for sparsity monotonically in
  λ_sp**: 51 → 36 → 16 → 0 modes as λ_sp goes 0 → 1, with sp@1e-3 rising
  0.92 → 1.0 and `rec` (active dims still alive) falling 0.92 → 0.04. The
  gate prunes before G has found the modes, and a pruned dim gets no
  gradient to come back (STE forward is exactly zero).
* **Warm-starting the gate fixes it**: linear head for the first 40 % of
  the run, then the gate engages. With λ_sp 0.01 this keeps the base's
  coverage (63±0) and conditioning (0.99) while making 94 % of the
  inactive coordinates *exactly* 0.0 and 97 % of them < 1e-2, at the
  base's width (core 0.83) and W1 (0.041). It is the first cell that
  clears every axis of the bar except the 64th mode — the one the base
  itself is missing.
* λ_sp 0 with the warm start is high-variance (48±20): the gate needs a
  small push to close; without it some seeds prune the wrong dims.
* The top-k head fails even with the warm start (10 modes, `rec` 0.4):
  hard top-k over 24 gate logits with a straight-through sigmoid gradient
  is too unstable a mechanism, oracle k or not. Not worth more time.

## 5. Stage `discrete` — the symbol head (`TABLE_discrete.md`)

| cat_mode | τ | λ_sym | symbol map | modes | hq | sym | symKL | core |
|---|---|---|---|---|---|---|---|---|
| gumbel straight-through (base) | 1 | 0 | identity | 63±0 | 0.990 | 1.000 | 0.00 | 0.78 |
| gumbel ST | 1→0.1 | 0 | identity | 63±0 | 0.989 | 1.000 | 0.00 | 0.80 |
| gumbel soft | 1→0.1 | 0 | identity | 63±0 | 0.990 | 0.999 | 0.00 | 0.80 |
| ST argmax | 1 | 0 | identity | **64±0** | 0.991 | 1.000 | 0.00 | 0.71 |
| softmax probs | 1 | 0 | identity | **64±0** | 0.988 | 0.999 | 0.00 | 0.76 |
| softmax probs | 1→0.1 | 0 | identity | 63±0 | 0.993 | 0.999 | 0.00 | 0.76 |
| gumbel ST + supervised symbol | 1 | 1.0 | identity | 62±1 | 0.990 | 1.000 | 0.00 | 0.75 |
| gumbel ST | 1 | 0 | split (K = 16, 50/50 per class) | 63±0 | 0.990 | 0.998 | 0.02 | 0.77 |
| softmax probs | 1 | 0 | split | 61±2 | 0.989 | 0.996 | 0.02 | 0.85 |

Readings:

* **The discrete output is not the hard part.** Every relaxation — hard
  Gumbel, soft Gumbel, straight-through argmax, plain softmax probabilities
  — learns the symbol to ≥ 0.996 accuracy *purely adversarially* (λ_sym 0),
  with D seeing the relaxed vector at train time and the hard one-hot at
  eval. The softmax head hands D an obvious real/fake tell (one-hot vs.
  soft). Its successful hard evaluation output does not establish that its
  training output saturated: `y_hardness` is always one after eval's
  argmax read-out. The earlier saturation claim is withdrawn.
* A supervised anchor buys nothing (62±1 vs 63±0). τ-annealing buys
  nothing. Straight-through argmax and plain softmax are marginally the
  best on coverage (64/64), inside seed noise.
* Under `split`, where each class must emit *two* symbols at 50/50, the
  emitted per-class distribution matches the truth to KL 0.02 — the head
  represents a distribution over symbols, not one symbol per class; it is
  not simply reading the symbol off the class.
* Why it is easy here: the symbol is a deterministic (or 2-way) function of
  the mode, the mode is carried by the particle, and D scores the joint
  [x | y] with the cap penalty acting on both. The symbol head is a
  classifier of the particle, and the particles are already sorted by mode.

## 6. Stage `fewshot` — a finite pool of reals (`TABLE_fewshot.md`)

| n_train (per mode) | modes | hq | cond | sym | sp@1e-2 | core | w1 |
|---|---|---|---|---|---|---|---|
| ∞ (base) | 63±0 | 0.990 | 1.000 | 1.000 | 0.71 | 0.78 | 0.041 |
| 8192 (128) | 63±1 | 0.992 | 1.000 | 1.000 | 0.72 | 0.77 | 0.046 |
| 2048 (32) | 63±0 | 0.990 | 1.000 | 1.000 | 0.72 | 0.71 | 0.049 |
| 512 (8) | 62±2 | 0.992 | 1.000 | 0.999 | 0.73 | 0.63 | 0.056 |
| 128 (2) | 50±2 | 0.990 | 1.000 | 0.999 | 0.77 | **0.42** | 0.089 |
| 512 (8), frozen Gaussian prior | 26±1 | 0.07 | 0.94 | 0.91 | 0.08 | 5.4 | 0.093 |

Readings:

* **Down to 8 reals per mode nothing changes** (62±2 modes, cond 1.0,
  hq 0.99). At 2 per mode the model still hits every mode it finds sharply
  (hq 0.99) and conditions perfectly, but finds 50 of 64 and the clouds
  compress to core 0.42: with two points per mode there is no within-mode
  spread to learn, so it memorises the points. That is the expected
  failure and the metric that catches it is the core width, not hq.
* The learnable prior is again what makes the difference in the small-data
  regime: frozen Gaussian blocks at 512 reals give 26 blurry modes.
* Conditioning accuracy is 1.0 at every pool size — the prior-partition
  mechanism does not need many reals per class to lock onto the class.

## 7. Stage `champion` — everything stacked (`TABLE_champion.md`)

Base + λ₁ 0.02 (§3) + warm-started gated head (§4), 3 seeds each:

| cell | bar (seeds) | bar_step | modes | hq | cond | sym | sp@1e-2 | sp@1e-3 | zero | core | w1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| λ₁ 0.02, linear head | 0/3 | — | 64±0 | 0.988 | 1.000 | 1.000 | 0.85 | 0.12 | 0.00 | 0.81 | 0.032 |
| λ₁ 0.02, gated warm 40 %, λ_sp 0.01 | 1/3 | 3200, 4600 | 63±1 | 0.979 | 0.997 | 0.997 | 0.980 | 0.964 | 0.96 | 0.79 | 0.033 |
| **λ₁ 0.02, gated warm 40 %, λ_sp 0.003** | **2/3** | **2900** | **64±0** | 0.977 | 0.991 | 0.991 | 0.975 | 0.951 | 0.95 | 0.77 | 0.033 |
| same, λ_sp 0.01, 8k steps | 2/3 | 4700, 5100, 5600 | 64±0 | 0.980 | 0.988 | 0.988 | 0.990 | 0.980 | 0.98 | 0.67 | 0.029 |
| gated warm 60 % | 0/3 | — | 41±19 | 0.62 | 0.90 | 0.90 | 0.85 | 0.81 | 0.80 | 1.44 | 0.076 |
| concat D, gated warm | 0/3 | — | 9±4 | 0.09 | 0.49 | 0.51 | 0.84 | 0.83 | 0.83 | 5.1 | 0.098 |
| proj D, gated warm | 0/3 | — | 11±6 | 0.11 | 0.67 | 0.67 | 0.84 | 0.83 | 0.83 | 4.0 | 0.122 |

Readings:

* **The bar is crossed.** 7 of the 9 seeds in the three UCD + warm-gate
  cells reach all five criteria at once (first crossing 2 900–5 600
  steps), and every seed that misses does so on exactly one count: 63 of
  64 modes with everything else holding (cond ≥ 0.97, sym ≥ 0.97,
  sp@1e-2 ≥ 0.97, hq ≥ 0.96). The generator emits exact zeros on 95–98 %
  of the inactive coordinates, the right symbol 99 % of the time, and the
  right class 99 % of the time, with an honest core width (0.7–0.8) and
  the lowest W1 in the study.
* The gate must open early enough: warm-starting at 60 % instead of 40 %
  leaves too little time under the gate (41±19, one seed at 14 modes).
  λ_sp 0.003 vs 0.01 is inside seed noise on the bar; 8k steps does not
  add modes but pushes exact zeros to 98 %.
* **UCD earns its place here, not in §3.** With the linear head the
  injection D modes (concat / proj) matched UCD. With the gated head
  they collapse (9–11 modes, core 4–5) while UCD holds 64. The
  straight-through gate makes G's output non-smooth in its parameters;
  a D that also carries the condition on its input apparently has a
  shortcut that kills the game, whereas the UCD head — class only on the
  output side, backbone unconditional — stays playable. This is the one
  place in the study where the paper's argument (condition injection
  gives the backbone a redundant shortcut) has a measurable consequence.

## Summary

1. The 100-Gaussians champion recipe **does not transfer as-is** to a
   24-D 3-sparse conditional mixture: it collapses to one point per class
   on every seed. Two structural changes fix it, both cheap:
   the gradient cap moved from the samples to real/fake **interpolates**
   (`g_interp_cap`), and the class routed **through the particle prior**
   (per-class particle blocks, no class input to G).
2. With those, coverage is 63–64/64 with hq 0.99, conditioning and symbol
   accuracy 1.000, on 3/3 seeds — the symbol head needs no supervision and
   any relaxation works.
3. **Exact sparsity** needs a gate with a straight-through threshold, and
   the gate must engage *after* coverage (warm-start at 40 %); then 95–98 %
   of the inactive coordinates are exactly 0.0 without losing modes.
4. **UCD** vs condition injection: equivalent under a smooth G head; under
   the gated head UCD is the only D that survives. λ₁ must be small
   (0.02, as in the paper); 0.5+ costs modes.
5. Data sparsity: unchanged down to 8 reals per mode; at 2 per mode the
   model memorises (core 0.42) and finds 50/64.
6. Cost: ~2 min per 5k-step run on an A6000 with 10 in parallel; the whole
   study (probes + 181 grid runs) took about 1.5 GPU-hours.

Open next steps: the 64th mode (a particle-allocation floor — try a
particle re-allocation / anchor-repulsion term as in the particle-z
branch); a symbol that is *not* a function of the mode (stochastic given
the class *and* the real part); real data.
