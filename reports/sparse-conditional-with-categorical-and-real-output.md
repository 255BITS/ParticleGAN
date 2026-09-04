# Sparse conditional toy with mixed real + categorical output: design note

Proposal for a synthetic follow-up to the 100-Gaussians recipe
(`examples/100gaussians.py`, `lib/toy_models.py`). Purely seeded RNG, no data
files, minutes on one GPU, judged by eye + a few numbers. Three axes, each its
own research question: sparsity, conditioning, mixed output.

## 1. Generative process

Defaults (all overridable flags; all RNG via an explicit `torch.Generator`):

| symbol | value | note |
|---|---|---|
| `d` | 24 | real dims |
| `k` | 3 | active dims per mode |
| `N` | 64 | modes |
| `C` | 8 | conditioning classes (8 modes each) |
| `K` | 8 | categorical symbols, deterministic in v1: `s(m) = c(m)` |
| `sigma` | 0.05 | noise on active dims only |
| `A` | {-2, -1, 1, 2} | active-mean alphabet |

Construction (seeded, once at startup):

1. For each mode `m`: sample support `S_m ⊂ {1..d}`, `|S_m| = k`, without
   replacement; sample active values uniform from `A`. Inactive dims are
   **exactly 0** — that is the point of the toy.
2. Assign class `c(m) = m mod C` (balanced by construction). Assign symbol
   `s(m) = c(m)` in v1 (1:1 map, so joint coherence is checkable by equality).
3. Sample: `m ~ Uniform(1..N)`; `x[S_m] = mu_m + sigma * N(0, I_k)`;
   `x[~S_m] = 0`; `y = onehot(s(m))`, optionally flipped with `p_flip = 0.02`
   in v2.

Why this shape: `d = 24, k = 3` keeps a `d × C` mean-`|x|` heatmap readable;
`N = 64` keeps nearest-center assignment cheap (`cdist` on 64 centers, same
pattern as `mode_coverage`); `sigma = 0.05` against `|mu| >= 1` separates the
"sparsity error" (mass on inactive dims) from ordinary blob width. Supports
should overlap partially across modes (shared dims, different values) so the
union is not trivially separable — check overlap stats at seed time and reseed
if any dim is used by zero modes.

Open: v2 stochastic symbol (`s | c` a 2-way split per class) to separate
"right distribution over symbols" from "right single symbol". Keep v1
deterministic until the metrics below pass.

## 2. Generator / discriminator I/O

Keep the surrounding recipe fixed: RpGAN logistic + `b_cap` (coeff 1.0),
Fourier-2 D, EMA eval, Adam β1=0, delayed cosine anneal, particle prior. Only
the heads change.

**G:** `(z, c) -> (x_hat ∈ R^d, l ∈ R^K)`.
- `z_dim = 8` (overcomplete vs. nothing in particular; 4 sufficed for 2-D
  data, take 8 for `d = 24`). Particle table sized `~32 × N` (≈2000).
- Class enters as a learned embedding `E_c ∈ R^{C×8}`, concatenated with `z`.
  Baseline v0 concatenates; v1 also tries projection-style scaling
  (`h *= (1 + W E_c)`) if v0 ignores the condition (see §4).
- Real head: v0 plain linear (expected to smear: small-but-nonzero on
  inactive dims — the baseline failure). v1 gated: `x = m ⊙ v` with mask
  `m = STE(sigmoid(logits_m))` (threshold 0.5 forward, sigmoid backward) plus
  an L0-ish penalty `mean(sigmoid(logits_m))`. Exact zeros come from the
  forward threshold, not from shrinkage.
- Categorical head: linear → logits `l`. Train-time: Gumbel-softmax
  (τ 1.0 → 0.1 anneal) or ST-argmax; eval-time: hard argmax. D receives the
  symbol as `stopgrad(embedding(y_hard)) + onehot(y_hard)` (embedding
  `E_s ∈ R^{K×8}` learned jointly).

**D:** `(x, y) -> (score, class_logits)`.
- Real path: Fourier-2 on `x` only (symbol path stays un-Fouriered), same MLP
  sizes as `SimpleMLPDiscriminator` (128 × 3). Concat `[fourier(x), E_s[y]]`.
- Class output two ways (pick one per run, compare): (a) projection head
  `score += f(x,y)^T E_c[c]`; (b) auxiliary classifier head with CE loss
  (UCD-style). v0: auxiliary classifier — simpler to instrument
  (per-class accuracy falls out for free).

**Losses** (additive, one dial per axis):

| term | form | dial |
|---|---|---|
| `L_gan` | RpGAN logistic on joint `(x, y)` score, `b_cap` on reals+fakes | as now |
| `L_cls` | CE(D class head, true `c`) on reals; G pays CE(D class head on fakes, input `c`) | `λ_cls ≈ 1.0` |
| `L_sym` | CE(G logits `l`, true `s(m)`) as a supervised anchor (v0 on; ablate to 0 to test pure adversarial symbol learning) | `λ_sym ∈ {0, 1.0}` |
| `L_sp` | gate sparsity: `mean(sigmoid(logits_m))`, or L1 on `x_hat[~S]` in v0 | `λ_sp` log-sweep |

Open: whether `L_sym > 0` trivializes the mixed-output question (it does, by
design — it is a curriculum knob: converge with it, then anneal to 0 and see
if the joint holds); whether the cap penalty should see the symbol embedding
norm (probably yes — penalize ∇ over the concatenated joint input).

## 3. What to plot (eye test)

1. **Mean-|x| heatmap, `d × C`** (fake, conditioned per class, vs. real side
   by side). Healthy: same `k`-sparse column pattern. Smear failure: columns
   light up everywhere at 0.05–0.2.
2. **Symbol confusion matrix, `K × C`** (rows requested class, columns emitted
   symbol). Healthy v1: diagonal. Conditioning-collapse signature: identical
   rows regardless of requested `c`.
3. **Per-class 2-D scatter** on 2 hand-picked active dims + a PCA panel over
   the active union. Cheap; catches "right sparsity, wrong values".

## 4. Metrics table

All on `n_eval = 20000` EMA samples, half conditioned uniformly over `C`.
`eps = 1e-3` absolute (≈ 2% of `sigma`, 0.05% of min `|mu|`) as the
near-zero threshold; report `eps ∈ {1e-3, 1e-2}` to show threshold-robustness.

| metric | definition | healthy | failure signature |
|---|---|---|---|
| `modes` / `hq` | as `mode_coverage`, nearest of 64 centers, 3σ ball, active dims only | 64/64, hq ≥ 0.9 | recall < 1 → transport failure (same reading as now) |
| `sparse_prec` | frac of `\|x̂_j\| < eps` over truly-inactive dims | ≥ 0.99 at `1e-3` | 0.5–0.9 at `1e-3` but ~1.0 at `1e-1` → smear, not sparsity (v0 expected) |
| `sparse_rec` | frac of `\|x̂_j\| > 10σ` over truly-active dims | ≥ 0.95 | low → gate over-pruned; check with `λ_sp` sweep |
| `active_bias` | `mean \|mean(x̂_j) − mu\|` over active dims, assigned modes | ≤ σ | large with good `hq` → values right, placement wrong (support mismatch) |
| `cls_acc` | D aux-head accuracy on fakes (input `c` vs. predicted) | ≥ 0.95 | ~1/C → conditioning ignored; all mass one class → collapse |
| `cond_collapse` | mean pairwise symmetric KL (or sliced-W1 on reals) between per-class fake clouds; 0 = identical | well above 0, near real-data value | ≈ 0 while marginals look fine → conditioning collapse (the failure this axis exists to catch) |
| `sym_acc` | frac `argmax(l) == s(m)` (v1: `== c`) | ≥ 0.98 | high `sym_acc` + low `cls_acc` → symbol emitted but reals ignore class |
| `joint_acc` | frac (nearest-center mode `m̂` satisfies `c(m̂) == c_in` AND `s(m̂) == s_out`) | ≥ 0.9 | marginals pass, joint fails → coherent parts learned independently, not jointly |
| `hist_kl` / `hist_js` | as `mode_recall_and_hists`, over 64 modes | KL ≈ 0 | KL → log 64 → single-mode collapse (same reading as now) |
| `core_ratio` | `per_mode_core_ratio`, active dims | 0.8–1.2 | < 0.8 compressed, > 2 blurry (same guard as LEADERBOARD) |

## 5. Staged plan and open questions

1. **v0 baseline:** plain G head + `λ_sym = 1`, `λ_sp = 0`. Expect: symbols
   right, sparsity smeared, conditioning possibly working. Confirms the
   metrics move independently.
2. **v1 gated:** ST mask + `λ_sp` sweep. Question: does the gate find exact
   zeros without killing `sparse_rec`? Is L0-penalty needed at all, or does
   the adversarial signal suffice once the mechanism exists?
3. **Conditioning stress:** `λ_cls → 0` ablation; projection vs. aux-head.
   Question: which D design actually forces G to use `c`? Does the particle
   prior specialize per class on its own (cluster the table by requested `c`
   and look)?
4. **Joint stress:** anneal `λ_sym → 0`. Question: does the symbol stay glued
   to the reals, or drift? This is the mixed-output result.
5. Cost: same order as 100-Gaussians (batch 256, 5–10k steps). Failure budget:
   if `modes < 64` at defaults, shrink to `N = 32, d = 16` before touching the
   loss — coverage first, per `docs/convergence-tips.md` §1.

Non-goals: no new penalties, no optimizer changes, no real data. If the toy
needs more than the current recipe plus heads, that is itself the finding.
