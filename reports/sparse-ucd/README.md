# Sparse conditional mixed-output study (UCD) — protocol

Companion to the design note
[`../sparse-conditional-with-categorical-and-real-output.md`](../sparse-conditional-with-categorical-and-real-output.md).
This file is the *how*; `FINDINGS.md` next to it is the *what happened*.

## The problem in one paragraph

64 modes in R^24, each exactly 3-sparse (active dims take values in
{-2, -1, 1, 2} plus N(0, 0.05²); inactive dims are **exactly 0**), grouped
into 8 conditioning classes of 8 modes each, and each mode emits one
categorical symbol (v1: symbol = class; `symbol_map: split` gives each class a
50/50 pair of symbols). The generator must, given a class, produce a real
vector *and* a symbol that are jointly right. Three axes, three independent
ways to fail: smear (mass on inactive dims), conditioning collapse (class
ignored, or one point per class), symbol drift (symbol not glued to the real
part).

## Conditioning: UCD

The default discriminator is the *Unconditional Discriminator* of
arXiv:2510.00624 (Xia et al.): D never sees the class; it outputs one logit
per class, the adversarial logit for a sample of class `c` is `d(x)[c]`, and D
additionally minimises `λ₁ · (CE(d(x_real), c) + CE(d(x_fake), c))`. G's loss
is the ordinary (here: relativistic-pairing logistic) loss on `d(G(z,c))[c]`.
Controls: `scalar` (no class anywhere in D — the floor), `concat` (one-hot
class concatenated to D's input, the injection UCD argues against), `proj`
(projection discriminator).

Everything the class-conditional GAN literature does with the *generator* is
also a switch: class enters G as an embedding (`emb_dim > 0`) and/or through
the prior itself (`prior_partition: class` gives every class its own block of
particles; with `emb_dim: 0` the class reaches G **only** through which
particles it may draw — the ParticleGAN-native way to condition).

## Files

| path | role |
|---|---|
| `lib/sparse_toy.py` | problem instance (seeded), samplers, nearest-mode assignment |
| `lib/sparse_models.py` | G (linear / gated / top-k real head; gumbel_st / gumbel_soft / st_argmax / soft symbol head), D (scalar / concat / proj / ucd), critic adapters |
| `lib/sparse_metrics.py` | the metrics below and the convergence bar |
| `experiments/train_sparse.py` | one run from a YAML config; unknown keys are rejected |
| `experiments/gen_sparse_configs.py` | stages → `configs/sparse/<stage>/*.yaml` |
| `experiments/run_grid.py` | parallel runner (shared with the regularizer study) |
| `experiments/analyze_sparse.py` | `results/sparse/TABLE_<stage>.md` + curve plots |
| `experiments/sparse_pipeline.sh` | gen → grid → analyze per stage, **one log** |

## Run / follow

```bash
# whole study on GPU 1, 6 concurrent runs (each run is ~1-3 min)
GPUS=1 WORKERS=6 experiments/sparse_pipeline.sh
# one stage, with a base recipe applied to every config of the stage
GPUS=1 BASE="arm=g_interp_cap,coeff=1.0" experiments/sparse_pipeline.sh sparse
# follow along
tail -f results/sparse/PIPELINE.log
```

Every run writes `metrics.jsonl` (one row per eval, EMA read-out),
`summary.json`, `heatmap.png` (mean |x| per class, real vs fake, and their
log-ratio — smear shows as warm cells off the support), `confusion.png`
(requested class × emitted symbol), `pca.png` (per-class scatter on the real
data's top-2 PCs), `final_samples.npz`, `ckpt.pt`.

## Metrics (all on 8 192 EMA samples per eval, 20 000 at the end, class uniform)

| column | definition | healthy | what a bad value means |
|---|---|---|---|
| `modes` / `hq` | nearest of 64 centers over all 24 dims; hq = within 3σ√k = 0.26 | 64 / ≥ 0.9 | modes missing → transport; hq low with modes full → blur or smear |
| `mode_recall`, `hist_kl` | nearest-center mass balance | 1.0 / ≈ 0 | starved modes / collapse |
| `cond` (`cond_acc`) | P(nearest mode's class == requested class) | ≥ 0.95 | ≈ 1/8 → class ignored |
| `sep` (`cond_sep_ratio`) | mean pairwise sliced-W1 between per-class fake clouds ÷ same on reals | ≈ 1 | ≈ 0 → all classes emit one cloud; ≫ 1 → per-class point collapse |
| `cond_var_ratio` | between-class variance / total variance of x | ≈ real value | 0 → conditioning collapse |
| `sym` (`sym_acc_mode`) | emitted symbol == symbol of the mode the real part landed in | ≥ 0.95 | symbol not glued to the reals |
| `sym_acc_cond` | emitted symbol allowed under p(s \| c) | ≥ 0.98 | symbol ignores class |
| `symKL` (`sym_cond_kl`) | mean_c KL(p(s\|c) ‖ emitted) | ≈ 0 | wrong symbol *distribution* (matters under `split`) |
| `joint` | cond ∧ sym | ≥ 0.9 | parts learned independently |
| `sp@1e-2`, `sp@1e-3` | P(\|x_j\| < ε) over truly-inactive dims | ≥ 0.99 | 0.5–0.9 at 1e-2 → smear, not sparsity |
| `rec` (`sparse_rec`) | P(\|x_j\| > 10σ) over truly-active dims | ≥ 0.95 | gate over-pruned |
| `zero` (`exact_zero_frac`) | inactive coordinates that are exactly 0.0 | 1.0 (gated) / 0 (linear) | — |
| `smear` | mean \|x_j\| over inactive dims | ≈ 0 | the v0 failure in one number |
| `active_bias` | \|mean x_j − μ_j\| over active dims | ≤ σ | values placed wrong |
| `core` (`core_ratio`) | median-centred core width on active dims ÷ σ (chi-median inverted) | 0.8–1.2 | < 0.8 compressed, > 2 blurry |
| `w1` | sliced W1, fake vs real, all dims | small | — |
| `ucd r/f` | D's argmax class == true class on reals / requested class on fakes (UCD's Nash probe) | high | low on fakes → G's samples are not class-typical |
| `ppur` | particle class purity (shared table only) | — | 1.0 = particles specialised per class |

**Convergence bar** (per run, all must hold at the end): `modes == 64`,
`hq ≥ 0.9`, `cond ≥ 0.95`, `sym ≥ 0.95`, `sp@1e-2 ≥ 0.95`. `bar_step` is the
first eval at which all held. Per-axis bars are logged separately so a run can
be read as "coverage yes, sparsity no".

## Stages

See `experiments/gen_sparse_configs.py`; each stage is one question, 3 seeds
per cell, 5 000 steps (≈ 1 min per run on an A6000, ~6 in parallel).
