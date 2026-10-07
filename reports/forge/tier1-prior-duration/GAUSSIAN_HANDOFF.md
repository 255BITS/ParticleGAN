# Gaussian handoff after ring16 repair

Continue on PR [#316](https://github.com/255BITS/ParticleGAN/pull/316), branch
`research/tier1-prior-smoke`, worktree
`/home/martyn/dev/ParticleGAN-tier1-prior-smoke`, based on develop `7183d65d`.
The user selected the passing ring conditions; see
[the adopted smoke](../ring16-smoke-v2/README.md). The user deferred Gaussian
research until after compaction. No additional Gaussian trial is authorized by
this handoff itself.

Read `EXPERIMENTATION.md` and compiled `reports/forge/EXPERIMENT_MEMORY.md`
before choosing a bounded question. Do not vary seeds, repeat an unchanged
study for a merge, or generate another leaderboard. Use seed 0, public
deterministic initialization, GPU training/sampling, isolated checkpointed RNG
streams, fixed numerical gates and one whole trainer configuration across tasks
when comparing trainers. Store raw logs/checkpoints locally and make logs tail-able.

Evidence already complete:

- [Prior grid](../tier1-prior-smoke/README.md): 18 CUDA cells, counts
  256/1024/4096 crossed with cloud and MoG sigma .025/.1. All original full
  gates fail. More particles do not repair acquisition within the fixed budget.
- [Duration study](README.md): exact saved checkpoints resumed, no prefix
  updates repeated. Gaussian with 256 particles and MoG sigma .1 extends from
  1,000 to 4,000 updates. Only three of 96 individual full checks pass; no five
  consecutive full passes. KS at 1,000/2,000/3,000/4,000 is
  .04297/.16982/.13952/.09594. Final mean error .27536 sigma and std ratio
  1.74796 fail; largest observed std ratio is 3.71290.
- The full gate remains mean error <=.2 sigma, std ratio [.8,1.2], KS <=.05,
  sample count >=4096 and finite fraction 1, with five terminal passes.
  A moments-only smoke can accept non-Gaussian impostors and cannot replace it.

Exact baseline: target N(2,.5²), z_dim 2, 256 learned uniform MoG locations,
sigma .1, initialization scale 1, no standardization, batch 128, MLP width
32/depth 2, critic Fourier 2. Selected candidate is
`bcap-dualnorm--7beb7378d81dc3be2c648438661e0376fe2805298232f5c2398be835ddaad6f9`.
Rates .012/.018/.03 for G/D/prior, zero momentum, both LR floors 1,
no training noise, no prior regularizer or EMA. The original task-bound recipe
horizon is 1,000; duration changes only the external cap to 4,000.
The separate standalone scalar K3P caller uses another initialization/RNG
cohort; do not silently substitute it for this BCAP baseline.

The scalar target is simple, but this is a coupled optimization of generator,
critic and learned prior. It reaches acceptable snapshots and then drifts;
that supports a stability hypothesis more than a need for more particles or
updates. Constant normalized update sizes are a plausible contributor, but
the existing evidence does not isolate optimizer, prior motion or critic as
the cause. Inspect saved moments, prior movement and critic behavior before
declaring a bounded force-response or role-isolation experiment. Any such change
is a new bounded trainer comparison, not a continuation of the unchanged recipe.

The user has now requested further investigation on PR #316. The
[saved-state diagnosis](../gaussian1d-diagnosis/README.md) verifies the dimensions
above against checkpoint tensors, correcting this handoff's earlier 4/64 typo.
Its same-network/unchanged-initial-prior affine capacity control passes 24/24
checks with zero training. Saved states show increasing skew and mostly
between-component output variance. The user requires a continuous learner and
rejects learning-rate annealing. Recommendations now cover a separately declared
frozen-prior diagnostic, force-response damping, spread restraint and position
springs; continued quality retention and adaptation must both be measured.
The current Gaussian task and all historical verdicts remain unchanged.

Local evidence:

- Parent raw: `runs/api/tier1-prior-smoke-v1/mog100-n256/gaussian1d_acquisition/`.
- Continued raw: `runs/api/tier1-prior-duration-v1/mog100-n256-gaussian1d_acquisition/`.
- Curves, saved outputs and named-stream checkpoints are present; use metrics
  before pictures. GIFs render actual saved training samples.
- Parent archive SHA256:
  `0ab6a34f0dbb27b7bd6647ef44918a1f022abc2732826c0d50190064d96ccd7d`.
- Duration archive SHA256:
  `18e339a2e78b73d95798b871914ac7a9301057e8b49a89c610f34dd7992d2480`.
- Scientific commits: `ec9be602b2b8d587b8c8c3f8bc10e98c96580dc1`
  (prior grid), `b4d1f95a074ffa64ac8f64e12aa0a049f836c0af` (continuation).
- Torch/CUDA training environment: `/usr/bin/python` (Python 3.14).
  Rendering environment: `/home/martyn/dev/ParticleGAN/.venv/bin/python`
  (Python 3.12 with Matplotlib/Pillow). Two RTX A6000s; studies used cuda:0.
