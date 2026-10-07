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

The completed [batch 128 versus 512 comparison](../tier1-batch-size/README.md)
adds 10,400 GPU updates, reusing Gaussian 128 and the ring 128 prefix. Neither
Gaussian acquires five consecutive passes through 4,000; batch 512 final KS
.12660 is worse than batch 128 .09594. Ring 128 retains all 144/144 post-acquisition
checks, while ring 512 acquires earlier but fails one of 144 hold checks. Keep
the adopted batch 128 ring settings. Larger batches also increase the frequency
of normalized prior-row motion, so this does not isolate gradient noise alone.

The user's preferred next hypothesis is **extrapolation from the past**, after
compaction. [Gidel et al. §3.3](https://arxiv.org/pdf/1802.10551) reuse the previous
gradient for a lookahead, evaluate a fresh gradient there, and correct from the
original state. A prospective BCAP augmentation should explicitly define a
joint G/D/prior lookahead, checkpoint cached gradients and optimizer histories,
and retain batch 128, constant rates, seed 0 and the matched public initialization.
Its interaction with unit-normalized prior updates is unknown. Score live-model
acquisition, continued stationary hold and a separately declared target-shift
response; averaged-model or best-checkpoint success cannot stand in for these.
This is a preference and implementation question, not an executed experiment or
a frozen budget. Declare the supported mechanism and finite study before spend.

Local evidence:

- Parent raw: `runs/api/tier1-prior-smoke-v1/mog100-n256/gaussian1d_acquisition/`.
- Continued raw: `runs/api/tier1-prior-duration-v1/mog100-n256-gaussian1d_acquisition/`.
- Batch comparison raw: `runs/api/tier1-batch-size-v1/`.
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

PR316 is now merged into develop `d91c8d867b06435e79c25f65cf46754e8eabbe69`.
The user authorized a new branch, `research/bcap-extrapolation`, and the bounded
[extrapolation-from-the-past investigation](../bcap-past-extrapolation/README.md).
Its executed source is `f987656a49c7477322c058d98e6e1c180518fa36`; all seven new
GPU trials complete22000 updates for279.351 loop seconds, with no scientific
retry. The shared public Recipe/GANTrainer capability is implemented and tested;
the paper's equations20–21 operate on the selected recipe's normalized field.

It does not fix Gaussian: acquisition FAIL, hold3/72, longest passing streak2,
final KS.26799. Simultaneous control passes0/96 checks. The adopted alternating
ring remains the only combined acquisition/strict-hold pass; extrapolation's
strong late ring fit misses1600 acquisition and retains115/144 hold checks.
All three Gaussian mean2→3 continuations pass0/48 full shift checks. Extrapolation
responds to the new mean but ends narrow, KS.13998. No history reset, annealing,
average or best-checkpoint substitution is used.

The final scalar prior cache has97 moving rows with nearly unit direction norms;
its implied average row step remains.0299996. The saved-state shape improvement
does not meet target-law fidelity or isolate a cause. Stop this exact revision
as a scalar repair. The next hypothesis is response magnitude near fit at fixed
nominal rates, possibly a magnitude-sensitive prior or explicit frozen-prior role
control. No follow-up experiment or budget has been frozen or executed. Preserve
the ordinary sigma.025 Gaussian task and the original qualification board.

New local raw: `/home/martyn/dev/ParticleGAN-bcap-extrapolation/runs/api/bcap-past-extrapolation-v1/`.
Archive SHA256: `10d61ee47ea4339f97060db2514cb086e32e63e50f1da6d2e28da18b37fe5b09`.
