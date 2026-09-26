# Toy train-speed search (opencode)

Optimize wall-clock training speed for the toy example by optimizing the
winning config and how it works. Implementation search only: the trained
result must stay numerically the same game, not a cheaper different game.

## Winning config (frozen starting point)

`configs/toy100/constraints_simple_regularization.json`
(SHA-256 `4af9863a319378b362bfb925b161d9ae8b8b07c9ecf1a452bb645570e04b99b7`),
the simpler22 shared recipe: **22/22** frozen toys in one fresh
production-runner replay. See `reports/toy100/simpler22/README.md`.

| Setting | Value |
| --- | --- |
| Objective / optimizer | Logistic relativistic pairing, ordinary Adam, β=(0, .999) |
| Learning rates | G .00425, D .00425, particles .0085 (×2 prior mult) |
| Discriminator regularizer | `b_cap`, κ=1, coefficient 1, exact autograd **every** update |
| Particle regularizer | Disabled (`prior_reg=0`) |
| Networks | Width 128, depth 3, Fourier 3; batch 2048; 7,000 updates |
| Schedules | Network cosine from 60% of `min(host_budget,1600)`, floor .01; particle cosine from 60% of host budget, floor .05 |
| Noise | Input σ=.5 → 0 over first 10%; output σ rises to .029 over first 20%, then held |
| Init / resources | Identity 2D affine G; 20,000 particles uniform on [−5,5]²; seed 1234 |
| EMA | .995, diagnostic only, never supplies a PASS |

## How it works (what the speed search is optimizing)

Per training update: sample an unlabelled real batch → forward G (affine on
native toys) from sampled particles → score real/fake with the Fourier MLP
critic → relativistic logistic G/D losses → `b_cap` penalty
(`0.5 × (mean_real relu(‖∇D‖₂−1)² + mean_fake relu(‖∇D‖₂−1)²)`, exact double
backward, every step) → three Adam updates (G, D, particles) → periodic
evaluation (every 250 steps: 20k-sample gate draws), snapshots, logs, renders.
Read `benchmarks/toy100/train.py`, `particlegan/training.py`,
`particlegan/grad_regularizers.py`, `particlegan/recipes.py`.

Known cost centers, in expected order: (1) exact `b_cap` double backward every
D update; (2) 20k-sample evaluations + snapshots every 250 steps plus GIF
rendering; (3) three Adam updates over G/D/20k-particle table at batch 2048;
(4) real-batch sampling + input/output noise generation on one CPU thread.
Optimize the implementation of these steps, not the game: same losses, same
update counts, same seeds, same thresholds.

## Allowed vs forbidden

Allowed: lazy/batched exact `b_cap` with rescaled weight **declared**
(`reg_every` × coefficient, same optimizer step, no separate phase);
fused/foreach Adam; reduced eval/snapshot/log frequency for *timed* runs with
the full gate still passing on the *qualifying* run; caching/reuse that keeps
gradients intact; compile/vectorization/threading discipline under the pinned
profile below; I/O and render avoidance during timing.
Forbidden: changing losses, κ, coefficients (except declared lazy rescaling),
learning rates, schedules, noise, batch size, architectures, seeds, budgets,
thresholds, or gate criteria; finite-difference `b_cap` (experimental, known
quality cost — see `reports/cifar-ddgan/speed/READOUT.md`); seed sweeps;
metric feedback or target statistics in training; image-only evidence.

## Measurement protocol (pinned)

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2
export ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
python -u -m benchmarks.toy_suite run \
  --config configs/toy100/constraints_simple_regularization.json \
  --output artifacts/toy-suite/<candidate> 2>&1 | tee <candidate>.live.log
python -m benchmarks.toy_suite regrade --output artifacts/toy-suite/<candidate>
```

Primary metric: `train_seconds` wall clock for the full 22-toy gate on the
pinned CPU profile above, one thread, fixed seed 1234. Report the
train/eval/IO split where the runner records it. Secondary: per-problem
`total_seconds` for grid100/rotated100/staggered100 and `samples/s` where
meaningful. A timing claim without a full-gate PASS at frozen thresholds is
not a winner. Baseline (checked-in evidence, same profile):

| Problem | Live result | CPU elapsed |
| --- | --- | ---: |
| grid100 | 100 modes | 98.8 s |
| rotated100 | 100 modes | 99.9 s |
| staggered100 | 100 modes | 176.1 s |
| toy100 ×3 incl. eval + GIF | 3/3 PASS | 374.8 s |

## Agent contract (opencode lanes)

- Start from the frozen config above; declare every implementation change and
  its added/removed forward/backward work per update.
- Distinguish equal-step results from equal-compute results.
- Fixed seeds, no seed sweeps. No gate weakening.
- Tail-friendly logs (`tee *.live.log`), small focused diffs, concise
  `result.md` per attempt with exact replay commands.
- Leaderboard over images: update `reports/toy100/train-speed/LEADERBOARD.md`
  on this branch — append rows, never rewrite history; promote only on a full
  22/22 regrade PASS that is faster than the current best.
- Rank acquisition of speed first, then sustained quality (identical gates),
  with runtime and changed lines visible.
