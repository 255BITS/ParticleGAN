# Independent CB64-RA acceptance protocol, frozen before candidate runs

The pre-outcome CPU-AMENDMENT.md supersedes GPU/device and canonical fixture
identity clauses below. Canonical GPU acceptance is unavailable; unchanged
full-budget CPU diagnostic scorer verdicts are reported separately.

Candidate: the actual pkg-CB64-RA and configs/overrides-CB64-RA.json after the
integration owner issues READY. No candidate import or run occurs before its
source/config receipt is copied into this directory. Reference E22, all hosts,
scorers, previous data, evaluator and models remain read only. No package repair,
threshold change, seed sweep, reduced native budget or outcome-guided tuning.

## Learned nonlinear and real-image comparison

Run E22 and CB64-RA once each on the previous nonlinear 25-Gaussian MLP and
actual MNIST convolutional GAN. Reuse the previous validation's models_metrics.py,
toy/image real streams, MNIST files and trained real-only evaluator read only:
/ml2/hypergan/gan-attempts/scaling-portability-20260929/validation.

Same architecture, initial parameters, stream seed 314159, N 1024, z_dim 128,
batch 128, two real batches per update, exactly 2000 updates per run. Assert
initial generator/critic/prior hashes against the previous E22 receipts.
Only N, latent dimension and batch size override each frozen config. Architecture,
loss, serving, noise and controller state come from each package/config.

Checkpoints at 0,100,250,500,750,1000,1250,1500,1750,2000. Toy metrics are unchanged:
8192 samples, nearest-mode precision within 3σ, supported mode coverage (≥1%
of all draws per mode), mass TV including unsupported bucket, centre distance
and covered-centroid RMS, plus clean particle-centre metrics. The original
useful-quality gate remains precision≥.9, all 25 covered, massTV≤.1. Report
that this is an independent toy gate, not a native/frozen-harness PASS.

MNIST: same real-only evaluator, test accuracy 98.08%, fixed heldout halves,
4096 generated samples, class mass TV, confident-class coverage/fraction,
mean classifier confidence and pixel clipping. Use the already declared raw
64-dimensional learned embedding and 39 training-active standardized coordinates
from the previous evaluator diagnostic. The active mask depends only on 5000
real training images (std > 1e-6×maximum training std); the inactive-coordinate
normalization artifact is avoided. Report Fréchet distance and k 5 manifold
precision/recall against the same heldout reference and real-vs-real controls.
No Inception FID claim. There was no numerical image-quality acceptance gate
in the original experiment; compare the quality/diversity Pareto tradeoff.

Compare at 2000 updates and the latest checkpoint within the minimum final
training-only elapsed time; report unused time due to checkpoint granularity.
Timers include complete CUDA-synchronized trainer updates/controllers, exclude
evaluation/checkpoint I/O; report whole-run seconds, throughput, peak allocated
and reserved GPU memory separately. Record all ordinary reaction, isolation,
parent-selection and row-evidence diagnostics. The candidate must execute its
ordinary cell reaction path and make at least one ordinary BD move on a declared
training or frozen task; support-only/no-op results do not validate that path.
The reference row window 50 caps effective n 99<3×128=384; report its inactive
gate without editing it. Capture checkpoint continuation on saved 1000-step
states with the same next 10 real updates, separately from the quality budgets.

## Frozen acceptance suites

Use documented direct screen.py API, with owned output directories. Do not
submit to or alter the global pool. Original frozen screen:
/ml2/hypergan/lrfree-20260926/harness/screen.py.
Its source matches harness-absence/screen.py, but original task/host files
avoid the larger-table unequal-mass and withholding modifications.

Candidate options: eval_output_noise=true, save_final_state=true,
strict_streams=true, diagnostics=true. Keep every host's initialization,
task/stream seed, observation schedule, update budget and gate unchanged.
Unset ABSENT/ABSENT_START/ABSENT_END and LRFREE_NATIVE_TEST_STEPS.

**13 portability tasks:** mode_hold (1200 updates, N 12), four image-shaped
hosts (600,N 32), vector_two_broad/vector_unequal_mass/vector_unequal_width/
vector_anisotropic/vector_overlap (1200,N 256), vector_spiral (1600,N 256),
ring_shift (4600,N 20000), stationary (7500,N 20000). These image-shaped host
fixtures are frozen portability gates, distinct from the actual MNIST run.
Pass iff every frozen result.json says PASS; errors and failures count
separately and no historical 13/13 is inherited by CB64-RA.

**3 native tasks:** grid100,rotated100,staggered100, each 7000 updates,
seed 1234, N 20000/z_dim 2/batch 2048, frozen QR initialization and original host.
No reduced-step plumbing run substitutes for acceptance. Use all five terminal
20k quality clouds plus independent 100k holdout, official live noisy scorer.
Report coverage and accuracy status, terminal-check booleans, holdout precision,
centre/trace/KS/eigenvalue metrics, margins, live versus EMA and branch counts.
Pass iff all three official frozen native verdicts PASS with valid evidence.
S 1b 14k, S 4rate variants, broad constant sensitivity and full admissibility
ceremony are outside this request; do not call this a full S 1–S 6 solution.

Archived E22 native confirmations are read-only comparators; their hashes,
config, metrics and runtimes are archived. Their timings are noncontemporary.
No new E22 native run is needed. The candidate 13-task results may compare to
archived E 19a/E22 evidence only when source/task/options identities are recorded.

## Resources and receipts

Validation alone owns physicalGPU 0. Serialize the four learned-model runs,
13 portability tasks and 3 native tasks. At most 2 CPU threads (harness uses 1),
CUDA memory fraction .2 (~9.8GB), no GPU 1 use or other-process signals.
All commands, protocol/config/package/host hashes, logs, per-task results,
controller activity, checkpoints and an honest report/leaderboard live here.
API incompatibility/admissibility rejection is recorded separately from quality;
continue other authorized runs without editing any gate or candidate.
