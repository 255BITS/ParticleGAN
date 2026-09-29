# CB64-RA independent static validation protocol

Written before candidate comparisons, 2026-09-29. CPU only, two Torch/BLAS
threads. This lane owns only this directory. No seed sweeps, failed-gate tuning,
oracle controller inputs, gate overrides or forced ordinary-reaction markers.
Reference packages, prior fixtures/bundles and other agents' files are read only.

## Candidate and comparison contract

Wait for integration READY and its frozen package/config receipt. Import the
actual `pkg-CB64-RA/particlegan/feature_cells.py` implementation and
`configs/overrides-CB64-RA.json`; do not test the copied E22 stub or duplicate the
candidate. Record and verify those hashes before/after tests. Public proposed
API: `FeatureCellSnapshot.fit(real_features, generator=..., cells=64, rank=8,
chunk=256)`, `transform`, `assign`, `support`, `select_parents`, and
`FeatureCellBirthDeath.observe_real/maybe_apply/_move/diagnostics/state_dict`.
Only adapt argument names if the actual frozen API differs; record corrections.

Candidate constants are integration's fixed rank8/K64, four power/Lloyd passes,
chunk256, real-anchor parent policy, at most64 eligible parents per cell,
nearest four real-cell representatives and at most256 local parents per child,
uniform parent within the fixed feature ball/rank16 cap. Native candidate
sampling/fake/repair jitter is the integrated Gaussian .025/norm-cap .05 law.
Ordinary cell reaction uses its actual exact hypergeometric test/Bonferroni
Q/K and row actuation, not E22's kNN Beta law. Final frozen config/source is the
authority for implementation details; no validation-specific candidate knob.

E22 is the unchanged reference source/config. Both controllers receive exactly
the same initial latents, real stream, generator/critic and observable head
features. Candidate support scores differ from E22 by design; each backend
uses its own actual unmodified flags and Q=.05. We compare against identical
frozen acceptance thresholds, rather than feeding reference flags into the
candidate. No support oracle, intended mode, raw output distance, or target
mass enters a backend. Oracle evaluation occurs after an action is returned.

## Fixed families and scopes

1. **Cost family**, seed90229: unchanged `scaling_a/shared_toy.py`, scenarios
   nominal/highdim/rare_hole, N1024/2048/4096/8192. Frozen toy generator/head are
   wrapped as torch modules so actual package score-head extraction is used.
   Cost fixture latent bandwidth .025 and output noise .029 remain fixed.
2. **Geometry family**, seed20260929: unchanged `geometry_a/toy_family.py` and
   bundle.pt; linear/fold/fold_nuisance, dimensions2/16/64/128 at N2048, plus
   N1024/4096 at d128/fold_nuisance. Main features use the critic trained600steps.
   Separately report the already preregistered frozen-initialization critic
   control at the same sole seed, zero training updates. Model states stay fixed.

For every fixture/backend report three scopes:

- **Isolation mechanics**: actual backend support and parent selector, ordinary
  children empty, exact center cloning. This intentionally disables ordinary
  moves only to isolate parent selection. It is not full-reaction evidence.
- **Common jitter diagnostic**: reset original table/private streams, then four
  isolation rounds using actual backend support/parents plus the unchanged
  E22/DV12 half-nearest-nonidentical-latent cap and common row-indexed normal
  noise. Geometry bandwidth .0125; cost .025. This is the original comparative
  jitter gate and intentionally costs dense latent queries; report separately
  from the bounded candidate's native jitter.
- **Full integrated reaction**: reset original table/private streams and run
  four unforced actual `maybe_apply` reservoir turnovers with frozen networks.
  Both ordinary and isolation mechanisms execute exactly as integrated, with
  native backend jitter/output noise, actual `_move`, EMA/optimizer/history
  row actuation and native invalidation. Replay the same frozen N-row real
  reservoir at each turnover; fake pools are actual private-stream iid samples.
  Do not inject evidence, marker tensors or eligibility state. First-turnover
  controller wall time is the original per-evaluation cost comparison; report
  all four turnovers' timing/metrics and whether ordinary moves really happened.
  This bounded static replay is not independent fresh real data or GAN training.

Use actual config recipe settings, overriding only fixture structural N/z_dim.
Reference full-reaction adapter supplies its unchanged prior/controller bandwidth
and the fixture's fixed .029 output noise. Source/config consumers and every
scope difference are recorded. Read-only profiling hooks may count network
forwards, cdist dimensions, matmul work and actual moves without changing
selection, random streams, copied latents, flags or backend evidence.

## Acceptance criteria: retain original gates

Cost efficacy at each nominal N requires detector recall>=.85, precision>=.95,
common-supported FPR<=.005, original rare-particle retention>=.98, repair
recall>=.85, repair precision>=.95, valid parent support>=.99, and post-repair
mass TV<=.05 and <=matched E22 TV+.01. The integrated candidate intentionally
changes the detector/reaction, so exact E22 flag/radius agreement is diagnostic,
not a requirement reserved for the earlier exact-tree alternative. Mandatory
highdim/rare_hole stresses use the same absolute efficacy gates for any claimed
portability pass. A no-op or matching a failed baseline cannot qualify.

Cost scaling claim additionally requires N8192 nominal first full-evaluation
speedup>=2x and log(time)/log(N) slope<=1.5 over all four nominal sizes. Include
all setup, projection, support, ordinary reaction, model/head forwards, parent
search, native jitter/actuation and invalidation. Count peak distance matrices,
process RSS, retained arrays and exact work. No hidden child-by-all-parent
matrix, N-by-N projection covariance, stale traversal or latent nearest-table
pass may be ignored in the candidate's bounded-cost claim.

Geometry detector receipt: unsupported recall>=.90, precision>=.95, legitimate
FPR<=.002 and zero falsely flagged supported rare rows. Exact repair also
requires conditional realized and actual-candidate-set expected intended-mode
validity>=.95, unsupported mass<=.005, rare mass/target>=.90, mass TV<=.01.
Common four-round jitter gate requires final unsupported mass<=.01, rare
ratio>=.90, mass TV<=.015 and repeated-write fraction among original planted
rows<=.20. Detector failures disable end-to-end qualification. Full reaction
gets the same final support/mass/churn gates, with its initial detector receipt
and all actual row actions reported. Ordinary actions alone cannot validate
isolation parent semantics; report their valid-parent and repair metrics.

Report support and intended-mode repair precision/recall across ALL original
unsupported rows, correct parents, full mode masses, rare retention, exact vs
native/common jitter, target/candidate sizes, inaccessible deficit, false moves,
ordinary/isolation move counts, timing/forward/work/memory scaling, code and
config receipts. Qualification must hold across the entire required fixed
family, not selected sizes or the separate feature-map diagnostic.

## Deliverables and commands

Save `run.log` (unbuffered, easy to tail), source/config/input hashes, JSON/CSV,
per-case controller traces, a concise leaderboard/report and artifact manifest.
Freeze protocol before comparisons and harness/source receipt before execution.
Any implementation/API correction is disclosed without changing thresholds.

Run from this directory:

`env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /tmp/pr38-default-env/bin/python -u run_validation.py > run.log 2>&1`
