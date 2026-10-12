# Round 2: smaller constant global updates

The concluded [round 1 readout](round1-readout.md) found that every constant policy passes the clock probe, but none passes Ring16 density. Saved observations show sharp late acquisition bursts. This separately registered final round tests whether a smaller constant update magnitude reduces those bursts enough to pass the original gates.

Each of three one-point schema2 searches changes only `Recipe.lr`, from 0.0085 to 0.00425, on its exact phase1 configuration parent:

| Constant parent | Reason to retain this point | Search |
| --- | --- | --- |
| beta2=0.99, gamma=0.1 | Preserved two-pole transport margin; late Ring16 quality dip | `r1r2-tier1-half-rate-b99-g01-v1` |
| beta2=0.99, gamma=0.3 | Best observed Ring16 component covariance error, 1.568014, still above 0.85; late mode collapse | `r1r2-tier1-half-rate-b99-g03-v1` |
| beta2=0.9, gamma=0.1 | Passing word endpoint; Gaussian acquisition failure makes the global tradeoff explicit | `r1r2-tier1-half-rate-b90-g01-v1` |

All other resolved recipe fields remain unchanged. Each child uses one whole global configuration across the six required Tier1 tasks and its separate clock probe. Native Adam, learned prior, initial prior distribution/capacity, direct-particle optimizer moments, penalty coefficient, zero prior regularization, zero noise and zero EMA remain fixed. No schedules return. Task architectures, target/data laws, seen batch sequences, initializer, protocol seed0, sampling, update budgets, evaluation cadence and numerical gates retain the round1 identities.

The fourth role is the exact cached scheduled baseline under a new READY study. Its original task compatibility keys, source, runtime and candidate revision match round1, so the shared queue attaches its seven certified attempts without new training. Its numerical study prediction is terminal Ring16 `component_covariance_error>0.85`, with falsifier `<=0.85`; these signatures are distinct from full scientific qualification. Each constant child's parent evidence also remains saved under its exact original identity. No scientific baseline or parent rerun is authorized.

The scientific source digest remains `25b483a604e58ca6a6e4b949896de88c6309bce7d0ef8267799a084dc8843507`. The project interpreter and runtime are identical to round1. Preparation freezes the [manifest](phase2.json), new declarations, source/runtime/task projections, [first registration](registration.json), [first numerical publication](publication/results.json) and this rationale before admission.

Round2 reserves 10,080 seconds across four roles, retaining 2,520 seconds per role. New scientific work is limited to three recipes. Across both rounds the declared full reservations total 27,720 seconds and charged work is capped at 5,400 seconds, including round1's 1,259.1354882339656 seconds. The aggregate recipe ceiling remains twelve. Each task keeps its full allowance; unavailable full reservations leave explicit unmeasured cells. The shared archive retains its 8 GiB cap and 5 GiB free-space floor. This is the final bounded round: no seed variations, retries, Tier2, automatic refinement or default adoption.

Reused attempts retain their original paid charge and request owner in the collector. They add zero cost to round2; the aggregate cost counts each charged attempt once. Certified wall time is checked against the original charge even on reuse. The numerical readout must distinguish an improved endpoint or smaller burst from all six ordinary gates plus the clock probe passing. The current generated technique inventory remains the single leaderboard.

Run these commands from the checkout root using the exact project interpreter. Preparation launches no workers; admission requires all frozen inputs committed and an explicit execution SHA.

```sh
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-tier1-search/workflow.py prepare --manifest reports/forge/r1r2-tier1-search/phase2.json --registration reports/forge/r1r2-tier1-search/round2-registration.json
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-tier1-search/workflow.py run --manifest reports/forge/r1r2-tier1-search/phase2.json --registration reports/forge/r1r2-tier1-search/round2-registration.json --source-commit COMMITTED_SHA --submit --drain
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-tier1-search/workflow.py collect --manifest reports/forge/r1r2-tier1-search/phase2.json --registration reports/forge/r1r2-tier1-search/round2-registration.json --output reports/forge/r1r2-tier1-search/round2-publication --media
tail -f /mnt/ml7tb/ParticleGAN-forge/r1r2-tier1-search-20261012/round2-run.log
```

Collection validates saved results and exports actual-training GIFs without optimizer updates or sampling draws. Final readout and concise generated memory summarize the whole global comparison, including failures, costs and stopping decision.
