# Named KA2/K3P defaults study

This is one finite pair of whole configurations: `ka2` and `k3p`, each with the shared tuple `lr=0.006375`, `prior_lr_mult=1.0`, `d_lr_mult=1.0`. Each tuple stays unchanged across eight required public API questions. The two smoke prerequisites precede six quality cases; the first non-PASS stops that family. Unreached cells remain UNKNOWN in the full 16-cell denominator. A negative capacity outcome blocks its own family, while a fully supported sibling can proceed.

These are explicitly family-owned laws. The unchanged public factories use fast-only serving, no DV12 controller, no particle birth/death or continuous row evidence, AMSGrad false, and fixed output sigma warmed from zero to 0.029 over the first 20% of the full host horizon. Learning rates retain the named recipe schedules. `ParticlePrior` reads its stored rows directly; the serialized Recipe `standardize` flag is not applied to those particle reads. Image/vector primary observations omit training output noise; native primary observations include the fixed scheduled output noise, with clean diagnostics separate. This is a new sampling/formulation cohort, and supplies no Atlas/E22, historical positive, Forge MoG, default-adoption or speed credit.

`protocol.py` mirrors the original factory Recipe adaptation, including `total_steps=600` for images, `1200` for ordinary vectors, and `7000` for the three native hosts. These schedules stay fixed even when a software control uses a short execution prefix. The narrow Recipe-aware grade adapter uses the unchanged public retained-receipt verifier, numerical scorer, checkpoint-health checks and first-window reducer. It corrects the legacy search resolver's omission of host horizons; it patches no old modules, private globals, factory, numerical gate or training loop. All training remains in the public `api_run.main()`.

| Required order | Case | Tier | Full updates | Evaluation samples | Acquisition cap |
| --- | --- | --- | --- | --- | --- |
| 1 | `image-develop-img_intensity2-source-transpose12` | Smoke | 600 | 1,024 | 180s |
| 2 | `api-vector-two-broad` | Smoke | 1,200 | 4,096 | 180s |
| 3 | `api-grid100` | Quality | 7,000 | 20,000 | 2,100s |
| 4 | `api-rotated100` | Quality | 7,000 | 20,000 | 2,100s |
| 5 | `api-staggered100` | Quality | 7,000 | 20,000 | 2,100s |
| 6 | `api-vector-unequal-mass` | Quality | 1,200 | 4,096 | 180s |
| 7 | `api-vector-anisotropic` | Quality | 1,200 | 4,096 | 180s |
| 8 | `image-develop-img_bars4-source-transpose12` | Quality | 600 | 1,024 | 180s |

All eight original case hashes, numerical gates, full budgets, seed 24002, evaluation seed 34002, 24 post-update scoring observations, five terminal observations and nine actual-training GIF frames remain fixed. The added study grade requires the first five consecutive primary PASS observations, at least five later checks, and every later primary check passing. A later recovery does not replace a failed first hold. Original and added grades appear together beside each GIF. Native scope is 24 noisy-fast 20,000-output observations with five terminal checks; it supplies no independent 100,000-output gate.

Before ordinary execution, root must commit/freeze these helpers and capture fresh candidate-bound capacity outcomes for all 16 cells. The binder checks the current public sampler, full Recipe/source/state/array identities and zero-update provenance. Capacity is necessary representability evidence, not learning or robustness evidence. Unknown, blocked and valid numerical failures remain visible; prior witnesses or grades cannot replace fresh cells.

The finite campaign retains its original 15,360-second ceiling. The one-time prior debit is 113.99425188452005 seconds: prior scientific intervals total 109.23634317959659 seconds, while the distinct startup ERROR costs 4.757908704923466 seconds. The latest generator-step combined/certification/publication and both family studies are pinned, with the original critic-rate studies and startup durable evidence checked recursively. Bookkeeping slot `ka2` inherits the former Atlas debit 59.797148591605946 seconds, leaving 7,620.202851408394 seconds; `k3p` inherits the former E22 debit 54.19710329291411 seconds, leaving 7,625.802896707086 seconds. The mapping carries paid cost only and grants no family grades. The pair has 15,246.00574811548 seconds remaining. Current paid time and conservative interruption reservation remain separate; prior costs are charged once.

Each task reserves its unchanged complete acquisition allowance plus 60 seconds of export grace before launch. Insufficient remaining allowance yields INCOMPLETE with deeper cells UNKNOWN; no shorter schedule, failed retry, seed change or extra budget is allowed. Physical GPU1 alone is admitted through the existing shared coordinator, with at least 12,288 MiB free, temperature at most 82 C, one CPU thread, deterministic CUDA settings and a 0.2 process memory fraction. External contention prevents speed ranking. The source snapshot includes the original pinned ring discovery JSON and explicitly binds this protocol, new runner/binder and unchanged delegated generator/critic helpers.

Root owns capacity captures, queues, source freezes and ordinary launches. The following commands document the supported path; preparing these files executes none of them. Replace the capacity path after root's committed capture. The global shared queue path must be the same for both family owners.

```bash
# Read-only fixed spec, after fresh capacity has been captured.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python reports/forge/ka2-k3p-defaults-20261003/run_family_defaults.py spec \
  /ml2/hypergan/forge-ka2-k3p-defaults-20261003/capacity/capacity.json \
  --output /ml2/hypergan/forge-ka2-k3p-defaults-20261003/spec.json

# Explicit CPU-only plan/replay; missing/invalid outcomes fail before queue creation.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python reports/forge/ka2-k3p-defaults-20261003/run_family_defaults.py plan \
  /ml2/hypergan/forge-ka2-k3p-defaults-20261003/spec.json \
  --output /ml2/hypergan/forge-ka2-k3p-defaults-20261003/plan.json

# Root alone launches one family owner at a time on physical GPU1.
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python reports/forge/ka2-k3p-defaults-20261003/run_family_defaults.py run \
  /ml2/hypergan/forge-ka2-k3p-defaults-20261003/spec.json --family ka2 \
  --queue-root /ml2/hypergan/ParticleGAN-single-recipe/runs/forge \
  --output /ml2/hypergan/forge-ka2-k3p-defaults-20261003/ka2

# Repeat the owner command once with --family k3p and a distinct .../k3p output.
# After both archives are immutable, root explicitly recertifies/composes them.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python reports/forge/ka2-k3p-defaults-20261003/run_family_defaults.py combine \
  /ml2/hypergan/forge-ka2-k3p-defaults-20261003/ka2/study.json \
  --archive /ml2/hypergan/forge-ka2-k3p-defaults-20261003/k3p/study.json \
  --output /ml2/hypergan/forge-ka2-k3p-defaults-20261003/combined.json
```

Software validation uses CPU-only isolated fixtures and explicit synthetic receipt controls; those are not trained qualifications. Controls exercise exact resolver-versus-factory equality for all 16 hosts, real two-update optimizer ownership, the genuine primary numerical grader plus first-window semantics, corrupt learned-state rejection, source/snapshot/discovery/CLI identity, family-specific blockers, full denominators, finite budgets, immutable prior debit and durable interruption costs.
