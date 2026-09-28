# lrfree screening harness + GPU pool (2026-09-26)

Goal: find public-`GANTrainer` configs that need **no learn-rate adjustment** (no horizon/clock-driven
LR or noise schedule, one setting for every task, runs indefinitely). This directory holds a fast,
reusable multi-task screen that reproduces the PR155 new-init (`batch_feature_zero`) API screens
**bitwise**, plus a job pool that keeps both A6000s busy.

Python: `/tmp/pr38-default-env/bin/python` (torch 2.13.0+cu126). Abbreviated below as `$PY`.

## Layout

| path | what |
|---|---|
| `harness/screen.py` | one candidate x one task -> `result.json`, `metrics.jsonl`, `rates.jsonl`, `job-header.json` |
| `harness/hosts/` | verbatim copies of the frozen hosts: `mode_hold_host.py` (+ 1200-row batch-receipt fixture), `image_host.py`, `vector_host.py`, `frozen_shared_critic.py`, `ring_host.py` |
| `harness/tasks/` | frozen task cards: `mode_hold_protocol.json`, `image_task_specs.json`, `vector_task_specs.json`, `ring_reference_declaration_dv2.json` |
| `harness/pool.py` | daemon: claims `queue/*.json`, runs screen.py with N slots per GPU, appends `ledger.jsonl`, rebuilds `LEADERBOARD.md` after every job |
| `harness/submit.py` | candidate + tasks -> job files (refuses exact duplicates) |
| `harness/wait.py` | blocks until a candidate's jobs finish, prints a compact matrix |
| `harness/compare.py` | exact comparison of a run against archived evidence metrics/rates |
| `harness/validate_all.py` | re-checks every finished run that has archived PR155 evidence (prints the table below) |
| `harness/lrlib.py` | shared paths, task presets, hashing, ledger/leaderboard code |
| `harness/native100_score.py`, `hosts/native100/` | native100 frozen gates (separate process) and verbatim host copies |
| `harness/custom22.py`, `harness/components.py`, `hosts/custom/` | the 8 custom hosts of the 22-check suite: host-loop bindings, the candidate-policy engine, verbatim host copies (see *Custom22 hosts*) |
| `harness/components_parity.py` | L1 gate: engine vs the package's own `GANTrainer.step`, bitwise every update; cached in `runs/_parity/` |
| `pool-config.json` | slots per GPU, free-memory guard, timeouts (re-read every 2 s; edit live) |
| `runs/<cand>/<task>/` | per-job output: `log.txt` (compact JSON line per observation + final `RESULT` line), `metrics.jsonl` (full observation rows incl. EMA + diagnostics), `rates.jsonl` (per-update LRs/noise), `result.json` |
| `runs/<cand>/candidate.json` | registered config (package root + content hash, overrides, options, note) |
| `candidates/<name>/` | put modified package copies here (`candidates/<name>/particlegan/...`) + `overrides.json` |
| `validation/` | direct reproduction runs of the PR evidence (see below) |
| `ledger.jsonl`, `LEADERBOARD.md`, `pool.log` | results; `tail -f pool.log` shows START/DONE lines |

## Commands

```bash
cd /ml2/hypergan/lrfree-20260926
PY=/tmp/pr38-default-env/bin/python

# single screen (foreground)
$PY harness/screen.py --package-root DIR_WITH_particlegan --overrides '{"lr":0.002}' \
    --task mode_hold --output /tmp/x --device cuda:0 [--candidate-options '{...}'] [--cand NAME]

# pool (already running; start again only if pool.pid is stale)
nohup $PY harness/pool.py >> pool.log 2>&1 &
touch queue/STOP          # drain: finish running jobs, then exit
kill $(python3 -c "import json;print(json.load(open('pool.pid'))['pid'])")  # stop launching now; running children keep going and are adopted on restart

# submit a candidate (tasks: names or presets quick|images|vectors|gates|ring|all|native; default gates)
$PY harness/submit.py --cand my-cand --package-root candidates/my-cand --overrides candidates/my-cand/overrides.json \
    --tasks gates --note "what it is" [--priority 5] [--candidate-options '{"strict_streams": false}']

# block until done, print matrix
$PY harness/wait.py my-cand [other-cand ...] [--timeout 3600]

# compare against archived evidence
$PY harness/compare.py runs/API-RP12/mode_hold <evidence>/metrics.jsonl.gz --rates <evidence>/learning-rates.jsonl.gz
```

`--overrides` / `--candidate-options` take inline JSON or a file. A PR `declaration.json` is also accepted as
`--overrides`: its `recipe_overrides` are used and its `evaluation_generate` / `serial_backward_argument`
become the options. Presets: `gates` = mode_hold + 4 images + 6 vectors (11 jobs), `ring` = ring_shift + stationary.

## Tasks (frozen PR155 new-init definitions)

| task | host resources forced over the candidate recipe | updates | observations | pass rule |
|---|---|---:|---|---|
| `mode_hold` | 12 particles, z 4, batch 128; prior `make_prior(init_std=.5, generator=data stream)`; MLP G 4-96x3-2, D 96x3 Fourier3 | 1200 | 24 (every 50), 4096 samples, live primary | final obs has 8/8 modes and HQ>=.90 **and** final passing suffix >=5 |
| `img_intensity2`, `img_blobs4`, `img_stripes2`, `img_bars4` | 32 particles, z 8, batch 32, residual_upsample width 16 | 600 | 24 (every 25), all 32 particles | modes (>=min_mode_fraction) = all, HQ>=.90 (rmse<=.1), final suffix >=5 |
| `vector_two_broad`, `vector_unequal_mass`, `vector_unequal_width`, `vector_anisotropic`, `vector_overlap` | 256 particles, z 4, batch 128, frozen critic card per task | 1200 | 24 (ceil(i*N/24)) | every card threshold (sw1, mass_tv, hq, component covariance/eigen, min_mass_ratio ...) + final suffix >=5 |
| `vector_spiral` | same | 1600 | 24 | sw1 / mean_error / covariance_error + final suffix >=5 |
| `ring_shift` | original recovery ring: 20000 particles, z 2, batch 2048, G/D 96x3 Fourier3, trainer-owned prior | 4600, target +[1,0] after 2400 | every 10 (460) | harness rule: both segments arrive (8 modes, HQ>=.9) and each ends with >=5 passing checks. Reported: arrival, retention since arrival, changed-target delay, retention after arrival, departures |
| `stationary` | same ring, no change | 7500 | every 10 (750) | arrival + final suffix >=5; reported retention since arrival/departures |

**Note:** the public `Recipe` default is `total_steps=7000`; a candidate that does not set `"total_steps": null`
is horizon-scheduled and its trainer raises on `stationary` (7500 updates) -> leaderboard cell `ERROR budget`.

The candidate's own recipe policy (rates, noise, controller, EMA, total_steps ...) is left untouched; only the
host resource fields above are overwritten, exactly as the PR screens did. `initialization='batch_feature_zero'`
is injected unless the overrides set it.

Kept from the frozen harnesses: deterministic CUDA (`use_deterministic_algorithms`, cudnn deterministic,
no TF32, highest matmul precision, CUBLAS_WORKSPACE_CONFIG=:4096:8, 1 CPU thread), identical construction
order and RNG consumption, identical data/latent streams (mode_hold: shared stream0 with peek/commit of the G
real batch, checked against the frozen 1200-row receipt fixture every update; image: shared global CUDA
stream with real -> D idx -> G idx cursor check every update; vector: data0 / latent1 / penalty2 streams, two
real batches per update via the `generator_real` callback; ring: stream0 + trainer-owned streams), the
public `GANTrainer.step` path under serial autograd with the same `collect_stats` schedule, identical
evaluation code (live model scored; EMA reported separately under `ema`), identical scorers and pass rules.

Dropped: source sealing/hash manifests, declarations, CPU zero-step preflights, the 1200/600-row dry
sampling passes, per-step optimizer-placement proofs, state digests around each evaluation, checkpoints
(opt-in `save_final_state`), the ring frozen-copy control (opt-in `ring_frozen_control`), Torch-version/GPU
asserts (recorded in `job-header.json` instead).

### Candidate options (`--candidate-options`)

| option | default | meaning |
|---|---|---|
| `evaluation_generate` | `auto` | `plain` / `indexed` call of `trainer._generate` in evaluation. auto = `indexed` iff `GANTrainer._generate` has an `indices` parameter (matches all 51 PR declarations) |
| `serial_backward_argument` | `auto` | pass `serial_backward=True` to `GANTrainer`. auto = iff `GANTrainer.__init__` accepts it (matches all PR declarations) |
| `strict_streams` | `true` | error (status ERROR) if the candidate changes the frozen stream transaction (extra/missing latent draws on the shared stream, construction consuming the data stream, not calling `generator_real`). `false`: continue, keep the frozen data order, count `stream_deviations` (cell marked `*dev`) |
| `initialization` | `batch_feature_zero` | injected into overrides unless overrides set it (`null` = don't inject) |
| `diagnostics` | `true` | controller/precision/penalty/game_stats diagnostics in `metrics.jsonl` at observation steps (verified not to perturb training) |
| `save_final_state` | `false` | save `final-state.pt` |
| `ring_frozen_control` | `false` | ring_shift: also score a frozen copy of the 2400 state (diagnostic only; restores global RNG) |
| `eval_output_noise` | `false` | `false`: primary score on clean generator samples, old noisy score under `noisy`. `true`: old behaviour primary, clean under `clean` (see below) |
| `image_prior_perturb` | `false` | image tasks only: pass `prior.perturb(prior.z, generator=stream)` to `_generate` so packages with latent jitter in `ParticlePrior.sample` score the same jittered law as training; included in the config hash |

### Clean evaluation (fix of 2026-09-27)

Before this fix every task scored samples with the recipe's **training output noise** added (`output_noise_std`,
e.g. .029 for DV12/RP15): mode_hold `fake + sigma*randn` after `_generate`, image/vector `_generate(..., sigma, stream)`,
ring `GANTrainer.sample()` (which adds it itself). That is a scoring bug (img_intensity2 HQ needs rmse <= .06).
Now the primary score uses clean samples (sigma 0); the old noisy score is still computed at every observation
from an identically seeded evaluation stream (noisy call first, in the frozen order) and stored as
`metrics.jsonl` row key `noisy` (live + `ema` + `pass`), with `noisy_status`, `noisy_passing_checks`,
`noisy_first_arrival`, `noisy_final_streak`, `noisy_final` (ring: `noisy_segments`) in `result.json`.
Evaluation runs under `fork_rng` on private generators, so training is untouched. Checked directly on GPU
(new harness vs old pool runs): API-DV12 img_intensity2, API-RP15 mode_hold, API-DV12 vector_two_broad:
`rates.jsonl` byte-identical and the new `noisy` scores bitwise equal to the old primary scores (24/24 obs,
live+EMA, diag/lr identical); ring path smoke-tested (noisy == old obs). DV12-class controllers still apply
their latent perturbation (`controller.perturb_latent`) inside `_generate` at evaluation; that is not output noise
and is unchanged. `eval_output_noise` is always part of the config hash, so no new run collides with a pre-fix
run; pre-fix candidate names are refused by submit.py (resubmit under a new name, e.g. `NAME-ce`).
LEADERBOARD.md has an `eval` column: rows without the option in their recorded options (all pre-fix rows) are `noisy`.

The PR declaration fields `initial_optimizer_state` (eager/lazy) and `optimizer_step_devices` were only
assertions about what the package does (eager Adam comes from the recipe's `adam_eager_state`); they do not
change behaviour and are not needed. Observed LRs are logged every update in `rates.jsonl`.

### Native 100-Gaussian tasks (added 2026-09-27): `grid100`, `rotated100`, `staggered100`

Opt-in preset `native` (not in `gates`/`all`; no config hash changed). With these the harness covers 14 of the
frozen 22-toy suite (mode_hold, 4 images, 6 vectors, 3 native); the other 8 (two_pole, trajectory,
residual_student, unipolar, mid_scale_identity, ae_gan_hold, cover_leftover, unused_token_hold) need
component-controller bindings and are not GANTrainer routes (`api-rp2-frozen22-route-map.md`).
`ring_shift`/`stationary` are harness-only extras, not part of the 22.

Host = `configs/toy100/constraints_simple_regularization.json` resources (copied to `tasks/`, hash-checked):
7000 updates, seed 1234, 20000 particles, z 2, batch 2048, `affine_square_v1` G (Linear(2,2) identity/zero),
`SimpleMLPDiscriminator(2,128,3,fourier=3)` (verbatim `hosts/native100/toy_models.py`). Canonical native CUDA
fixture (gpu-known-winner-control worker): construction under the CUDA default device, prior N(0,1) draw then
uniform[-5,5], Linear defaults then identity, D defaults then Xavier; the archived prior/G raw hashes and prior
range are asserted (`tasks/native100_fixture.json`), all tensors hashed to `native-fixture.json`. The candidate's
`batch_feature_zero` then keeps the identity G, zero biases and the supplied prior and replaces D's four
Xavier weights (as on every harness task). Learner = candidate `get_recipe(**overrides, num_particles=20000,
z_dim=2, batch_size=2048)` + public `GANTrainer.step`; 7000 is the evaluator budget only. Data: own CUDA stream
seed 1234, D-real then a fresh G-real batch per update via `generator_real` (strict: missing call = deviation).
Evaluation: 34 observations (0,1,10,25,50,100, every 250 to 7000), live+EMA, 20000 prior draws with replacement
(latent 1637) through the candidate `_generate` with sigma 0 (= public `sample` without output noise) = clean
cloud; noisy cloud = clean + candidate sigma * randn on the forked global seed 1636 (paired live/EMA, the
frozen protocol). Target 1635; final-five quality clouds 6000..7000; 100k holdout (2835/2836/2837).
Both clouds are written in the native run-directory format (`runs/<cand>/<task>/native-clean/`,
`native-noisy/`) and graded by the UNCHANGED `benchmarks/toy100/gate.py::score_run` (coverage) and
`accuracy_gate.py::score_run` (final five + holdout) via `harness/native100_score.py` (separate process, source
hashes checked). Cell status = accuracy gate status (PASS needs coverage PASS too); `passing_checks` = passing
coverage observations of 34; `@` = first passing coverage observation; `final` also carries `acc_*`
(final-observation fidelity) and `holdout_*`; `native` in result.json has both gates' summaries.

Validation (no archived native100 run through a continuous public `GANTrainer.step` learner exists; the only
A6000/torch 2.13 native runs are `gpu-known-winner-control/simpler22_reference`, a legacy LegacyRecipe +
cosine/noise-hook learner that cannot be expressed as package+recipe): (1) the frozen gates reproduce the three
archived gate + accuracy-gate verdicts exactly from the saved evidence; (2) archived prior/G hashes and prior
range reproduced; (3) observation target, 100k holdout target and step-0 target bitwise equal to the archive
(all 3 problems); (4) API-RP15 (no latent controller) step-0 live/EMA clean clouds bitwise equal the archived
step-0 snapshots (sha 07cc7aed...); (5) the noisy adapter at sigma .029 is bitwise equal to the frozen legacy
`make_trainer` + `trainer.sample` path under seeds 1636/1637; (6) 100-step plumbing runs
(`LRFREE_NATIVE_TEST_STEPS`, test only) pass every evidence-integrity check of both gates.

Runtime: 20000 particles make DV12's `perturb_latent` (20000-center nearest-neighbour search per latent) and
KA2's per-particle surprise the cost: ~0.6 steps/s DV12, ~0.9 steps/s RP15 under a full pool (~2-3.5 h per
job); GPU ~0.6-1.2 GB in training, DV12 ~4.3 GB peak at the 100k holdout. Pool timeout 43200 s per native task.

**2026-09-27 DV12 `perturb_latent` speedup:** the dv12-family candidate packages (dv12-ams, dv12-nr, dv12-st, dv12-pfloor, dv12-const) now use a broadcast squared-distance nearest-neighbour search instead of the `cdist(donot_use_mm)` loop. It is ~10x faster per native100 update (0.21 -> 0.02 s uncontended) and uses less memory. Results are **bitwise identical across the patch**: rates, metrics and every native cloud/verdict are byte-equal on real runs (mode_hold, img_intensity2, grid100). Package hashes changed, so results from before and after the patch are the same config under different hashes. See `reports/perturb-latent-speedup.md`.

### native_steps (added 2026-09-27)

Candidate option `native_steps` (int >= 7000, native tasks only; unset = unchanged behaviour and config hashes,
hashed only when set): N updates, the same schedule extended (0,1,10,25,50,100, every 250 to N), final five
N-1000..N, holdout at N, frozen gates scored with budget N (they read it from `config.json` `steps`; thresholds
untouched; recipe `total_steps` untouched, so training up to any step is identical). Every multiple of 7000 below N
is also scored as its own budget from the same run: `native-<kind>-b<B>/` (snapshots hard-linked, events to B,
final five B-1000..B, holdout drawn at B) -> result.json `native_budgets[B][clean|noisy]`. Check: n21k runs
reproduce dv12-ams-rc3-ce bitwise to 7000 (rates, observation rows, b7000 quality clouds and holdout).

### Recorded-only sharpness + schedule diagnostics (added 2026-09-27)

No effect on training, verdicts or config hashes (harness-only; evaluation tensors already drawn, no RNG):
- `sharp_clean` / `sharp_noisy` (mode_hold, native100): median over samples of distance to the nearest mode
  centre / that mode's data std (mode_hold .07, native .03). Ideal isotropic 2-D Gaussian: sqrt(2 ln 2) = 1.1774.
- `sharp_rmse_clean` / `sharp_rmse_noisy` (images): median per-sample rmse to the nearest template.
- Both kinds are in every observation row (live, `ema`, and the secondary `noisy`/`clean` dict) and in
  result.json `final` / `ema_final`.
- `diag.sched` at observation steps: `pe`, `m`, `gt` (controller payoff_error / mobility / game_trust),
  `critic_payoff_factor` (1/(1+pe^2)), `g_lr_scale` (per G-optimizer group), `d_lr_scale` (applied D LR / initial),
  `output_sigma` (the sigma the candidate samples with), `log_output_sigma` + `output_noise_mode` when the package
  has them, `real_grad_norm` = mean_i ||grad_x D(x_i)|| on that update's D-real batch (post-update critic).
- Output sigma everywhere (evaluation noisy scoring, `rates.jsonl` `out_noise`) comes from `GANTrainer.output_sigma()` /
  `last_output_sigma` when the package defines them (learnable / mobility output noise, e.g. `candidates/dv12-nr`),
  else from the module-level `output_noise_std` exactly as before.

### Custom22 hosts (added 2026-09-27): `two_pole`, `trajectory`, `residual_student`, `unipolar`, `ae_gan_hold`, `cover_leftover`, `unused_token_hold`, `mid_scale_identity`

These complete the frozen 22-check suite: preset `all22` = `gates` (11) + `custom` (8) + `native` (3). The presets
`gates`/`all`/`native` and every config hash are unchanged (custom tasks are opt-in). Full design, decisions,
verification and open policy questions: `reports/custom22-design.md` (section 9 "Implementation").

- **Task** = the frozen PR #155 behavioral host (budgets 80/400/400/400/250/800/200/800; specs and live thresholds
  from `default_comparison.json` -> `tasks/custom22_specs.json`). Host sources are byte-identical copies in
  `hosts/custom/benchmarks/` (3-line provenance header; sha256 checked at every run; import-only shims for
  `gan_v3.py`, `legacy/recipe.py`, `toy100/__init__.py`). `custom22.py` re-expresses only each host's training loop,
  with `# Lnnn` references to the original lines. Data, models, auxiliary losses, scorers and budgets are the host's.
- **Learner** = the candidate package's GANTrainer policy via `components.py`: `make_optimizers` role groups
  (g / prior / d), amsgrad, A2, KA2 penalty + EMA critic + spike guard, DataDriftController hooks fired in `_step`
  order, StationarityLR, learnable sigma, EMA. Host Adam, legacy loss/b_cap, host LR schedules,
  `schedule_optimizer` and host EMA are removed; LRs are written only by the engine (asserted each step).
  Unbound policy is refused (cell `ERROR refused`): particle_birth_death, dv11, non-gan/scalar, critic input noise,
  unknown recipe fields or GANTrainer hooks.
- **Parity gate**: before its first custom job a package + overrides must pass L1 (engine re-expression ==
  its own `GANTrainer.step`, bitwise after every update, 1000 updates, 2 configs, CPU; ~40 s, run automatically
  under a lock and cached in `runs/_parity/<pkg>-<overrides>-<engine>-<gate>.json`). FAIL -> `ERROR parity`.
  A package whose `_step` differs from the engine's re-expression fails here instead of running wrongly.
- **CPU jobs** (1 thread, deterministic, no CUDA context), a few seconds each; they take a pool slot but no GPU.
- **Scoring of record is NOISY for custom tasks regardless of `eval_output_noise`**: the candidate's output noise
  is added at the frozen `legacy_noise_adapters` sites (evaluation: global seed 402+step, private noise stream
  +1901); clean scores are kept under `clean` / `clean_*`. The 5 parameter-scored hosts have noisy == clean;
  trajectory, residual_student and ae_gan_hold differ.
- **Verdict** = verbatim `protocol.test_verdict`: 24 observations at ceil(i*budget/24), passing suffix >= 5 and
  every final live threshold. EMA is reported only.
- **Outputs** as for every task: `log.txt` (one compact JSON line per observation: thresholds, `ok`, EMA, LRs,
  sigma, pe/m/gt; then `RESULT`), `metrics.jsonl` (+ `clean`, `ema`, controller/KA2/A2 diagnostics), `rates.jsonl`
  (every update, every group), `result.json` (screen schema + `thresholds`, `clean_*`, `custom22` receipts:
  sources, parity id, groups/roles, resources, aux coefficients, policy extensions applied).
- **Policy extensions awaiting sign-off** (design section 8, recorded per task in `custom22.blockers_applied`):
  B1 MoG latent kernel on `means()`, B2 ae_gan_hold outside GANTrainer scope, B3 one role-union KA2 call on the
  multi-scale hosts, B4 learnable sigma trained by auxiliary terms, B5 one bandwidth for cover_leftover's two tables.
- Tests: `$PY -m unittest -v harness/tests/test_components_parity.py`.
- Submit: `$PY harness/submit.py --cand NAME --package-root DIR --overrides FILE --tasks custom` (or `all22`).

### Duplicate / naming rules (submit.py)

Config hash = sha256(package `particlegan/*.py` contents + normalized recipe overrides + behaviour options
`evaluation_generate`, `serial_backward_argument`, `strict_streams`, `eval_output_noise` (always)). A task already run (non-ERROR) or queued
with the same hash is skipped under any name; a candidate name registered with a different hash is refused
(use a new name). `--rerun-errors` re-queues ERROR tasks. Editing a package after submit is flagged in the
ledger (`warning: package changed between submit and run`).

## Validation (exact reproduction of PR155 new-init evidence)

Every archived new-init API run that the harness re-ran (via `validation/` direct runs and pool runs in `runs/`)
reproduces **bitwise**: every numeric field of every live and EMA observation, and the applied learning rates of
every update (`harness/validate_all.py` regenerates this table; `harness/compare.py` does one run).
Options were auto-detected in the pool runs (plain declarations only in `validation/`), so the auto-detection
of `evaluation_generate` (incl. DV13 `indexed`) and `serial_backward_argument` is covered too.

Requested checks:

| check | PR evidence | harness | exact? |
|---|---|---|---|
| RP12 mode_hold | PASS 19/24, first 300, final 8/8 HQ .9983 | PASS 19/24, first 300, final 8/8 HQ .998291 | bitwise, 24/24 obs, 1200/1200 LR rows |
| RP12 img_bars4 | FAIL 0/24 | FAIL 0/24 (final 2 modes, HQ .97) | bitwise |
| RP12 img_intensity2 | PASS 12/24 @325 | PASS 12/24 @325 | bitwise |
| DV12 vector_unequal_mass | FAIL 0/24, cov err .8793, min mass ratio .2319 | FAIL 0/24, .879316 / .231934 | bitwise |
| public ka2-constant mode_hold | FAIL 0/24, final 4/8 | FAIL 0/24, final 4/8 HQ 1.0 | bitwise |
| DV2 ring_shift (4600, change 2400) | no arrival by 2400; +380, 183/183 | same | bitwise, 460 obs, 4600 LR rows |
| DV2 stationary (7500) | no archived new-init run | first 240 observations (to 2400) identical to ring_shift | consistency only |

All archived comparisons (pool + validation runs):

| candidate | task | evidence result | harness result | observations exact | LR steps exact |
|---|---|---|---|---|---|
| API-DV12 | mode_hold | PASS 12/24 @650 final 8/8 hq 0.9834 | PASS 12/24 @650 final 8/8 hq 0.9834 | 264/264 (bitwise) | 1200/1200 |
| API-DV12 | vector_unequal_mass | FAIL 0/24 cov 0.8793 mmr 0.2319 | FAIL 0/24 [c.covariance_error,min_mass_ratio] cov 0.8793 mmr 0.2319 | 1272/1272 (bitwise) | 1200/1200 |
| API-DV13 | mode_hold | FAIL 0/24 @None final 8/8 hq 0.8662 | FAIL 0/24 (8m hq0.87) final 8/8 hq 0.8662 | 264/264 (bitwise) | 1200/1200 |
| API-DV2 | ring_shift | arr None ret 0/0 / arr 2780 ret 183/183 | FAIL no arrival | +380 183/183 | 5060/5060 (bitwise) | 4600/4600 |
| API-RP12 | img_bars4 | FAIL 0/24 @None | FAIL 0/24 (2m hq0.97) | 600/600 (bitwise) | 600/600 |
| API-RP12 | img_blobs4 | PASS 17/24 @200 | PASS 17/24 @200 | 600/600 (bitwise) | 600/600 |
| API-RP12 | img_intensity2 | PASS 12/24 @325 | PASS 12/24 @325 | 408/408 (bitwise) | 600/600 |
| API-RP12 | img_stripes2 | PASS 17/24 @200 | PASS 17/24 @200 | 408/408 (bitwise) | 600/600 |
| API-RP12 | mode_hold | PASS 19/24 @300 final 8/8 hq 0.9983 | PASS 19/24 @300 final 8/8 hq 0.9983 | 264/264 (bitwise) | 1200/1200 |
| API-RP14 | img_bars4 | FAIL 0/24 @None | FAIL 0/24 (1m hq0.34) | 600/600 (bitwise) | 600/600 |
| API-RP14 | img_blobs4 | PASS 6/24 @475 | PASS 6/24 @475 | 600/600 (bitwise) | 600/600 |
| API-RP14 | img_intensity2 | PASS 9/24 @375 | PASS 9/24 @375 | 408/408 (bitwise) | 600/600 |
| API-RP14 | img_stripes2 | PASS 13/24 @300 | PASS 13/24 @300 | 408/408 (bitwise) | 600/600 |
| API-RP14 | mode_hold | PASS 12/24 @600 final 8/8 hq 0.9998 | PASS 12/24 @600 final 8/8 hq 0.9998 | 264/264 (bitwise) | 1200/1200 |
| API-RP15 | img_bars4 | FAIL 0/24 @None | FAIL 0/24 (3m hq0.72) | 600/600 (bitwise) | 600/600 |
| API-RP15 | img_blobs4 | PASS 17/24 @200 | PASS 17/24 @200 | 600/600 (bitwise) | 600/600 |
| API-RP15 | img_intensity2 | PASS 5/24 @500 | PASS 5/24 @500 | 408/408 (bitwise) | 600/600 |
| API-RP15 | img_stripes2 | PASS 21/24 @75 | PASS 21/24 @75 | 408/408 (bitwise) | 600/600 |
| API-RP15 | mode_hold | PASS 14/24 @550 final 8/8 hq 0.9998 | PASS 14/24 @550 final 8/8 hq 0.9998 | 264/264 (bitwise) | 1200/1200 |
| public-k3p | mode_hold | FAIL 0/24 @None final 6/8 hq 0.9709 | FAIL 0/24 (6m hq0.97) final 6/8 hq 0.9709 | 264/264 (bitwise) | 1200/1200 |
| public-ka2 | mode_hold | FAIL 0/24 @None final 6/8 hq 0.9993 | FAIL 0/24 (6m hq1) final 6/8 hq 0.9993 | 264/264 (bitwise) | 1200/1200 |
| public-ka2-constant | mode_hold | FAIL 0/24 @None final 4/8 hq 1.0000 | FAIL 0/24 (4m hq1) final 4/8 hq 1.0000 | 264/264 (bitwise) | 1200/1200 |

Determinism across placement: 44 benchmark copies of the same job on GPU0 and GPU1 at 1-12 concurrent jobs per
GPU produced one identical metric trace. The only non-bitwise outputs are wall-clock `seconds` fields.

## Pool sizing and runtimes

Per-job GPU memory is ~0.4 GB (CUDA context; torch reserves <=60 MB for gate tasks, ~130 MB ring). Jobs are
CPU/launch-bound (profiling: >95% of time inside `GANTrainer.step`, harness overhead negligible), so the limit is
CPU cores and GPU time-slicing, not memory. Benchmark (public ka2-constant mode_hold, 1 CPU thread per job):

| concurrent jobs | per-job train time | total throughput |
|---|---:|---:|
| 1 per GPU (2) | 15.8 s | 396 jobs/h |
| 4 per GPU (8) | 22.4 s | 1108 jobs/h |
| **7 per GPU (14)** | 34.8 s | **1226 jobs/h** |
| 10 per GPU (20) | 48.4 s | 1279 jobs/h (load avg 25 on 24 cores) |
| GPU1 only: 4 / 8 / 12 | 20.2 / 32.0 / 46.3 s | 640 / 829 / 872 jobs/h |

Chosen: **7 slots per GPU** (`pool-config.json`), 96% of the 10/GPU throughput while leaving ~5 cores for the
other users' processes; free-memory guard 1500 MiB per GPU before each launch. Both GPUs then run at 99-100%.
Edit `pool-config.json` to change slots live (e.g. `{"slots": {"0": 9, "1": 9}}`).

Per-task train time under that full load (14 concurrent jobs), fast public KA2/K3P-class learners vs. the
slow two-field secant-resolvent RP12/RP14 class (unloaded times are ~2.5-3x shorter):

| task | KA2/K3P class | DV12 | RP12/RP14 class | unloaded reference |
|---|---:|---:|---:|---:|
| mode_hold (1200) | 18-34 s | 44 s | 185-211 s | ka2 16 s, RP12 64 s |
| image, each (600) | 12-16 s | 19-23 s | 83-101 s | RP12 29 s |
| vector, each (1200; spiral 1600) | 17-46 s | 48-74 s | 183-259 s | DV12 23 s |
| ring_shift (4600) | DV2 131 s | - | 670 s | DV2 90 s |
| stationary (7500) | DV2 205 s | - | 1023 s | - |

A full `gates` submission (11 jobs) finishes in ~1-2 min for KA2-class candidates and ~5 min for RP12-class
when the pool is otherwise idle; `ring` adds 2-20 min.

## Leaderboard ranking

Rows = candidates, columns = tasks. Rank: number of PASSed tasks (gates), then mode_hold passing checks, then
total passing checks over the 24-observation tasks. Ring tasks count as gates but their 460/750 checks are not
added to the total.

## Reference rows already in the ledger

Submitted with the PR declarations' `recipe_overrides` (`candidates/<name>/overrides.json`), packages read in
place (read-only): `API-RP12` (all 13 tasks), `API-RP14`, `API-RP15`, `API-DV12` (gates),
`API-DV13` (mode_hold), `API-DV2` (ring), `public-ka2-constant`, `public-ka2` (`/ml2/hypergan/ParticleGAN-ka2-default`)
and `public-k3p` (package extracted from its evidence `source.zip` into `candidates/public-k3p/`, hash-verified).
These are references, not LR-free candidates: RP12/14/15 carry `network_lr_horizon_cap`, the public controls
carry `total_steps` (public-ka2/k3p refuse `vector_spiral`'s 1600 updates: `ERROR budget`).

To screen a new mechanism: copy a package (`cp -r <pkg>/particlegan candidates/NAME/`), edit it, write
`candidates/NAME/overrides.json`, then `submit.py --cand NAME --package-root candidates/NAME ...`. Do not edit a
package after submitting (the run records the content hash it actually imported).
