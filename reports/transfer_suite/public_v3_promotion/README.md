# Installed GAN v3 default: public-path verification

**19/19 live behavioral PASS using the installed 0.6.0 wheel and actual
`get_recipe()` default.** All 19 final metrics and complete 24-point live curves
match the previous native-loop winning profile exactly. This verifies the
public implementation, not just the research constructor.

The [versioned leaderboard](../unadjusted/README.md) is v1 **5/19**, v2 **8/19**,
v3 **19/19** with declared D choices (**15/19** on the reference D profile).
[Illustrated formula and settings](../../../docs/gan-v3.md).

## Results through the public API

Six vector and four image toys use `get_recipe().make_trainer(...)`.
The nine auxiliary behavioral hosts retain their custom loops with public
primitives; they are not unconditional `GANTrainer` applications.
Only frozen resource sizes/budgets and the declared D architecture vary.
Losses, regularizers, optimizer settings and schedule remain one global recipe.

PASS requires every metric at five or more consecutive final observations of
all 24 scheduled live checks. The suffix is that final uninterrupted run of
passing checks; confirmation is the first step at which it reaches five.
EMA never determines selection. N/A means that host does not record EMA.

| Toy | Public route | Live | Final passing checks | Confirmed step / budget | EMA |
| --- | --- | --- | ---: | ---: | --- |
| `two_pole` | Custom loop | **PASS** | 12/24 | 57/80 | N/A |
| `trajectory` | Custom loop | **PASS** | 6/24 | 384/400 | N/A |
| `residual_student` | Custom loop | **PASS** | 10/24 | 317/400 | N/A |
| `unipolar` | Custom loop | **PASS** | 18/24 | 184/400 | N/A |
| `ae_gan_hold` | Custom loop | **PASS** | 16/24 | 136/250 | N/A |
| `cover_leftover` | Custom loop | **PASS** | 15/24 | 467/800 | N/A |
| `unused_token_hold` | Custom loop | **PASS** | 6/24 | 192/200 | N/A |
| `mid_scale_identity` | Custom loop | **PASS** | 17/24 | 400/800 | N/A |
| `mode_hold` | Custom loop | **PASS** | 5/24 | 1200/1200 | N/A |
| `vector_two_broad` | Trainer | **PASS** | 20/24 | 450/1200 | PASS |
| `vector_unequal_mass` | Trainer | **PASS** | 7/24 | 1100/1200 | FAIL |
| `vector_unequal_width` | Trainer | **PASS** | 5/24 | 1200/1200 | PASS |
| `vector_anisotropic` | Trainer | **PASS** | 8/24 | 1050/1200 | PASS |
| `vector_overlap` | Trainer | **PASS** | 10/24 | 950/1200 | PASS |
| `vector_spiral` | Trainer | **PASS** | 23/24 | 400/1600 | PASS |
| `img_stripes2` | Trainer | **PASS** | 5/24 | 600/600 | PASS |
| `img_bars4` | Trainer | **PASS** | 9/24 | 500/600 | PASS |
| `img_blobs4` | Trainer | **PASS** | 7/24 | 550/600 | FAIL |
| `img_intensity2` | Trainer | **PASS** | 6/24 | 575/600 | FAIL |

The mode-hold toy has **8/8 live modes**. The image cases retain **2/2, 4/4,
4/4 and 2/2** modes respectively. The unequal-mass case has seven final passes,
with worst-final-five normalized minimum component variance **.42385 ≥ .15**.

EMA passes **7/10 measured cases** and fails unequal mass, image blobs and image
intensity; nine custom hosts have no EMA curve. All six vector EMA endpoints
match the native runs exactly. Image EMA endpoints differ slightly because
`GANTrainer` uses multiply/add while the native host uses `lerp_`; the public
EMA curves are independently scored and retain the same verdicts.

## Resolved attributes and shapes

| Attribute | Verified installed default |
| --- | --- |
| Canonical recipe | `gan_v3`; same through `get_recipe()`, `get_recipe("gan")` and `Recipe()` |
| Loss / cap / spread | Rp logistic; b_cap coefficient 6, κ1.25; spread .05; no particle L2 |
| Adam G / D / particles | .00425 / .00425 / .0085; betas (0, .99) |
| Schedule | 60% full rate; cosine toward 5%; actual groups checked at every trainer step |
| Generic resources | 20,000 particles, latent dimension 4, batch 256, 7,000 steps |
| Default particle table | `[20000, 4]` |
| Public batch-distance D | 19,013 trainable parameters; `[B, 2] → [B]` for B=1,7,128,256 |
| Vector host data / generated samples | `[B, 2]` |
| Image host data / generated samples | `[B, 1, 8, 8]` |
| Quickstart samples | `[1024, 2]`, live weights |

The 19 toys keep their smaller original resource sizes. The optional public
`BatchDistanceDiscriminator` is the unequal-mass witness and quickstart D; the
19/19 profile also declares other D architectures. The API still accepts the
application's networks. Fixed neighbor scales and quadratic batch cost limit
what this result establishes about transfer.

The quickstart was copied outside the checkout and run against the installed
wheel. An eight-step continuous run and a four-step checkpoint resumed to eight
have exactly matching models, optimizer state, resolved recipe, completed step,
initial rates, data RNG and latent/penalty RNGs. Intermediate sampling advances
the separate evaluation RNG; equality of subsequent sample draws is not claimed.
Historical version fields and checkpoint restoration have focused regression tests.

## Provenance and checks

- Numerical source: commit `940af7f`; Python 3.12.13, Torch 2.13.0+cu126 on CPU,
  one thread, fixed seed 0. No seed or budget search.
- Wheel: `particlegan-0.6.0-py3-none-any.whl`, SHA-256
  `2b6cf695673eb551063b41e13f7bf2eab7bad801fbb83c5745df4d3c88ce2888`.
- Public imports resolve to `/tmp/pr38-v3-installed/particlegan`; all 12 loaded
  public modules match checkout bytes. Every host records origins, source hashes,
  recipes, optimizer receipts, shapes and live/EMA curves. Source changes and
  checkout fallback are rejected.
- The three new public-route integration tests pass. The full repository test
  result is documented below; the separate inherited native-2D failure is not
  included in the 19/19 claim.
- Both wheel and source distribution build; `twine check --strict` passes.
- Actual GitHub preview inspected at desktop and mobile widths: 22 rendered math
  expressions, seven display equations, both illustrations loaded, no math errors
  or document-width overflow. Unsupported macros found in the first preview were
  corrected before review passed.

Full local repository run: **665 passed, 9 skipped, one inherited failure**
(`test_particle_native_2d`: EMA action MSE .248969 > .18). Optional gym, image
metric and CUDA integrations account for the skips. On GitHub, Python 3.10–3.12
also fail only that existing gate (the current Torch 2.14 CPU run measures
.193090 > .18). No gate was weakened or skipped. See [CI evidence](../../ci_status.md).

## Reproduce from a checkout

Install the checkout's experiment dependencies in a suitable Python environment,
then build a wheel and force the benchmark to import that installed package:

```bash
python -m pip install '.[dev,experiments]'
repo_dir="$(pwd)"
verification_dir="$(mktemp -d)"
python -m pip wheel --no-deps . --wheel-dir "$verification_dir/wheel"
python -m pip install --no-deps --target "$verification_dir/site" \
  "$verification_dir"/wheel/particlegan-*.whl
cd /tmp
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH="$verification_dir/site:$repo_dir" \
python -u -m benchmarks.transfer_suite.public_default_verification \
  --require-installed-root "$verification_dir/site" \
  --output "$verification_dir/results" > "$verification_dir/run.log" 2>&1
cat "$verification_dir/results/summary.json"
```

In another terminal, `tail -f` the printed log path while the run proceeds.
The summary must say `attempted: 19`, `passed: 19`, `overall: PASS`.
Generated `protocol.json`, `index.json`, `summary.json`, compressed episodes,
source archive and supplementary receipts remain local under the
[artifact policy](../../README.md). This readable report and the runner stay in Git.
