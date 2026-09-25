# One common default: winner replay after removing presets

**19/19 live PASS through the installed package, unchanged from PR #38.**
Every complete live/EMA curve, final metric, convergence result, learning-rate
action and recorded loss matches the previous installed-package run exactly
(excluding elapsed time). This is verification of the existing winner, not an
additional candidate or selection trial.

## What changed

`get_recipe(**overrides)` now returns `Recipe(**overrides)` directly. The public
package has no version/domain preset table, legacy recipe helper or branching
by recipe ID. `name` is report/checkpoint metadata only. The winning fields
remain Rp logistic, b_cap6/κ1.25, spread .05, no particle L2, Adam (0,.99),
G/D LR .00425, particle LR .0085, 60% hold then cosine toward 5%.

```python
from particlegan import get_recipe

recipe = get_recipe()  # One common winning default.
# Component choices are explicit and inherit the same training defaults:
mog = get_recipe(prior_kind="mog", sigma_rel=.025, num_particles=400)
ddgan = get_recipe(model="ddgan", conditioning="ucd", num_classes=4)
```

Positional preset names are removed. Complete saved recipe dictionaries still
restore with `Recipe(**saved_fields)`. Earlier leaderboard controls read their
complete recorded fields from benchmark data outside the installable package.
Applications and the quickstart use the shared default directly; fixed
historical experimental controls keep explicit settings where needed.

## Replicated leaderboard

| Configuration | Required toys | Data toys | Image toys | Live total |
| --- | ---: | ---: | ---: | ---: |
| PR #38 winner, installed package | 9/9 | 6/6 | 4/4 | **19/19** |
| Same winner, single public default | 9/9 | 6/6 | 4/4 | **19/19** |

The declared discriminator profile, data, generators, initialization rules,
resources, training budgets, 24 observations and thresholds are unchanged.
Six vector and four image toys use `get_recipe().make_trainer`; the nine
auxiliary hosts use public primitives in their required custom loops.
The reference D profile's earlier score remains 15/19; a universal D is not
claimed. This run repeats the supported winning profile only.

PASS requires every metric at five or more consecutive final live observations.
A suffix is the uninterrupted sequence of passing observations at the end.
EMA is separate and cannot determine selection. N/A means the host does not
record EMA.

| Toy | Public route | Live | Final passing observations | EMA |
| --- | --- | --- | ---: | --- |
| `two_pole` | Custom loop | **PASS** | 12/24 | N/A |
| `trajectory` | Custom loop | **PASS** | 6/24 | N/A |
| `residual_student` | Custom loop | **PASS** | 10/24 | N/A |
| `unipolar` | Custom loop | **PASS** | 18/24 | N/A |
| `ae_gan_hold` | Custom loop | **PASS** | 16/24 | N/A |
| `cover_leftover` | Custom loop | **PASS** | 15/24 | N/A |
| `unused_token_hold` | Custom loop | **PASS** | 6/24 | N/A |
| `mid_scale_identity` | Custom loop | **PASS** | 17/24 | N/A |
| `mode_hold` | Custom loop | **PASS** | 5/24 | N/A |
| `vector_two_broad` | Trainer | **PASS** | 20/24 | PASS |
| `vector_unequal_mass` | Trainer | **PASS** | 7/24 | FAIL |
| `vector_unequal_width` | Trainer | **PASS** | 5/24 | PASS |
| `vector_anisotropic` | Trainer | **PASS** | 8/24 | PASS |
| `vector_overlap` | Trainer | **PASS** | 10/24 | PASS |
| `vector_spiral` | Trainer | **PASS** | 23/24 | PASS |
| `img_stripes2` | Trainer | **PASS** | 5/24 | PASS |
| `img_bars4` | Trainer | **PASS** | 9/24 | PASS |
| `img_blobs4` | Trainer | **PASS** | 7/24 | FAIL |
| `img_intensity2` | Trainer | **PASS** | 6/24 | FAIL |

Mode hold retains **8/8 modes**; the image cases retain **2/2, 4/4, 4/4, 2/2**.
The unequal-mass case retains seven final passes and worst-final-five minimum
component variance **.42385 ≥ .15**. EMA passes **7/10 measured cases**, with nine
N/A; unequal mass, image blobs and image intensity remain EMA failures.

## Validation and provenance

- Installed wheel: public imports resolve outside the checkout, and all 12
  loaded package modules match source hashes. Numerical run source is `18d7a66`;
  the later schedule-search import correction does not change this run or any
  installed package module.
- Python 3.12.13, Torch 2.13.0+cu126 on CPU, one thread, seed 0. No seed sweep,
  retuning, threshold change or budget extension.
- Public fields exactly match the recorded winner. Default prior shape
  `[20000, 4]`; public batch-distance D has 19,013 parameters and maps
  `[B, 2]` to `[B]` for B=1,7,128,256. Every vector/image host's shapes and
  actual optimizer rates are checked.
- Installed quickstart, PyTorch loop and AE/VAE reconstruction examples pass.
  Eight continuous updates and four plus checkpoint resume to eight have equal
  model weights, optimizer state, resolved recipe and training RNGs. The
  separate evaluation RNG is not part of that resume-equality claim.
- Wheel and source distribution build; strict Twine validation passes.

Full repository tests: **666 passed, nine skipped, one inherited failure**
(and 27 passing subtests). Optional gym/image/CUDA integrations account for
the skips. The native-2D EMA test still fails (.248969 against ≤.18); its
original numerical arm is explicitly preserved. The 19/19 leaderboard is not
a claim that this unrelated repository test passes.

Changing the component no longer silently chooses old optimizer settings.
MoG/DDGAN/application convergence under the new common defaults is not
established by the 19-toy result; the application smoke tests verify execution
and checkpoint contracts.

## Reproduce

Use the [installed-wheel commands](../public_v3_promotion/README.md#reproduce-from-a-checkout)
with a new output directory. They call the current `get_recipe()` directly.
The [historical candidate-search workflow](../../../benchmarks/transfer_suite/UNADJUSTED_SEARCH.md)
keeps older comparison inputs as data. [All attempted candidates](../unadjusted/README.md).

Full curves, source archives, module hashes, per-step settings and comparison
receipts remain local under the [artifact policy](../../README.md). This report,
benchmark inputs and source stay in Git.

Verified wheel SHA-256: `93377144a29eaab75988e8ec46d94401ef1f81c2f410ef01b566f30847ebff85`.
