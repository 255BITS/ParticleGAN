# Forge host provenance audit — 2026-09-29

The corrected intensity failure is valid evidence for Forge's declared **transpose12** host. It is not a failed reproduction of the historical K3P positive, which used **residual16**. Selecting a known passing formulation requires binding its architecture and initialization, not only its task name. This audit read existing source and receipts; it changed no production files and launched no training.

The current attempt is `reports/forge/attempts/65265b43758f4dbd85b77ef2ed46ac4b/`: source `fc012b47b3714f8b7f67e21ce4d1bf3ff2809b3ff5aa001cff00b36663d1d39d`, origin `f35be792137dfb30cfb9cb3033a888099a3e8747`, candidate revision `ef34dc1682c082151c202c4549af6343fa8b309ac5cf71c8ad29098dc55e77e4`. All 24 observations fail; final modes 0, HQ 0, RMSE 0.1750000268. Finite updates and zero unintended RNG deviations support a scientific failure, not an infrastructure error. A zero image has RMSE exactly 0.175 against the low-intensity template: 16 of 64 pixels at 0.35. The curve is consistent with near-zero output, but saved metrics alone do not prove sigmoid saturation or locate its cause.

## What was selected and what was executed

`configs/forge/tasks/img_intensity2.json` has contained `architecture: transpose`, `width: 12` since its introduction in commit `3a3ef24a`. Those values match the raw `benchmarks/transfer_suite/image_tasks.py: TASKS`. The adapter at `experiments/forge/adapters.py:_image` passes that exact host definition to `image_tasks.Generator` and `Discriminator`; these constructors consume architecture and width. The adapter is not silently ignoring a residual16 declaration. The mismatch occurred when selecting the raw task defaults instead of the published passing host profile. I found no Forge rationale documenting that choice as a deliberate replacement of the passing architectures; source history establishes the choice, not the author's intent.

The published route is explicit:

1. `benchmarks/transfer_suite/compare_defaults.py:plan` loads `plans/default_comparison.json`, extracted from passing architectures at `d77e9e8`.
2. Its intensity row declares residual upsampling, width 16, and reference `solvability/images/stage1/episodes/residual16__img_intensity2.json.gz`, original reference SHA-256 `127f9da08ea919d4ce2d319626308ae326da3456579d7b13c1708b0d178fae7d`.
3. `public_default_verification.py:load_declaration/declared_spec` applies the separate declared discriminator profile without reverting the image architecture.
4. The archived K3P `sources/k3p/probe.py` uses that same declaration loader before `toy100_compatibility.run_image`.

Relevant current file hashes:

| File | SHA-256 |
| --- | --- |
| `benchmarks/transfer_suite/plans/default_comparison.json` | `2e62a935dd702c9851165354cb580d18ffd4c47b72e0568562d0f50256028883` |
| `reports/transfer_suite/unadjusted/leading_profile.json` | `38327c94dd8c1ad2729ab07a5866191d2c4ec05b3acb841498d5abd786d691dd` |
| `benchmarks/transfer_suite/plans/residual16.json` | `8f6aa589f80ded8970297bfedfc335a8c79a077f48c8b88ad4c6d5521bf7fbd0` |
| `benchmarks/transfer_suite/image_tasks.py` | `4f070d0879cbaaaa076f82b4683cfe74ea9ed8d85480d1922e249444a752ce58` |
| `configs/forge/tasks/img_intensity2.json` | `6a00da8c76921f40726aefdf6a66c38507a89ef88bde07c6ea4df153955cfa93` |

## The actual positive and its limits

`reports/toy100/gap-fill-20260925/results/k3p-toy-img_intensity2.json.gz` records CUDA PASS, live modes 2, HQ 1, RMSE 0.0327395275, seven final passing checks; first pass 375, stable from 450, confirmed at 550 of 600 updates. Compressed-file SHA-256 is `3e62e45d26ab06ba753ea874e713523cfdfb058f8c2deb2e23ac79c7e7ebdbae`; **decompressed original JSON** SHA-256 is `b8956714ff6082879989c939947452281f6d9ddab4ed0d7296261f4a44f52792`, matching `qualification-summary.json`'s artifact hash.

Its spec explicitly contains residual16. The initial optimizer tensor shapes also prove it: G input `[64,8]`, two `[16,16,3,3]` convolution weights, prior `[32,8]`. The receipt binds supplied initialization fixture SHA-256 `2a559722f8b3120a218759118327c4da3122fe8bd7fb9b1901c10e356096d1f2`; the recorded initial prior tensor hash is `5bf65657c73e0f7f175debdbd8dba1f4335dd1dadcff23de6c8c2901c03866ef`. A digest alone cannot reconstruct that fixture.

Both this positive and current Forge use G/D LR 0.00425, prior LR 0.0085, betas (0,0.999), K3P coefficient/kappa 1, network floor 0.01, prior floor 0.05, input noise 0.5, and output noise 0.029. Thus changing LR to the raw host provenance value 0.0017 is not a K3P replication. A2 had zero scoped applications in the historical positive and zero eligible/applied updates in the current failure; disabling A2 is not the first evidence-backed explanation for their difference.

Other differences remain explicit: the historical image scorer evaluated an output-noise wrapper; Forge now scores clean outputs. Historical construction used the host's G/D/prior draw order and a supplied fixture; Forge uses named component/parameter initialization and isolated streams. Historical runtime was Torch 2.13.0+cu126; current is 2.14.0+cu130. The image prior is a 32-row finite cloud in both; the global historical `standardize:true` field is not evidence that `ParticlePrior.z` enumeration was standardized. The learned-MoG default is intentionally excepted for this image host.

I found no current-cohort transpose12 intensity GAN positive. The specifically bound original-host studies provide contrary evidence: `reports/transfer_suite/calibration/images/README.md` records modes 0/HQ 0 for both ordinary intensity controls; G-half and D-half do not sustain a pass. `reports/transfer_suite/solvability/images/combined/README.md` records no original-architecture intensity GAN pass among its declared loss/prior controls, while residual12 and residual16 pass. Transpose16/24 and doubling the original budget also fail. A supervised MSE control passes transpose12 intensity; it demonstrates representability, not successful adversarial training. These are scoped historical findings, not a proof that transpose12 can never work.

## The complete reference matrix also changes hosts

The historical 22/22 does not supply a current 16-task positive label:

| Current reference subset | Historical positive versus current Forge |
| --- | --- |
| Image stripes/blobs | Archived residual16 versus raw transpose12, as with intensity/bars. |
| Vector unequal mass | Archived batch-distance critic, width96/layers3/Fourier0; current plain `SimpleMLPDiscriminator`64/layers2/Fourier2. |
| Vector unequal width, anisotropic, overlap | Archived declared softplus critics: respectively 128/layers3, additive raw/Fourier with linear skip, and 96/layers3. Current plain MLP substitutes do not implement these cards. |
| All six vectors | Historical finite prior initialized with `init_std=.5`; current learned MoG sigma .025, default location `init_std=1`, named initialization. Two-broad and spiral otherwise retain their basic MLP dimensions. |
| Three native100 tasks | Archived `affine_square_v1`: identity affine G, prior coordinates uniform[-5,5], D Fourier3 with Xavier weights/zero biases. Current `_native` hardcodes MLP G128/layers3 and D Fourier2 with named public initialization and a MoG prior. |
| Five behavioral references | Objectives are retained; named initialization/RNG and shared public component changes still require their own current evidence. |

For the vector cases, current cards themselves omit the historical `research_discriminator` definitions, so the immediate issue is again host selection. Additionally, `_models` only constructs simple MLPs: merely adding a research-discriminator dictionary would silently fail to construct it unless dispatch/preflight is extended. Native model selection is hidden in `_native` rather than explicit task metadata. `docs/initialization.md` and `benchmarks/toy100/train.py:_init_linear` independently warn that replacing the native host's Xavier initialization with the public default-scale orthogonal initialization failed its recorded three native gates.

MoG defaults, component-scoped RNG, public API execution and explicitly changed initialization are intentional Forge protocol choices. Dropping a published architecture profile is a distinct host change and must be named separately. Keep the complete 16-reference denominator and all original evidence. The two newly measured vector pilot passes remain valid for their own frozen tasks/source; they do not certify the remaining hosts or convert an archived positive into a new full-suite positive.

## Minimal next action

First make canonical host selection explicit and shared: one versioned host-profile resolver should distinguish `raw_transfer_defaults` from the published passing profile and emit exact G/D architecture, prior-location initialization, resource, and source/hash declarations. Both task freezing and constructor dispatch should consume that resolved card. Unsupported architecture or initialization declarations must block, not fall back to a simple MLP. Preserve the current tasks and negative receipts under their existing identities.

Then preregister **one 600-update residual16 intensity diagnostic** using the current shared K3P formulation, fixed screening seed/named streams, explicit finite-cloud exception, clean live scoring, and unchanged 24-check/final-five gates. This is an architecture-profile transfer probe, not an exact historical replay or a new known-positive certificate. Before updates, verify constructor shapes/parameter counts and record initial output/logit range. Preserve early losses, G/D/prior gradient/update norms and output range at steps 0/1/5/25 to distinguish early collapse if it recurs; retain the full original 600-step schedule horizon even for early diagnostic observations. Do not alter rates, thresholds, seeds, EMA selection or budgets to manufacture a pass, and do not launch the remaining matrix merely to fill cells.

A pass would justify a subsequent frozen profile audit and deliberately selected next task. A failure would localize the remaining question to current formulation/initialization/RNG/sampling transfer on the historically solvable architecture; it would not establish that the historical positive was false. Neither outcome retroactively qualifies old rows or licenses a blanket positive calibration label.
