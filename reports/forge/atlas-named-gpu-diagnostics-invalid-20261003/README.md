# Named GPU diagnostics: preserved startup failures

**Two INVALID engineering attempts; six adapted questions NOT_RUN.** Both lanes halted before model allocation or optimizer updates. There are no numerical PASS/FAIL results, checkpoints, observations or goal GIFs in this cohort.

Source commit `aee59bea7f1a629d71052fbb010d930650a4ed26`; snapshot `73b00a75863383e4038a3731c8213a7fb072971712a05aab79efa181a93c0280`; protocol `44f2a206e1ea1f4bd0b5f9a89a96ac58a60af2ab95eb5a856da2f1385f5ceffb`. [Exact compact receipt](summary.json) retains task/source/Recipe/runtime and durable supervisor pins.

The fixed pair is Atlas `lr=.0053125`, `prior_lr_mult=1.5`, seed `0`. Each of five separately named families keeps all 26 required questions (Tier 1/2/3: **5/19/2**). Across these separate ledgers there are **2 INVALID and 128 NOT_RUN cells**; the adapted eight comprise 2 INVALID and 6 NOT_RUN. This count does not pool family evidence or provide ordinary qualification, calibration, shipping-default or speed credit.

| Physical GPU | Family | Adapted status | Measured paid | Reserve | Charged |
|---|---|---|---:|---:|---:|
| 0 | `atlas_conditional` | 1 INVALID, 3 NOT_RUN | 12.873334386 s | 0 s | 12.873334386 s |
| 0 | `atlas_ae_routed` | 1 NOT_RUN | 0.000000000 s | 0 s | 0.000000000 s |
| 1 | `atlas_routed` | 1 INVALID | 12.449620655 s | 0 s | 12.449620655 s |
| 1 | `atlas_multibank` | 1 NOT_RUN | 0.000000000 s | 0 s | 0.000000000 s |
| 1 | `atlas_word_joint_min11` | 1 NOT_RUN | 0.000000000 s | 0 s | 0.000000000 s |

Total actual paid and charged: **25.322955040959641 seconds**; conservative reserve: **0 seconds**. Both durable supervisors recorded a completed process with exit code 1, so only measured paid time is charged. This includes startup/teardown. The prospective ceilings remain GPU0 7500 + GPU1 3000 = 10500 seconds; they were not all spent or reserved. Raw adapter/runner subclocks are not added again.

## Exact boundary failure

The planner attached `field_ownership` (a dictionary) and `preflight_blockers` (an empty list) to each task. The recorded task equals its frozen canonical declaration exactly after removing those two inert compiler annotations. The strict producer constructor compared the annotated object to the canonical declaration and rejected it. The observed exception therefore identifies a metadata transport boundary; it supplies no evidence of inadequate representation, failed optimization or a changed scientific objective.

The raw reporter's `execution_path=public_trainer` is a generic post-exception fallback, not evidence that a GANTrainer owned this attempt. The intended task declares `public_components`; no producer ownership receipt or host update was reached. The unchanged ordinary v2 decision contract stays inactive and hash-bound, with ordinary admission BLOCKED. The structural readiness card provides no capacity or learned-quality proof.

### `trajectory_conditional_policy_selected_cloud_v1`

`ValueError: conditional task changed its original objective, host, initializer, gates or resources`

[Byte-identical compact raw error](errors/atlas_conditional-raw-error.json). Frozen producer `experiments/forge/conditional_policy_adapters.py` calls strict validator `experiments/forge/conditional_policy_contracts.py` before allocating models. Recorded CUDA peak allocated/reserved bytes are both zero. Source-bound full traceback and durable supervisor paths/hashes remain in `summary.json`; bulk stdout stays outside Git.

Attempt `9a997cef3257b682601a27ef6754a26becd8357c3fac6b59de03848f819c08bf`; task compatibility `6dbe3fa54a0ab5348e49f28c9a57e899c6ebad836b99f37f5d37f2163f6ae56a`. Physical GPU0: RTX A6000; pre-admission free 45324 MiB, 67 C. Actual runtime Python 3.12.13, Torch 2.13.0+cu126; CPU thread 1, process memory fraction .2.

### `unused_token_hold_routed_policy_selected_cloud_v1`

`ValueError: routed task changed its original objective, host, initialization or gates`

[Byte-identical compact raw error](errors/atlas_routed-raw-error.json). Frozen producer `experiments/forge/routed_policy_adapters.py` calls strict validator `experiments/forge/routed_policy_contracts.py` before allocating models. Recorded CUDA peak allocated/reserved bytes are both zero. Source-bound full traceback and durable supervisor paths/hashes remain in `summary.json`; bulk stdout stays outside Git.

Attempt `adad38e926837b78ad4f58015f7a6a6a1af333a818b2f60471c09e0455a19a56`; task compatibility `5043c3e34f5b6fbc941d177a60c1021119363c4393d8446c8e4127ca47ed8e97`. Physical GPU1: RTX A6000; pre-admission free 31005 MiB, 52 C. Actual runtime Python 3.12.13, Torch 2.13.0+cu126; CPU thread 1, process memory fraction .2.

## Preserved questions and planned budgets

| Family | Original question | Declared updates | Full allowance | Status |
|---|---|---:|---:|---|
| `atlas_conditional` | `trajectory` | 400 | 1800 s | INVALID |
| `atlas_conditional` | `residual_student` | 400 | 1800 s | NOT_RUN |
| `atlas_conditional` | `unipolar` | 400 | 1800 s | NOT_RUN |
| `atlas_conditional` | `mid_scale_identity` | 800 | 1800 s | NOT_RUN |
| `atlas_ae_routed` | `ae_gan_hold` | 250 | 300 s | NOT_RUN |
| `atlas_routed` | `unused_token_hold` | 200 | 300 s | INVALID |
| `atlas_multibank` | `cover_leftover` | 800 | 1800 s | NOT_RUN |
| `atlas_word_joint_min11` | `five_word_joint_acquisition` | 20001 | 900 s | NOT_RUN |

Targets, objectives, gates, full horizons, original 24-check cadence, terminal-five requirements and task-owned sampling laws remain pinned as declarations. None of these gates was evaluated in this cohort. The named AE fixed-width MoG, conditional/routed banks and min11 word resources retain separate ownership/law identities. Original N5 word eligibility remains BLOCKED; unexecuted min11 receives no old-source or resource-equivalence credit.

## Full 26-slot ledgers

The columns below are separate family ledgers. Exact actual variant IDs, parent mappings, task execution/evaluation fingerprints and retained preflight blockers are in `summary.json`. Three downstream family archives were never registered; their ledger states here are planned NOT_RUN, with `state_materialized=false`. They are not fabricated raw study files.

| Parent question | Conditional | AE routed | Unused routed | Multibank | Word min11 |
|---|---|---|---|---|---|
| `two_pole` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `unused_token_hold` | NOT_RUN | NOT_RUN | INVALID | NOT_RUN | NOT_RUN |
| `ae_gan_hold` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `ring16_acquisition` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `five_word_joint_acquisition` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `trajectory` | INVALID | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `residual_student` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `unipolar` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `cover_leftover` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `mid_scale_identity` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `mode_hold` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `vector_two_broad` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `vector_unequal_mass` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `vector_unequal_width` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `vector_anisotropic` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `vector_overlap` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `vector_spiral` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `img_stripes2` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `img_bars4` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `img_blobs4` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `img_intensity2` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `grid100` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `rotated100` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `staggered100` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `ring_hold` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| `ring_extension` | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

The finite dispatcher correctly halted each lane on INVALID and did not retry. Any producer-boundary repair must use a separately frozen source/cohort; these raw failures, paid costs and unreached denominators remain unchanged. No software-test result or scientific success is claimed by this publication.

Publication reads only frozen bytes, JSON, logs and source manifests. It performs no model construction/restore, sampling, grading, training, GPU operation, queue mutation, Git action or artifact hydration. Only two small original raw-error JSONs are copied; checkpoints, tensor dumps and bulk logs remain outside Git.

The [independent implementation audit](independent-review/AUDIT.md) and
[exact audit receipt](independent-review/audit.json) are copied byte for byte
from the separate frozen review. They reproduce the validation mismatch using
metadata only and verify the original source, errors and paid costs.
