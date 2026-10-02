# PR239–243: generated caption-edit questions

All five questions are useful, distinct **4/5 definitions**, merged into develop
at `038c399376f3c22c49a86a225a02b16fa3c643d5` after their exact-head CI checks
passed. The shared-head baseline fails; the four untied-head variants pass their
original fixed endpoint gates. These are generated velocity-prediction tasks,
with numerical goals and actual-training GIFs. They do not demonstrate decoded
image quality, full Supra transfer, or a generally qualified config.

| PR | Question the test attempts to falsify | Original result and reason | Why retain it / goal media |
| --- | --- | --- | --- |
| [239](https://github.com/255BITS/ParticleGAN/pull/239) | Can a shared-head routed particle adapter beat ordinary BF16 LoRA on held-out physical error while protecting each source, using useful codes and live routing? | **FAIL:** particle RMSE 0.00650694 vs ordinary 0.00648992, 0.2623% worse; five sources exceed the harm bound. Code benefit and bank/router activity pass. | Failed baseline separates live, useful codes from an actual accuracy win. [Six-state goal GIF](../../../../docs/e22_routed_caption_accuracy/goal.gif), [protocol/results](../../../../docs/e22_routed_caption_accuracy.md). |
| [240](https://github.com/255BITS/ParticleGAN/pull/240) | Does separating main and particle output heads beat both ordinary LoRA and shared heads while retaining all code, source and liveness controls? | **PASS:** untied RMSE 0.00644943; 0.6239% / 0.8838% better than ordinary / shared, with every source improved. | Tests the shared-head bottleneck hypothesis. Extra 82,944 parameters and separate BF16 rounding prevent attributing the gain to one cause. [Six-state goal GIF](../../../../docs/e22_routed_caption_untied/observed-training.gif), [protocol/results](../../../../docs/e22_routed_caption_untied.md). |
| [241](https://github.com/255BITS/ParticleGAN/pull/241) | Does replacing the fixed .04 normalization floor with untrained FIT coordinate standard deviations remove the untied-head advantage? | **PASS:** untied RMSE 0.00639454; 2.2733% / 2.0083% better than the matched ordinary / shared controls. | Tests whether the earlier advantage depends on arbitrary residual units. [Six-state goal GIF](../../../../docs/e22_routed_caption_normalization/goal.gif), [protocol/results](../../../../docs/e22_routed_caption_normalization.md). |
| [242](https://github.com/255BITS/ParticleGAN/pull/242) | Does correlating Gaussian latents across time, preserving their theoretical marginals and recomputing FIT units, remove the advantage? | **PASS:** untied RMSE 0.00640259; 2.5579% / 2.4864% better than the matched controls. | Tests dependence on independently generated time points; the derived normalization change is explicit. [Six-state goal GIF](../../../../docs/e22_routed_caption_flow/goal.gif), [protocol/results](../../../../docs/e22_routed_caption_flow.md). |
| [243](https://github.com/255BITS/ParticleGAN/pull/243) | Does matching 29 frozen-host parameter sampling distributions to retained tensor means and population standard deviations remove the advantage? | **PASS:** untied RMSE 0.73716391; 4.9304% / 1.1039% better than the matched controls. | Tests sensitivity to initialization amplitude. It copies scalar sampling moments, not learned directions, covariance or weights. [Six-state goal GIF](../../../../docs/media/e22_routed_caption_frozen_stats/goal.gif), [protocol/results](../../../../docs/e22_routed_caption_frozen_stats.md). |

The 4/5 ratings assess the bounded questions, independently of the scientific
PASS/FAIL. No row is a duplicate: each changes a substantive proposed explanation
of the generated-fixture advantage. Passing a relative endpoint gate does not
prove zero error or sustained convergence. Errors across different host,
normalization and latent laws cannot be pooled; PR243's absolute RMSE especially
uses a different amplitude law.

## Numerical contract and visual meaning

Every protocol fixes physical GPU 0, 512 updates per arm and a 300-second
startup-through-final-write budget. PR239 has two arms; each later variant has
three. Matched caller data, Gaussian and native penalty streams were checked
through all 512 updates. Training uses the public initialization, `get_recipe`,
`Recipe` loss/optimizer factories, `E22Policy`, `RoutedRows` and checkpoint
contracts. Held-out physical accuracy does not drive training, structural guards,
stopping or checkpoint selection.

The terminal numerical gate requires at least 0.1% aggregate RMSE improvement
over the declared control, with no source harmed by more than 1e-6. Untied
variants must beat both controls. Removing codes must worsen aggregate RMSE by
at least 0.1% and worsen every source; required bank/router activity must cover
at least 90% of the 511 eligible updates, with finite positive gradients at the
required site heads. The scorer uses all 48 TEST contexts, eight per source.

The GIFs show saved physical velocity-error heatmaps beside a zero-error target,
with fixed initial-only scales. Six selected source cameras illustrate the goal;
the full numerical gate uses all TEST contexts. Frames correspond to actual
states at updates 0, 64, 128, 256, 384 and 512. They are error views, not decoded
caption images or a claim that every output reaches the target.

## Separate actual-caption transfer evidence

PR240 also retains a [three-state actual-caption GIF](../../../../docs/e22_routed_untied_caption/observed-training.gif)
and [depth-one Supra readout](../../../../docs/e22_routed_untied_caption_results.md).
That 6,400-update attempt **FAILS**: absolute gain 0.00020536 misses the required
0.0005, and source harms remain despite 0.407% mean improvement. Its frames are
updates 0, 5120 and 6400. The earlier ordinary/tied controls are cached histories
with terminal TEST240 prediction replay; they are not independently complete
training qualifications. The original full-Supra top score remains unbeaten.

This result depends on external frozen assets and executable sources not shipped
in the PR. It supplies context and a sixth GIF, not an extra portable question.
The five primary questions add five rows to the catalog: **119 questions and
187 GIFs** overall. Original failed transfers and reduction-error history remain.

## Reproduce a declared portable test

Use a fresh exclusive output directory from the repository root. For example:

```sh
timeout 300s env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python examples/e22_routed_caption_untied.py --run \
  --out runs/caption-untied-new
```

The corresponding modules are `e22_routed_caption_accuracy.py`,
`e22_routed_caption_normalization.py`, `e22_routed_caption_flow.py` and
`e22_routed_caption_frozen_stats.py`. Their linked protocols freeze bounds,
sampling laws and reproduction commands. Keep the external watchdog: cooperative
checks cannot interrupt a stalled native call. Timeout exits nonzero and is
incomplete. Source/package identity reads occur before the handled try; missing
bound files can exit 1 without a completed receipt. Interpret a scientific FAIL
only when a complete, identity-bound terminal receipt exists. Preflight and short
software checks are not quality PASS results.

These examples accept fixed E22/E22_routed recipes and their declared overrides.
`--protocol` validates the fixed task/gates; it is not a config override. There is
no arbitrary `--recipe` or Atlas/KA2 config selection. Their cloud/R2 bank and
clean-live sampling law cannot fill a learned-MoG or other Forge task. Use the
[read-only Forge selector](../../CONFIG_SELECTION_READINESS.md) for current
config requirements; a future generic comparison needs a separately frozen
policy-aware task and fair config protocol.

## Independent review

[PR239/240 review](pr239-240-review.json) and its [artifact bindings](pr239-240-bindings.json)
checked source/native/raw/media hashes, full stream correspondence, controls,
media meaning and exact-head CI. Forty focused software checks passed in 3.08
seconds, including eight tiny replay updates; no scientific campaign was rerun.
[PR241–243 review](pr241-243-review.json) independently reduced saved terminal
metrics within 2e-15 and checked all source/fixture/raw/media identities. Twelve
selected software checks passed; they performed no native training updates.
The proofs record the pre-merge snapshot; the integration commit above records
the subsequent authorized merge. Raw logs and checkpoints remain outside Git.
