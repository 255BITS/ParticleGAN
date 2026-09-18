# TransGAN preflight and cost estimate

Three512-step scratch runs completed through the experiment pipeline. All certified. These are stability/performance checks with no FID measurement. Ten tests passed, including full-state8vs4+4 resume for both transformer reconstruction routes.

| Arm | G parameters | Updates/s after warmup | Peak allocated GiB | Estimated50k training hours | Live pixels saturated at±0.99 |
|---|---:|---:|---:|---:|---:|
| transgan_all | 18,851,023 | 2.92 | 11.06 | 4.75 | 0.0000% |
| transgan_e_only | 18,851,023 | 3.03 | 10.50 | 4.58 | 0.0000% |
| cnn_e_only | 645,123 | 23.53 | 1.55 | 0.59 | 0.0008% |

Recommendation: launch the three50k scouts at this transformer size. Both transformer arms retain nontrivial variation across256 fixed prior samples and have no saturated pixels in live or EMA output probes. This establishes that the immediate failure was resolved; it does not establish good FID or guarantee longer-term stability.

The original bounded-head port saturated during the initial preflight. Corrected upstream Linear initialization was necessary for fidelity but insufficient for stable training here. Final token LayerNorm plus RGB Xavier gain0.1 resolved immediate saturation without changing learning rates. These adaptations are shared by both transformer arms and explicitly differ from the unbounded official output head. Earlier failed preflight evidence is retained.

Transformer initial weights match between routing arms. D/E initialization matches across allthree arms; the CNN full initialization matches the historical hash exactly. Data/prior training RNG end states match, lazy bcap remainsN8×8, frozen features and sigma remain unchanged, optimizer counters equal512, and checkpoint hashes are unchanged after audit.

Training estimates exclude five FID50k evaluations, test10k reconstruction diagnostics, startup and checkpoint writes. At one worker perGPU, both transformers start together and the CNN follows the first completion. Allow roughly5–6hours for the whole queue and aboutonehour for the first transformer FID measurement. These are short-preflight estimates, not a runtime guarantee.

Tail: `tail -F runs/cifar_particle_ae/transgan_scout/PIPELINE.log`.

Raw validation: `VALIDATION.json`. Tests: `TESTS.txt`. Architecture and interpretation: `PLAN.md`.
