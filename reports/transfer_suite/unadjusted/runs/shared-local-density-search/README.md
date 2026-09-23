# Shared-cap6 local-density architecture results

22 architecture cards, 44 full training episodes, 1056 live observations. One unchanged shared_c6 formulation and optimizer recipe throughout. D architecture is the only varying axis; both original 1200-step budgets, 256 particles, batch 128 and all metric thresholds remain fixed. Seed 0 only; no seed experiments.

**Sustained live passes: 0/44 episodes.** This bounded round provides no additional task support unless a row below passes. Final metrics that pass without a five-check suffix remain FAIL. EMA is independent.

| D architecture | Parameters | Rare live / suffix | Rare minimum eigen ratio | Width live / suffix | Width minimum eigen ratio |
| --- | ---: | --- | ---: | --- | ---: |
| local_rbf64_direct | 67 | [FAIL / 0](screen/episodes/shared_c6__local_rbf64_direct__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_rbf64_direct__vector_unequal_width.json.gz) | 0 |
| local_rbf128_direct | 131 | [FAIL / 0](screen/episodes/shared_c6__local_rbf128_direct__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_rbf128_direct__vector_unequal_width.json.gz) | 0 |
| local_rbf256_direct | 259 | [FAIL / 0](screen/episodes/shared_c6__local_rbf256_direct__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_rbf256_direct__vector_unequal_width.json.gz) | 0 |
| local_rbf128_adaptive | 515 | [FAIL / 0](screen/episodes/shared_c6__local_rbf128_adaptive__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_rbf128_adaptive__vector_unequal_width.json.gz) | 0 |
| local_cauchy128_direct | 131 | [FAIL / 0](screen/episodes/shared_c6__local_cauchy128_direct__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_cauchy128_direct__vector_unequal_width.json.gz) | 0 |
| local_rbf128_softplus64 | 12609 | [FAIL / 0](screen/episodes/shared_c6__local_rbf128_softplus64__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_rbf128_softplus64__vector_unequal_width.json.gz) | 0.0359985 |
| local_quad16_width1 | 99 | [FAIL / 0](screen/episodes/shared_c6__local_quad16_width1__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_quad16_width1__vector_unequal_width.json.gz) | 0.000721248 |
| local_quad32_width1 | 195 | [FAIL / 0](screen/episodes/shared_c6__local_quad32_width1__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_quad32_width1__vector_unequal_width.json.gz) | 0 |
| local_quad32_width05 | 195 | [FAIL / 0](screen/episodes/shared_c6__local_quad32_width05__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_quad32_width05__vector_unequal_width.json.gz) | 0 |
| local_quad32_adaptive | 291 | [FAIL / 0](screen/episodes/shared_c6__local_quad32_adaptive__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_quad32_adaptive__vector_unequal_width.json.gz) | 0 |
| local_product_silu64_l2 | 8769 | [FAIL / 0](screen/episodes/shared_c6__local_product_silu64_l2__vector_unequal_mass.json.gz) | 0.00888562 | [FAIL / 0](screen/episodes/shared_c6__local_product_silu64_l2__vector_unequal_width.json.gz) | 0.00315605 |
| local_product_silu96_l2 | 19297 | [FAIL / 0](screen/episodes/shared_c6__local_product_silu96_l2__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_product_silu96_l2__vector_unequal_width.json.gz) | 0.0179285 |
| local_product_silu128_l2 | 33921 | [FAIL / 0](screen/episodes/shared_c6__local_product_silu128_l2__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_product_silu128_l2__vector_unequal_width.json.gz) | 0.0515565 |
| local_squared_silu64_l3 | 8577 | [FAIL / 0](screen/episodes/shared_c6__local_squared_silu64_l3__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_squared_silu64_l3__vector_unequal_width.json.gz) | 0 |
| local_squared_silu96_l3 | 19009 | [FAIL / 0](screen/episodes/shared_c6__local_squared_silu96_l3__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_squared_silu96_l3__vector_unequal_width.json.gz) | 0 |
| local_product_softplus96_l2 | 19297 | [FAIL / 0](screen/episodes/shared_c6__local_product_softplus96_l2__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](screen/episodes/shared_c6__local_product_softplus96_l2__vector_unequal_width.json.gz) | 0.00355356 |
| curvature_raw_silu128_l3_q32_w1p0 | 33729 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_silu128_l3_q32_w1p0__vector_unequal_mass.json.gz) | 0.00348406 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_silu128_l3_q32_w1p0__vector_unequal_width.json.gz) | 0.0014115 |
| curvature_raw_silu128_l3_q64_w1p0 | 33921 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_silu128_l3_q64_w1p0__vector_unequal_mass.json.gz) | 0.0110184 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_silu128_l3_q64_w1p0__vector_unequal_width.json.gz) | 0.00114008 |
| curvature_raw_silu128_l3_q32_w0p5 | 33729 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_silu128_l3_q32_w0p5__vector_unequal_mass.json.gz) | -1.45519e-11 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_silu128_l3_q32_w0p5__vector_unequal_width.json.gz) | 0.00126018 |
| curvature_raw_softplus96_l3_q32_w1p0 | 19201 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q32_w1p0__vector_unequal_mass.json.gz) | 0.0148711 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q32_w1p0__vector_unequal_width.json.gz) | 0.000910163 |
| curvature_raw_softplus96_l3_q64_w1p0 | 19393 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q64_w1p0__vector_unequal_mass.json.gz) | 0.00399649 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q64_w1p0__vector_unequal_width.json.gz) | 0.00221124 |
| curvature_raw_softplus96_l3_q32_w0p5 | 19201 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q32_w0p5__vector_unequal_mass.json.gz) | 0 | [FAIL / 0](refinement/episodes/shared_c6__curvature_raw_softplus96_l3_q32_w0p5__vector_unequal_width.json.gz) | 0.000310319 |

The required minimum component eigen ratio is .15. Fixed radial and standalone local quadratic heads often lose entire modes. Adding local curvature to an initialized raw MLP generally restores occupancy and several distribution metrics, but still leaves a collapsed narrow covariance direction. More explicit local features alone did not repair the two blockers at these shared learning rates.

Initial 16-card screen: 32 episodes. Six approved initially-zero residual-curvature refinements: 12 episodes. Every failed trial, complete live/EMA curve, action trace and actual optimizer receipt is retained. Architecture-only spec diffs and every verdict were recomputed during packaging. The second source archive adds two modules and preserves every byte of the first archive’s numerical source.

Total episode CPU wall time: 466.993s. 25 focused tests passed, including exact initial base-model weights, score and global RNG preservation for residual branches, nonzero cap gradients into the branch and actual unchanged optimizer receipts.

[Initial screen matrix](screen/README.md) · [Refinement matrix](refinement/README.md) · [All episode metadata](index.json) · [Frozen methodology](PROTOCOL.md) · [Validation](validation.json) · [File hashes](inventory.json)

Code commit: 19bf7b8c6b196d1c058a95f7c26903e690d071ba, based on f10cfb1b025aa6c843b61ea77da5f474444bc7cc. No production components or shared runner were edited.

For the primary importer, use screen/index.json and refinement/index.json separately; each has its required sibling protocol.json and source.tar.gz. The aggregate index.json is for browsing and has stage-prefixed artifact links; it is not a primary-import input. Original compressed bytes are unchanged. Source archives contain exact executed code; protocol.json records Python/Torch/build/CPU details. Inventory lists original and compressed SHA256.

To rerun a whole stage from the checkout, use its plan.json with the module in PROTOCOL.md. To reproduce one retained episode without external historical report files, extract that stage source.tar.gz, then run:

```sh
mkdir /tmp/local-density-source
tar -xzf refinement/source.tar.gz -C /tmp/local-density-source
PYTHONPATH=/tmp/local-density-source OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python reproduce.py \
  refinement/episodes/shared_c6__curvature_raw_silu128_l3_q64_w1p0__vector_unequal_mass.json.gz \
  --output /tmp/local-density-replay.json
```

The helper checks source hashes first, then compares numerical results, curves, actions, receipts and verdicts exactly, excluding runtime fields. It is supplied for reproduction but was not executed in this search round; the 44 retained episodes are the complete GAN-run count.
