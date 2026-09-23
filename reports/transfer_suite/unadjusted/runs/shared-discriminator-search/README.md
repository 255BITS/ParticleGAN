# Shared cap6 architecture results

**Two additional task witnesses; rare mass and unequal width remain unresolved.** One unchanged formulation and optimizer recipe throughout; architecture selection per case is explicit. The existing reference profile is 15/19. Combining its retained successes with the new anisotropic and overlap architecture witnesses supports 17/19, pending the primary importer. This bundle alone is a six-data study, not a fresh run of all19 or a single-discriminator universal pass.

28 search episodes, 672 live observations, 266.469 total single-CPU episode seconds. One additional exact overlap replay verifies portability. All use seed0; there are no seed sweeps, resource changes, target-derived features or objective additions. Five search episodes pass; all23 failures remain.

Live PASS requires 24 recorded checkpoints and at least5 passing checkpoints at the end. Cells show live status and final passing suffix; missing cases stay untested.

| Architecture | D parameters | two_broad | unequal_mass | unequal_width | anisotropic | overlap | spiral |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| raw_softplus96_l3 | 19009 | [PASS 15/24](completion/episodes/shared_c6__raw_softplus96_l3__vector_two_broad.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__raw_softplus96_l3__vector_unequal_mass.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__raw_softplus96_l3__vector_unequal_width.json.gz) | [FAIL 2/24](cross/episodes/shared_c6__raw_softplus96_l3__vector_anisotropic.json.gz) | [PASS 10/24](cross/episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz) | [PASS 9/24](completion/episodes/shared_c6__raw_softplus96_l3__vector_spiral.json.gz) |
| raw_silu128_l3 | 33537 | [FAIL 0/24](completion/episodes/shared_c6__raw_silu128_l3__vector_two_broad.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_mass.json.gz) | [FAIL 1/24](screen/episodes/shared_c6__raw_silu128_l3__vector_unequal_width.json.gz) | [PASS 6/24](cross/episodes/shared_c6__raw_silu128_l3__vector_anisotropic.json.gz) | [FAIL 4/24](cross/episodes/shared_c6__raw_silu128_l3__vector_overlap.json.gz) | [FAIL 0/24](completion/episodes/shared_c6__raw_silu128_l3__vector_spiral.json.gz) |
| quadratic_softplus96_l2 | 9985 | not tested | [FAIL 0/24](screen/episodes/shared_c6__quadratic_softplus96_l2__vector_unequal_mass.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__quadratic_softplus96_l2__vector_unequal_width.json.gz) | not tested | not tested | not tested |
| quadratic_tanh96_l3 | 19297 | not tested | [FAIL 0/24](screen/episodes/shared_c6__quadratic_tanh96_l3__vector_unequal_mass.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__quadratic_tanh96_l3__vector_unequal_width.json.gz) | [FAIL 4/24](cross/episodes/shared_c6__quadratic_tanh96_l3__vector_anisotropic.json.gz) | [FAIL 0/24](cross/episodes/shared_c6__quadratic_tanh96_l3__vector_overlap.json.gz) | not tested |
| residual_raw_softplus96_l3 | 19009 | not tested | [FAIL 0/24](screen/episodes/shared_c6__residual_raw_softplus96_l3__vector_unequal_mass.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__residual_raw_softplus96_l3__vector_unequal_width.json.gz) | not tested | not tested | not tested |
| residual_lowfreq_softplus96_l3 | 19393 | not tested | [FAIL 0/24](screen/episodes/shared_c6__residual_lowfreq_softplus96_l3__vector_unequal_mass.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__residual_lowfreq_softplus96_l3__vector_unequal_width.json.gz) | not tested | not tested | not tested |
| halfscore_fourier_skip96_l2 | 10467 | not tested | [FAIL 0/24](screen/episodes/shared_c6__halfscore_fourier_skip96_l2__vector_unequal_mass.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__halfscore_fourier_skip96_l2__vector_unequal_width.json.gz) | not tested | not tested | not tested |
| additive_raw_fourier64_l2 | 5796 | not tested | [FAIL 0/24](screen/episodes/shared_c6__additive_raw_fourier64_l2__vector_unequal_mass.json.gz) | [FAIL 0/24](screen/episodes/shared_c6__additive_raw_fourier64_l2__vector_unequal_width.json.gz) | [PASS 8/24](cross/episodes/shared_c6__additive_raw_fourier64_l2__vector_anisotropic.json.gz) | [FAIL 1/24](cross/episodes/shared_c6__additive_raw_fourier64_l2__vector_overlap.json.gz) | not tested |

The completed raw-SiLU128 profile passes only anisotropic (1/6); it loses broad and spiral. The completed raw-Softplus96 profile passes overlap, broad and spiral (3/6). Explicit per-case architecture support is therefore essential to the17/19 accounting. Raw-SiLU128 width ends within all bounds but has only one passing checkpoint; its overlap ends within bounds but has suffix4. Quadratic-Tanh anisotropic also has suffix4. All remain FAIL.

Raw-SiLU128 anisotropic passes with HQ.98364, covariance error.30926 and minimum eigen ratio.36505 (suffix6). The smaller additive raw/Fourier critic also passes anisotropic (5796D parameters, suffix8), but its final minimum eigen ratio.16660 is closer to the.15 bound. Raw-Softplus96 overlap passes with SW.10615, mean error.10610 and covariance error.32536 (suffix10).

Rare-mode minimum variance still fails for every tested card, including nonperiodic critics. These results do not establish periodic features as the sole cause under the higher shared learning rates. No optimizer setting was selected separately by toy.

EMA is recorded independently in every artifact. Raw-SiLU128 EMA passes anisotropic, overlap and spiral, while its live model passes only anisotropic. Raw-Softplus96 live spiral passes but EMA fails. EMA cannot rescue live failures.

[Frozen methodology and card definitions](PROTOCOL.md) · [All episode metadata](index.json) · [Stage1 matrix](screen/README.md) · [Stage2 matrix](cross/README.md) · [Stage3 matrix](completion/README.md) · [SHA256 inventory](inventory.json)

Reproduction from a checkout uses the module command in PROTOCOL.md and the retained stage plan.json files. For a standalone exact replay, extract that stage source.tar.gz, install the versions recorded in protocol.json, and run from outside a repository:

```sh
mkdir /tmp/shared-d-source
tar -xzf cross/source.tar.gz -C /tmp/shared-d-source
PYTHONPATH=/tmp/shared-d-source python reproduce.py \
  cross/episodes/shared_c6__raw_softplus96_l3__vector_overlap.json.gz --output /tmp/overlap-replay.json
```

The replay checks source hashes before training and exact numerical curves, actions and optimizer receipts afterward (timings excluded). The first replay comparator incorrectly compared nested convergence timestamps; its original script/log are retained. The corrected recursive comparator verified the already-retained replay without another training run. The retained overlap-replay.json.gz/log show this was executed from an extracted source bundle. Reference artifact paths are provenance labels; replay uses the canonical original_spec retained in the episode and does not read historical report files.

Code commit: b13c69491a3f3195b8d3f4a214dadc79ba00d0e5, based on78c872236b70f6e5527169db7143559536e052a1. The parent-owned shared_variants.py is included byte-for-byte in source archives but excluded from that commit. Focused tests:11 passed. Numerical sources were frozen before all28 episodes and unchanged throughout.
