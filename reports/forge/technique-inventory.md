# Forge family leaderboard

Recorded passes / required experiments, grouped by family and view. Click a family for its technique, pseudocode and training details; click any count for the experiment results. Each family uses one current selected configuration and source. Earlier runtime cohorts remain on its detail page.

A family is the high-level implementation and formulation. Optimizers, learning rates, momentum and other configuration choices stay within that family. Current benchmarks use the explicit family selection; BCAP uses the selected DualNorm recipe. Historical optimizer variants, formulations and ablations retain their separate evidence on the detail pages. Source and runtime differences remain explicit on the detail pages; this selection grants no new qualification.

The [BCAP leader prior searches](bcap-prior-search/README.md) compare five opt-in mechanisms and fifteen global strengths, with every task measurement, score, target and source-bound repair or regression. Their [interpretation and recommendations](bcap-prior-search/findings.md) explain the results. The selected family configuration and its recorded counts remain unchanged.

Family totals sum the view rows. A shared experiment counts once per view requiring it; these totals measure requirements across views, not unique training runs or scientific rank.

| Family / view | Tier 1 | Tier 2 | Tier 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| **[BCAP](families/bcap-pure.md)** | **[22/22](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[49/116](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[71(*)/152](families/bcap-pure.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [11(*)/23](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [4/4](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [12(*)/30](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [6/6](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [9/21](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [15(*)/29](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [11(*)/24](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [11(*)/24](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [8/19](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [11/22](families/bcap-pure.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[Atlas](families/atlas.md)** | **[0(*)/22](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/atlas.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[0(*)/152](families/atlas.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation) | [0(*)/3](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0(*)/23](families/atlas.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0(*)/4](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0(*)/30](families/atlas.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0(*)/6](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0(*)/29](families/atlas.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0(*)/3](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0(*)/24](families/atlas.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0(*)/3](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0(*)/24](families/atlas.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0(*)/3](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0(*)/22](families/atlas.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[E22](families/e22.md)** | **[0(*)/22](families/e22.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/e22.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/e22.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[0(*)/152](families/e22.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation) | [0(*)/3](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [0(*)/23](families/e22.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [0(*)/4](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [0(*)/30](families/e22.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [0(*)/6](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [0(*)/29](families/e22.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [0(*)/3](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [0(*)/24](families/e22.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [0(*)/3](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [0(*)/24](families/e22.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [0(*)/3](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [0(*)/22](families/e22.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[GAN v3 release 0.7](families/release07-gan-v3.md)** | **[20/22](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20(*)/152](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5(*)/29](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[K3P](families/k3p.md)** | **[20/22](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20(*)/152](families/k3p.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5(*)/29](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[KA2](families/ka2.md)** | **[20/22](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20(*)/152](families/ka2.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5(*)/29](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[R1/R2](families/r1r2.md)** | **[20/22](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20(*)/152](families/r1r2.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](families/r1r2.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](families/r1r2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5(*)/29](families/r1r2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](families/r1r2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](families/r1r2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](families/r1r2.md#cohort-cuda-1bf9d7d34422-quality_coverage) |

Runtime cohorts and actual per-task devices are recorded on the family pages and in receipt provenance.

(*) means at least one required experiment has no recorded execution for the selected configuration and source, including preflight-blocked tests. Recorded PASS and FAIL are both executed results. Attempted ERROR, INVALID or INCOMPLETE results show their cause and execution evidence on the family page and earn no pass credit. Zero recorded passes always displays as 0, including unrun families.

Counts retain recorded verdicts under their original recipe, prior, initialization, budget, serving law and source. Changes to today's test definition and unavailable recorded definitions are shown separately on the family pages and do not add (*); recorded passes grant no new qualification. Required lower tiers must pass before later work is eligible. The declared calibration and eligibility requirements appear on each family page.

Separately scoped cohort views, diagnostic-only views and historical/API studies are available on the family pages and excluded from totals.

Atlas/E22 retain (*) for blocked, unrun tests in their selected main-table cohort. The family pages list the host, prior, serving and parameter-ownership blockers. Their separately scoped policy measurements keep their own results and do not fill these cells.

## Browse by technique tag

Tags describe properties shared by a family's supported variants, not performance or qualification. Configuration-dependent behavior is explained on each family page. Historical pages retain their own scope.

<a name="tag-adaptive-training-policy"></a>

### `adaptive-training-policy`

Uses an observation-driven training policy rather than a fixed update schedule.

- [Atlas](families/atlas.md)
- [E22](families/e22.md)

<a name="tag-adversarial-training"></a>

### `adversarial-training`

Trains a generator against a critic; task-owned auxiliary objectives may also apply.

- [Atlas](families/atlas.md)
- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [E22](families/e22.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)
- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)
- [R1/R2](families/r1r2.md)

<a name="tag-capped-input-gradients"></a>

### `capped-input-gradients`

Penalizes critic input-gradient norms above a threshold on real and generated inputs; this is a soft loss penalty.

- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-constant-learning-rate"></a>

### `constant-learning-rate`

Declared baseline uses constant learning rates; resolved per-role rates remain configuration-specific.

- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)

<a name="tag-critic-gradient-penalty"></a>

### `critic-gradient-penalty`

Uses a penalty on critic input gradients; its formula and strength are configuration-specific.

- [Atlas](families/atlas.md)
- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [E22](families/e22.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)
- [R1/R2](families/r1r2.md)

<a name="tag-historical-cohort"></a>

### `historical-cohort`

A retained historical family with its original source, prior and scoring contracts.

- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-learning-rate-annealing"></a>

### `learning-rate-annealing`

Declared baseline decreases learning rates during training; horizons and floors are configuration-specific.

- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-optimizer-interventions"></a>

### `optimizer-interventions`

Includes formulation-specific interventions around optimizer steps; availability depends on the host and resolved recipe.

- [Atlas](families/atlas.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md) (historical cohort)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md) (historical cohort)
- [BCAP nsgda_global](families/bcap-nsgda-global.md) (historical cohort)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md) (historical cohort)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md) (historical cohort)
- [BCAP sgda](families/bcap-sgda.md) (historical cohort)
- [BCAP with K3P](families/bcap.md) (historical cohort)
- [E22](families/e22.md)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)
- [KA2](families/ka2.md)

<a name="tag-selectable-loss"></a>

### `selectable-loss`

The family explicitly supports multiple adversarial loss formulations.

- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)

<a name="tag-structural-ablation"></a>

### `structural-ablation`

Removes a named mechanism from a parent technique and retains a separate evidence identity.

- [K3P without A2](families/k3p-no-a2.md) (historical cohort)
- [K3P without critic anchor](families/k3p-no-anchor.md) (historical cohort)
- [K3P without critic penalty](families/k3p-no-penalty.md) (historical cohort)
- [K3P without training output noise](families/k3p-no-training-noise.md) (historical cohort)

<a name="tag-tensorflow-v1-adam"></a>

### `tensorflow-v1-adam`

Uses the declared dense TensorFlow-v1 Adam update law with checkpointed application clocks and epsilon before second-moment bias correction.

- [Halloween optimizer/loss transfer](families/halloween.md) (historical cohort)

## Refresh

```sh
python reports/forge/regenerate_technique_inventory.py
```

This regenerates the leaderboard, family pages and experiments-by-tier report from committed evidence and declarations. Register newly measured evidence with `--source-commit <executed-commit>`; advancing the recorded view policy also requires `--advance-policy`.

[Experiments, criteria and tier assignments](EXPERIMENTS_BY_TIER.md) · [Complete numerical publication and provenance](technique-inventory.json)

Publication input digest `3cc0aeaa3d3bb5aaf0c0fd5f73168b3311c39ff4ff99ba96c1d1c78289030102`.
[Performance by test and complete recipe](test-performance.md): compatible passes, failures, execution errors and missing witnesses, with sortable evidence drill-down.

## Scoped non-Atlas discoveries

[Verified reconciled receipts, complete recipes and audit gaps](test-performance-evidence/nonatlas-reconciled-204.json): **204 full runs** across **66 recipe IDs**, **75 source-bound revisions** and **7 source cohorts**. Recorded outcomes are **107 PASS / 97 FAIL**. These are disjoint ledger counts across exact protocols; **no global survivor or pooled score** is established.

**Evidence limits:** six mode-hold transfers lack saved sample panels; raw archives and full-run CUDA packets are not publicly available. Their original logged-metric FAIL grades remain explicit.

<details>
<summary>Open the 204-run cohort table, exact clocks and audit limitations</summary>

| Round | Full PASS / FAIL | Source cohorts | Inclusive attempt wall seconds | Audit |
| --- | ---: | --- | ---: | --- |
| cell-moment-round | 14 / 1 | `f90962a5b351` | 746.970010416 | PASS_INDEPENDENT_SAVED_OUTPUT_AND_BINDING_AUDIT |
| cell-moment-transfer-round | 13 / 2 | `f90962a5b351` | 730.446546641 | PASS_INDEPENDENT_SAVED_OUTPUT_AND_BINDING_AUDIT |
| cell-moment-acquisition-round | 13 / 2 | `a21587094267` | 745.665159237 | PASS_INDEPENDENT_SAVED_OUTPUT_AND_BINDING_AUDIT |
| gaussian-stability-round | 0 / 3 | `a21587094267` | 386.763673001 | PASS_INDEPENDENT_SAVED_OUTPUT_CONTINUITY_AND_BINDING_AUDIT |
| cell-moment-retention-architecture-round | 6 / 6 | `421df4e58168` | 1083.554060910 | PASS_INDEPENDENT_SAVED_OUTPUT_AND_BINDING_AUDIT |
| cell-memory-round | 12 / 12 | `47b8aff9e9ef` | 2487.138272296 | PASS_INDEPENDENT_SAVED_OUTPUT_OWN_CONTINUITY_AND_MEMORY_BINDING_AUDIT |
| cell-quadrature-round | 0 / 12 | `8220bfed22b6` | 3554.116458337 | PASS_INDEPENDENT_SAVED_OUTPUT_AND_BINDING_AUDIT |
| cell-optimizer-round | 12 / 12 | `47b8aff9e9ef` | 2950.390634309 | PASS_INDEPENDENT_SAVED_OUTPUT_OWN_CONTINUITY_AND_MEMORY_BINDING_AUDIT |
| cell-mixed-role-round | 12 / 12 | `d6b7ca5a0d78` | 2919.869744607 | PASS_INDEPENDENT_SAVED_OUTPUT_OWN_CONTINUITY_AND_MEMORY_BINDING_AUDIT |
| cell-gradient-signal-round | 17 / 7 | `11cf56e3462e` | 3236.489829691 | PASS_INDEPENDENT_SAVED_OUTPUT_OWN_CONTINUITY_AND_MEMORY_BINDING_AUDIT |
| cell-signal-transfer-round | 2 / 10 | `11cf56e3462e` | 708.227286190 | PARTIAL_SAVED_SAMPLES_FULL_BINDING_AND_LOGGED_GATE_AUDIT |
| cell-signal-reverse-transfer-round | 3 / 9 | `11cf56e3462e` | 730.770430319 | PARTIAL_SAVED_SAMPLES_FULL_BINDING_AND_LOGGED_GATE_AUDIT |
| cell-signal-moment-transfer-round | 3 / 9 | `11cf56e3462e` | 713.859442260 | PARTIAL_SAVED_SAMPLES_FULL_BINDING_AND_LOGGED_GATE_AUDIT |

The original 45 geometry/transfer/acquisition runs remain indexed once: aggregate 35/36 PASS and five of nine fresh controls PASS, with Bures ring FAIL retained. All cumulative representatives, all 12 memory corrections and all 12 mixed-role recipes fail full Gaussian stability; all 12 quadrature recipes fail mode retention. Gradient attenuation passes five of 12 full Gaussian runs, but all five passing recipes fail essential transfer gates. Forward, reverse and moment transfer cohorts pass 2/12, 3/12 and 3/12; these cannot fill one another's protocols.

Gaussian continuations retain the complete 6,000-step clock and their own pinned 1,000-step producers: only 5,000 new updates and the new attempt charge are counted. Complete recipes expose objective, reference policy, cubature, optimizer, G/D/prior role assignments and signal attenuation. Published/base architectures and source revisions remain separate.

**12 pre-runtime ERROR cells** cost **30.168105392 seconds**, with zero training updates. Three CPU-nonfinite SGDA rejections, semantic dedup exclusions, preparation/audit failures and the CPU gradient probe remain separate diagnostics. The probe's REJECT decision supplies no optimizer or feasibility PASS.

The amended CPU reference ablation keeps primary **REJECT** and secondary posthoc quota **SUPPORT** separate, with original registrations and transparent quota/directional-measurement errata. These add zero trained rows and no qualified recipe or feasibility witness; 4.349590721 timed CPU seconds remain separate from attempt charges.

[Two paired64 cells](test-performance-evidence/nonatlas-finite-realization-prefix64.json) remain **INCOMPLETE** against original 7,000-step staggered100 gates. Full-run wall charge **20994.261548214 seconds**; prefix **20.104729663 seconds**; disjoint full/prefix/failed-supervisor total **21044.534383269 seconds**. CPU preparation/probe costs are recorded separately; no exclusive CUDA-time or cross-protocol speed ranking is claimed.

**Six mode-hold transfer jobs have unavailable saved sample panels**. Their cells explicitly retain original logged-metric gate regrades; score-history GIFs are not sample trajectories. The other 384 saved sample observations per transfer cohort regrade exactly in the supplied audits. Full-run CUDA packets and contemporaneous causal gradient/reference histories were not retained. Source/artifact hashes were verified; bulk archives remain local-only.

Display coverage changes only. Private prototype implementations are not public API releases or adopted defaults; selected family rows, qualification, experiments and training code remain unchanged.

</details>


## Quota-transition and Astra portfolio follow-up

[Verified cumulative receipts and original submetrics](test-performance-evidence/astra-cumulative-24.json): **24 additional full runs**, 8 predecessor quota runs plus 16 Astra runs. The six interim Astra receipts occur once. Family execution ledger: **228 full runs** and **470,400 new accepted updates**; no global survivor.

**Finding:** all twelve Astra hypotheses fail full unequal-mass gates. Three endpoint covariance submetrics pass, but their 1/5 terminal suffixes still fail the original sustained requirement.

<details>
<summary>Open the 24-run follow-up: controls, predictors and separate cost clocks</summary>

| Round | Full PASS / FAIL | Source | Queue seconds | Supervisor wall seconds |
| --- | ---: | --- | ---: | ---: |
| cell-quota-transition-round | 5 / 3 | `9bcc5a8eef26` | 462.898204478 | 480.149192524 |
| astra-quota-followup | 0 / 4 | `db1f08de2d98` | 148.701697541 | 157.336667785 |
| astra-fractional-transport | 0 / 2 | `1adee8703f13` | 65.182648164 | 68.483431482 |
| astra-shape-distribution | 0 / 4 | `05e7cf2d2e6c` | 178.974365511 | 187.800058275 |
| astra-remaining-mechanisms | 0 / 6 | `b95b46ed9444` | 301.378773134 | 312.712553871 |

All **12 Astra hypotheses FAIL** the unchanged full unequal-mass gate. Four fresh source-bound controls are separate paid runs. Swap, slow covariance and lookahead pass the final covariance submetric, but each ends with a one-check suffix against five required checks; their full verdict stays FAIL. The predecessor minimum-change recipe passes Gaussian acquisition/stability and mode retention, but fails unequal mass.

Added work: **36,000 accepted updates** plus **1,200 temporary predictor evaluations**. Lookahead's temporary predictor is not another accepted optimizer update; its work is already included in both clocks. Original horizons, budgets, quotas, assignment/transport semantics, reference policies and covariance-memory changes remain bound.

The 24-run queue charge is **1157.135688828 seconds**; inclusive supervisor wall charge is **1206.481903937 seconds**. These clocks are distinct and are not added together. Inherited budgets and earlier charges are not counted again. Preparation errors and predecessor failed-audit CPU costs remain diagnostic exclusions; unavailable preparation elapsed times stay null.

5 additional source cohorts and all saved-output/state/trace pins were verified. Fractional transport uses row-argmax labels with exact soft capacities; integer assignment counts are not substituted for those labels. Controls' model equality is certified by the supplied audits. Raw arrays, tensor states and traces stay local-only; the import runs no research code, training or metric rescore and grants no qualification.

</details>


## Verified successor packet

[Twenty-one new non-Atlas full runs](test-performance-evidence/successor-nonatlas-21.json) add **9 PASS / 12 FAIL** and **39,000 new updates** to the existing 228-run execution ledger, for **249 non-Atlas full runs**. The cumulative total counts distinct source-bound attempts across recipes; it is not a pooled winner or global qualification.

[Three separate Atlas two-pole diagnostics](test-performance-evidence/atlas-seed0-diagnostics-3.json) add **1 PASS / 2 FAIL** over 240 updates. Native fails, RMS cap passes, and paired-score centering fails at the original 80-update, 24-observation seed-0 screen. These formulations retain distinct source and recipe identities; none establish current MoG, held-out transfer, robustness or release qualification.

The saved request, raw result, original grade and terminal pins were checked against the local packet. Bulk states and observations remain local-only. Earlier 24 packet attempts already indexed by Astra are counted once.


[BCAP initial Adam study](pure-bcap/README.md): five adversarial losses at two constant Adam rates; 3/6 required Tier 1 passes for one selected whole recipe. 109 unique attempts cost 2549.509 paid seconds, counted once across the shared campaign. Executed source cohorts `af75a3fea19aa6e4d1ca2be867b9c50a02931e33`, `44cc66d78495cb913e0ea064840e2de37c8a4ae5`; each candidate keeps its complete cohort. No default adoption.
