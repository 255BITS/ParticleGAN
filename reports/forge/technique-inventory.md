# Forge family leaderboard

Recorded passes / required experiments, grouped by family and view. Click a family for its technique, pseudocode and training details; click any count for the experiment results. Each family uses one current selected configuration and source. Earlier runtime cohorts remain on its detail page.

A family is the high-level implementation and formulation. Optimizers, learning rates, momentum and other configuration choices stay within that family. Current benchmarks use the explicit family selection; BCAP uses the selected DualNorm recipe. Historical optimizer variants, formulations and ablations retain their separate evidence on the detail pages. Source and runtime differences remain explicit on the detail pages; this selection grants no new qualification.

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
| **[GAN v3 release 0.7](families/release07-gan-v3.md)**<br>Recorded source measurement · **stale**; no current qualification | **[20/22](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20(*)/152](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5(*)/29](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](families/release07-gan-v3.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[K3P](families/k3p.md)**<br>Recorded source measurement · **stale**; no current qualification | **[20/22](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/k3p.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20(*)/152](families/k3p.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](families/k3p.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](families/k3p.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5(*)/29](families/k3p.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](families/k3p.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](families/k3p.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](families/k3p.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[KA2](families/ka2.md)**<br>Recorded source measurement · **stale**; no current qualification | **[20/22](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/ka2.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20(*)/152](families/ka2.md#cohort-cuda-1bf9d7d34422)** |
| ↳ [adaptation](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-2) | [0(*)/1](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation-tier-3) | [3(*)/23](families/ka2.md#cohort-cuda-1bf9d7d34422-adaptation) |
| ↳ [clockfree_continuous](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) | [3/4](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-2) | [0(*)/7](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous-tier-3) | [3(*)/30](families/ka2.md#cohort-cuda-1bf9d7d34422-clockfree_continuous) |
| ↳ [discriminator_stability](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) | [5/6](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-1) | [0(*)/21](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-2) | [0(*)/2](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability-tier-3) | [5(*)/29](families/ka2.md#cohort-cuda-1bf9d7d34422-discriminator_stability) |
| ↳ [formulation_comparison](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-2) | [0(*)/2](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison-tier-3) | [3(*)/24](families/ka2.md#cohort-cuda-1bf9d7d34422-formulation_comparison) |
| ↳ [host_profile_transfer](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-2) | [0(*)/2](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer-tier-3) | [3(*)/24](families/ka2.md#cohort-cuda-1bf9d7d34422-host_profile_transfer) |
| ↳ [quality_coverage](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage) | [3/3](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-1) | [0(*)/19](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-2) | [0/0](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage-tier-3) | [3(*)/22](families/ka2.md#cohort-cuda-1bf9d7d34422-quality_coverage) |
| **[R1/R2](families/r1r2.md)**<br>Recorded source measurement · **stale**; no current qualification | **[20/22](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-1)** | **[0(*)/116](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-2)** | **[0(*)/14](families/r1r2.md#cohort-cuda-1bf9d7d34422-tier-3)** | **[20(*)/152](families/r1r2.md#cohort-cuda-1bf9d7d34422)** |
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
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md)
- [BCAP historical configuration: develop integration combined](families/bcap-develop-integration-combined-v1.md)
- [BCAP historical configuration: develop integration winner](families/bcap-develop-integration-winner-v1.md)
- [BCAP historical configuration: three phase cap margin](families/bcap-three-phase-cap-margin-v1.md)
- [BCAP historical configuration: three phase finite cap](families/bcap-three-phase-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability incumbent](families/bcap-tier1-stability-incumbent-v1.md)
- [BCAP historical configuration: tier1 stability projection](families/bcap-tier1-stability-projection-v1.md)
- [BCAP historical configuration: tier1 stability projection global](families/bcap-tier1-stability-projection-global-v1.md)
- [BCAP historical configuration: tier1 stability projection local](families/bcap-tier1-stability-projection-local-v1.md)
- [BCAP historical configuration: tier1 stability repairs cap margin](families/bcap-tier1-stability-repairs-cap-margin-v1.md)
- [BCAP historical configuration: tier1 stability repairs finite cap](families/bcap-tier1-stability-repairs-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability transport](families/bcap-tier1-stability-transport-v1.md)
- [BCAP nsgda_global](families/bcap-nsgda-global.md)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md)
- [BCAP sgda](families/bcap-sgda.md)
- [BCAP with K3P](families/bcap.md)
- [E22](families/e22.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md)
- [K3P without critic anchor](families/k3p-no-anchor.md)
- [K3P without critic penalty](families/k3p-no-penalty.md)
- [K3P without training output noise](families/k3p-no-training-noise.md)
- [KA2](families/ka2.md)
- [R1/R2](families/r1r2.md)

<a name="tag-capped-input-gradients"></a>

### `capped-input-gradients`

Penalizes critic input-gradient norms above a threshold on real and generated inputs; this is a soft loss penalty.

- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md)
- [BCAP historical configuration: develop integration combined](families/bcap-develop-integration-combined-v1.md)
- [BCAP historical configuration: develop integration winner](families/bcap-develop-integration-winner-v1.md)
- [BCAP historical configuration: three phase cap margin](families/bcap-three-phase-cap-margin-v1.md)
- [BCAP historical configuration: three phase finite cap](families/bcap-three-phase-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability incumbent](families/bcap-tier1-stability-incumbent-v1.md)
- [BCAP historical configuration: tier1 stability projection](families/bcap-tier1-stability-projection-v1.md)
- [BCAP historical configuration: tier1 stability projection global](families/bcap-tier1-stability-projection-global-v1.md)
- [BCAP historical configuration: tier1 stability projection local](families/bcap-tier1-stability-projection-local-v1.md)
- [BCAP historical configuration: tier1 stability repairs cap margin](families/bcap-tier1-stability-repairs-cap-margin-v1.md)
- [BCAP historical configuration: tier1 stability repairs finite cap](families/bcap-tier1-stability-repairs-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability transport](families/bcap-tier1-stability-transport-v1.md)
- [BCAP nsgda_global](families/bcap-nsgda-global.md)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md)
- [BCAP sgda](families/bcap-sgda.md)
- [BCAP with K3P](families/bcap.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)

<a name="tag-critic-gradient-penalty"></a>

### `critic-gradient-penalty`

Uses a penalty on critic input gradients; its formula and strength are configuration-specific.

- [Atlas](families/atlas.md)
- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md)
- [BCAP historical configuration: develop integration combined](families/bcap-develop-integration-combined-v1.md)
- [BCAP historical configuration: develop integration winner](families/bcap-develop-integration-winner-v1.md)
- [BCAP historical configuration: three phase cap margin](families/bcap-three-phase-cap-margin-v1.md)
- [BCAP historical configuration: three phase finite cap](families/bcap-three-phase-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability incumbent](families/bcap-tier1-stability-incumbent-v1.md)
- [BCAP historical configuration: tier1 stability projection](families/bcap-tier1-stability-projection-v1.md)
- [BCAP historical configuration: tier1 stability projection global](families/bcap-tier1-stability-projection-global-v1.md)
- [BCAP historical configuration: tier1 stability projection local](families/bcap-tier1-stability-projection-local-v1.md)
- [BCAP historical configuration: tier1 stability repairs cap margin](families/bcap-tier1-stability-repairs-cap-margin-v1.md)
- [BCAP historical configuration: tier1 stability repairs finite cap](families/bcap-tier1-stability-repairs-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability transport](families/bcap-tier1-stability-transport-v1.md)
- [BCAP nsgda_global](families/bcap-nsgda-global.md)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md)
- [BCAP sgda](families/bcap-sgda.md)
- [BCAP with K3P](families/bcap.md)
- [E22](families/e22.md)
- [GAN v3 release 0.7](families/release07-gan-v3.md)
- [GAN v3 release 0.7 (MoG)](families/release07-gan-v3-mog.md) (historical cohort)
- [GAN v3 release 0.7 (cloud)](families/release07-gan-v3-cloud.md) (historical cohort)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md)
- [K3P without critic anchor](families/k3p-no-anchor.md)
- [K3P without training output noise](families/k3p-no-training-noise.md)
- [KA2](families/ka2.md)
- [R1/R2](families/r1r2.md)

<a name="tag-historical-cohort"></a>

### `historical-cohort`

A retained historical family with its original source, prior and scoring contracts.

- [BCAP historical configuration: develop integration combined](families/bcap-develop-integration-combined-v1.md)
- [BCAP historical configuration: develop integration winner](families/bcap-develop-integration-winner-v1.md)
- [BCAP historical configuration: three phase cap margin](families/bcap-three-phase-cap-margin-v1.md)
- [BCAP historical configuration: three phase finite cap](families/bcap-three-phase-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability incumbent](families/bcap-tier1-stability-incumbent-v1.md)
- [BCAP historical configuration: tier1 stability projection](families/bcap-tier1-stability-projection-v1.md)
- [BCAP historical configuration: tier1 stability projection global](families/bcap-tier1-stability-projection-global-v1.md)
- [BCAP historical configuration: tier1 stability projection local](families/bcap-tier1-stability-projection-local-v1.md)
- [BCAP historical configuration: tier1 stability repairs cap margin](families/bcap-tier1-stability-repairs-cap-margin-v1.md)
- [BCAP historical configuration: tier1 stability repairs finite cap](families/bcap-tier1-stability-repairs-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability transport](families/bcap-tier1-stability-transport-v1.md)
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
- [BCAP ada_nsgda](families/bcap-ada-nsgda.md)
- [BCAP dualnorm (experimental starting point)](families/bcap-dualnorm.md)
- [BCAP dualnorm_D_only](families/bcap-dualnorm-d-only.md)
- [BCAP historical configuration: develop integration combined](families/bcap-develop-integration-combined-v1.md)
- [BCAP historical configuration: develop integration winner](families/bcap-develop-integration-winner-v1.md)
- [BCAP historical configuration: three phase cap margin](families/bcap-three-phase-cap-margin-v1.md)
- [BCAP historical configuration: three phase finite cap](families/bcap-three-phase-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability incumbent](families/bcap-tier1-stability-incumbent-v1.md)
- [BCAP historical configuration: tier1 stability projection](families/bcap-tier1-stability-projection-v1.md)
- [BCAP historical configuration: tier1 stability projection global](families/bcap-tier1-stability-projection-global-v1.md)
- [BCAP historical configuration: tier1 stability projection local](families/bcap-tier1-stability-projection-local-v1.md)
- [BCAP historical configuration: tier1 stability repairs cap margin](families/bcap-tier1-stability-repairs-cap-margin-v1.md)
- [BCAP historical configuration: tier1 stability repairs finite cap](families/bcap-tier1-stability-repairs-finite-cap-v1.md)
- [BCAP historical configuration: tier1 stability transport](families/bcap-tier1-stability-transport-v1.md)
- [BCAP nsgda_global](families/bcap-nsgda-global.md)
- [BCAP nsgda_layer](families/bcap-nsgda-layer.md)
- [BCAP particle_rownorm_only](families/bcap-particle-rownorm-only.md)
- [BCAP sgda](families/bcap-sgda.md)
- [BCAP with K3P](families/bcap.md)
- [E22](families/e22.md)
- [K3P](families/k3p.md)
- [K3P without A2](families/k3p-no-a2.md)
- [K3P without critic anchor](families/k3p-no-anchor.md)
- [K3P without critic penalty](families/k3p-no-penalty.md)
- [K3P without training output noise](families/k3p-no-training-noise.md)
- [KA2](families/ka2.md)

<a name="tag-selectable-loss"></a>

### `selectable-loss`

The family explicitly supports multiple adversarial loss formulations.

- [BCAP](families/bcap-pure.md)
- [BCAP](families/bcap-pure-configuration.md)

<a name="tag-structural-ablation"></a>

### `structural-ablation`

Removes a named mechanism from a parent technique and retains a separate evidence identity.

- [K3P without A2](families/k3p-no-a2.md)
- [K3P without critic anchor](families/k3p-no-anchor.md)
- [K3P without critic penalty](families/k3p-no-penalty.md)
- [K3P without training output noise](families/k3p-no-training-noise.md)

## Refresh

```sh
python reports/forge/regenerate_technique_inventory.py
```

This regenerates the leaderboard, family pages and experiments-by-tier report from committed evidence and declarations. Register newly measured evidence with `--source-commit <executed-commit>`; advancing the recorded view policy also requires `--advance-policy`.

[Experiments, criteria and tier assignments](EXPERIMENTS_BY_TIER.md) · [Complete numerical publication and provenance](technique-inventory.json)

Publication input digest `cd7c6acab56a6bb9c3c6912d34bba55c326321c5ff66e3f60ea8e8f6ee70a3fc`.
[BCAP initial Adam study](pure-bcap/README.md): five adversarial losses at two constant Adam rates; 3/6 required Tier 1 passes for one selected whole recipe. 109 unique attempts cost 2549.509 paid seconds, counted once across the shared campaign. Executed source cohorts `af75a3fea19aa6e4d1ca2be867b9c50a02931e33`, `44cc66d78495cb913e0ea064840e2de37c8a4ae5`; each candidate keeps its complete cohort. No default adoption.
