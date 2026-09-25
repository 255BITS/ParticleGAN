# Configuration search ledger

All 62 tested configurations are retained, including screened failures. CPU, seed 0, unchanged source/protocol/runtime fingerprint `ec8da4732b918b382069db2f040c918ffbccdfb17f7c507a34d13abf85b8d5c8`. No seed sweep.

Every screening trial trains two-pole, trajectory and ring. The remaining six hosts run when those three pass; the VICReg branch additionally required full eight-mode final coverage before continuing. `MISSING` means untested, never a pass. The initial eight comparison arms run all nine hosts. Shared checks are independent of candidate and are recorded in the main report.

The main [leaderboard](../README.md) contains every completed nine-host run. This ledger includes the incomplete screening trials. Final live weights determine the original bounds; eight-mode coverage and late stability remain separate selection evidence. No best-checkpoint selection, per-toy config or changed threshold.

| Family | Config | Tested toys | Passed tested bounds | Failed screen toys | Live modes | Live HQ | EMA modes / HQ | Tail 8 modes + HQ≥90% | Worst tail HQ |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| bcap-exploration | `locked_shared` | 9/9 | 26/29 | trajectory, mode_hold | 5/8 | 82.30% | 3/8 / 49.49% | 0/5 | 17.43% |
| bcap-exploration | `no_particle_l2` | 9/9 | 28/29 | mode_hold | 8/8 | 74.05% | 8/8 / 100.00% | 0/5 | 57.76% |
| bcap-exploration | `r1_r2_0_1` | 9/9 | 29/29 | None | 7/8 | 100.00% | 5/8 / 65.16% | 0/5 | 16.60% |
| bcap-exploration | `r1_r2_0_1_no_l2` | 9/9 | 28/29 | trajectory | 8/8 | 100.00% | 7/8 / 83.33% | 3/5 | 42.94% |
| bcap-exploration | `b_cap_no_l2_lr_half` | 9/9 | 26/29 | two_pole, trajectory | 7/8 | 92.33% | 7/8 / 83.91% | 1/5 | 58.59% |
| bcap-exploration | `b_cap_no_l2_lr_quarter` | 9/9 | 17/29 | two_pole, trajectory, mode_hold | 5/8 | 75.29% | 3/8 / 49.32% | 0/5 | 40.41% |
| bcap-exploration | `b_cap_no_l2_coeff_2` | 9/9 | 26/29 | trajectory, mode_hold | 6/8 | 65.84% | 8/8 / 100.00% | 0/5 | 42.65% |
| bcap-exploration | `b_cap_no_l2_coeff_5` | 9/9 | 27/29 | mode_hold | 4/8 | 48.12% | 8/8 / 91.38% | 1/5 | 8.33% |
| search_caps | `b_cap_k0_5_c1_0_no_l2` | 3/9 | 1/5 | two_pole, trajectory, mode_hold | 5/8 | 56.84% | 8/8 / 92.07% | 0/5 | 33.33% |
| search_caps | `b_cap_k0_75_c1_0_no_l2` | 3/9 | 4/5 | trajectory | 7/8 | 91.72% | 7/8 / 100.00% | 0/5 | 49.34% |
| search_caps | `b_cap_k1_25_c1_0_no_l2` | 3/9 | 3/5 | mode_hold | 4/8 | 49.83% | 4/8 / 74.63% | 0/5 | 23.88% |
| search_caps | `b_cap_k1_0_c0_5_no_l2` | 3/9 | 2/5 | trajectory, mode_hold | 4/8 | 66.43% | 4/8 / 83.84% | 0/5 | 8.28% |
| search_caps | `b_cap_k0_5_c0_5_no_l2` | 3/9 | 2/5 | two_pole, trajectory, mode_hold | 7/8 | 82.57% | 8/8 / 100.00% | 0/5 | 33.62% |
| search_caps | `b_cap_k0_5_c2_0_no_l2` | 3/9 | 1/5 | two_pole, trajectory, mode_hold | 5/8 | 57.25% | 8/8 / 100.00% | 0/5 | 16.70% |
| search_caps | `b_cap_k0_75_c0_5_no_l2` | 3/9 | 2/5 | trajectory, mode_hold | 6/8 | 82.13% | 5/8 / 92.16% | 0/5 | 26.05% |
| search_caps | `b_cap_k0_75_c2_0_no_l2` | 3/9 | 2/5 | trajectory, mode_hold | 6/8 | 66.36% | 8/8 / 100.00% | 0/5 | 15.77% |
| search_caps | `b_cap_k1_25_c0_5_no_l2` | 3/9 | 3/5 | mode_hold | 4/8 | 39.87% | 6/8 / 83.45% | 0/5 | 39.87% |
| search_caps | `b_cap_k1_25_c2_0_no_l2` | 9/9 | 29/29 | None | 8/8 | 91.75% | 8/8 / 100.00% | 3/5 | 75.22% |
| search_caps | `b_cap_k0_25_c1_0_no_l2` | 3/9 | 1/5 | two_pole, trajectory, mode_hold | 5/8 | 66.31% | 6/8 / 82.30% | 0/5 | 34.06% |
| search_caps | `b_cap_k0_25_c2_0_no_l2` | 3/9 | 1/5 | two_pole, trajectory, mode_hold | 4/8 | 49.24% | 6/8 / 84.06% | 0/5 | 49.24% |
| search_caps | `b_cap_k1_0_c0_25_no_l2` | 3/9 | 3/5 | mode_hold | 6/8 | 74.29% | 7/8 / 100.00% | 0/5 | 33.03% |
| search_caps | `b_cap_k0_75_c5_0_no_l2` | 3/9 | 3/5 | trajectory, mode_hold | 7/8 | 75.49% | 7/8 / 83.91% | 1/5 | 56.91% |
| search_caps | `b_cap_k1_25_c5_0_no_l2` | 3/9 | 3/5 | mode_hold | 4/8 | 41.06% | 8/8 / 100.00% | 0/5 | 41.06% |
| search_lr | `b_cap_k1_25_c2_lr0_75` | 3/9 | 2/5 | trajectory, mode_hold | 4/8 | 57.91% | 6/8 / 100.00% | 0/5 | 49.85% |
| search_lr | `b_cap_k1_25_c2_lr0_85` | 9/9 | 29/29 | None | 8/8 | 100.00% | 8/8 / 100.00% | 3/5 | 41.89% |
| search_lr | `b_cap_k1_25_c2_lr0_9` | 3/9 | 4/5 | mode_hold | 7/8 | 82.98% | 6/8 / 92.16% | 0/5 | 32.64% |
| search_lr | `b_cap_k1_25_c2_lr0_95` | 3/9 | 4/5 | mode_hold | 7/8 | 83.03% | 7/8 / 82.13% | 0/5 | 9.01% |
| search_lr | `b_cap_k1_25_c2_lr1_05` | 3/9 | 3/5 | mode_hold | 5/8 | 57.74% | 7/8 / 100.00% | 0/5 | 34.23% |
| search_lr | `b_cap_k1_25_c2_lr1_1` | 3/9 | 3/5 | mode_hold | 5/8 | 74.05% | 4/8 / 49.61% | 0/5 | 41.55% |
| search_lr | `b_cap_k1_25_c2_lr1_25` | 3/9 | 3/5 | mode_hold | 2/8 | 24.83% | 4/8 / 57.25% | 0/5 | 16.72% |
| search_prior | `bcap_k1p25_c2p0_v0p025` | 3/9 | 5/5 | None | 7/8 | 100.00% | 7/8 / 100.00% | 0/5 | 34.08% |
| search_prior | `bcap_k1p25_c2p0_v0p075` | 3/9 | 4/5 | trajectory | 7/8 | 90.99% | 7/8 / 90.99% | 0/5 | 24.54% |
| search_prior | `bcap_k1p25_c2p0_v0p1` | 3/9 | 2/5 | trajectory, mode_hold | 5/8 | 65.26% | 4/8 / 57.01% | 0/5 | 50.73% |
| search_prior | `bcap_k1p25_c2p0_v0p2` | 3/9 | 3/5 | mode_hold | 6/8 | 74.17% | 8/8 / 100.00% | 0/5 | 65.77% |
| search_prior | `bcap_k1p25_c2p0_v0p01` | 3/9 | 5/5 | None | 7/8 | 92.33% | 7/8 / 84.06% | 0/5 | 16.75% |
| search_prior | `bcap_k1p25_c2p0_v0p5` | 3/9 | 3/5 | mode_hold | 5/8 | 66.04% | 5/8 / 65.77% | 0/5 | 41.72% |
| search_prior | `bcap_k1p25_c2p0_v0p0` | 3/9 | 3/5 | mode_hold | 6/8 | 73.88% | 8/8 / 91.72% | 0/5 | 41.58% |
| search_prior | `bcap_k1p0_c1p0_v0p025` | 3/9 | 2/5 | trajectory, mode_hold | 5/8 | 57.67% | 8/8 / 100.00% | 0/5 | 41.26% |
| search_prior | `bcap_k1p0_c1p0_v0p1` | 3/9 | 4/5 | mode_hold | 7/8 | 84.01% | 8/8 / 100.00% | 0/5 | 48.83% |
| search_prior | `bcap_k1p0_c1p0_v0p2` | 3/9 | 2/5 | trajectory, mode_hold | 5/8 | 74.02% | 7/8 / 91.72% | 0/5 | 49.07% |
| search_prior | `bcap_k1p0_c1p0_v0p5` | 3/9 | 2/5 | trajectory, mode_hold | 5/8 | 48.73% | 8/8 / 100.00% | 0/5 | 48.73% |
| search_prior | `bcap_k1p0_c1p0_v0p0` | 3/9 | 2/5 | trajectory, mode_hold | 4/8 | 57.59% | 7/8 / 100.00% | 0/5 | 23.85% |
| search_losses | `hinge_rp_b_cap_k1_0_c1_0_no_l2` | 3/9 | 3/5 | trajectory, mode_hold | 2/8 | 100.00% | 2/8 / 100.00% | 0/5 | 0.00% |
| search_losses | `hinge_rp_b_cap_k1_25_c2_0_no_l2` | 3/9 | 3/5 | mode_hold | 5/8 | 74.61% | 6/8 / 100.00% | 0/5 | 24.37% |
| search_losses | `hinge_rp_b_cap_k1_0_c2_0_no_l2` | 3/9 | 1/5 | two_pole, trajectory, mode_hold | 6/8 | 82.30% | 6/8 / 82.71% | 0/5 | 50.44% |
| search_losses | `hinge_rp_b_cap_k1_25_c1_0_no_l2` | 3/9 | 2/5 | trajectory, mode_hold | 5/8 | 82.74% | 6/8 / 83.81% | 0/5 | 32.45% |
| search_losses | `lsgan_rp_b_cap_k1_0_c1_0_no_l2` | 3/9 | 3/5 | mode_hold | 4/8 | 58.52% | 5/8 / 92.33% | 0/5 | 58.52% |
| search_losses | `lsgan_rp_b_cap_k1_25_c2_0_no_l2` | 3/9 | 3/5 | mode_hold | 6/8 | 75.10% | 6/8 / 75.10% | 0/5 | 42.16% |
| search_losses | `lsgan_rp_b_cap_k1_0_c2_0_no_l2` | 3/9 | 3/5 | mode_hold | 5/8 | 75.22% | 4/8 / 59.45% | 0/5 | 43.07% |
| search_losses | `lsgan_rp_b_cap_k1_25_c1_0_no_l2` | 3/9 | 3/5 | mode_hold | 5/8 | 65.77% | 6/8 / 83.47% | 0/5 | 48.90% |
| search_refine | `bcap_k1p25_c2p0_lr0p8` | 9/9 | 29/29 | None | 8/8 | 91.67% | 8/8 / 100.00% | 2/5 | 34.40% |
| search_refine | `bcap_k1p25_c2p0_lr0p825` | 3/9 | 2/5 | trajectory, mode_hold | 4/8 | 49.66% | 7/8 / 91.72% | 0/5 | 42.24% |
| search_refine | `bcap_k1p25_c2p0_lr0p875` | 3/9 | 3/5 | mode_hold | 4/8 | 33.67% | 8/8 / 100.00% | 0/5 | 25.46% |
| search_refine | `bcap_k1p25_c1p5_lr0p85` | 3/9 | 2/5 | trajectory, mode_hold | 4/8 | 74.63% | 5/8 / 83.64% | 0/5 | 49.68% |
| search_refine | `bcap_k1p25_c2p5_lr0p85` | 3/9 | 3/5 | mode_hold | 6/8 | 74.19% | 6/8 / 74.61% | 0/5 | 67.38% |
| search_refine | `bcap_k1p25_c3p0_lr0p85` | 9/9 | 29/29 | None | 8/8 | 100.00% | 8/8 / 100.00% | 4/5 | 83.91% |
| search_refine | `bcap_k1p5_c2p0_lr0p85` | 3/9 | 3/5 | mode_hold | 2/8 | 31.54% | 6/8 / 91.67% | 0/5 | 31.54% |
| search_refine | `bcap_k1p5_c2p0_lr1p0` | 3/9 | 3/5 | mode_hold | 5/8 | 75.49% | 8/8 / 100.00% | 0/5 | 50.05% |
| search_strength_fine | `bcap_k1p25_c2p75_lr0p85` | 3/9 | 2/5 | trajectory, mode_hold | 5/8 | 58.30% | 6/8 / 76.12% | 0/5 | 42.43% |
| search_strength_fine | `bcap_k1p25_c3p25_lr0p85` | 3/9 | 4/5 | trajectory | 7/8 | 100.00% | 7/8 / 91.72% | 0/5 | 43.02% |
| search_strength_fine | `bcap_k1p25_c3p5_lr0p85` | 3/9 | 2/5 | trajectory, mode_hold | 4/8 | 75.05% | 6/8 / 91.72% | 0/5 | 66.85% |
| search_strength_fine | `bcap_k1p25_c4p0_lr0p85` | 3/9 | 4/5 | mode_hold | 7/8 | 84.23% | 8/8 / 90.99% | 1/5 | 50.32% |

## Raw evidence and reproduction

Each family contains exact resolved configs and every raw measurement, including live curves and enumeration of the complete twelve-particle output support. The full numerical suite can reproduce all screened settings using the same runner; it will also evaluate the previously untested six hosts.

```bash
python -m benchmarks.locked_shared.baseline \
  --configs reports/behavioral_baseline/search/search_caps/configs.json \
  --reference /path/to/conceptmod --output /tmp/reproduce_cap_search
```

Replace the config path with the desired family below. The main report preserves each original source artifact hash and attaches its source path to every completed row.

- [bcap-exploration configs](bcap-exploration/configs.json) · [raw results](bcap-exploration/results.json)
- [search_caps configs](search_caps/configs.json) · [raw results](search_caps/results.json)
- [search_lr configs](search_lr/configs.json) · [raw results](search_lr/results.json)
- [search_prior configs](search_prior/configs.json) · [raw results](search_prior/results.json)
- [search_losses configs](search_losses/configs.json) · [raw results](search_losses/results.json)
- [search_refine configs](search_refine/configs.json) · [raw results](search_refine/results.json)
- [search_strength_fine configs](search_strength_fine/configs.json) · [raw results](search_strength_fine/results.json)
