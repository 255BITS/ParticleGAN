# Focused rare-mode architecture search

**1 sustained rare-mode winner among 58 new discriminator architectures.** 5 additional data-toy cross-checks. All runs retain Rp logistic, b_cap3 / κ1.25, prior regularization .05, no particle L2, Adam(0,.99), G LR .001, D LR .0015, prior LR .01, cosine, the original G, 256 particles, batch128 and 1,200 rare-toy updates. Architecture may vary by toy.

A final PASS is insufficient: every metric must pass at all five late checks. The spread ratio is the smallest covariance eigenvalue relative to the target, across all four components. It detects a component flattened into a line even when occupancy and sample quality look good. EMA is measured separately and never determines the live result.

**The winner is linear_skip_d96_beta5: D96×2, Softplus(beta5), Fourier2, plus a learned raw-coordinate linear skip initialized to zero.** Its two-parameter skip brings D to 10,467 parameters. Live bounds hold from update 950 through 1,200, confirm at 1,150, and pass 6 final checks. This supplies the missing rare-toy witness: the same formulation now supports **9/9 required + 10/10 practical = 19/19 behavioral toys**. Different toys use different architectures; the winning D itself passes 3/6 data toys.

The winner's EMA has only 4 final passing checks and remains **FAIL** under the same sustained rule. Live weights supply this PASS.

[Reproduction command and reusable discriminator](../../../benchmarks/transfer_suite/linear_skip_research.md) · [Independent exact replay](winner_replay/parity.json.gz) · [CLI exact numerical replay](cli_replay/parity.json.gz).

| Architecture family | Candidates | Sustained live passes | Closest live candidate | Final passing streak / required 5 | Sustained EMA passes (separate) |
| --- | ---: | ---: | --- | ---: | ---: |
| linear_refinement | 4 | 1 | linear_skip_d96_beta5 | 6 | 0 |
| skip | 8 | 0 | fourier_softplus5_linear_skip | 4 | 0 |
| softplus | 10 | 0 | softplus6_d96_l2_f2 | 1 | 0 |
| layernorm | 6 | 0 | ln_pre_silu_d64 | 0 | 0 |
| rbf | 8 | 0 | rbf128_full_softplus5 | 0 | 0 |
| spectral | 10 | 0 | b5_fourier05 | 0 | 0 |
| ensemble | 4 | 0 | ensemble4_softplus6_d96 | 0 | 0 |
| frequency | 8 | 0 | b5_raw_only | 0 | 0 |

<details><summary>Every architecture, including all failures</summary>

| D architecture | Family | Sustained live | Final passing streak / required 5 | Final spread / minimum .15 | Final failing metrics |
| --- | --- | --- | ---: | ---: | --- |
| [linear_skip_d96_beta5](linear_refinement/screen/episodes/linear_skip_d96_beta5__vector_unequal_mass.json.gz) | linear_refinement | PASS | 6 | 0.2698 | All final bounds pass |
| [fourier_softplus5_linear_skip](skip/screen/episodes/fourier_softplus5_linear_skip__vector_unequal_mass.json.gz) | skip | FAIL | 4 | 0.4909 | All final bounds pass |
| [softplus6_d96_l2_f2](softplus/episodes/softplus6_d96_l2_f2__vector_unequal_mass.json.gz) | softplus | FAIL | 1 | 0.2080 | All final bounds pass |
| [ln_pre_silu_d64](layernorm/episodes/ln_pre_silu_d64__vector_unequal_mass.json.gz) | layernorm | FAIL | 0 | 0.1546 | component_covariance_error |
| [rbf128_full_softplus5](rbf/screen/episodes/rbf128_full_softplus5__vector_unequal_mass.json.gz) | rbf | FAIL | 0 | 0.1312 | component_min_eigen_ratio |
| [b5_fourier05](spectral/episodes/b5_fourier05__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.1275 | component_min_eigen_ratio |
| [softplus5_d80_l2_f2](softplus/episodes/softplus5_d80_l2_f2__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.1148 | component_min_eigen_ratio |
| [fourier_residual2_softplus5_quadratic_skip](skip/screen/episodes/fourier_residual2_softplus5_quadratic_skip__vector_unequal_mass.json.gz) | skip | FAIL | 0 | 0.1054 | component_min_eigen_ratio |
| [linear_skip_d64_beta6](linear_refinement/screen/episodes/linear_skip_d64_beta6__vector_unequal_mass.json.gz) | linear_refinement | FAIL | 0 | 0.3406 | mass_tv, component_covariance_error |
| [rbf64_full_softplus5](rbf/screen/episodes/rbf64_full_softplus5__vector_unequal_mass.json.gz) | rbf | FAIL | 0 | 0.0448 | component_min_eigen_ratio |
| [b5_raw4](spectral/episodes/b5_raw4__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.3243 | component_covariance_error |
| [ensemble4_softplus6_d96](ensemble/episodes/ensemble4_softplus6_d96__vector_unequal_mass.json.gz) | ensemble | FAIL | 0 | 0.0375 | component_min_eigen_ratio |
| [b6_fourier05](spectral/episodes/b6_fourier05__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.0359 | component_min_eigen_ratio |
| [b5_raw_only](frequency/episodes/b5_raw_only__vector_unequal_mass.json.gz) | frequency | FAIL | 0 | 0.0297 | component_min_eigen_ratio |
| [softplus4_d96_l2_f2](softplus/episodes/softplus4_d96_l2_f2__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.0281 | component_min_eigen_ratio |
| [b5_harmonic2_025](spectral/episodes/b5_harmonic2_025__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.0214 | component_min_eigen_ratio |
| [ensemble2_softplus6_d96](ensemble/episodes/ensemble2_softplus6_d96__vector_unequal_mass.json.gz) | ensemble | FAIL | 0 | 0.0197 | component_min_eigen_ratio |
| [fourier_softplus5_quad_mlp_skip](skip/screen/episodes/fourier_softplus5_quad_mlp_skip__vector_unequal_mass.json.gz) | skip | FAIL | 0 | 0.0184 | component_min_eigen_ratio |
| [b6_frequency0p125](frequency/episodes/b6_frequency0p125__vector_unequal_mass.json.gz) | frequency | FAIL | 0 | 0.0176 | component_min_eigen_ratio |
| [b6_harmonic2_025](spectral/episodes/b6_harmonic2_025__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.0165 | component_min_eigen_ratio |
| [b5_fourier025](spectral/episodes/b5_fourier025__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.0137 | component_min_eigen_ratio |
| [softplus4_d112_l2_f2](softplus/episodes/softplus4_d112_l2_f2__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.0126 | component_min_eigen_ratio |
| [b5_harmonic2_05](spectral/episodes/b5_harmonic2_05__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.0086 | component_min_eigen_ratio |
| [b5_frequency0p125](frequency/episodes/b5_frequency0p125__vector_unequal_mass.json.gz) | frequency | FAIL | 0 | 0.0072 | component_min_eigen_ratio |
| [b5_raw2_fourier05](spectral/episodes/b5_raw2_fourier05__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.0065 | component_min_eigen_ratio |
| [rbf128_full_silu](rbf/screen/episodes/rbf128_full_silu__vector_unequal_mass.json.gz) | rbf | FAIL | 0 | 0.0056 | component_min_eigen_ratio |
| [b6_raw_only](frequency/episodes/b6_raw_only__vector_unequal_mass.json.gz) | frequency | FAIL | 0 | 0.0054 | component_min_eigen_ratio |
| [ensemble4_softplus5_d96](ensemble/episodes/ensemble4_softplus5_d96__vector_unequal_mass.json.gz) | ensemble | FAIL | 0 | 0.0061 | component_covariance_error, component_min_eigen_ratio |
| [fourier_softplus5_raw_mlp_skip](skip/screen/episodes/fourier_softplus5_raw_mlp_skip__vector_unequal_mass.json.gz) | skip | FAIL | 0 | 0.0043 | component_min_eigen_ratio |
| [softplus5_d112_l2_f2](softplus/episodes/softplus5_d112_l2_f2__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.0036 | component_min_eigen_ratio |
| [linear_skip_d96_beta6](linear_refinement/screen/episodes/linear_skip_d96_beta6__vector_unequal_mass.json.gz) | linear_refinement | FAIL | 0 | 0.0033 | component_min_eigen_ratio |
| [raw_softplus5_quadratic_skip](skip/screen/episodes/raw_softplus5_quadratic_skip__vector_unequal_mass.json.gz) | skip | FAIL | 0 | 0.0032 | component_min_eigen_ratio |
| [raw_residual2_silu_quadratic_skip](skip/screen/episodes/raw_residual2_silu_quadratic_skip__vector_unequal_mass.json.gz) | skip | FAIL | 0 | 0.0020 | component_min_eigen_ratio |
| [softplus8_d96_l2_f2](softplus/episodes/softplus8_d96_l2_f2__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.0000 | component_min_eigen_ratio |
| [fourier_silu_quadratic_skip](skip/screen/episodes/fourier_silu_quadratic_skip__vector_unequal_mass.json.gz) | skip | FAIL | 0 | 0.0004 | component_covariance_error, component_min_eigen_ratio |
| [b5_raw2](spectral/episodes/b5_raw2__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.0000 | component_min_eigen_ratio, min_mass_ratio |
| [b6_raw2](spectral/episodes/b6_raw2__vector_unequal_mass.json.gz) | spectral | FAIL | 0 | 0.0000 | component_min_eigen_ratio, min_mass_ratio |
| [ensemble2_softplus5_d96](ensemble/episodes/ensemble2_softplus5_d96__vector_unequal_mass.json.gz) | ensemble | FAIL | 0 | 0.0000 | component_min_eigen_ratio, min_mass_ratio |
| [rbf64_full_silu](rbf/screen/episodes/rbf64_full_silu__vector_unequal_mass.json.gz) | rbf | FAIL | 0 | 0.0019 | sw1_normalized, mass_tv, component_min_eigen_ratio |
| [rbf64_residual2_silu](rbf/screen/episodes/rbf64_residual2_silu__vector_unequal_mass.json.gz) | rbf | FAIL | 0 | 0.0091 | component_covariance_error, component_min_eigen_ratio |
| [rbf32_fixed_silu](rbf/screen/episodes/rbf32_fixed_silu__vector_unequal_mass.json.gz) | rbf | FAIL | 0 | 0.0070 | component_covariance_error, component_min_eigen_ratio |
| [softplus2_d96_l2_f2](softplus/episodes/softplus2_d96_l2_f2__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.0150 | component_covariance_error, component_min_eigen_ratio |
| [softplus6_d80_l2_f2](softplus/episodes/softplus6_d80_l2_f2__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.3390 | component_covariance_error |
| [fourier_softplus5_quadratic_skip](skip/screen/episodes/fourier_softplus5_quadratic_skip__vector_unequal_mass.json.gz) | skip | FAIL | 0 | 0.5854 | component_covariance_error |
| [softplus5_d96_l2_f3](softplus/episodes/softplus5_d96_l2_f3__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.8488 | component_covariance_error |
| [rbf64_centers_silu](rbf/screen/episodes/rbf64_centers_silu__vector_unequal_mass.json.gz) | rbf | FAIL | 0 | 0.0000 | sw1_normalized, mass_tv, component_min_eigen_ratio, min_mass_ratio |
| [ln_pre_softplus_d96](layernorm/episodes/ln_pre_softplus_d96__vector_unequal_mass.json.gz) | layernorm | FAIL | 0 | 0.0000 | component_covariance_error, component_min_eigen_ratio, min_mass_ratio |
| [ln_pre_silu_d96](layernorm/episodes/ln_pre_silu_d96__vector_unequal_mass.json.gz) | layernorm | FAIL | 0 | 0.0000 | sw1_normalized, component_min_eigen_ratio, min_mass_ratio |
| [linear_skip_d64_beta10](linear_refinement/screen/episodes/linear_skip_d64_beta10__vector_unequal_mass.json.gz) | linear_refinement | FAIL | 0 | 0.0475 | component_covariance_error, component_min_eigen_ratio |
| [b5_frequency0p25](frequency/episodes/b5_frequency0p25__vector_unequal_mass.json.gz) | frequency | FAIL | 0 | 0.0015 | sw1_normalized, mass_tv, component_covariance_error, component_min_eigen_ratio |
| [softplus5_d96_l2_f1](softplus/episodes/softplus5_d96_l2_f1__vector_unequal_mass.json.gz) | softplus | FAIL | 0 | 0.0078 | component_covariance_error, component_min_eigen_ratio |
| [rbf64_fixed_silu](rbf/screen/episodes/rbf64_fixed_silu__vector_unequal_mass.json.gz) | rbf | FAIL | 0 | 0.0048 | component_covariance_error, component_min_eigen_ratio |
| [ln_post_softplus_d64](layernorm/episodes/ln_post_softplus_d64__vector_unequal_mass.json.gz) | layernorm | FAIL | 0 | 0.0000 | sw1_normalized, component_covariance_error, component_min_eigen_ratio, min_mass_ratio |
| [b6_frequency0p25](frequency/episodes/b6_frequency0p25__vector_unequal_mass.json.gz) | frequency | FAIL | 0 | 0.0000 | sw1_normalized, mass_tv, component_covariance_error, component_min_eigen_ratio, min_mass_ratio |
| [ln_post_softplus_d96](layernorm/episodes/ln_post_softplus_d96__vector_unequal_mass.json.gz) | layernorm | FAIL | 0 | 0.0000 | sw1_normalized, mass_tv, component_covariance_error, component_min_eigen_ratio, min_mass_ratio |
| [ln_pre_softplus_d64](layernorm/episodes/ln_pre_softplus_d64__vector_unequal_mass.json.gz) | layernorm | FAIL | 0 | 0.0000 | sw1_normalized, mass_tv, component_covariance_error, component_min_eigen_ratio, min_mass_ratio |
| [b5_frequency0p5](frequency/episodes/b5_frequency0p5__vector_unequal_mass.json.gz) | frequency | FAIL | 0 | 0.0000 | sw1_normalized, mass_tv, component_min_eigen_ratio, min_mass_ratio |
| [b6_frequency0p5](frequency/episodes/b6_frequency0p5__vector_unequal_mass.json.gz) | frequency | FAIL | 0 | 0.0000 | sw1_normalized, mass_tv, component_covariance_error, component_min_eigen_ratio, min_mass_ratio |

</details>

## Five late measurements

Closest candidates by sustained result, final passing streak, then final normalized bound shortfall. All 24 observations and every failure are retained in the machine-readable report.

| Architecture | Metric | 1,000 | 1,050 | 1,100 | 1,150 | 1,200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| linear_skip_d96_beta5 | component_min_eigen_ratio | 0.5567 | 0.3669 | 0.5543 | 0.2684 | 0.2698 |
| linear_skip_d96_beta5 | component_covariance_error | 0.4057 | 0.4532 | 0.3522 | 0.4452 | 0.4413 |
| linear_skip_d96_beta5 | hq | 0.9922 | 0.9851 | 1.0000 | 0.9919 | 0.9971 |
| linear_skip_d96_beta5 | mass_tv | 0.0468 | 0.0468 | 0.0468 | 0.0468 | 0.0468 |
| linear_skip_d96_beta5 | min_mass_ratio | 0.9149 | 0.9149 | 0.9149 | 0.9149 | 0.9149 |
| fourier_softplus5_linear_skip | component_min_eigen_ratio | 0.0388 | 0.4212 | 0.4430 | 0.4981 | 0.4909 |
| fourier_softplus5_linear_skip | component_covariance_error | 0.6274 | 0.5805 | 0.5482 | 0.6768 | 0.5576 |
| fourier_softplus5_linear_skip | hq | 0.9885 | 0.9875 | 0.9875 | 0.9795 | 0.9802 |
| fourier_softplus5_linear_skip | mass_tv | 0.1107 | 0.1107 | 0.1107 | 0.1107 | 0.1107 |
| fourier_softplus5_linear_skip | min_mass_ratio | 0.8314 | 0.8314 | 0.8314 | 0.8314 | 0.8314 |
| softplus6_d96_l2_f2 | component_min_eigen_ratio | 0.1059 | 0.1326 | 0.2028 | 0.1350 | 0.2080 |
| softplus6_d96_l2_f2 | component_covariance_error | 0.5778 | 0.6429 | 0.6617 | 0.5638 | 0.5578 |
| softplus6_d96_l2_f2 | hq | 0.9883 | 0.9895 | 0.9824 | 0.9932 | 0.9932 |
| softplus6_d96_l2_f2 | mass_tv | 0.0150 | 0.0150 | 0.0150 | 0.0150 | 0.0150 |
| softplus6_d96_l2_f2 | min_mass_ratio | 0.9761 | 0.9761 | 0.9761 | 0.9761 | 0.9761 |

## Full six-data profiles for rare-mode winners

| Architecture | Toy | Sustained live | Final passing streak | Confirmed step |
| --- | --- | --- | ---: | ---: |
| linear_skip_d96_beta5 | vector_anisotropic | FAIL | 0 | — |
| linear_skip_d96_beta5 | vector_overlap | FAIL | 0 | — |
| linear_skip_d96_beta5 | vector_spiral | PASS | 23 | 400 |
| linear_skip_d96_beta5 | vector_two_broad | PASS | 15 | 700 |
| linear_skip_d96_beta5 | vector_unequal_width | FAIL | 0 | — |
| linear_skip_d96_beta5 | vector_unequal_mass | PASS | 6 | 1150 |

[Failure diagnosis and exact replay](DIAGNOSIS.md) · [Current formulation leaderboard](../formulations/README.md) · [Final live/EMA metrics, specs and full-curve artifact links](leaderboard.json) · [Artifact audit](validation.json).

The search adapts to inspected development results. Fixed cards are recorded before each batch. No seed sweeps, extra training, metric relaxation or diagnostic controls are counted as fixes. Source archives and drivers retain every attempted architecture; runtime measurements include concurrent CPU work.

```bash
python -m reports.transfer_suite.rare_focus.build
python -m reports.transfer_suite.rare_focus.verify
```

The current verifier understands the compressed repository layout. Commands inside frozen agent handoffs refer to their original layouts. Use the reusable command above for the winning architecture.
