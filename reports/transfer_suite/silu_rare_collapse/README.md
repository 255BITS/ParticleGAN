# Why axis_silu collapses the rare component on vector_unequal_mass

Protocol v4, fixed cosine recipe, seed 0, sustained rule. Everything is the same as
`../anisotropic_core_metric/run.py`. The target is four corners at σ=.18 with masses
.55/.30/.13/.02, and the 2% component sits at (1.5, 1.5). Each episode runs on the CPU
with 1 thread and takes 10–25 s.

```
python run.py --diagnose [ARM ...]   # per-observation rare-component probe -> diagnose_<arm>.log
python run.py --arms                 # 17 arms on unequal_mass -> arms.log / arms.jsonl
python run.py --crosscheck ARM ...   # arms x 8 dev vector tasks -> crosscheck.log / crosscheck.jsonl
python summarize.py diagnose|arms|crosscheck   # the tables below
python oracle.py                     # protocol v5 finite-atom oracle -> oracle.log / oracle.jsonl
python run.py --v5                   # protocol v5 rescore (18 episodes) -> v5.log / v5.jsonl
python summarize.py v5
```

Research code only. No default critic, recipe or protocol changes. Two opt-in options were
added to `benchmarks/transfer_suite/smooth_critic_research.py`: `first_activation`
(`axis_leaky1_silu`) and `raw_skip` (`axis_silu_raw_skip`). Neither changes the MLP's
initialization, and `tests/test_smooth_critic_research.py` pins this.

## Diagnosis

The generator's output is atomic. `ParticlePrior.sample` indexes into 256 fixed particles
with no per-sample noise, so each particle is one output point, and the 2% component gets
about 5 of them. The probe maps all 256 particles through G at each observation. Numbers
below are medians over the last 8 observations unless stated otherwise.

| arm | obs with 0 rare particles | rare particles | core eig | minor/major std | rare z RMS | min z pair dist | ‖J‖ rare / big comp | J·Δz spread /σ | D(center) − D(1σ ring) | inward pull on reals | ‖∇D‖ fake / real rare | D real rare / real big |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| axis_silu | 7 | 3 | 0.0003 | 0.13 | 0.42 | 0.43 | 2.0 / 1.0 | 0.32 | **+0.006** | **+0.19** | 0.11 / 0.27 | 0.60 / 0.32 |
| leaky_orig | 3 | 6 | 0.21 | 0.47 | 1.56 | 0.63 | 1.8 / 1.3 | 5.2 | −0.007 | +0.13 | 0.13 / 0.27 | −0.50 / −0.40 |
| linear_skip_d96_beta5 | 4 | 7 | 0.32 | 0.58 | 0.37 | 0.07 | 1.7 / 1.4 | 1.6 | −0.003 | +0.01 | 0.06 / 0.08 | −0.07 / 0.26 |
| silu_raw_skip_d96 | 4 | 7 | 0.00 | 0.00 | 1.27 | 0.03 | 1.6 / 1.0 | 6.3 | **+0.029** | **+0.43** | 0.15 / 0.47 | 0.89 / 0.39 |

**Timing: the collapse spans the whole recovered phase.** SiLU over-covers the rare corner early
(20–25 particles at steps 150–200). It then drops the corner entirely: 0 particles at every
observation from step 250 to 550. At step 600 it recovers only 3 particles, and core eig stays
≤.05 at all 13 observations from step 650 to 1200. Final mass ratio is .65.

**Cause: particle motion, not a collapsed prior or a flat generator.** The 3 rare particles are
distinct in z: RMS spread .42, minimum pair distance .43. G's Jacobian at those particles is not
small. Its operator norm is 2.0, twice the big component's. Operator norm times z spread would
allow up to 4.7σ of output spread. At the observations with 3 rare particles, the first-order
output spread J·(z_i − z̄) is only 0.24–0.46σ, and the actual spread is 0.10–0.37σ. The z offsets lie almost entirely in J's null space. The
adversarial gradient on a particle, Jᵀ∇D, lies in J's row space. So the particles have moved
along that gradient until their row-space differences are gone. The null-space differences it
cannot touch are what remains. `rare_focus/DIAGNOSIS.md` found the same thing for Softplus5:
particle updates, not generator updates, contract the cluster.

**Critic: it sees the component but not the spread inside it.** SiLU scores real rare samples
above the big mode (.60 vs .32), which is correct for a component running a 35% mass deficit.
It expresses that as a smooth bump wider than σ: D(center) − D(1σ ring) = +0.006, and D rises
toward the center (inward pull +0.19 at real samples). Every rare particle is pulled to the top
of the bump. A point mass there gets the weakest push of anything in the component: ‖∇D‖ is 0.11
at the fakes vs 0.27 at real rare samples. A batch of 128 holds about 2.5 real rare samples, too
few for the critic to carve a σ-scale dip under the fakes. The critics that keep the spread have
no peak:

- LeakyReLU: center − ring −0.007. Its 6 particles stay spread but leak, with 5σ output RMS and
  41.8% spill.
- linear_skip_d96_beta5: center − ring −0.003, inward pull 0.01. It is locally flat.

Adding a raw skip to SiLU sharpens the peak (+0.029, pull +0.43) and the core still collapses.

**Floor.** At about 5 atoms the core-eig gate is fragile even for an ideal sampler. With n i.i.d.
N(0, I) atoms, P(min whitened eig ≥ .15) is .30 for n=3, .72 for n=5 and 1.00 for n=20. Passing
needs at least 3 non-collinear atoms inside the 4σ core, held for 5 consecutive observations.
In the diagnosis run, LeakyReLU's gated core eig flickers between .008 and .45 from step 300
onward.

## Arm leaderboard (vector_unequal_mass)

Kind: critic = architecture-only (promotable); reg = regularizer setting; recipe = diagnostic
only (the suite fixes the recipe). `core eig (min)` is the gated min over components. In every
SiLU arm the rare component is that min.

| arm | kind | verdict (suffix) | sw1 | mass_tv | core eig (min) | rare core eig | min_mass_ratio | spill | rare mass | failing |
|---|---|---|---|---|---|---|---|---|---|---|
| linear_skip_d96_beta5 | critic | fail (3) | 0.066 | 0.047 | 0.27 | 0.56 | 0.91 | 0.021 | 0.029 | - |
| leaky_orig | ref | fail (0) | 0.081 | 0.052 | 0.45 | 0.45 | 0.91 | 0.418 | 0.024 | spill |
| silu_dlr3 | recipe | fail (0) | 0.092 | 0.055 | 0.08 | 0.08 | 0.90 | 0.011 | 0.025 | core eig |
| silu_steps2x | recipe | fail (0) | 0.128 | 0.086 | 0.00 | 0.00 | 0.65 | 0.009 | 0.013 | core eig |
| axis_silu | ref | fail (0) | 0.137 | 0.086 | 0.00 | 0.00 | 0.65 | 0.046 | 0.013 | core eig |
| silu_batch512 | recipe | fail (0) | 0.053 | 0.047 | 0.01 | 0.01 | 0.88 | 0.225 | 0.027 | core eig, spill |
| silu_raw_skip_d96 | critic | fail (0) | 0.054 | 0.027 | 0.00 | 0.00 | 0.95 | 0.718 | 0.021 | core eig, spill |
| leaky1_silu_fourier3 | critic | fail (0) | 0.073 | 0.060 | 0.00 | 0.00 | 0.84 | 0.290 | 0.017 | core eig, spill |
| silu_hidden128 | critic | fail (0) | 0.103 | 0.052 | 0.00 | 0.00 | 0.74 | 0.557 | 0.015 | core eig, spill |
| leaky1_silu | critic | fail (0) | 0.114 | 0.087 | 0.00 | 0.00 | 0.84 | 0.296 | 0.033 | core eig, spill |
| silu_cap_kappa2.5 | reg | fail (0) | 0.134 | 0.075 | 0.00 | 0.00 | 0.44 | 0.528 | 0.009 | core eig, spill |
| silu_raw_skip | critic | fail (0) | 0.080 | 0.067 | 0.00 | 0.00 | 0.82 | 0.843 | 0.020 | core cov, core eig, spill |
| silu_cap_kappa0.5 | reg | fail (0) | 0.084 | 0.048 | 0.00 | 0.00 | 0.94 | 0.293 | 0.032 | core cov, core eig, spill |
| silu_fourier3 | critic | fail (0) | 0.109 | 0.069 | 0.00 | 0.00 | 0.23 | 0.041 | 0.005 | core cov, core eig, min mass |
| silu_prior_reg0.5 | reg | fail (0) | 0.050 | 0.038 | 0.00 | 0.00 | 0.00 | 1.000 | 0.000 | core cov, core eig, spill, min mass |
| oriented8_silu | critic | fail (0) | 0.141 | 0.121 | 0.00 | 0.00 | 0.00 | 1.000 | 0.000 | core cov, core eig, spill, min mass |
| silu_particles1024 | recipe | fail (0) | 0.167 | 0.144 | 0.00 | 0.00 | 0.00 | 1.000 | 0.000 | core cov, core eig, spill, min mass |

- **No SiLU arm keeps the rare core.** Rare core eig is ≤.08 in all 14 SiLU arms: critic,
  regularizer and recipe changes alike. Fixes that correct mass (kappa .5, batch 512, raw skip
  d96 reach mass ratio .88–.95) still collapse the core and add spill. Three arms (higher
  prior_reg, oriented8, 1024 particles) drop the corner entirely.
- **silu_dlr3 comes closest among the SiLU arms** (eig .08, mass .90, spill .011). Its faster
  critic closes the mass deficit that sustains the bump. It is a recipe change, so it is not
  promotable here.
- **linear_skip_d96_beta5** is the rare_focus Softplus5 winner under the older protocol. It is
  the only arm that clears every final bound, but its passing suffix is 3 of the 5 required, so
  it is not sustained. Its locally flat critic is what keeps 7 spread particles.

## Cross-check (8 dev vector tasks, current protocol)

| task | axis_silu | linear_skip_d96_beta5 | silu_raw_skip_d96 |
|---|---|---|---|
| vector_two_broad | SUST (21) | SUST (15) | SUST (18) |
| vector_unequal_mass | fail (0) core eig | fail (3) - | fail (0) core eig, spill |
| vector_unequal_width | **SUST (8)** | fail (0) spill | fail (2) - |
| vector_anisotropic | **SUST (15)** | fail (0) spill | fail (0) core eig |
| vector_overlap | fail (4) | fail (0) cov | fail (2) |
| vector_spiral | SUST (21) | SUST (23) | SUST (23) |
| vector_scale_drift (diag) | SUST (11) | SUST (12) | SUST (14) |
| vector_narrow (diag) | fail (0) | fail (0) | fail (0) |
| **passes** | **5/8** | 3/8 | 3/8 |

axis_silu reproduces its 5/8. Both alternatives lose unequal_width and anisotropic, the two
tasks where SiLU's margin came from, and neither gains unequal_mass. No candidate keeps SiLU's
five passes and adds unequal_mass.

## Recommendations

1. **Stop tweaking the SiLU architecture for unequal_mass.** The collapse comes from the smooth
   bump the critic forms over an under-filled rare mode, combined with the roughly 5 atoms that
   256 particles give a 2% component. Features, width, a first LeakyReLU layer, a raw skip and
   cap strength all leave the rare core at eig ≤.08.
2. **Promotion.** axis_silu can become the default critic for the vector transfer suite only.
   Its passes are a strict superset of LeakyReLU's (5/8 vs 4/8), and its one ranking-task
   failure is still unequal_mass: a rare-mode collapse where LeakyReLU spills. That failure is
   qualitatively worse (the mode is lost for 7 of 24 observations), so flag it in the leaderboard
   instead of treating the switch as neutral. Do not promote it as a global default yet. What a
   promotion would touch:
   - `benchmarks/transfer_suite/vector_tasks.py`: the critic constructor in `run_episode`, and a
     `fingerprint()` version bump, since it hashes `lib/toy_models.py` but not the research
     module.
   - `SmoothFourierCritic`: move the `axis_silu` card out of `smooth_critic_research.py` into a
     non-research module, or add a `critic="silu"` field to `DEFAULTS`.
   - Tests: no test pins the vector critic class today, so add one to
     `tests/test_transfer_vector_tasks.py`. Rerun `tests/test_transfer_*.py` and
     `tests/test_compare_defaults.py`, which drive `vector_tasks`.
   - The v4 leaderboard rows.
   - Leave `lib/toy_models.SimpleMLPDiscriminator` as it is. toy100 (`benchmarks/toy100/train.py`,
     with the explicit xavier init from #207), `benchmarks/toy_suite.py`,
     `examples/100gaussians.py`, `benchmarks/locked_shared` and `experiments/` share it, and
     SiLU has not been screened on any of them.
3. **Next experiments for the rare mode**, each a distinct configuration:
   - Critic LR 3× with axis_silu across the whole suite, as a recipe proposal. It is the only
     lever that closed the mass deficit without adding spill.
   - A prior-side floor on rare-mode atoms: a minimum particle count per occupied region, or a
     local repulsion between particles whose outputs coincide. This targets the particle motion
     directly.
4. **Protocol (separately versioned).** With about 5 atoms, `component_core_min_eigen_ratio ≥.15`
   fails an ideal i.i.d. sampler 28% of the time per observation. Consider a finite-atom-aware
   bound for components below about 10 particles. Pair it with the spill normalization suggested
   in the v4 section of `../anisotropic_core_metric/README.md`.

## Protocol v5: particle-resolution floor

The rule reads only the spec. A component is under-resolved when `masses[k] * particles` is below
`PARTICLE_FLOOR = 32`. Its core error, core eigenvalue and spill are still reported, flagged by
`component_resolved`, but the gate uses `resolved_*` aggregates over the resolved components.
Mass (`mass_tv`, `min_mass_ratio`) and `hq` still cover every component. At 256 particles, only
the unequal_mass 2% component (5.1 particles) is exempt. The 13% component (33.3 particles) is
still gated. `resolve()` refuses a gated spec where no component is resolved, or where an exempt
component has no `min_mass_ratio` bound. The `resolved_*` values for an empty set (0, 1, 0)
therefore never reach a verdict. The rule applies to anisotropic, unequal_width and unequal_mass.
Every anisotropic and unequal_width component has ≥64 particles, so their gated values are
identical to v4.

### Oracle (how 32 was chosen)

`ParticlePrior.sample` is pure indexing into a fixed particle table with no latent noise, and G
is deterministic, so a generated component is exactly n atoms. The oracle is a perfect sampler
under that model. It draws n atoms i.i.d. from the true component Gaussian, and gives each atom a
Binomial(4096, 1/256) multiplicity from the suite's evaluation multinomial. It scores them with
the suite's per-component core/spill statistics; `oracle.py` checks its replica against
`score_samples`. There are 20k trials per cell. It covers the isotropic shape and each of the 3
anisotropic covariances, and each column below is the worst over those 4 shapes. The grid and
the selection rule were declared in `oracle.py` before it ran. The rule picks the smallest n at
which every gated statistic's per-observation false-fail rate is ≤5% at that n and every larger
grid n. The rule is per component. The gate averages core error over components, so this is
conservative for core error. "sust." is the rate of failing the 5-observation suffix, assuming
independent atom draws per observation. Frozen atoms give the per-observation rate instead.

| n atoms | core err >.5 | core eig <.15 | spill >.05 | any | any, sust. |
|---|---|---|---|---|---|
| 3 | 0.942 | 0.713 | 0.033 | 0.943 | 1.000 |
| 5 | 0.782 | 0.297 | 0.056 | 0.786 | 1.000 |
| 8 | 0.560 | 0.070 | 0.087 | 0.575 | 0.986 |
| 10 | 0.439 | 0.028 | 0.110 | 0.469 | 0.958 |
| 12 | 0.354 | 0.010 | 0.124 | 0.401 | 0.923 |
| 16 | 0.226 | 0.002 | 0.135 | 0.296 | 0.827 |
| 20 | 0.146 | 0.000 | 0.110 | 0.211 | 0.694 |
| 24 | 0.102 | 0.000 | 0.072 | 0.146 | 0.545 |
| **32** | **0.046** | 0.000 | **0.048** | 0.085 | 0.358 |
| 40 | 0.026 | 0.000 | 0.042 | 0.063 | 0.277 |
| 48 | 0.017 | 0.000 | 0.027 | 0.039 | 0.179 |
| 64 | 0.006 | 0.000 | 0.015 | 0.019 | 0.093 |

- **Core error binds, not core eig.** Eig alone would give a floor of 10. The relative Frobenius
  error of a covariance estimated from n atoms shrinks only as about 1/√n, and it drops below .5
  reliably only at about 32 atoms. At 5 atoms a perfect sampler fails core error 78% of the time.
- **Spill is non-monotone.** With k of n atoms beyond 3σ (p=1.1% each), spill is k/n. For n<20,
  one stray atom is already >5%, so the false-fail rate climbs to 13.5% at n=16. It falls only
  once two strays are needed (n≥20).
- The rule chooses per-metric bounds, but the combined rate at 32 is 8.5% per observation. Even at
  64 atoms, a perfect sampler misses the sustained rule 9% of the time per component.

### Rescore (vector_unequal_mass, fixed cosine recipe, seed 0, sustained rule)

The v4 and v5 verdicts come from the same episode (`run.py --v5`). Every metric reproduces the
earlier runs exactly. Resolved columns cover the 3 big components. The rare component is exempt,
and its core eig is shown for reference.

| arm | kind | v4 → v5 verdict (suffix) | sw1 | mass_tv | min_mass_ratio | resolved core err | resolved core eig | resolved spill | rare mass | rare core eig (exempt) | v4 failing (final) | v5 failing (final) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| silu_dlr3 | recipe | fail (0) → **SUST (10)** | 0.092 | 0.055 | 0.90 | 0.39 | 0.44 | 0.011 | 0.025 | 0.08 | core eig | - |
| linear_skip_d96_beta5 | critic | fail (3) → **SUST (7)** | 0.066 | 0.047 | 0.91 | 0.36 | 0.27 | 0.021 | 0.029 | 0.56 | - | - |
| axis_silu | ref | fail (0) → fail (2) | 0.137 | 0.086 | 0.65 | 0.19 | 0.70 | 0.046 | 0.013 | 0.00 | core eig | - (spill, earlier obs) |
| leaky_orig | ref | fail (0) → fail (1) | 0.081 | 0.052 | 0.91 | 0.18 | 0.77 | 0.048 | 0.024 | 0.45 | spill | - (spill, earlier obs) |

**What fails now:** only `resolved_max_component_spill`, from the big components, not the rare
one. At the final observation it is the 55% component. LeakyReLU's big component spills 4–5% beyond 3σ. That bound fails in 8 of its last
12 observations, but not at the final one. axis_silu's big component spills 4.6% and fails 5 of
its last 12. The 42% "spill" LeakyReLU showed under v4 came from the exempt rare component. The
55% component has about 140 particles. Extrapolating the oracle, a perfect sampler would fail
there far less than 1% of the time, so this is genuine marginal spill, not particle noise.

### Suite (8 dev vector tasks, sustained)

LeakyReLU and axis_silu were rerun on all 8 tasks. Every non-unequal_mass verdict and suffix is
identical under v4 and v5, and they match #208/#210/#212.
`test_particle_floor_leaves_other_tasks_unchanged` pins the equality of the gated values.

| task | leaky_orig v4 → v5 | axis_silu v4 → v5 |
|---|---|---|
| vector_two_broad | SUST (19) → SUST (19) | SUST (21) → SUST (21) |
| vector_unequal_mass | fail (0) → fail (1) | fail (0) → fail (2) |
| vector_unequal_width | fail (0) → fail (0), spill | SUST (8) → SUST (8) |
| vector_anisotropic | SUST (5) → SUST (5) | SUST (15) → SUST (15) |
| vector_overlap | fail (4) → fail (4) | fail (4) → fail (4) |
| vector_spiral | SUST (23) → SUST (23) | SUST (21) → SUST (21) |
| vector_scale_drift (diag) | SUST (22) → SUST (22) | SUST (11) → SUST (11) |
| vector_narrow (diag) | fail (0) → fail (0) | fail (0) → fail (0) |
| **passes** | 4/8 → **4/8** | 5/8 → **5/8** |

linear_skip_d96_beta5 moves from 3/8 to 4/8. Its only change is unequal_mass; the other 7 tasks
reuse the cross-check above. It still loses unequal_width and anisotropic on spill.

### Recommendations (v5)

1. **Keep v5.** It removes a failure mode that a perfect sampler hits 79% of the time, and it
   changes no other verdict. unequal_mass now names a real, fixable cause: spill of about 5%
   from the 55% component.
2. **What v5 no longer detects.** axis_silu's rare core is still a point collapse (eig 0.00,
   about 3 particles, mass ratio .65), and v5 no longer penalizes it. Only its mass is gated.
   Resolving a 2% component needs ≥1600 particles, so rare-mode shape at 256 particles is
   unmeasurable, not merely unscored. If rare-mode shape matters, declare a separate task with
   more particles. Do not lower the floor.
3. **Next experiment for unequal_mass.** Both reference critics now miss only on big-component
   spill, suffix 1–2. silu_dlr3 (critic LR 3×) passes with suffix 10 and the lowest spill
   (.011). Critic LR 3× with axis_silu on the whole suite is the single most informative
   follow-up. It is a recipe change, so it is a suite-wide proposal, not a critic promotion.
4. **Protocol noise floor elsewhere.** The oracle shows core/spill gates are not free even for
   well-resolved components. A perfect 64-atom sampler misses the sustained rule 9% of the time
   per component. The spill bound (.05, versus a 1.1% expectation) dominates that at mid n.
   Weigh this before reading one marginal spill failure, such as LeakyReLU unequal_width at
   .055, as a real difference.
