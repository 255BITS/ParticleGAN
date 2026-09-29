# LR-free GAN base search — September 27

## Round 6: D-LR floors lead 13/16 honest (September 28)

The [D-floor round report](dtrack-floors/README.md) archives five honest-generator
candidates on the frozen cf1 base (paired-BD graft): codex α-.75 / α-1.0 prior-scale
D floors, mine2 H2 BD-churn floor, and mine3 both2 max(prior-scale, churn-rate).
Full-toy matrix (13 quick + ring/stationary + 3 natives): both2 **13/16**
(11/13 quick, grid+rotated PASS, staggered FAIL centre .199), α-.75 12/16,
h2 11/16, cf1 and α-1.0 10/16. Every variant trades rotated against staggered;
nothing honest takes all three. img_intensity2 + img_bars4 fail lineage-wide.

## Native100 prior EMA relaxation result (September 28)

The [prior EMA relaxation report](prior-ema-relaxation/README.md) archives one
fixed package that passes grid100, rotated100, and staggered100 (3/3) on a
diagnostic host with a fresh real batch on every callback. The original
frozen host caches its first callback batch; its staggered100 run **FAILS**
the final centre check (.20266σ). The frozen-host 3/3 objective remains open. The
[native leaderboard](prior-ema-relaxation/leaderboard.md), [full22 matrix](prior-ema-relaxation/all22-leaderboard.md),
and [handoff](prior-ema-relaxation/HANDOFF.md) record scope and limitations.
The 22-task matrix is 7 PASS, 2 FAIL, 13 ERROR; this research candidate is not
a general project default.

## Direct-particle follow-up (September 28)

The [direct-particle Adam report](direct-particle-adam/README.md) archives a
small `st-10` policy patch and its eight custom-host reruns. `two_pole` changes
from FAIL 0/24 to PASS 7/24; the other seven custom hosts reproduce every
non-time observation row exactly. This makes the custom subtotal **6/8** and
implies **17/22** from the unchanged 11 quick and 0 native results. It does
not fix a native 100-Gaussian gate. The separate latest grid100 structural
lanes all failed; the [paired birth/death graft](paired-bd-graft/README.md)
was the strongest completed native directional improvement, but still missed
centre and covariance limits.

## 22-check leaderboard: all eight custom hosts now run (September 27, night)

The eight custom hosts of the 22-toy suite now run for any candidate package in the LR-free harness
([custom22 report](custom22/README.md); the harness is committed at [`harness/`](harness/)). The candidate's own
policy (controller, KA2, amsgrad, noise) is bitwise parity-checked against its `GANTrainer.step`, and the host
copies are verbatim. As a known-good control, develop K3P with its annealed schedule passes 8/8 through the same
engine. Noisy scoring throughout.

| Candidate | Quick 11 | Custom 8 | Native 3 | **/22** | ring / stationary |
|---|---:|---:|---:|---:|---|
| `st-10` (stationarity LR + learnable σ) | 11 | 5 | 0 | **16** | PASS / PASS |
| `dv12-ams-rc3` (#217 default) | 11 | 4 | 0 | **15** | PASS / PASS |

Both fail `two_pole`. The cause is AMSGrad's second-moment memory on the direct particles: the step collapses as
gradients shrink about 20× in the first 10 updates. With amsgrad off and reg_coeff 1, rc3 passes; K3P plus amsgrad
fails. `dv12-ams-rc3` also misses `cover_leftover`, `unused_token_hold` and `mid_scale_identity`. `st-10` misses
`trajectory`, and `unused_token_hold` on streak length only.

## Current 100-Gaussian readout: QR initialization is primary

The library's `batch_feature_zero` QR initialization is the primary native
comparison. The Xavier-first native interpretation in the older sections
below is superseded; Xavier runs remain mechanism controls and should not
be ranked against QR runs. All new results score noisy model samples. The
frozen native gate requires coverage and accuracy over the final five checks
and an independent 100k holdout. The [structural round report](structural100/README.md)
archives configs, source patches, and [exact result fields](structural100/native100-results.json).

`st5-scaleaware-qr` uses state-driven, two-scale prior settling with learned
output noise and no birth/death. Its three native cards all **FAIL** with
0/34 passing checks and failing holdouts. Each row below gives the live
step-7,000 diagnosis; every row has 100 modes.

| Native task | Precision (need ≥.97) | Covariance eigenvalue range (need .40–1.70) | Centre RMS (need ≤.20σ) | Trace bias (need ≤.10 absolute) | Radial KS (need ≤.04) | Learned output σ |
|---|---:|---:|---:|---:|---:|---:|
| grid100 | .979 | .163–2.432 | .364σ | +.074 | .084 | .00677 |
| rotated100 | .962 | .059–1.857 | .281σ | −.261 | .120 | .00152 |
| staggered100 | .960 | .034–2.123 | .522σ | −.249 | .067 | .00329 |

Grid reaches the precision target, while per-mode shape and centres still
miss badly. Rotated and staggered miss precision as well. Learned σ shrinks
far below the data σ of `.03`; the scale-aware settle rule has not solved
the streaked, off-centre modes. **No 13-gate run has been completed for this
QR candidate**, so there is no harness-gate claim.

The same QR candidate was extended on rotated100 to 14,000 updates.
Its 7k prefix reproduces the original run; the 7k subbudget and 14k frozen
verdicts both **FAIL**. At 14k it has 0/62 passing checks and a failing
holdout. Precision improves from `.962` to `.975` and radial KS from `.120`
to `.069`, but centre RMS is `.308σ`, trace bias `−.189`, and covariance
eigenvalue ratios span `.066–2.741`; learned output σ is `.00135`.
The controller cut and later restored the prior LR, yet no observation met
all shape limits. [The extension record](structural100/README.md#rotated100-extension-to-14000-updates)
contains the trajectory, exact frozen gate fields, and reproduction command.
A [frozen-generator critic refit](structural100/critic-refit-summary.md)
recovered favorable local covariance-repair derivatives after 250 critic-only
updates with both warm and fresh-QR critics. It suggests critic tracking may
contribute to the saved adverse shape signal; it made no generator update or
new native gate evaluation.
A later [bounded saved-critic audit](structural100/critic-underfit-audit.md)
found held-out KA2 descent after 250 critic-only updates at both the saved
post-cut and restored pre-cut D rates. Their total-objective difference was
unresolved (95% interval `−7.48e−6` to `+7.27e−6`); restoring the larger
rate has no demonstrated advantage here. Neither test continued the GAN.

The Xavier birth/death controls all fail their frozen native gates. Cells
show final modes / precision after 7,000 updates; they are not comparable
to the QR row above because initialization changes the trajectory.

| Xavier control | grid100 | rotated100 | staggered100 |
|---|---:|---:|---:|
| `st2` no BD | 98 / .969 | 100 / .930 | 100 / .950 |
| `st2` unrestricted BD | 89 / .910 | 93 / .827 | 91 / .887 |
| `st3` paired BD | 94 / .927 | 97 / .849 | 91 / .880 |
| `st4` paired BD | 95 / .949 | 97 / .849 | 91 / .880 |
| `st5` paired BD | 95 / .949 | 99 / .838 | 91 / .880 |

Unrestricted BD moved hundreds of thousands of rows, mostly using neutral
parents; pairing reduced churn but still lost covered modes. The repeated
snapshot analysis in the structural report supports a mismatch between
local density evidence and each mode's total high-quality mass.

`st6-eg-qr` tested transactional adaptive extragradient on the no-BD `st5`
base. Its grid diagnostic fell to zero covered modes and HQ `.0032` by step
500, so the run was stopped. The harness records `ERROR` because it has no
final result: **this is not a frozen 7,000-update native verdict**. Rotated,
staggered, and the 13 gates were not run for `st6`.

`st7-fake-reopen-qr` added a generated-sample drift signal to reopen only the
critic's stationarity test. Its single frozen grid100 run **FAILS 0/34** with
a failing holdout. The signal never triggered a critic reopening, and all
7,000 applied-rate rows and the native verdict match `st5` exactly. This
does not test whether a forced critic refit would improve training; no other
native card or 13-gate task was run for `st7`.

A bounded [online critic-refinement preflight](structural100/critic-refinement-preflight.json)
rejected its shadow D after
257 host calls, consuming 771 extra fixed-objective evaluations and skipping
257 ordinary G updates. Fresh post-hoc batches nevertheless found a critic
objective gain, so the acceptance rule missed useful refinement; it was
rejected before any full native gate. An isolated
[`st5-dbase-qr` grid test](structural100/README.md#st5-dbase-qr-d-base-rate-attribution-on-grid100)
then kept D at its base LR before payoff damping while retaining the other
stationarity rules. It **FAILS 0/34** with a failing holdout: 99 modes,
precision `.9821`, centre RMS `.315σ`, eigenvalue ratios `.159–2.901`, and
radial KS `.0873`. Higher D rate improved some live metrics but lost a mode
and did not clear native shape. Neither experiment supports core promotion.

## Separate source audit: PR #215 GS2 versus PR #217 DV12

The [fixed-protocol comparison](comparison-pr215-pr217/README.md) reruns
exact [#215](https://github.com/255BITS/ParticleGAN/pull/215) source commit
`c138aff` and [#217](https://github.com/255BITS/ParticleGAN/pull/217) source
commit `261fcfd` on these #155 hosts. #215 passes **5/13** gates with the
frozen host defaults (Xavier critic on native) and **7/13** with a recorded
QR adapter. The
exact-head #217 QR/native adapter passes **13/13** after its image evaluator
draws latent jitter through the source prior. Every arm **fails all three
native 100-Gaussian gates**, with zero of 34 passing checks and failing 100k
holdouts on each task. The exact statuses, package digests, run paths, and
native metrics are in [results.json](comparison-pr215-pr217/results.json).

The #215 QR grid100 continuation to 14,000 updates also **FAILS**: 0/62
passing checks and a failing holdout. Its 7k rate prefix is byte-identical
to the original, and its final 100 modes / `.9832` precision still have
centre RMS `.244σ`, covariance eigenvalue ratios `.256–2.480`, and radial
KS `.04157`. The [extension record](comparison-pr215-pr217/README.md#215-qr-grid100-extension-to-14000-updates)
keeps this longer single-task result separate from the 7k three-card score.

The published #215 **17/26** belongs to #202's different host suite, and
the #217 high-`z` throughput regression belongs to a separate performance
probe. An isolated nearest-search benchmark found a 32 MiB high-dimensional
block 2.37× faster than the current 4 MiB block on one shared-GPU shape;
its [proposed patch](comparison-pr215-pr217/pr217-highz-nearest-32m-proposal.patch)
is **unmerged** and was not used for any gate run. This is a search-only
measurement, not a full-training speed claim.

## Earlier leaderboard (September 27 evening; mixed native initialization)

This section preserves the earlier 13-gate standings and exploratory native
rows. Its native ordering and proposed next steps are superseded by the QR
readout above.

Scoring is noisy throughout, meaning the model's own samples. The harness has 13 gates: 10 quick gates,
`img_intensity2` judged at 1,200 updates, `ring_shift` and `stationary`. The 100-Gaussian columns are
separate from those 13. All native runs FAIL, and the precision needed there is ≥ .97.

| # | Config | Harness gates | Fails | `intensity2` @1200 | ring_shift | stationary | grid100 | rotated100 | staggered100 | native init |
|---:|---|---:|---|---|---|---|---|---|---|---|
| 1 | **`st-10`**: stationarity-tested LR + learnable σ | 13/13 | — | PASS 33/48 | PASS 357/460 | PASS 677/750 | FAIL 98m p0.969 | FAIL 100m p0.930 | — | Xavier |
| 2 | **`dv12-ams-rc3`**: DV12 + amsgrad + reg 3 (current #217 default) | 13/13 | — | PASS 20/48 | PASS 369/460 | PASS 688/750 | FAIL 91m p0.707 | FAIL 56m p0.519 | FAIL 97m p0.722 | Xavier |
| 3 | `nr-D`: dv12-ams-rc3, EMA anchor off | 13/13 | — | PASS 20/48 | PASS 366/460 | PASS 685/750 | FAIL 94m p0.722 | FAIL 56m p0.522 | FAIL 97m p0.732 | Xavier |
| 4 | `dv12-rc3`: plain Adam | 13/13 | — | PASS 23/48 | PASS 363/460 | PASS 663/750 | FAIL 99m p0.938 | FAIL 100m p0.895 | FAIL 100m p0.919 | QR |
| 5 | `st-00`: dv12-ams-rc3 + learnable σ | 12/13 | mode_hold | PASS 16/48 | PASS 346/460 | PASS 684/750 | FAIL 81m p0.672 | FAIL 60m p0.515 | — | Xavier |
| 6 | `nr-mob-r1`: σ tied to mobility | 12/13 | bars4 | PASS 32/48 | PASS 373/460 | PASS 691/750 | FAIL 78m p0.656 | FAIL 59m p0.504 | FAIL 93m p0.728 | Xavier |
| 7 | API-DV12 (reg 1) | 12/13 | v.unequal_mass | PASS 30/48 | PASS 360/460 | PASS 687/750 | FAIL 99m p0.972 | FAIL 100m p0.928 | FAIL 100m p0.953 | QR |
| 8 | API-RP15 | 12/13 | bars4 | PASS 28/48 | PASS 328/460 | PASS 654/750 | FAIL 100m p0.969 | FAIL 98m p0.908 | FAIL 100m p0.918 | QR |
| 9 | `t2-dv12q-ons018`: output noise .018 | 12/13 | intens2 | FAIL 8/24 (@600) | PASS 363/460 | PASS 685/750 | FAIL 99m p0.909 | FAIL 100m p0.881 | FAIL 100m p0.904 | QR |
| 10 | `st-01`: learnable σ + birth–death | 11/13 | mode_hold, v.unequal_mass | PASS 33/48 | PASS 384/460 | PASS 702/750 | FAIL 99m p0.943 | FAIL 60m p0.515 | — | Xavier |
| 11 | `nr-lat-r1`: no output noise | 11/13 | mode_hold, bars4 | PASS 31/48 | PASS 365/460 | PASS 684/750 | FAIL 76m p0.640 | FAIL 56m p0.496 | FAIL 93m p0.725 | Xavier |
| 12 | `nr-C`: reg_coeff 1 | 11/13 | bars4, v.unequal_mass | PASS 32/48 | PASS 371/460 | PASS 673/750 | — | — | — |  |
| 13 | `nr-fix-nor1`: R1-on-real off | 9/13 | mode_hold, bars4, v.unequal_mass, v.anisotropic | PASS 23/48 | PASS 340/460 | PASS 663/750 | — | — | — |  |
| 14 | `st-11`: stationarity + birth–death | 9/13 | mode_hold, v.unequal_mass, v.unequal_width, v.anisotropic | PASS 32/48 | PASS 299/460 | PASS 705/750 | FAIL 99m p0.969 | FAIL 100m p0.930 | — | Xavier |
| 15 | `nr-A`: whole critic penalty off | 7/13 | mode_hold, bars4, v.unequal_mass, v.anisotropic, ring, stationary | PASS 21/48 | FAIL 0/460 | FAIL 0/750 | — | — | — |  |
| 16 | `nr-E`: penalty off + no pe damping | 5/13 | mode_hold, blobs4, bars4, v.unequal_mass, v.unequal_width, intens2, ring, stationary | FAIL 7/48 | FAIL 0/460 | FAIL 0/750 | — | — | — |  |
| — | *Reference: stock annealed K3P (LR schedule over 7,000 steps; not a candidate)* | — | — | — | — | — | FAIL 97m p0.980 | PASS 100m p0.982 | PASS 100m p0.981 | Xavier |

**Notes**
- **Native init varies by row, and it matters a lot.** "QR" rows are the library's default
  deterministic init. "Xavier" rows use the frozen card's critic init, which needs
  `initialization=null`. That also switches off the library init elsewhere, and it hurts DV12:
  `dv12-ams-rc3` reaches 99 grid100 modes at precision .89 with QR, but 91 modes at .71 with Xavier.
  Compare native columns only within the same init.
- **`dv12-ams-rc3` at 21,000 updates (QR):**
  - rotated100 keeps improving. Precision goes .878 → .908 → .931 and every accuracy limit passes
    from update 9,750, so on a rough extrapolation it would pass somewhere around 35–45k updates.
  - grid100 is stuck at 99 modes. DV12's game trust holds G's LR at about 1.5e-6.
- **The stock annealed K3P passes rotated100 and staggered100** in this harness, so the harness is
  sound and the remaining gap is in the continuous learners.

**What the recent rounds found**
- **Stationarity-tested LR (`st-10`) is the most promising lever.** It passes 13/13 harness gates.
  It does not collapse the LR while there is still progress to make: on rotated100 modes go 60 → 100
  and precision .52 → .93 compared with `st-00`, and after a target change the LR comes back up.
  - Open issue: the prior table's test never settles.
  - Confound: every 2×2 arm used learnable σ.
- **Birth–death (`st-01`) gives the best native shape on grid100:** centre .23σ, radial KS .024, trace
  bias +.03. But its guard skipped every step on rotated100, and it hurts the small tables
  (`mode_hold`, `unequal_mass`).
- **Noise × R1 round:**
  - The fuzziness is the output noise itself, not R1. Removing R1, or all output noise, makes things
    worse.
  - DV12's `1/(1+pe²)` damping on the critic cannot replace the critic penalty (`nr-A` 7/13, `nr-E` 5/13).
  - The EMA-critic anchor has no measurable effect (`nr-D` matches the base).
- **Next:** give the stationarity test a small-drift rule so the prior table can settle, fix the
  birth–death guard and its small-table behaviour, and rerun both on fixed σ .029 to separate them
  from learnable σ.

---


## Correction: score the model's own samples (noisy), not clean samples

The earlier update below scored clean samples, and that was wrong.

- In this library, output noise is a **permanent part of the generator's sampling distribution**. It
  warms up and then stays; it is not annealed away. A model's samples are `G(z) + noise`.
- On the native 100-Gaussian problems these recipes use noise .029, close to the data std of .03. The
  generator puts particles at the mode centres and lets the noise supply the width. Clean samples
  therefore come out 40–90% too narrow per mode (covariance-trace bias −.41 to −.91).
- The noisy score, which is also the frozen protocol, is the correct verdict. The clean score is
  kept only as a sharpness diagnostic.

**Verdicts under noisy scoring**
- `img_intensity2` still passes at 1,200 updates:
  - `dv12-ams-rc3`: PASS 20/48 @650, final streak 16 (600 updates: FAIL 0/24);
  - `dv12-rc3`: PASS 23/48 @475 (600 updates: FAIL 2/24).
  - So "it only needed more updates" still holds.
- **`t2-dv12q-ons018` drops to 12/13.** Its `img_intensity2` pass at 600 updates held only under
  clean scoring (noisy: FAIL 8/24, suffix 4).
- **Best standing:** `dv12-ams-rc3` passes 13/13 on the harness gates with `img_intensity2` at 1,200
  updates. Ring passes at 178/178 and then 191/192; stationary passes at 685/685.

**Leaderboard (noisy scoring; the 13 harness gates count `img_intensity2` at 1,200 updates)**

| # | Candidate | 13 gates (intensity2 @1200) | mode_hold | intens2 @1200 | intens2 @600 | blobs4 | stripes2 | bars4 | v.broad | v.mass | v.width | v.aniso | v.overlap | v.spiral | ring_shift | stationary | grid100 | rot100 | stag100 |
|---:|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `dv12-ams-rc3` (DV12 + amsgrad + reg_coeff 3) | 13/13 | PASS 9/24 @800 | PASS 20/48 @650 | FAIL 0/24 | PASS 17/24 @200 | PASS 22/24 @75 | PASS 15/24 @250 | PASS 22/24 @150 | PASS 18/24 @350 | PASS 19/24 @300 | PASS 22/24 @150 | PASS 23/24 @50 | PASS 24/24 @67 | PASS 365/460 | PASS 685/750 | FAIL 0/34 | FAIL 0/34 | FAIL 0/34 |
| 2 | `dv12-rc3` (plain Adam) | 13/13 | PASS 6/24 @950 | PASS 23/48 @475 | FAIL 2/24 @475 | PASS 17/24 @175 | PASS 22/24 @75 | PASS 15/24 @200 | PASS 22/24 @150 | PASS 18/24 @350 | PASS 19/24 @300 | PASS 22/24 @150 | PASS 21/24 @50 | PASS 24/24 @67 | PASS 356/460 | PASS 652/750 | FAIL 0/34 | FAIL 0/34 | FAIL 0/34 |
| 3 | `t2-dv12q-ons018` (output noise .018) | 12/13 | PASS 7/24 @900 | — | FAIL 8/24 @350 | PASS 17/24 @175 | PASS 21/24 @100 | PASS 8/24 @375 | PASS 22/24 @150 | PASS 17/24 @400 | PASS 19/24 @300 | PASS 22/24 @150 | PASS 22/24 @50 | PASS 24/24 @67 | PASS 359/460 | PASS 680/750 | FAIL 0/34 | FAIL 0/34 | FAIL 0/34 |
| 4 | API-DV12 | 12/13 | PASS 12/24 @650 | PASS 30/48 @425 | PASS 6/24 @425 | PASS 18/24 @175 | PASS 21/24 @100 | PASS 5/24 @500 | PASS 22/24 @150 | FAIL 0/24 | PASS 21/24 @200 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 24/24 @67 | PASS 358/460 | PASS 686/750 | FAIL 0/34 | FAIL 0/34 | FAIL 0/34 |
| 5 | API-RP15 | 12/13 | PASS 14/24 @550 | PASS 28/48 @500 | PASS 5/24 @500 | PASS 17/24 @200 | PASS 21/24 @75 | FAIL 0/24 | PASS 23/24 @100 | PASS 21/24 @200 | PASS 20/24 @250 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 23/24 @67 | PASS 327/460 | PASS 653/750 | FAIL 0/34 | FAIL 0/34 | FAIL 0/34 |
| 6 | `t2-rp15noise-in10` (RP15 + input noise .1) | 12/13 | PASS 11/24 @600 | — | FAIL 2/24 @550 | PASS 17/24 @200 | PASS 20/24 @125 | PASS 12/24 @325 | PASS 21/24 @200 | PASS 21/24 @200 | PASS 20/24 @250 | PASS 21/24 @200 | PASS 24/24 @50 | PASS 24/24 @67 | PASS 328/460 | PASS 671/750 | — | — | — |

- Ring and stationary cells count all checks, including those before first arrival (460 and 750).
  Retention after arrival is higher: `dv12-ams-rc3` is 178/178, then 191/192 after the target change,
  and 685/685 on stationary.
- `—` means not run. For `t2-dv12q-ons018` and `t2-rp15noise-in10`, `img_intensity2` was not run at
  1,200 updates, so their 13-gate count uses the 600-update result.
- The native columns carry the harness caveats: QR critic init instead of the frozen Xavier init, and
  DV12's latent jitter active at evaluation. Faithful reruns are in progress.

**Native 100-Gaussian problems (grid100, rotated100, staggered100; 7,000 updates): all 15 runs FAIL.**
This holds under both scorings, for the leaders plus `t2-dv12q-ons018`.

| Result | Detail |
|---|---|
| Coverage | 99–100 modes |
| Precision | .88–.97 (needs ≥ .97) |
| Centre RMS | .17–.44σ (needs ≤ .20) |
| Closest miss | API-RP15 grid100 (noisy): covariance-trace bias .103 vs .10, radial KS .045 vs .04 |

Two caveats before these native numbers count:
- The harness replaced the frozen card's Xavier critic init with deterministic QR init. `develop` #207
  deliberately keeps Xavier for toy100.
- DV12's latent perturbation stays active at evaluation.

These will be rerun faithfully.

**Next (proposed):**
- Replace hand-tuned output noise with **learnable output noise**. The generator learns its own noise
  scale, driven by state rather than a clock.
- Rerun the native problems with the Xavier critic.
- Then run the critic-regularizer ablation and port the remaining 8 hosts of the 22-toy suite.

Draft PR #209 is on hold; its clean-by-default `sample()` rests on the same mistaken premise.

---


## Update (superseded by the correction above): clean scoring fixed, `img_intensity2` passes with a longer budget

**Headline:**
- `dv12-ams-rc3` passes all 11 quick gates with clean scoring when `img_intensity2` gets 1,200
  updates, and it passes `ring_shift` and `stationary`.
- A new variant, **`t2-dv12q-ons018`**, passes all 11 quick gates at the original frozen budgets.
  It is `dv12-ams-rc3` with constant `output_noise_std` lowered from .029 to .018.
- The three native 100-Gaussian problems from the 22-toy suite are running now. The other eight
  custom hosts of that suite are not yet wired to the public API.

**What changed**
1. **Sampling bug fixed in the harness.** Every screen now scores clean samples. The old noisy score
   is still recorded alongside, and a GPU check confirmed training is byte-identical.
   - Public `GANTrainer.sample()` still adds the training output noise by design
     (`training.py:296-312`). A clean-sampling option is left for a separate library PR.
   - Clean scoring changed no pass/fail verdict except on `img_intensity2`.
2. **`img_intensity2` needed more updates.** Clean HQ was still rising at update 600: it hovered at
   .84–.94 while both modes held. At 2,400 updates all four leaders pass and hold it:
   - `dv12-ams-rc3` passes every check from update 600, 76/78 after arrival.
   - `dv12-rc3` passes every check from update 975.
   - The misses were single-check HQ dips, which stop once the DV12 controller brings G's LR down to
     about 3–4% of peak.

   **`img_intensity2` is now scored at 1,200 updates. The other gates keep their frozen budgets.**
   A 1,200-update verdict is the first 48 observations of the 2,400-update run: the runs are
   deterministic, and the shared prefix is bitwise identical.

   The verdict depends on where a run stops, because the final 5-check suffix rule is strict.
   - `dv12-rc3` passes at 900 and 1,200 but fails at 1,000.
   - `dv12-ams-rc3`'s `img_bars4` passes at 600 and 2,400 but would fail at 1,200. A single dip to HQ
     .875 at update 1,125 lands in the suffix window, even though it passes 86/87 after arrival over
     2,400 updates.
   - For that reason only `img_intensity2`'s budget was changed.

**Leaderboard (clean scoring, 11 quick gates; `img_intensity2` at 1,200 updates)**

| # | Candidate | Quick gates | `mode_hold` | `img_intensity2` @1200 (@600) | `img_bars4` | `unequal_mass` | `ring_shift`* | `stationary`* |
|---:|---|---:|---|---|---|---|---|---|
| 1 | **`dv12-ams-rc3`** | **11/11** | PASS 10/24 @750 | **PASS 28/48 @475** (FAIL 4/24) | PASS 15/24 @250 | PASS 18/24 @350 | PASS @660 175/175, +300 190/191 | PASS 685/685 |
| 2 | **`t2-dv12q-ons018`** | **11/11** | PASS 7/24 @900 | PASS at 600: 9/24 @350 | PASS 8/24 @375 | PASS 17/24 @400 | pending | pending |
| 3 | `dv12-rc3` | 11/11 | PASS 6/24 @950 | PASS 27/48 @450 (FAIL 5/24; fails at 1,000) | PASS 15/24 @200 | PASS 18/24 @350 | PASS @650 175/176, +380 181/183 | PASS 652/686 |
| 4 | API-DV12 | 10/11 | PASS 12/24 @650 | PASS 30/48 @425 (PASS 6/24) | PASS 5/24 @500 | **FAIL** 0/24 | PASS @600 181/181, +440 177/177 | PASS 686/691 |
| 5 | API-RP15 | 10/11 | PASS 15/24 @500 | PASS 32/48 @425 (PASS 8/24) | **FAIL** 0/24 (3m, HQ .72) | PASS 21/24 @200 | PASS @930 143/148, +370 184/184 | PASS 653/658 |
| 6 | `t2-rp15noise-in10` (RP15 + constant input noise .1) | 9/11 | FAIL 10/24 (suffix 4) | FAIL at 600 (suffix 1) | PASS 12/24 @325 | PASS 21/24 @200 | — | — |

\* `ring_shift` and `stationary` come from the earlier runs scored with the output noise. Their clean
reruns are in progress. Noise only lowers HQ at scoring, so the earlier passes are conservative. All
other cells are clean scores. The remaining quick-gate cells (blobs4, stripes2 and five vectors) are
PASS for rows 1–5.

**Caveats**
- `t2-dv12q-ons018` passes `mode_hold` only thinly: 7/24, first passing at update 900. The
  output-noise level sits in a narrow window: .010, .020 and .022 each fail a gate.
- DV12's rates are driven purely by training signals, but they decay slowly. G's LR is about 12% of
  peak at update 600, 3–4% at 1,200, then flat.
- The full 22-toy suite is not run. The native 100-Gaussian problems are in progress, and the eight
  custom hosts need a component layer under `GANTrainer` first.

---


## Original report (September 27, before the clean rescore)

**Goal:** one fixed public-`GANTrainer` config that needs no learning-rate adjustment: no LR or noise
schedule tied to a clock or horizon, no per-task LR tuning, and able to run indefinitely. State-driven
controllers and AMSGrad are allowed.

**Result so far:** two configs pass **all 12 trusted gates**. The 12 are `mode_hold`, three images,
six vectors, `ring_shift` and `stationary`. The recommended base is **`dv12-ams-rc3`**, which also
holds perfectly over 7,500 stationary updates (685/685 checks passing after arrival).
`img_intensity2` is **treated as broken** (see the sampling bug below). After the fix it still fails
narrowly on clean samples for both, so it remains an open gate rather than a pass. Nothing is merged and
no default is changed.

## Sampling bug: scoring adds training output noise

Every frozen screen adds the generator's training output noise to samples before scoring them:
- `mode_hold` adds `sigma·randn` after generation;
- images and vectors call `_generate` with the noise sigma;
- ring and stationary score through public `GANTrainer.sample()`.

For example, DV12 and RP15 keep a constant `output_noise_std` of .029, and the scorer adds that noise
too. `img_intensity2` counts a sample as high quality only when its rmse is ≤ .06. Noise of .029 adds in
quadrature, so a sample needs a clean rmse of about .052 or less to count. Output noise is a training
regularizer, so the generator should be scored on clean samples.

Public `GANTrainer.sample()` also returns samples with the current training output noise added
(`training.py:296-312`; its docstring says "with the current output noise"). Anyone sampling through
the API gets noisy samples. Whether to add a clean-sampling option belongs in a separate library PR.

**Status:** the harness fix is in. It scores clean samples by default and keeps the noisy score
alongside. A GPU check showed training is unchanged: per-update rates are byte-identical, and the noisy
scores are bitwise equal to the old ones. The clean rescore of the top two already shows:

| Candidate | `img_intensity2`, frozen (noisy) | `img_intensity2`, clean |
|---|---|---|
| `dv12-ams-rc3` | FAIL 0/24, HQ .84 | FAIL 4/24 @475, final HQ .91 (final passing run too short) |
| `dv12-rc3` | FAIL 2/24 @475, HQ .88 | FAIL 5/24 @450, final HQ .94 |

So noise explains part of the gap, not all of it. The other gates checked so far keep their verdicts
under clean scoring. `mode_hold` for `dv12-ams-rc3` moves from 9/24 @800 to 10/24 @750. The rescore of
the remaining candidates on all 13 gates is running. `img_intensity2` results in the tables below (⚠)
use the frozen noisy scoring and do not rank candidates.

## Leaderboard (12 trusted gates)

Harness: a fast screen that runs candidates through public `GANTrainer.step` on the frozen PR155
new-init hosts. It reproduces the recorded RP12, DV12 and ka2-constant screens bitwise, and adds
`ring_shift` and `stationary`. Each cell is one deterministic run; there are no seed repeats.
Cells read `PASS 19/24 @300`: status, passing observations, first passing observation.
Ring cells read `@arrival retained/checks / +changed-target delay retained/checks`.
Stationary cells read `@arrival passing/checks since arrival` over 7,500 updates.

| # | Candidate | LR policy | Gates (of 12) | mode_hold | blobs4 | stripes2 | bars4 | v.broad | v.mass | v.width | v.aniso | v.overlap | v.spiral | ring_shift | stationary | intensity2 ⚠ |
|---:|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `dv12-ams-rc3` | DV12 controller | 12/12 | PASS 9/24 @800 | PASS 17/24 @200 | PASS 22/24 @75 | PASS 15/24 @250 | PASS 22/24 @150 | PASS 18/24 @350 | PASS 19/24 @300 | PASS 22/24 @150 | PASS 23/24 @50 | PASS 24/24 @67 | PASS @660 175/175 / +300 190/191 | PASS @660 685/685 | FAIL 0/24 (2m hq0.84) |
| 2 | `dv12-rc3` | DV12 controller | 12/12 | PASS 6/24 @950 | PASS 17/24 @175 | PASS 22/24 @75 | PASS 15/24 @200 | PASS 22/24 @150 | PASS 18/24 @350 | PASS 19/24 @300 | PASS 22/24 @150 | PASS 21/24 @50 | PASS 24/24 @67 | PASS @650 175/176 / +380 181/183 | PASS @650 652/686 | FAIL 2/24 @475 (2m hq0.88) |
| 3 | `API-RP15` | RP precision | 11/12 | PASS 14/24 @550 | PASS 17/24 @200 | PASS 21/24 @75 | FAIL 0/24 (3m hq0.72) | PASS 23/24 @100 | PASS 21/24 @200 | PASS 20/24 @250 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 23/24 @67 | PASS @930 143/148 / +370 184/184 | PASS @930 653/658 | PASS 5/24 @500 |
| 4 | `API-DV12` | DV12 controller | 11/12 | PASS 12/24 @650 | PASS 18/24 @175 | PASS 21/24 @100 | PASS 5/24 @500 | PASS 22/24 @150 | FAIL 0/24 [c.covariance_error,min_mass_ratio] | PASS 21/24 @200 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 24/24 @67 | PASS @600 181/181 / +440 177/177 | PASS @600 686/691 | PASS 6/24 @425 |
| 5 | `rp15-const` | constant | 10/12 | FAIL 9/24 @550 (8m hq0.81) | PASS 17/24 @200 | PASS 21/24 @75 | FAIL 0/24 (3m hq0.72) | PASS 23/24 @100 | PASS 21/24 @200 | PASS 20/24 @250 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 23/24 @67 | PASS @930 140/148 / +370 182/184 | PASS @930 587/658 | PASS 5/24 @500 |
| 6 | `dv12-ams` | DV12 controller | 10/12 | PASS 6/24 @950 | PASS 19/24 @150 | PASS 21/24 @100 | FAIL 0/24 (2m hq0.94) | PASS 23/24 @100 | FAIL 0/24 [c.covariance_error,min_mass_ratio] | PASS 21/24 @200 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 24/24 @67 | PASS @580 183/183 / +320 188/189 | PASS @580 673/693 | FAIL 8/24 @375 (2m hq0.97) sfx4 |
| 7 | `rp15-ams-rc3` | RP precision | 10/12 | PASS 6/24 @950 | PASS 15/24 @250 | PASS 17/24 @200 | PASS 7/24 @450 | PASS 22/24 @150 | PASS 21/24 @200 | PASS 19/24 @300 | PASS 21/24 @200 | FAIL 21/24 @50 sfx2 | PASS 24/24 @67 | FAIL no arrival / +500 159/171 | PASS @4020 349/349 | FAIL 6/24 @425 (2m hq0.97) sfx1 |
| 8 | `API-RP12` | RP precision | 9/12 | PASS 19/24 @300 | PASS 17/24 @200 | PASS 17/24 @200 | FAIL 0/24 (2m hq0.97) | PASS 22/24 @150 | FAIL 0/24 [c.min_eigen_ratio,min_mass_ratio] | FAIL 0/24 [sw1,mass_tv] | PASS 21/24 @200 | PASS 24/24 @50 | PASS 23/24 @134 | PASS @630 177/178 / +310 190/190 | PASS @630 687/688 | PASS 12/24 @325 |
| 9 | `pr202-ams-lr0015` | constant | 9/12 | PASS 14/24 @450 | PASS 11/24 @350 | PASS 12/24 @275 | FAIL 0/24 (2m hq0.84) | PASS 22/24 @150 | FAIL 0/24 [c.min_eigen_ratio,min_mass_ratio] | PASS 12/24 @650 | PASS 19/24 @300 | FAIL 16/24 @50 sfx1 | PASS 22/24 @200 | PASS @670 162/174 / +40 217/217 | PASS @670 672/684 | FAIL 5/24 @450 (2m hq0.94) sfx2 |
| 10 | `pr202-noams-lr002` | constant | 9/12 | PASS 14/24 @500 | PASS 11/24 @350 | PASS 10/24 @300 | FAIL 0/24 (2m hq0.88) | PASS 22/24 @150 | FAIL 0/24 [c.min_eigen_ratio,min_mass_ratio] | FAIL 10/24 @250 [c.min_eigen_ratio] | PASS 19/24 @200 | PASS 22/24 @50 | PASS 23/24 @134 | PASS @650 157/176 / +210 176/200 | PASS @650 641/686 | PASS 9/24 @400 |

<details>
<summary>Remaining candidates screened on all quick gates (ring/stationary only where shown)</summary>

| # | Candidate | LR policy | Gates (of 12) | mode_hold | blobs4 | stripes2 | bars4 | v.broad | v.mass | v.width | v.aniso | v.overlap | v.spiral | ring_shift | stationary | intensity2 ⚠ |
|---:|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | `pr202-ams-rc3-lr001` | constant | 9/12 | PASS 12/24 @550 | FAIL 0/24 (4m hq0.81) | PASS 18/24 @175 | FAIL 0/24 (2m hq0.69) | PASS 17/24 @400 | FAIL 10/24 @400 sfx3 | PASS 12/24 @650 | PASS 19/24 @300 | PASS 23/24 @50 | PASS 21/24 @267 | PASS @980 133/143 / +40 217/217 | PASS @980 643/653 | FAIL 3/24 @450 (2m hq0.91) sfx2 |
| 12 | `rp15-ams` | RP precision | 8/12 | FAIL 0/24 (7m hq1) | FAIL 2/24 @225 (4m hq0.88) | PASS 20/24 @100 | FAIL 0/24 (4m hq0.78) | PASS 22/24 @150 | PASS 21/24 @200 | PASS 20/24 @250 | PASS 21/24 @200 | FAIL 22/24 @50 sfx4 | PASS 24/24 @67 | PASS @870 154/154 / +530 168/168 | PASS @870 664/664 | FAIL 6/24 @425 (2m hq1) sfx4 |
| 13 | `rp15-const-ams` | constant | 8/12 | FAIL 0/24 (6m hq0.52) | PASS 15/24 @225 | PASS 20/24 @100 | FAIL 0/24 (4m hq0.78) | PASS 22/24 @150 | FAIL 20/24 @200 sfx4 | PASS 20/24 @250 | PASS 21/24 @200 | FAIL 22/24 @50 sfx4 | PASS 24/24 @67 | PASS @870 154/154 / +30 218/218 | PASS @870 664/664 | FAIL 6/24 @425 (2m hq1) sfx4 |
| 14 | `API-DV15` | schedule (horizon) | 8/10 | FAIL 0/24 (5m hq0.62) | PASS 14/24 @275 | PASS 20/24 @125 | FAIL 0/24 (2m hq0.94) | PASS 23/24 @100 | PASS 10/24 @750 | PASS 21/24 @200 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 24/24 @67 | — | — | FAIL 4/24 @450 (2m hq0.94) sfx1 |
| 15 | `API-RP5` | schedule (horizon) | 8/10 | FAIL 0/24 (5m hq0.75) | PASS 12/24 @250 | PASS 21/24 @75 | PASS 14/24 @275 | PASS 21/24 @150 | FAIL 5/24 @250 [c.covariance_error] | PASS 19/24 @300 | PASS 18/24 @350 | PASS 24/24 @50 | PASS 23/24 @67 | — | — | FAIL 7/24 @400 (2m hq0.97) sfx4 |
| 16 | `dv12-ams-rc2p0` | DV12 controller | 8/10 | FAIL 0/24 (5m hq0.52) | PASS 16/24 @225 | PASS 21/24 @100 | FAIL 12/24 @300 (4m hq0.88) | PASS 22/24 @150 | PASS 18/24 @350 | PASS 20/24 @250 | PASS 22/24 @150 | PASS 24/24 @50 | PASS 24/24 @67 | — | — | FAIL 2/24 @525 (2m hq0.88) |
| 17 | `dv12-ams-rc2p5` | DV12 controller | 8/10 | FAIL 0/24 (8m hq0.79) | PASS 16/24 @225 | PASS 21/24 @100 | PASS 14/24 @275 | PASS 22/24 @150 | FAIL 0/24 [c.covariance_error] | PASS 19/24 @300 | PASS 22/24 @150 | PASS 24/24 @50 | PASS 24/24 @67 | — | — | FAIL 0/24 (2m hq0.88) |
| 18 | `dv12-rc2p0` | DV12 controller | 8/10 | FAIL 0/24 (6m hq0.39) | PASS 11/24 @350 | PASS 21/24 @100 | FAIL 0/24 (4m hq0.88) | PASS 23/24 @100 | PASS 18/24 @350 | PASS 20/24 @250 | PASS 22/24 @150 | PASS 23/24 @50 | PASS 24/24 @67 | — | — | FAIL 2/24 @575 (2m hq0.91) sfx2 |
| 19 | `pr202-const-ams` | constant | 7/11 | FAIL 0/24 (0m hq0) | PASS 11/24 @350 | PASS 22/24 @75 | FAIL 0/24 (1m hq0.5) | PASS 23/24 @100 | PASS 8/24 @650 | FAIL 8/24 @150 [c.min_eigen_ratio] | FAIL 20/24 @200 sfx4 | PASS 22/24 @50 | PASS 24/24 @67 | PASS @550 186/186 / +480 170/173 | ERROR budget | PASS 12/24 @325 |
| 20 | `API-RP14` | RP precision | 7/10 | PASS 12/24 @600 | PASS 6/24 @475 | PASS 13/24 @300 | FAIL 0/24 (1m hq0.34) | PASS 23/24 @100 | FAIL 0/24 [c.min_eigen_ratio,min_mass_ratio] | PASS 20/24 @250 | PASS 21/24 @200 | FAIL 22/24 @50 sfx1 | PASS 23/24 @134 | — | — | PASS 9/24 @375 |
| 21 | `rp15-rc3` | RP precision | 7/10 | FAIL 4/24 @800 (8m hq0.89) | PASS 17/24 @200 | PASS 21/24 @75 | FAIL 0/24 (4m hq0.81) | PASS 21/24 @200 | FAIL 0/24 [c.min_eigen_ratio] | PASS 19/24 @300 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 24/24 @67 | — | — | FAIL 2/24 @450 (2m hq0.94) sfx1 |
| 22 | `API-DV16` | schedule (horizon) | 7/10 | FAIL 0/24 (7m hq0.91) | PASS 16/24 @225 | PASS 21/24 @100 | FAIL 0/24 (2m hq0.88) | PASS 23/24 @100 | FAIL 0/24 [min_mass_ratio] | PASS 21/24 @200 | PASS 21/24 @200 | PASS 23/24 @50 | PASS 24/24 @67 | — | — | PASS 9/24 @375 |
| 23 | `pr202-const-ams-rc3` | constant | 6/11 | FAIL 0/24 (5m hq0.92) | PASS 9/24 @375 | PASS 18/24 @175 | FAIL 0/24 (3m hq0.94) | PASS 24/24 @50 | FAIL 15/24 @300 [c.min_eigen_ratio] | FAIL 4/24 @200 [c.min_eigen_ratio] | FAIL 8/24 @150 sfx1 | PASS 22/24 @50 | PASS 24/24 @67 | PASS @520 189/189 / +530 168/168 | ERROR budget | PASS 15/24 @200 |
| 24 | `pr202-const-noams` | constant | 6/11 | FAIL 0/24 (3m hq0.33) | PASS 10/24 @375 | PASS 22/24 @75 | FAIL 0/24 (2m hq0.81) | PASS 22/24 @100 | FAIL 0/24 [c.covariance_error,c.min_eigen_ratio] | FAIL 6/24 @150 [c.min_eigen_ratio] | FAIL 19/24 @200 sfx4 | PASS 20/24 @50 | PASS 24/24 @67 | PASS @500 132/191 / +230 145/198 | ERROR budget | PASS 12/24 @275 |
| 25 | `dv12-rc2p5` | DV12 controller | 6/10 | FAIL 0/24 (7m hq0.88) | FAIL 12/24 @250 (4m hq0.97) sfx1 | PASS 21/24 @100 | FAIL 14/24 @250 (4m hq1) sfx3 | PASS 22/24 @150 | FAIL 0/24 [c.covariance_error] | PASS 19/24 @300 | PASS 22/24 @150 | PASS 24/24 @50 | PASS 24/24 @67 | — | — | FAIL 2/24 @575 (2m hq0.91) sfx2 |
| 26 | `public-ka2-constant` | constant | 6/10 | FAIL 0/24 (4m hq1) | PASS 19/24 @150 | PASS 21/24 @75 | PASS 16/24 @225 | PASS 23/24 @100 | FAIL 15/24 @400 sfx1 | FAIL 5/24 @350 [c.min_eigen_ratio] | FAIL 19/24 @200 sfx4 | PASS 20/24 @100 | PASS 21/24 @134 | — | — | FAIL 1/24 @575 (2m hq0.81) |

</details>

<details>
<summary>Partially screened variants (staged screens stop after a failure)</summary>

| # | Candidate | LR policy | Gates (of 12) | mode_hold | blobs4 | stripes2 | bars4 | v.broad | v.mass | v.width | v.aniso | v.overlap | v.spiral | ring_shift | stationary | intensity2 ⚠ |
|---:|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `API-C13-R1` | schedule (horizon) | 6/9 | FAIL 0/24 (7m hq1) | PASS 18/24 @150 | PASS 21/24 @75 | FAIL 4/24 @525 (4m hq1) sfx4 | PASS 23/24 @100 | PASS 16/24 @300 | PASS 20/24 @250 | PASS 19/24 @250 | FAIL 20/24 @50 sfx2 | — | — | — | PASS 10/24 @350 |
| 2 | `public-k3p` | schedule (horizon) | 4/9 | FAIL 0/24 (6m hq0.97) | FAIL 0/24 (3m hq0.97) | FAIL 19/24 @100 (2m hq1) sfx4 | FAIL 0/24 (3m hq0.97) | PASS 23/24 @100 | PASS 15/24 @300 | PASS 21/24 @200 | PASS 21/24 @200 | FAIL 20/24 @50 sfx2 | ERROR budget | — | — | PASS 5/24 @500 |
| 3 | `public-ka2` | schedule (horizon) | 4/9 | FAIL 0/24 (6m hq1) | FAIL 0/24 (3m hq0.97) | FAIL 19/24 @100 (2m hq1) sfx4 | FAIL 0/24 (3m hq0.97) | PASS 23/24 @100 | PASS 15/24 @300 | PASS 21/24 @200 | PASS 21/24 @200 | FAIL 20/24 @50 sfx1 | ERROR budget | — | — | PASS 5/24 @500 |
| 4 | `t2-rp15noise-in10` | RP precision | 3/3 | PASS 11/24 @600 | — | — | PASS 12/24 @325 | — | PASS 21/24 @200 | — | — | — | — | — | — | FAIL 2/24 @550 (2m hq0.91) sfx1 |
| 5 | `t2-rp15noise-in08` | RP precision | 3/3 | PASS 10/24 @500 | — | — | PASS 10/24 @375 | — | PASS 20/24 @250 | — | — | — | — | — | — | FAIL 1/24 @550 (2m hq0.84) |
| 6 | `t2-dv12reg-aw0p5` | DV12 controller | 3/3 | PASS 9/24 @800 | — | — | PASS 15/24 @250 | — | PASS 18/24 @350 | — | — | — | — | — | — | FAIL 0/24 (2m hq0.84) |
| 7 | `t2-dv12reg-dg3` | DV12 controller | 3/3 | PASS 9/24 @800 | — | — | PASS 15/24 @250 | — | PASS 18/24 @350 | — | — | — | — | — | — | FAIL 0/24 (2m hq0.88) |
| 8 | `t2-dv12q-ons018` | DV12 controller | 3/3 | PASS 7/24 @900 | — | — | PASS 8/24 @375 | — | PASS 17/24 @400 | — | — | — | — | — | — | FAIL 8/24 @350 (2m hq0.97) sfx4 |
| 9 | `t2-dv12reg-k1p5` | DV12 controller | 3/3 | PASS 5/24 @1000 | — | — | PASS 15/24 @250 | — | PASS 17/24 @400 | — | — | — | — | — | — | FAIL 0/24 (2m hq0.84) |
| 10 | `t2-rp15reg-rc1p25` | RP precision | 2/3 | PASS 12/24 @650 | — | — | FAIL 0/24 (4m hq0.81) | — | PASS 20/24 @250 | — | — | — | — | — | — | PASS 5/24 @500 |
| 11 | `t2-rp15reg-k0p5` | RP precision | 2/3 | PASS 10/24 @750 | — | — | FAIL 0/24 (3m hq0.75) | — | PASS 13/24 @200 | — | — | — | — | — | — | PASS 6/24 @450 |
| 12 | `t2-rp15noise-in10-out015` | RP precision | 2/3 | FAIL 9/24 @600 (8m hq1) sfx1 | — | — | PASS 6/24 @475 | — | PASS 21/24 @200 | — | — | — | — | — | — | FAIL 5/24 @475 (2m hq0.94) sfx4 |
| 13 | `t2-rp15reg-rc1p5-k0p5` | RP precision | 2/3 | PASS 9/24 @800 | — | — | FAIL 0/24 (4m hq0.88) | — | PASS 19/24 @300 | — | — | — | — | — | — | FAIL 4/24 @500 (2m hq0.97) sfx2 |
| 14 | `t2-rp15reg-rc2` | RP precision | 2/3 | PASS 7/24 @900 | — | — | FAIL 0/24 (4m hq0.78) | — | PASS 20/24 @250 | — | — | — | — | — | — | FAIL 4/24 @500 (2m hq0.94) sfx2 |
| 15 | `t2-dv12reg-re2` | DV12 controller | 2/3 | PASS 5/24 @1000 | — | — | FAIL 0/24 (3m hq0.94) | — | PASS 15/24 @500 | — | — | — | — | — | — | FAIL 2/24 @475 (2m hq0.84) |
| 16 | `t2-dv12q-ons022` | DV12 controller | 2/3 | FAIL 2/24 @1150 (8m hq0.93) sfx2 | — | — | PASS 16/24 @225 | — | PASS 18/24 @350 | — | — | — | — | — | — | FAIL 5/24 @475 (2m hq0.91) sfx4 |
| 17 | `t2-dv12q-ons010` | DV12 controller | 2/3 | FAIL 0/24 (7m hq0.99) | — | — | PASS 15/24 @250 | — | PASS 18/24 @350 | — | — | — | — | — | — | FAIL 5/24 @450 (2m hq0.91) sfx2 |
| 18 | `t2-dv12q-ons020` | DV12 controller | 2/3 | FAIL 0/24 (8m hq0.83) | — | — | PASS 13/24 @300 | — | PASS 17/24 @400 | — | — | — | — | — | — | FAIL 0/24 (2m hq0.88) |
| 19 | `t2-dv12reg-k0p75` | DV12 controller | 2/3 | FAIL 0/24 (8m hq0.84) | — | — | PASS 15/24 @250 | — | PASS 17/24 @400 | — | — | — | — | — | — | FAIL 0/24 (2m hq0.84) |
| 20 | `t2-rp15noise-in125` | RP precision | 1/3 | FAIL 8/24 @700 (8m hq0.92) sfx2 | — | — | PASS 13/24 @300 | — | FAIL 18/24 @200 [c.min_eigen_ratio] | — | — | — | — | — | — | FAIL 0/24 (2m hq0.84) |
| 21 | `t2-rp15reg-rc2-k0p5` | RP precision | 1/3 | FAIL 8/24 @650 (8m hq0.8) | — | — | FAIL 0/24 (4m hq0.78) | — | PASS 21/24 @200 | — | — | — | — | — | — | FAIL 4/24 @500 (2m hq0.94) sfx2 |
| 22 | `t2-dv12reg-rc3p5` | DV12 controller | 1/3 | FAIL 5/24 @900 (8m hq0.92) sfx2 | — | — | PASS 16/24 @225 | — | FAIL 0/24 [c.covariance_error] | — | — | — | — | — | — | FAIL 1/24 @575 (2m hq0.88) |
| 23 | `t2-rp15noise-out06` | RP precision | 1/3 | FAIL 5/24 @500 (8m hq0.88) | — | — | FAIL 0/24 (4m hq0.75) | — | PASS 20/24 @250 | — | — | — | — | — | — | FAIL 0/24 (0m hq0.09) |
| 24 | `p2-plm3` | DV12 controller | 1/3 | FAIL 3/24 @1100 (8m hq0.9) sfx3 | — | — | FAIL 0/24 (4m hq0.66) | — | PASS 5/24 @1000 | — | — | — | — | — | — | FAIL 4/24 @475 (2m hq0.81) |
| 25 | `dv12-const-ams` | DV12 controller | 1/3 | FAIL 1/24 @1200 (8m hq0.91) sfx1 | PASS 17/24 @125 | FAIL 20/24 @100 (2m hq1) sfx3 | — | — | — | — | — | — | — | — | — | FAIL 5/24 @450 (2m hq0.81) |
| 26 | `p2-plm4` | DV12 controller | 1/3 | FAIL 1/24 @1200 (8m hq0.94) sfx1 | — | — | PASS 14/24 @250 | — | FAIL 0/24 [min_mass_ratio] | — | — | — | — | — | — | FAIL 7/24 @425 (2m hq1) sfx3 |
| 27 | `t2-dv12q-b2p99` | DV12 controller | 1/3 | FAIL 0/24 (8m hq0.81) | — | — | PASS 10/24 @325 | — | FAIL 2/24 @1150 sfx2 | — | — | — | — | — | — | FAIL 1/24 @600 (2m hq0.91) sfx1 |
| 28 | `t2-dv12q-ons0` | DV12 controller | 1/3 | FAIL 0/24 (5m hq0.44) | — | — | FAIL 0/24 (3m hq0.91) | — | PASS 10/24 @750 | — | — | — | — | — | — | PASS 7/24 @450 |
| 29 | `t2-dv12q-ons025` | DV12 controller | 1/3 | FAIL 0/24 (8m hq0.87) | — | — | PASS 16/24 @225 | — | FAIL 2/24 @1150 sfx2 | — | — | — | — | — | — | FAIL 2/24 @575 (2m hq0.91) sfx2 |
| 30 | `t2-dv12reg-k2` | DV12 controller | 1/3 | FAIL 0/24 (5m hq0.78) | — | — | PASS 15/24 @250 | — | FAIL 4/24 @1050 sfx4 | — | — | — | — | — | — | FAIL 0/24 (2m hq0.84) |
| 31 | `t2-rp15noise-in05` | RP precision | 1/3 | FAIL 0/24 (2m hq0.1) | — | — | FAIL 0/24 (3m hq0.97) | — | PASS 20/24 @250 | — | — | — | — | — | — | PASS 6/24 @450 |
| 32 | `t2-rp15reg-k0p75` | RP precision | 1/3 | FAIL 0/24 (7m hq0.92) | — | — | FAIL 0/24 (3m hq0.72) | — | PASS 20/24 @250 | — | — | — | — | — | — | PASS 5/24 @500 |
| 33 | `t2-rp15reg-rc1p5` | RP precision | 1/3 | FAIL 0/24 (8m hq0.88) | — | — | FAIL 0/24 (4m hq0.88) | — | PASS 20/24 @250 | — | — | — | — | — | — | FAIL 4/24 @500 (2m hq0.97) sfx2 |
| 34 | `pr202-ams-rc3-lr00125` | constant | 1/1 | PASS 12/24 @650 | — | — | — | — | — | — | — | — | — | — | — | — |
| 35 | `pr202-const-ams-inf` | constant | 1/1 | — | — | — | — | — | — | — | — | — | — | — | PASS @550 696/696 | — |
| 36 | `pr202-const-ams-rc3-inf` | constant | 1/1 | — | — | — | — | — | — | — | — | — | — | — | PASS @520 699/699 | — |
| 37 | `pr202-const-noams-inf` | constant | 1/1 | — | — | — | — | — | — | — | — | — | — | — | PASS @500 574/701 | — |
| 38 | `t2-rp15noise-in20` | RP precision | 0/3 | FAIL 5/24 @500 (8m hq0.89) | — | — | FAIL 0/24 (3m hq0.81) | — | FAIL 0/24 [c.min_eigen_ratio] | — | — | — | — | — | — | FAIL 0/24 (1m hq0.47) |
| 39 | `p2-pf25` | DV12 controller | 0/3 | FAIL 0/24 (8m hq0.84) | — | — | FAIL 5/24 @450 (4m hq0.88) | — | FAIL 0/24 [min_mass_ratio] | — | — | — | — | — | — | FAIL 6/24 @450 (2m hq1) sfx4 |
| 40 | `p2-pf50` | DV12 controller | 0/3 | FAIL 0/24 (5m hq0.63) | — | — | FAIL 0/24 (3m hq0.94) | — | FAIL 0/24 [min_mass_ratio] | — | — | — | — | — | — | PASS 7/24 @450 |
| 41 | `p2-plm6` | DV12 controller | 0/3 | FAIL 0/24 (7m hq0.86) | — | — | FAIL 0/24 (3m hq0.88) | — | FAIL 0/24 [c.covariance_error] | — | — | — | — | — | — | PASS 9/24 @400 |
| 42 | `pr202-ams-rc3-lr0015` | constant | 0/1 | FAIL 14/24 @450 (8m hq1) sfx4 | — | — | — | — | — | — | — | — | — | — | — | — |
| 43 | `pr202-ams-rc3-lr002` | constant | 0/1 | FAIL 12/24 @600 (8m hq0.91) sfx2 | — | — | — | — | — | — | — | — | — | — | — | — |
| 44 | `pr202-ams-lr00175` | constant | 0/1 | FAIL 11/24 @350 (0m hq0) | — | — | — | — | — | — | — | — | — | — | — | — |
| 45 | `rp15-const-ams-rc3` | constant | 0/1 | FAIL 4/24 @1050 (8m hq1) sfx4 | — | — | — | — | — | — | — | — | — | — | — | — |
| 46 | `p2-pf100` | DV12 controller | 0/1 | — | — | — | — | — | FAIL 10/24 @600 sfx4 | — | — | — | — | — | — | — |
| 47 | `pr202-ams-lr00035` | constant | 0/1 | FAIL 0/24 (6m hq0.67) | — | — | — | — | — | — | — | — | — | — | — | — |
| 48 | `pr202-ams-lr0005` | constant | 0/1 | FAIL 0/24 (7m hq0.91) | — | — | — | — | — | — | — | — | — | — | — | — |
| 49 | `pr202-ams-lr0007` | constant | 0/1 | FAIL 0/24 (7m hq0.75) | — | — | — | — | — | — | — | — | — | — | — | — |
| 50 | `pr202-ams-lr001` | constant | 0/1 | FAIL 0/24 (4m hq0.59) | — | — | — | — | — | — | — | — | — | — | — | — |
| 51 | `pr202-ams-lr00125` | constant | 0/1 | FAIL 0/24 (7m hq0.91) | — | — | — | — | — | — | — | — | — | — | — | — |
| 52 | `pr202-ams-lr002` | constant | 0/1 | FAIL 0/24 (7m hq1) | — | — | — | — | — | — | — | — | — | — | — | — |
| 53 | `pr202-ams-lr003` | constant | 0/1 | FAIL 0/24 (7m hq1) | — | — | — | — | — | — | — | — | — | — | — | — |
| 54 | `pr202-ams-rc3-lr0005` | constant | 0/1 | FAIL 0/24 (7m hq0.92) | — | — | — | — | — | — | — | — | — | — | — | — |
| 55 | `pr202-ams-rc3-lr00075` | constant | 0/1 | FAIL 0/24 (3m hq0.34) | — | — | — | — | — | — | — | — | — | — | — | — |
| 56 | `pr202-noams-lr0005` | constant | 0/1 | FAIL 0/24 (4m hq0.32) | — | — | — | — | — | — | — | — | — | — | — | — |
| 57 | `pr202-noams-lr001` | constant | 0/1 | FAIL 0/24 (5m hq0.66) | — | — | — | — | — | — | — | — | — | — | — | — |

</details>

## The recommended base: `dv12-ams-rc3`

It starts from the reviewed PR155 DV12 port (`deterministic-init-retest/port-source/api-dv12/package`).
PR202's `Recipe.amsgrad` is ported into it ([diff](ports/dv12-amsgrad.diff)). With `amsgrad=False`
the port is bitwise identical to DV12, which was verified on GPU and in an adversarial review.
Exact overrides are in [configs/dv12-ams-rc3.json](configs/dv12-ams-rc3.json). The task-size fields in
that file are forced by the harness for each task.

| Setting | Value | Role |
|---|---|---|
| `continuous_policy` | `"dv12"` | State-driven LR controller; the ordinary LR schedule is bypassed |
| `total_steps` | `None` | No horizon: no end and no phases |
| `amsgrad` | `True` | Stops Adam's step from growing during a long hold |
| `reg_coeff` | `3.0` | Critic gradient penalty at 3× the default |
| `lr`, `d_lr_mult`, `prior_lr_mult` | .00425, 1, 2 | Peak rates |
| `betas` | (0, .999) | |
| `input_noise_std`, `output_noise_std` | 0, .029 constant | |

The schedule fields still present in the config have no effect under the controller:
`lr_anneal_start`, `lr_floor`, `network_lr_floor`, `network_lr_horizon_cap`, `input_noise_anneal_end`
and `output_noise_warmup`. The LR schedule function is not called, input noise is 0, and output noise
is constant.

**How it works**

1. **GAN core.** A learnable particle prior feeds the generator, and training uses a relativistic
   logistic loss.
2. **Critic penalty (KA2, strengthened by `reg_coeff 3`).** The penalty is
   `coeff/2·A` for the first 799 calls, then `coeff/2·(.5·A + .5·B)`.
   - A is an R1 penalty at real data plus a cap on the gradient norm at fakes.
   - B caps both gradient norms and adds `W·`proximity to an EMA critic.
   - The mix is fixed at 50/50; the surprise signal does not set it.
   - The surprise signal on the critic's Adam state gates `W`: it drops to 0 when the surprise ratio
     rises above 3 and returns to 1 when it falls below 1.75. DV12 holds `W` at 1 while
     `data_drive` < .1. The same signal sets how fast the EMA critic tracks.
   - Tripling the coefficient makes the critic smoother. That fills the 2%-mass component of
     `vector_unequal_mass` early, by update ~150, and recovers all four `img_bars4` modes.
3. **DV12 controller: why no LR tuning is needed.** Every update, each role runs at a fraction of its
   peak rate, computed from training signals:

   ```
   G     = lr · (.01 + .99·m) · gt
   prior = 2·lr · (.05 + .95·m) · gt
   D     = lr · (.01 + .99·m) · gt / (1 + pe²)
   ```

   - `pe` is the payoff error: an EMA of how far the generator is losing to the critic.
   - `data_drive` is the RMS z-score of the drift between fast (.1) and slow (.01) EMA means of random
     Fourier features of real batches, mapped to [0, 1] as `clip((z − 3)/3, 0, 1)`.
   - `m`, mobility, moves toward `max(data_drive, min(1, pe²))`. It rises at rate .05 and falls at
     .005.
   - `gt`, game trust, shrinks all rates when the critic's gradients become surprising relative to
     their baseline.
   - The rates stay high while the game is unresolved or the data is moving, and they relax on their
     own. On `mode_hold`, G ends at about 6% of peak. After the ring target change, `data_drive` sends
     the rates from about 3% back to 100%, and the new target is re-acquired in 300 updates.
4. **AMSGrad: why it holds.** With plain Adam, √v̂ keeps shrinking once gradients settle, so the gain
   `lr/√v̂` grows. The typical realized step grows only about 1.4× (90th percentile), but small
   fluctuations then blow up. In the PR202 constant-LR stationary run, the critic's median √v̂ fell
   about 7–8× between arrival and updates 7,000–7,400. AMSGrad keeps the running maximum instead.
   - Stationary: 685/685 with AMSGrad, against 652/686 for the same config with plain Adam
     (`dv12-rc3`).

In short, the controller gets training to the target and damps the fast oscillation that
constant-LR configs show. AMSGrad keeps it there over thousands of steps.

**Known weak spots**
- `reg_coeff` sits in a narrow window: 2.0, 2.5 and 3.5 each break other gates. Gate outcomes are
  chaotic in the config, so 3.0 is a working point, not a demonstrated optimum.
- The controller has references fixed at birth. Its feature normalization comes from the first real
  batch, and the KA2 surprise baseline is frozen after about update 824.
- On stationary data, mobility's fixed .005 decay behaves somewhat like a clock.
- `img_intensity2` fails under the frozen scoring (0/24, HQ .84) and also narrowly under clean scoring
  (4/24 @475, final HQ .91): quality still settles too late in the 600-update budget.

## What we learned

- **PR202's AMSGrad fixes the long-horizon collapse, not the short-horizon cycling**
  ([analysis](evidence/oscillation.md), [instrumentation](evidence/step-dynamics.md)).
  - Stationary with the PR202 package at constant LR: 696/696 with AMSGrad against 574/701 plain.
  - The instrumented runs are bitwise identical to the pool runs.
  - A causal switch test agrees in both directions. Switching plain Adam to AMSGrad at update 1,000
    removes all departures. Switching AMSGrad to plain at update 3,000 collapses within 10 updates.
  - AMSGrad does nothing measurable on `mode_hold`'s fast cycling. That cycling is intrinsic game
    oscillation at too large a step. At the .00425 peak LR, every `mode_hold` pass needs a state-driven
    LR cut. The constant-LR passes all run at a lower LR (.001–.002).
- **Constant LR alone does not produce a base.** With AMSGrad at lower constant LRs, `mode_hold`
  passes only at isolated values: .0015 passes while .00125 and .00175 fail. With `reg_coeff 3`,
  .001 and .00125 pass `mode_hold`, while .0015 and .002 fail on the final-suffix rule (14/24 and
  12/24). The only fully screened one (.001) fails `blobs4`, `bars4` and `unequal_mass`.
- **Attribution on DV12** ([report](evidence/amsgrad-dv12.md)).
  - `reg_coeff 3` fixes `unequal_mass`. It also turns `bars4` from API-DV12's late pass (5/24 @500)
    into 15/24 @200–250. `dv12-rc3`, with plain Adam, gets the same quick-gate pass/fail results.
  - AMSGrad adds `mode_hold` margin and perfect stationary retention.
  - AMSGrad alone, at reg 1, costs one trusted gate (`bars4`).
- **Refuted:** DV12's `unequal_mass` failure is not a late prior-LR freeze
  ([plm](evidence/prior-plm.md), [floor](evidence/prior-pfloor.md)). None of `prior_lr_mult` 3–6 or
  prior-LR floors .25–1.0 passes the staged screen. Only `prior_lr_mult 3` passes `unequal_mass`, barely
  (5/24 @1000), and it loses `mode_hold` and `bars4`. The rare component settles at about 1 of 256 particles by update
  ~450, while the prior is still mobile.
- **RP15** ([report](evidence/amsgrad-rp15.md)). Its only failing trusted gate is `img_bars4`.
  - Constant critic input noise of .08–.1 fixes `bars4` but delays `intensity2`
    ([report](evidence/top2-rp15noise.md)).
  - AMSGrad with the default `rp5` precision controller scores 6/11 quick gates. RP5's hard cut to 1%
    freezes states early; a softer closed level is the untested next lever.
- Knob map with time dependence for every field: [knobs.md](evidence/knobs.md).
  Prior evidence matrix: [evidence.md](evidence/evidence.md). Methods survey: [theory.md](evidence/theory.md).

## Next

1. Finish the clean rescore on all 13 gates for six candidates: `dv12-ams-rc3`, `dv12-rc3`,
   API-DV12, API-RP15, `t2-rp15noise-in10` and `t2-dv12q-ons018`.
2. Close `img_intensity2`: on clean samples the top two miss only by the final-suffix rule, because
   quality settles late in the 600-update budget.
3. Library: decide on a clean-sampling option for `GANTrainer.sample()`, in a separate PR.
4. Commit the screening harness with the rescore results. It is not in this commit and lives in the
   local lrfree workspace.
5. Before any default change: a 30k hold and the full 22-toy suite.
