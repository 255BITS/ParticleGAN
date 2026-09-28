# 22-check leaderboard: the eight custom hosts ported to the LR-free harness

The 22 toy checks are the 19 frozen older hosts plus the three native 100-Gaussian tasks. Until now the
harness ran 14 of them: 11 quick gates and the 3 native tasks. The other eight hosts (`two_pole`,
`trajectory`, `residual_student`, `unipolar`, `ae_gan_hold`, `cover_leftover`, `unused_token_hold`,
`mid_scale_identity`) have their own training loops and only ran through the old scheduled route. They now
run for any candidate package: preset `custom` runs the 8, and `all22` runs all 22 checks.

All scores are **noisy**, the model's own sampling law. Output noise is applied at each host's frozen noise
sites, and only `trajectory`, `residual_student` and `ae_gan_hold` score samples. The other five score
parameters, so noisy equals clean there.

## Leaderboard

| Candidate | Quick gates | Custom 8 | Native 3 | **Total /22** | ring_shift / stationary |
|---|---:|---:|---:|---:|---|
| **`st-10`**: stationarity-tested LR + learnable σ | 11/11 | 5/8 | 0/3 | **16/22** | PASS / PASS |
| **`dv12-ams-rc3`**: #217 default | 11/11 | 4/8 | 0/3 | **15/22** | PASS / PASS |
| *Reference, not a candidate:* develop K3P, annealed, horizon = host budget | — | **8/8** | rotated and staggered PASS, grid FAIL | — | — |

- Quick gates count `img_intensity2` at 1,200 updates.
- Native cells are FAIL for every continuous learner; see the native sections of the main README.

Custom-host cells: passing checks /24, then the first passing update.

| Host (budget) | `st-10` | `dv12-ams-rc3` | K3P reference |
|---|---|---|---|
| two_pole (80) | FAIL 0/24 · mean_abs .031 (≥ .30) | FAIL 0/24 · mean_abs .011 | PASS 9/24 @54 |
| trajectory (400) | FAIL 0/24 · identity_mse .263 (≤ .02) | PASS 11/24 @234 | PASS 22/24 @50 |
| residual_student (400) | PASS 21/24 @50 | PASS 22/24 @50 | PASS 22/24 @34 |
| unipolar (400) | PASS 15/24 @167 | PASS 11/24 @234 | PASS 18/24 @117 |
| ae_gan_hold (250) | PASS 23/24 @21 | PASS 23/24 @21 | PASS 22/24 @32 |
| cover_leftover (800) | PASS 13/24 @400 | FAIL 0/24 · u_kept .680 (≥ .85), pole errors .212/.224 (≤ .20) | PASS 11/24 @467 |
| unused_token_hold (200) | FAIL 2/24 @192 · final values pass, streak 2 < 5 | FAIL 0/24 · concept_move .709 (≥ .85) | PASS 10/24 @125 |
| mid_scale_identity (800) | PASS 12/24 @434 | FAIL 0/24 · identity_at_0 .790 (≥ .85) | PASS 16/24 @300 |

The per-cell fields (final metrics, thresholds, config hash, engine and binding sha256, parity record) are in
[`results.json`](results.json).

## How the port works

- **The hosts stay verbatim.** The 13 pinned host sources are copied byte for byte, with sha256 checks, into
  [`../harness/hosts/custom/`](../harness/hosts/custom/). The bindings in
  [`../harness/custom22.py`](../harness/custom22.py) keep each host's data, models, auxiliary losses,
  observation schedule, scorers and frozen verdict (`protocol.test_verdict`: 24 observations, passing suffix of
  at least 5, every final live threshold). Line references point back to the originals.
- **The candidate's own policy replaces the host's.** [`../harness/components.py`](../harness/components.py)
  builds what the candidate package's `GANTrainer` builds, using the package's own functions and in the same
  order:
  - optimizers with g/prior/d role groups, amsgrad and A2;
  - the KA2 penalty with its EMA critic, spike guard and controller link;
  - the DV12 controller or the stationarity LR, learnable σ and EMA.

  It then fires the `GANTrainer._step` hooks at each host's D and G points. Host Adam, legacy loss, host LR
  schedules and `schedule_optimizer` are removed. Learning rates are written only by the engine, which is
  asserted on every update.
- **Unsupported settings are refused, not dropped.** These give `ERROR refused`: critic input noise,
  dv11, unknown recipe fields and unknown `GANTrainer` hooks. `particle_birth_death` runs only where
  the host binds plain trainable `ParticlePrior` tables (residual_student, cover_leftover, and the
  non-MoG trajectory/unipolar runs); it is refused on no-table hosts (two_pole, unused_token_hold,
  mid_scale_identity) and MoG ae_gan_hold.

## Why the numbers can be trusted

1. **The copies are verbatim.** With the engine switched off, the rewritten loops reproduce the original PR #155
   hosts exactly for all 8 hosts over 12 updates: the per-step sha256 of LRs, parameters and gradients, the
   curves and the final metrics. The training output-noise call sequence is also identical.
2. **The engine matches the package.** Before a package's first custom job, the engine is checked against that
   package's own `GANTrainer.step`, bitwise after every update for 1,000 updates on CPU. The check compares
   losses, every tensor, both optimizer states (KA2 record, A2 history), learning rates, controller and settle
   state, σ, and all streams and RNG. It passes for `dv12-ams-rc3`, `st-10` and develop K3P, and a one-ulp
   change is caught.
3. **Known-good control.** Develop K3P with its declared annealed schedule (horizon = host budget), running
   through the same engine, passes **8/8**. One caveat: critic input noise is off in this control, because the
   engine does not bind input noise. On the frozen route, K3P passes `two_pole` with input noise .5 and with 0
   (9/24 and 10/24), so input noise is not what decides these hosts.

One port defect was found by this control and fixed. `two_pole`'s particles are direct sample particles, and the
package defines their optimizer itself: `make_generator_optimizer(direct_particles=...)` with
`direct_particle_betas` (0, .9) and the up-to-2× agreement gain. The first binding wrapped them in a
`ParticlePrior` instead (betas (0, .999), no gain), so the step was half as large and every config failed. After the
fix, K3P passes (final mean_abs .60). See [design §9.6](design.md).

## What the failures say

- **`two_pole`: AMSGrad collapses the direct-particle step.** The particle gradient falls about 20× in the first 10
  updates, and AMSGrad keeps the update-1 second-moment maximum. The 12 particles move together and never split
  toward the poles; rc3's stronger penalty makes this worse. These diagnostic arms were run on the fixed
  binding; no candidate was changed:

  | Arm | Final mean_abs | Verdict |
  |---|---:|---|
  | rc3, amsgrad off | .478 | FAIL (suffix 4 of 5) |
  | rc3, amsgrad off, reg_coeff 1 | .584 | PASS |
  | rc3, reg_coeff 1, amsgrad on | .104 | FAIL |
  | K3P + amsgrad | .143 | FAIL |
  | K3P + reg_coeff 3 | .447 | PASS |

  AMSGrad fixed the long-horizon creep collapse (`stationary`). Here it costs the early fast phase, where
  gradients shrink quickly, so its max-memory is the structural issue.
- The other misses have not been diagnosed yet:
  - `dv12-ams-rc3` misses `cover_leftover`, `unused_token_hold` and `mid_scale_identity` by 6–20% of the
    threshold;
  - `st-10` misses `trajectory` badly (identity_mse .263) and `unused_token_hold` on streak length only.

## Binding decisions to review

These are not dictated by the sources. Each is documented in [design.md](design.md) §5 and §8:

- **B1.** `ae_gan_hold`'s MoG prior: the DV12 latent kernel reads the `means()` view, which leaves it effectively
  inactive on this host. The alternative is to turn it off explicitly.
- **B2.** `ae_gan_hold` is outside `GANTrainer`'s declared scope (AE encoder + MoG prior) and is labelled
  `extended_scope`. Its MoG table comes from the candidate recipe, not the host's own random draw: sampling width
  .018 vs .014.
- **B3.** The multi-scale critics (`unipolar` with 2 roles, `mid_scale_identity` with 4) make one role-union KA2
  call per update, which keeps `GANTrainer`'s clock.
- **B4.** `st-10`'s learnable σ also gets gradient from host auxiliary terms at the frozen noise sites.
- **B5.** `cover_leftover`'s two tables share one latent bandwidth.
- **Initialization.** 7 hosts use the candidate's `batch_feature_zero` instead of the PyTorch default, consistent
  with QR-primary native. `two_pole` has no network to initialize.

## Reproduce

The harness is committed at [`../harness/`](../harness/). It expects the working directory layout described in
its README: packages under `candidates/`, Python `/tmp/pr38-default-env/bin/python`.

```bash
PY=/tmp/pr38-default-env/bin/python
$PY harness/submit.py --cand NAME --package-root candidates/<pkg>/package --overrides <overrides.json> \
    --candidate-options '{"eval_output_noise": true}' --tasks custom      # or --tasks all22
$PY harness/wait.py NAME
$PY -m unittest -v harness/tests/test_components_parity.py              # port tests (about 8 min on CPU)
```

The custom hosts run on CPU and take 3 to 30 s each.
