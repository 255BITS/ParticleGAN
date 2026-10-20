# Stall-reach critic read (closed PR #107): falsified at Tier 1

Historical report from [closed source PR](https://github.com/255BITS/ParticleGAN/blob/de93dbf48cabe6bb2913b3eccf93029655da5db9/reports/forge/stall-reach/README.md). Its runtime options, tests and workflow are archived on that source; they are not imported by this recall-only publication. Original numerical verdicts and source identities are unchanged.

The rebuilt stall-reach read **fails `ring16_acquisition`**, which its matched
leader control passes. The falsifier is therefore met, and the candidate never
reaches Tier 2, so `mode_hold` (the primary hypothesis) stays unmeasured. The
reason is mechanical. BCAP holds the critic's input slope near its cap, so the
saturation-only stall rule fires on almost every update. The read stays at its
widest stencil (`.5 / kappa`), and the generator cannot resolve ring modes.
Keep the option opt-in (off by default). This round adds no default change, no
tuning, no retries and no seed variants.

| Task (tier) | Control: leader | stall_reach | Note |
| --- | --- | --- | --- |
| gaussian1d_smoke (1) | PASS | PASS | 985/1000 updates widened, 720 stalled |
| two_pole (1) | PASS | PASS | 54/80 widened, 0 stalled |
| unused_token_hold (1) | PASS | PASS | 142/200 widened, 3 stalled |
| ae_gan_hold (1) | PASS | PASS | 189/250 widened, 0 stalled |
| ring16_acquisition (1) | PASS (suffix 26, 16 modes, HQ .961) | **FAIL** (suffix 0/5, 14 modes, HQ .530) | 1597/1600 widened, 1545 stalled |
| five_word_joint_smoke (1) | INCOMPLETE (900 s limit at update 17,501/20,001) | INCOMPLETE (900 s limit at update 10,834/20,001) | Both acquired words (confirmed at 834 / 1667); the allowance expired |
| **Tier 1** | **5/6** | **4/6** | Lost: ring16_acquisition |
| All 21 Tier 2 tasks | BLOCKED | BLOCKED | A required Tier 1 non-PASS blocks Tier 2 for both arms |

Acceptance required 6/6 Tier 1, at least one matched Tier 2 FAIL→PASS, and no
lost control pass. The result is **FALSIFIED**: Tier 1 is 4/6, and
`ring16_acquisition` was lost.

## What ran

- **Recipes.** Both arms use the complete `recipe-prior-refactor-bcap-pure-v1`
  leader recipe: non-saturating BCAP, direction blend, zero-momentum DualNorm
  with smoothing .001, lr .012, D×1.5, prior×2.5, `reg_kappa` 1. The sole
  declared trainer delta is `generator_critic_read: none → stall_reach`
  ([spec](spec.json), [ideas](https://github.com/255BITS/ParticleGAN/blob/de93dbf48cabe6bb2913b3eccf93029655da5db9/configs/forge/ideas/bcap-stall-reach-base-v1.json)).
- **Protocol.** Protocol seed 0, the `deterministic_orthogonal` initializer and
  the `discriminator_stability` view at revision 9. Task architectures, data
  laws, priors, sampling, budgets, cadence and gates are unchanged.
- **Task cards.** Task cards changed only their host-source pins, recorded as
  evaluator revision `generator-critic-read-hooks-v1`.
- **Control.** One fresh control ran. Archived leader controls are bound to
  another source digest, so they could not be reused.
- **Executed source.** Source `ee0d05728ded`, source digest `cbba593bb98d`.
  Paid time was 1036.6 s for the control and 1048.1 s for the candidate.
- **Archive.** Bulk envelopes, checkpoints and logs are in
  `/mnt/ml7tb/ParticleGAN-forge/stall-reach-20261011/`, with one line per
  finished task in `run.log`. None of it is committed.

Mechanism (`particlegan/critic_read.py`):
- After each critic step, `observe` measures `s = RMS ||∇x D(fake)||` on the
  detached critic-phase fakes. It uses one autograd call, computes no
  parameter gradients and draws no RNG. It sets `u = s / kappa`.
- A counter counts consecutive updates with `u ≥ .6`.
- The width is `REACH/kappa = .5` once the counter reaches 50. Otherwise it is
  `max(.15, .5·min(u, 1/u)) / kappa`.
- The generator reads the mean critic logit over `x` and `x ± w·e_i` for every
  input coordinate (`2d+1` points). D's loss and penalty never use the read.

## Diagnostics (no gate)

- **ring16 acquisition.**
  - Control: first passing update 1184, confirmed at 1250, terminal suffix 26.
  - stall_reach: never passes. 1545/1600 updates are stalled, so the
    saturation window latched by about update 55 and almost never released. The width
    stays at .5, about 7× the 0.07 ring sigma.
- **Fraction widened / stalled.** Near-total on GANTrainer hosts: ring16 99.8%
  widened and 96.6% stalled; gaussian1d 98.5% widened and 72% stalled. On the
  small behavior hosts, 68–76% are widened, with 0–3 stalled updates. Maximum
  width is .5, or .44–.49 on the hosts that never stalled.
- **Wall cost of the 2d+1 stencil.**
  - Small hosts: +4–35% wall (ring16 75.2 vs 72.2 s, ae_gan_hold 13.0 vs
    9.8 s).
  - Word host (`d = 170`, a 341-point read): about 83 vs 51 ms per update
    (10,834 vs 17,501 updates in 900 s), about 1.6× per update. The full run
    would need about 1660 s against a 900 s allowance, so the word smoke
    cannot fit at this cost.
  - img_* tasks were not reached (Tier 2 blocked).
- **Control word smoke INCOMPLETE.** This is an allocation failure, not a
  measured quality failure. Historical leader controls finish in 788–896 s
  (see the [prior searches](../bcap-prior-search/README.md)). This round ran
  beside a full CPU test suite, which slowed the Python-bound word loop. The
  plan forbids retries, and the verdict rests on the measured ring16 FAIL, so
  the control was not rerun.

Public-API toy, run separately ([toy README](https://github.com/255BITS/ParticleGAN/blob/de93dbf48cabe6bb2913b3eccf93029655da5db9/reports/toy_audit/api_contract/stall_reach/README.md)):
on the 8-mode ring through `GANTrainer`, the off path ends with 7 modes and HQ
.917 (PASS). stall_reach ends with 6 modes and HQ .747 (FAIL; gate is modes ≥ 7
and HQ ≥ .90), with 1093/1200 updates stalled. Both arms oscillate on this toy,
so it only supports the ring16 result. (Corrected: the first toy version drew
its particle table from the global RNG and its numbers did not reproduce.)

## Why it failed

PR #107's stall condition had two parts: D near its slope limit, **and** G's
own-curvature trust factor averaging ≤ .1 over 50 updates. The current recipe
has no G trust factor, so the approved rebuild used slope saturation alone.
Under BCAP with `reg_kappa = 1`, the critic's measured slope on the fakes sits
close to the cap (median `u ≈ .90` on the ring8 toy) during ordinary training. The stall rule
therefore cannot tell "stalled" from "normal". Even before it latches,
`min(u, 1/u)` at `u ≈ .9` gives a width near the .5 maximum. The PR84 setting,
where `s ≤ .3` and the width was .15, does not occur under BCAP. At width .5 the
read averages the critic over a region wider than the gaps between ring modes,
which removes the mode-resolving gradient.

## Recommendations

1. **Do not adopt** `generator_critic_read="stall_reach"`. Keep it opt-in and
   off. The off path is bitwise identical to develop, so leaving it in costs
   nothing.
2. Any follow-up would need a stall signal that is selective under BCAP. Candidates:
   - Bring back PR #107's G-side trust factor.
   - Use acquisition progress measured from training losses only.
   - Scale the width law to `s / kappa` near the BCAP cap.

   Each would be a new mechanism. Screen it on the ring8 and ring16 toys first,
   and do not tune widths or windows on this one.
3. The `2d+1` stencil does not scale to the word task (`d = 170`), and the
   images would cost more still. Any successor should use a constant-cost read,
   for example a few random directions or an antithetic pair, rather than every
   axis.
4. The 900 s word-smoke allowance has no slack under host contention: even the
   unchanged control missed it. Future rounds should not run CPU-heavy work
   beside it.

Full per-task measurements: [measurements.md](https://github.com/255BITS/ParticleGAN/blob/de93dbf48cabe6bb2913b3eccf93029655da5db9/reports/forge/stall-reach/measurements.md) ·
[results.json](results.json) · actual-training GIFs in [media/](https://github.com/255BITS/ParticleGAN/blob/de93dbf48cabe6bb2913b3eccf93029655da5db9/reports/forge/stall-reach/media/README.md)
· reproduction: [workflow.py](https://github.com/255BITS/ParticleGAN/blob/de93dbf48cabe6bb2913b3eccf93029655da5db9/reports/forge/stall-reach/workflow.py).
