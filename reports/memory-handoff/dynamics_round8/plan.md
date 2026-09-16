# Local memory dynamics and G-side translation scouts

User authorized configurable implementations and scouts for slow/fast D memory,
local feedback stability, and a G-side memory repair/translator. No private G
recurrent state, full generated training rollout, seed sweep, clipping, or EMA.
Default public API exact B-cap remains unchanged. Fixed particle per episode.

Runtime remains `x = G(z, translate(M), Fourier(t)); M = D.write(M, x)`.
Translation is internal to G and never writes back into D memory. Only D trains
the writer; only G trains the adapter. All features default off, preserving
existing model state dicts and configs. Old checkpoints may omit new inactive
RNG streams; enabled new features require their streams for exact resume.

## Designs

- Slow/fast: same fully coupled GRU and parameters as the baseline; first N
  learned coordinates interpolate toward each proposed GRU update at rate alpha.
  Remaining coordinates retain the original update. No fixed circle parameters.
- Translator: identity-initialized residual MLP, M -> 64 -> bottleneck -> M,
  with SiLU, owned by G. No private persistent state. Pure translator controls
  have no additional loss and test capacity/representation adaptation alone.
- Repair: Gaussian-corrupted detached prefix memory is translated toward either
  the detached raw clean memory or its detached current translation. MSE trains
  only the adapter, on prefixes >=4. The translated target permits a new code
  but can collapse; raw targets anchor to D's current representation. Both are
  local synthetic corruption objectives, not guarantees for novel situations.
- Stability: at the same real-prefix M, fixed z, and time, evaluate F(M) and
  F(M+delta), where F(M)=D.write(M,G(z,M,time)). Delta has fixed coordinate RMS
  .03. Loss is mean relu(||delta_next||^2/||delta||^2 - max_gain^2).
  Prefix states and particles are detached anchors. D phase freezes G; G phase
  freezes D but differentiates through its write. Two parallel one-step branches
  per enabled phase, no second generated prediction. Not a worst-case bound,
  and restricting all directions may erase meaningful state distinctions.

## Grid

16 fresh scouts, 2,000 updates, unchanged 10,000-update schedule, batch128 real
episodes x4 sampled targets, M32, G64, D128, clock6 at .03125 rad/step base.
No feedback replacement or previous D prediction/temporal auxiliary losses.

| Group | Settings |
|---|---|
| Slow writes (4) | 16 coordinates at .25/.1/.03; 24 coordinates at .1 |
| Translator (2) | bottleneck16 /64, adversarial learning only |
| Repair (4) | raw target weight10/100 at noise.05; weight10 at noise.15; translated target weight10 at noise.05 |
| Stability (3) | G-only weight.1; D+G weight.1 each; D+G with max gain.9 instead of1.1 |
| Combinations (3) | slow16/.1 + raw repair10; slow16/.1 + D/G stability.1; raw repair10 + D/G stability.1 |

Historical controls: clock_fourier6, clock_static6, clock_fourier6_shared from
round7. These are comparisons only, not repeated seeds or new scout results.

## Evaluation and decisions

Use unchanged cold zero-memory and real-prefix8/32 autonomous continuation,
128 fixed particles, 256/1024 steps. Original-orbit warm fidelity and full cold
circle passes are primary; direction coverage, radial/speed/position errors,
late-only fits and stopping are separate diagnostics. Inspect only completed
jobs. Rank using existing completed-only reporter; don't invent a winner when
pass rates tie at zero.

After completion, measure real-context prediction, memory shuffle/zero, clock
reset, finite-perturbation feedback gain, and adapter bypass/noise sensitivity
on the saved evaluation panel. Also compare the next prediction after one generated
write versus a clean real write (evaluation only), to test predictive damage from
feedback. These establish local use/robustness, not proof
that autonomous behavior benefits from M. No images guide selection.

Promote up to two clear primary-metric improvements to exact 5k continuations.
If primary rates remain zero, consider a diagnostic extension only for >=25%
radial-error improvement at BOTH prefixes versus clock_fourier6, with no worse
direction agreement or stopping and improved early position error. Label it
explicitly as a diagnostic, not a solved circle. Otherwise use mechanism probes
to recommend the next change rather than indiscriminately extending failures.

Pipeline: runs/memory_path/dynamics_round8, both cuda:0 and cuda:1.
Central tail: `tail -F runs/memory_path/core_round1/train.log`.
Drain stdout is completion/failure notifications only; report refreshes after
each completed result. Configs live in experiments/configs/memory_handoff/dynamics_round8.
