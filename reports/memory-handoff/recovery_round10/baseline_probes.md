# Prior candidate: matched history and repair probes

Completed checkpoint: round9 proposal_clean_s25 at2k. All probes are evaluation
only; no generated paths or squared-error diagnostics enter training.

Change one history property while keeping z, clock, center, phase at handoff,
and observation-noise realization fixed. Radius/speed values stay inside
training support. Ideal normalized radius/speed response is1.

- radius_response_mean_ideal1: 0.050287
- radius_response_median_ideal1: 0.000228
- radius_correct_change_fraction: 0.523438
- speed_response_mean_ideal1: -0.018407
- speed_response_median_ideal1: -0.000114
- speed_correct_change_fraction: 0.468750
- direction_correct_in_both_fraction: 0.054688

Near-zero median responses and approximately chance directional changes show
that persistent output differences do not establish useful process retention.
The original-vs-flipped-direction criterion requires the late mean angular motion
to agree with BOTH intended directions for the same particle. This is weaker
than consistent per-step direction and is still only5.5%.

The repair nevertheless helps autonomous radial and joint-motion fidelity:

| Prefix | Normal Q | Adapter bypass Q | Normal radial | Bypass radial |
|---|---:|---:|---:|---:|
| 8 | 0.006190 | 0.000136 | 1.497 | 4.161 |
| 32 | 0.006426 | 0.000166 | 1.487 | 4.158 |

Bypass changes the trained model operating distribution. It is a checkpoint
intervention, not a substitute for separately trained no-adapter controls.
