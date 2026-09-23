# Lunar flight experts: fixed-cohort tuning

Environment: Gymnasium 1.2.3 `LunarLander-v3`, continuous actions, wind off,
Box2D physics. The bidirectional variant applies a downward center-of-mass
impulse for negative main commands, scaled to the stock full main-engine
impulse (`MAIN_ENGINE_POWER * MAIN_ENGINE_Y_LOCATION / SCALE`). It charges the
same 0.30 fuel penalty at full power. This is a named variant, not stock Lunar.

All candidates used the same 20 training resets, seeds 61000–61019. A landing
counts only if the simulator terminates with no crash flag, both legs in
contact, the lander sleeping, and its center within the helipad. Mean steps
includes every episode through termination; contact step is the first leg
contact. We varied controller gains and downward power, rather than repeating
the same experiment with different seeds.

| Candidate (height gain, velocity gain, downward power) | Train landings | Mean steps | Mean first contact |
| --- | ---: | ---: | ---: |
| 0.35, 0.50, 0.00 | 18/20 | 222.90 | 174.20 |
| 0.50, 0.50, 0.00 | 20/20 | 205.30 | 153.90 |
| 0.50, 0.50, 0.15 | 20/20 | 199.55 | 144.15 |
| **Fast: 0.50, 0.50, 0.30** | **20/20** | **194.85** | **138.90** |
| 0.50, 0.50, 0.50 | 19/20 | 187.25 | 134.25 |
| **Slow: 0.50, 0.65, 0.00** | **19/20** | **232.15** | **185.40** |

The stock heuristic's negative main action means “off” in stock Lunar. Applying
it directly to this variant turns that action into a downward burn and crashed
all 16 flights in the first calibration cohort. The experts explicitly output
zero for off. The fast expert uses the down thruster only above observation
height 0.4 when its vertical control calls for a strong descent; it uses stock
upward control near contact. A stronger 0.50 down command was quicker but lost
one training landing. A negative vertical-control shift of −0.06 without a down
thruster reached 198.8 mean steps but landed only 12/20 under the strict rule.

The selected pair was checked on the separate validation cohort, seeds
71000–71039. These seeds were not used to adjust the chosen policies.

| Policy | Train landing / steps / contact | Validation landing / steps / contact | Validation mean return |
| --- | --- | --- | ---: |
| Slow | 19/20 / 232.15 / 185.40 | 40/40 / 220.95 / 175.33 | 274.71 |
| Fast | 20/20 / 194.85 / 138.90 | 40/40 / 185.20 / 134.18 | 277.53 |

The fast expert used negative main thrust on every validation flight (24.83
steps per flight on average). It completed validation flights 16.2% sooner by
mean total steps and reached first leg contact 23.5% sooner. These are paired
fixed-cohort observations, not a claim about all possible terrains or starts.

Reproduce the selected rows by running `rollout_episode(seed, slow_expert)` and
`rollout_episode(seed, fast_expert)` from `lib.lunar_flight` over the seed ranges
above, then `summarize_episodes`. `tests/test_lunar_flight.py` checks the actual
Box2D downward acceleration, stock opt-out, rendered frames, and a small paired
success/speed cohort.
