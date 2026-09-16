# Local round 2: stabilize pointwise learning

Eight 2k scouts, same 10k schedule and fixed evaluation panel. No generated
training trajectory, fake write, path critic, clipping, EMA, or seed sweep.
Each example still uses a strictly causal real memory and one G prediction.
D exclusively trains the writer; G reads that memory and uses a fixed particle.

The completed baseline probes in `baseline_diagnostics.json` show that memory is
already useful. With prefix32, baseline G clean-target MSE is .00560; shuffling
memory increases it to 1.405. D prefers the correct point to the previous point
in 77.1% of examples, and to a three-step-later point in 76.2%. Thus failure to use
memory is not established. These are intervention/ranking diagnostics, not a
calibrated accuracy estimate or proof that memory is sufficient under feedback.

A stronger probe uses two real histories ending at the identical observed point
with opposite travel directions. At prefix32 the baseline D chooses the correct
directional continuation in 88.5% of forward and 89.1% of reverse cases; both
preferences hold in 78.1% of pairs. G reverse-target MSE is .00557 versus .15274
against the forward target. This supports memory carrying motion information,
not merely position. Accumulated local error / feedback instability is therefore
a stronger working hypothesis than complete failure to use temporal context.

| Scout | Change | Question |
|---|---|---|
| local_silu | G SiLU | Does a smooth reader improve feedback stability? |
| local_tanh | G tanh | Does another smooth reader improve feedback stability? |
| local_g128 | G width128 | Is reader capacity limiting local accuracy? |
| local_d256 | D width256 | Is discriminator capacity limiting temporal discrimination? |
| local_predict1 | D prediction weight1 | Does explicit predictive memory help? |
| local_predict10 | D prediction weight10 | Does stronger predictive pressure help? |
| local_temporal1 | D temporal BCE weight1 | Does explicit wrong-time discrimination help? |
| local_predict10_temporal1 | Both auxiliaries | Are the two local tasks complementary? |

The first four preserve the GAN objective. Auxiliary heads in the latter four
share only D's writer with the adversarial head. Their targets never enter M.
Prediction is next-real-point coordinate MSE; temporal classification compares
correct points against previous and three-step-offset points on the same circle.
At the end of a real episode, the second negative is three steps earlier.
Both tasks exclude prefixes shorter than3. Auxiliary loss is D-only; G retains
its unchanged GAN/prior objective. Wrong-time examples do not alter the GAN's
negative distribution. B-cap applies to the original GAN candidate head with
unchanged defaults; auxiliary heads have their separately specified MSE/BCE.
Auxiliary strengths are hypotheses, not requirements of conditional GANs.

Prior dense4 is a historical matched baseline. All modules shared with it retain
initialization and training RNG streams. Extra heads use isolated RNG scopes.
50 focused tests passed before launch, including auxiliary ownership, exact
resume, active B-cap, causal memory, and absence of generated training unrolls.

Queue: `runs/memory_path/local_round2`. Results are examined after completion.
The completion-triggered reporter refreshes `leaderboard.md` and `results.json`.
Only meaningful gains warrant longer runs; single-point fit alone is insufficient.

```sh
tail -F runs/memory_path/core_round1/train.log
```
