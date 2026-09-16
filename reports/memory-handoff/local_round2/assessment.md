# Completed: local round 2

Eight 2k scouts completed in 340.7 seconds on two GPUs, with no failures.
Training cost was 76.7–81.8 seconds per scout. No generated training trajectories,
fake training writes, or trajectory losses were used. All 50 focused tests passed.
Queue is sealed and empty; no longer runs have been launched.

| Change | Cold256 / 1024 | Original-orbit passes, both prefixes | Late stopped1024 |
|---|---:|---:|---:|
| D predictive auxiliary10 | 1.56% / 1.56% | 0% | 43.8% |
| D predictive auxiliary1 | 0% / 0% | 0% | 35.2% |
| D temporal auxiliary1 | 0% / 0% | 0% | 56.2% |
| Both auxiliaries10/1 | 0% / 0% | 0% | 60.9% |
| G SiLU | 0% / 0% | 0% | 51.6% |
| G tanh | 0% / 0% | 0% | 53.9% |
| G width128 | 0% / 0% | 0% | 82.0% |
| D width256 | 0% / 0% | 0% | 70.3% |
| Prior dense4 baseline | 0% / 0% | 0% | 68.8% |

Predictive10's two cold256 passes were both CCW. Predictive1 reduced continuous
warm radial error (prefix32 1.32 versus baseline1.92), but this remains far outside
the .1 threshold. None is a convincing winner for longer training. No promotion
based solely on two passing particles or improved stopping statistics.

## The user’s correction was supported

The original real/fake objective already provides a reason to use M. It was too
strong to diagnose these failures as insufficient motivation to encode motion.
The baseline actually uses temporal context:

- Prefix32 G clean next-point coordinate MSE: .00560, versus1.405 with shuffled M.
- Two valid histories ending at the identical point but moving in opposite
  directions: D prefers the appropriate continuation88.5% /89.1% of the time.
  Both preferences hold in78.1% of pairs.
- Reversing the history changes G appropriately: reverse-target MSE .00557,
  versus .15274 against the original forward target.

These are held-out, read-only probes on one shared512-example panel, not seed
experiments. They demonstrate useful motion dependence, not perfect temporal
modeling or guaranteed closed-loop stability. The selected histories are real;
this probe does not directly measure generated-state OOD shift.

New scouts did not materially improve clean point prediction: prefix32 MSE ranges
.00519 (predictive1) to .01149 (tanh). Improved D direction-pair preference does not
imply a stable G: tanh scores86.1% on that D probe but still fails every cold path.

## Recommendation

Preserve the ordinary GAN objective as the primary next direction. The current
working hypothesis is accumulated local error / feedback instability; OOD memory
remains plausible but unproven by these probes. Additional representation tasks
have not established a solution at this budget.

A useful next architecture scout would preserve recent observations explicitly
inside D-owned M and let G predict an increment from its most recent point.
This would change the representation / output parameterization while keeping
pointwise GAN training, fixed particles, D-owned writes, and expert-free runtime.
Compare it against the existing learned GRU memory using the same full cold and
warm metrics. This is a proposal, not implemented or queued here. More memory
capacity alone and smooth activations alone failed this round.

[Full leaderboard](leaderboard.md), [results](results.json),
[baseline probes](baseline_diagnostics.json),
[all new checkpoint probes](completed_diagnostics.json), [formulation](plan.md).
