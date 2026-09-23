# Lunar Lander: does learning state and successor help control?

This experiment trains one state-only encoder from scratch on the existing
47 heuristic training episodes (9,297 transitions). Each decision uses the
current observation and terrain context. The actual Gym simulator advances
the lander after the generated action.

```text
                    +-- G1 -> reconstructed st
st -> E -> z -------+-- G2 -> at
                    +-- G3 -> predicted st+1

simulator.step(at) -> observed st+1 -> E -> ...
```

Terrain also enters E and each generator. The latent uses 1,024 MoG components,
32 coordinates, and bounded encoder offsets. G1, G2, and G3 are independent
networks sharing that latent. E never receives an action or successor target.

The two arms have identical initial weights and minibatches:

| Arm | What trains E and the MoG prior? | What trains G1/G3? |
| --- | --- | --- |
| `probes` | Expert action loss, plus common prior regularization | State/successor losses through detached z |
| `auxiliary` | Action, state, and successor losses, plus common prior regularization | State/successor losses through z |

Both optimize expert action MSE in standardized coordinates. The state heads
use continuous-coordinate MSE and contact BCE, with weight one on each head.
Both train all three generators; the distinction is whether auxiliary losses
can change the representation used for control. This initial comparison has
no discriminator or synthetic reconstruction loop.

G3 learns successors under expert behavior. Sharing z with G2 does not guarantee
that its prediction matches the physical consequence of G2's action, so we
measure both held-out expert prediction and actual learner-rollout prediction.
G3 cannot answer arbitrary alternative-action queries without an action input.

Each arm runs 2,500 updates with batch size 256 on GPU 1. Validation selects
among updates 250, 1,000, and 2,500 by landing rate, then mean return. Twenty
fresh validation worlds and fifty fresh test worlds are paired across methods.
The hand-written heuristic and previously successful imitation fine-tune are
re-evaluated on these same worlds. These are evaluation episodes, not repeated
training runs with different seeds.

```sh
tail -f results/gym/lunar_lander_state_control/live.log
.venv/bin/python experiments/train_gym_state_control.py --config configs/gym/lunar_lander_state_control/probes.yaml
.venv/bin/python experiments/train_gym_state_control.py --config configs/gym/lunar_lander_state_control/auxiliary.yaml
```

Training requires fresh output directories. Full commands, measured results,
and the next recommendation are recorded in
[the experiment readout](../reports/gym/lunar_lander_state_control/READOUT.md).
The [saved plan](gym-state-encoder-plan.md) records the comparison before running.

The completed comparison landed **50/50** test worlds with detached probes and
**27/50** with joint auxiliary learning. Auxiliary training improved held-out G3
prediction about 12x, but reduced control success. The live viewer now defaults
to `state_probes`, selected on validation; all six controller options remain.

```sh
.venv/bin/python -u examples/gym_lander_live.py --seed 591000
.venv/bin/python experiments/evaluate_gym_state_control.py --freeze --baseline
.venv/bin/python experiments/evaluate_gym_state_control.py --arm probes --checkpoint results/gym/lunar_lander_state_control/probes/checkpoint_250.pt --checkpoint results/gym/lunar_lander_state_control/probes/checkpoint_1000.pt --checkpoint results/gym/lunar_lander_state_control/probes/checkpoint_2500.pt --final results/gym/lunar_lander_state_control/probes/final.pt
```

Use `--arm auxiliary` and the corresponding `auxiliary/` checkpoint paths for
that arm. Existing frozen reports are verified and cached scores reused.
[Open the playable simulator](http://localhost:8787).
