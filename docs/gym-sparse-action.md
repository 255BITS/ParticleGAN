# Lunar Lander with sparse action labels

Can learning from many observed transitions help control when few expert actions
are labeled? This experiment keeps all 9,297 state/successor pairs from 47 expert
training episodes, but exposes actions from only five episodes: 1,010 labels.
The episodes are selected by a fixed hash ordering, independently of outcomes.

```text
                    +-- G1 -> reconstructed st
st -> E -> z -------+-- G2 -> at
                    +-- G3 -> predicted st+1

simulator.step(at) -> observed st+1 -> E -> ...
```

Terrain enters E and all three generators. E receives the current state only,
with no action history or targets. Both arms start from scratch with MoG 1,024,
z32, width128, the default MoG optimizer and prior regularizer, and EMA.

- **Detached probes:** labeled action loss trains E/G2/prior. G1/G3 learn from
  all transitions, but their inputs are detached so they cannot train E/prior.
- **Joint auxiliary:** the same losses and examples also let G1/G3 train E/prior.

There is no discriminator in this comparison. Each update uses 256 labeled
action examples and 256 state/successor examples from independent, matched
sampling streams. State normalization uses all available training observations;
action normalization uses only the five labeled episodes. Each head has weight
1, and the common prior regularizer is applied once per update. Both runs use
2,500 updates on GPU 1, without simulation calls or trajectory unrolling.

The successor observations can indirectly reveal actions; that is the intended
extra supervision. This tests fewer **explicit action labels**, with the same
expert-generated transition collection. G3 does not receive an alternative
action, so it is an expert-behavior successor predictor rather than a general
counterfactual dynamics model.

Sharing z is the only connection between the action and successor objectives.
There is no loss requiring G2's chosen action to produce G3's predicted successor
in the simulator. This experiment tests representation sharing; it does not
explicitly learn inverse dynamics or enforce physical consistency between heads.

Twenty fresh validation worlds select among updates 250, 1,000, and 2,500 by
landing rate, then mean return, then earlier update. Fifty separate test worlds
compare selected controllers. The full-label state-only probe and heuristic
are evaluated on these same worlds as references. The two sparse arms are the
matched comparison; the full-label reference also has different normalization.

The completed results are **44/50 landings for sparse probes**, **34/50 for
auxiliary**, and **49/50 for the full-label reference** on the same fresh test
worlds. Auxiliary improves held-out expert action MSE by 25% but has higher
action error on learner-visited states. Full results and costs are in
[the experiment readout](../reports/gym/lunar_lander_sparse_action/READOUT.md).
The frozen design is in [the plan](gym-sparse-action-plan.md).

Follow progress:

```bash
tail -f results/gym/lunar_lander_sparse_action/live.log
```

Reproduction commands (training requires fresh output directories):

```bash
.venv/bin/python -u experiments/evaluate_gym_sparse_action.py --freeze --baseline
.venv/bin/python -u experiments/train_gym_sparse_action.py --config configs/gym/lunar_lander_sparse_action/probes.yaml
.venv/bin/python -u experiments/train_gym_sparse_action.py --config configs/gym/lunar_lander_sparse_action/auxiliary.yaml
for arm in probes auxiliary; do
  run_dir="results/gym/lunar_lander_sparse_action/$arm"
  .venv/bin/python -u experiments/evaluate_gym_sparse_action.py --arm "$arm" \
    --checkpoint "$run_dir/checkpoint_250.pt" \
    --checkpoint "$run_dir/checkpoint_1000.pt" \
    --checkpoint "$run_dir/checkpoint_2500.pt" --final "$run_dir/final.pt"
done
.venv/bin/python experiments/diagnose_gym_sparse_cross.py
.venv/bin/python experiments/diagnose_gym_sparse_actions.py
.venv/bin/python experiments/plot_gym_sparse_action.py
.venv/bin/python -u examples/gym_lander_live.py --seed 791000
```

Open http://localhost:8787 for the actual simulator. The controller menu retains
earlier models and adds both sparse-action models. Its default comes from this
round's comparable validation scores, excluding the heuristic. The full-label
reference won validation and remains the default.
