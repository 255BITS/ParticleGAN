# Action disagreement on identical saved inputs

Posthoc only: no simulator resets/steps or training. Each row compares with the heuristic on the recorded current state.

| Input source | Evaluated model | Physical action MSE | Main agreement | Side agreement |
| --- | --- | ---: | ---: | ---: |
| Expert states + expert previous actions | sliders_all | 0.021207 | 98.3% | 90.5% |
| Expert states + expert previous actions | previous_l2 | 0.013328 | 98.8% | 90.8% |
| sliders_all traces | sliders_all | 0.354068 | 80.6% | 35.0% |
| sliders_all traces | previous_l2 | 0.289661 | 83.7% | 50.4% |
| previous_l2 traces | sliders_all | 0.288700 | 84.9% | 43.1% |
| previous_l2 traces | previous_l2 | 0.283049 | 85.0% | 40.8% |

Within a trace source, both models receive exactly the same state, previous command, and terrain. Physical MSE is comparable across models; standardized MSE in JSON uses each model’s own scaler.

Expert and learner datasets differ in both current states and previous commands. This does not isolate previous-action feedback, establish a causal failure mechanism, or prove that the heuristic or the other model would recover from learner states.
