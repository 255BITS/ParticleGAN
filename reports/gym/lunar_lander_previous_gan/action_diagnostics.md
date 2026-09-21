# Action disagreement on identical saved inputs

Posthoc only: no simulator resets/steps or training. Each row compares with the heuristic on the recorded current state.

| Input source | Evaluated model | Physical action MSE | Main agreement | Side agreement |
| --- | --- | ---: | ---: | ---: |
| Expert states + expert previous actions | previous_marginals | 0.013328 | 98.8% | 90.8% |
| Expert states + expert previous actions | imitation | 0.006348 | 99.2% | 94.4% |
| previous_marginals traces | previous_marginals | 0.332475 | 81.2% | 33.7% |
| previous_marginals traces | imitation | 0.139397 | 91.3% | 86.9% |
| imitation traces | previous_marginals | 0.065689 | 89.7% | 76.7% |
| imitation traces | imitation | 0.047833 | 90.0% | 82.5% |

Within a trace source, both models receive exactly the same state, previous command, and terrain. Physical MSE is comparable across models; standardized MSE in JSON uses each model’s own scaler.

Expert and learner datasets differ in both current states and previous commands. This does not isolate previous-action feedback, establish a causal failure mechanism, or prove that the heuristic/imitation would recover from learner states.
