# Cross-policy state and successor diagnostics

Both validation-selected models receive the same measured states and terrain within each row pair. All successors come from existing real simulator traces. No new simulator calls or training.

| Trace controller | Model | Records | G1 state MSE | G3 successor MSE | Persistence MSE | Action MSE vs recorded command |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| probes | probes | 10095 | 0.257770 | 0.267315 | 0.019207 | 1.63864e-14 |
| probes | auxiliary | 10095 | 0.016088 | 0.028207 | 0.019207 | 0.0388273 |
| auxiliary | probes | 10692 | 1.026640 | 1.017716 | 0.047972 | 0.341689 |
| auxiliary | auxiliary | 10692 | 0.338229 | 0.377632 | 0.047972 | 1.00902e-14 |

State and successor MSE average the six continuous coordinates after the frozen training normalization; action MSE averages two standardized commands. JSON also contains physical errors and contact BCE, Brier score, and accuracy.

G1 has an identical target across models on each trace set, making its comparison direct. G3 cannot condition on the other policy’s recorded action: action disagreement limits interpretation of cross-policy successor errors. Action errors here measure agreement with recorded learner commands, not the heuristic. These posthoc results do not select checkpoints or establish causality.
