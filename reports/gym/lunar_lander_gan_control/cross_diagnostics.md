# Cross-policy state and successor diagnostics

Both validation-selected models receive the same measured states and terrain within each row pair. All successors come from existing real simulator traces. No new simulator calls or training.

| Trace controller | Model | Records | G1 state MSE | G3 successor MSE | Persistence MSE | Action MSE vs recorded command |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| joint | joint | 12156 | 0.807278 | 0.856795 | 0.046507 | 2.62354e-14 |
| joint | marginals | 12156 | 0.785566 | 0.894471 | 0.046507 | 0.299391 |
| marginals | joint | 11893 | 1.305579 | 1.372858 | 0.069521 | 0.472808 |
| marginals | marginals | 11893 | 1.268438 | 1.443321 | 0.069521 | 2.83703e-14 |

State and successor MSE average the six continuous coordinates after the frozen training normalization; action MSE averages two standardized commands. JSON also contains physical errors and contact BCE, Brier score, and accuracy.

G1 has an identical target across models on each trace set, making its comparison direct. G3 cannot condition on the other policy’s recorded action: action disagreement limits interpretation of cross-policy successor errors. Action errors here measure agreement with recorded learner commands, not the heuristic. These posthoc results do not select checkpoints or establish causality.
