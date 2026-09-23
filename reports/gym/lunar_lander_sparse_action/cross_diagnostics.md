# Cross-policy state and successor diagnostics

Both validation-selected models receive the same measured states and terrain within each row pair. All successors come from existing real simulator traces. No new simulator calls or training.

| Trace controller | Model | Records | G1 state MSE | G3 successor MSE | Persistence MSE | Action MSE vs recorded command |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| probes | probes | 15027 | 1.440771 | 1.469513 | 0.015672 | 5.87531e-14 |
| probes | auxiliary | 15027 | 0.308216 | 0.356844 | 0.015672 | 0.411049 |
| auxiliary | probes | 19374 | 1.947802 | 1.889776 | 0.014655 | 1.76678 |
| auxiliary | auxiliary | 19374 | 0.730875 | 0.749776 | 0.014655 | 2.12745e-14 |

State and successor MSE average the six continuous coordinates after the frozen training normalization; action MSE averages two standardized commands. JSON also contains physical errors and contact BCE, Brier score, and accuracy.

G1 has an identical target across models on each trace set, making its comparison direct. G3 cannot condition on the other policy’s recorded action: action disagreement limits interpretation of cross-policy successor errors. Action errors here measure agreement with recorded learner commands, not the heuristic. These posthoc results do not select checkpoints or establish causality.
