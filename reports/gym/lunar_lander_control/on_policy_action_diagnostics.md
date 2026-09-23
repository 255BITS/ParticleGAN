# Expert action disagreement on learner-visited states

Posthoc labels on the frozen selected test traces; no new rollouts, resets, training, or checkpoint selection. The installed heuristic was queried using each recorded state. Overall metrics weight transitions equally; episode-weighted summaries and detailed counts are in the JSON.

| Controller | Landings | Demo MSE | On-policy MSE | First 20 MSE | Later MSE | Joint engine-regime agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0/50 | 0.08234 | 3.83806 | 2.68211 | 4.22964 | 5.7% |
| imitation | 50/50 | 0.01740 | 0.12011 | 0.14094 | 0.11788 | 74.2% |
| joint | 12/50 | 0.02082 | 0.61224 | 0.13968 | 0.70917 | 54.0% |

MSE uses the unchanged training action scaler. Demo MSE uses held-out expert states and expert previous commands; on-policy MSE uses learner states and learner previous commands.

| Controller / phase | Occupancy | Action MSE | Main false-off given expert-on | Main false-on given expert-off | Lateral regime agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| original / flight | 89.1% | 3.89684 | 56.0% | 39.2% | 15.1% |
| original / approach | 8.3% | 3.95836 | 28.6% | 100.0% | 9.8% |
| original / contact | 2.6% | 1.42145 | 24.1% | 77.3% | 62.7% |
| imitation / flight | 44.7% | 0.19616 | 0.2% | 13.3% | 71.9% |
| imitation / approach | 28.3% | 0.10300 | 0.3% | 98.7% | 87.1% |
| imitation / contact | 27.0% | 0.01211 | 7.2% | 2.2% | 99.9% |
| joint / flight | 59.7% | 0.73187 | 22.1% | 0.0% | 47.4% |
| joint / approach | 21.8% | 0.48844 | 3.4% | 2.4% | 47.2% |
| joint / contact | 18.5% | 0.37170 | 25.7% | 1.3% | 99.6% |

Engine regimes follow the simulator dead zones: main on iff command >0; lateral off iff absolute command ≤0.5, otherwise signed direction. Phase uses the current state: either leg contact first, then approach y<0.25, otherwise flight.

These measurements show how each controller disagrees with this expert on states it actually visits. Different state distributions and episode lengths prevent a causal attribution from this comparison alone. The heuristic was not tested for recovery from these states. A later DAgger comparison can test whether labeling learner-visited states improves control; this analysis does not establish that it will.
