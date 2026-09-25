# GAN sample quality on held-out expert contexts

Both paths generate one complete record per each of the same 2,444 held-out expert terrain contexts. Prior noise/component draws and contact-sampling streams are matched across models. Contacts are binary samples. Checkpoints were already selected by control validation.

| GAN | Path | SW1 ↓ | Precision ↑ | Coverage ↑ | Contact TV ↓ | Reference radius |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| joint | prior | 0.19794 | 40.3% | 33.6% | 0.17635 | 0.66090 |
| joint | encoded | 0.08721 | 55.8% | 43.7% | 0.00777 | 0.66090 |
| marginals | prior | 0.17363 | 38.8% | 36.7% | 0.11702 | 0.66090 |
| marginals | encoded | 0.08209 | 58.5% | 48.1% | 0.00900 | 0.66090 |

Precision and coverage use the 95th percentile of real nearest-other-real distances as their common radius. SW1 uses 128 fixed projections. All distances use the common sparse-training scaler.

These are pooled joint-record metrics: terrain is excluded from metric distances. They do not prove conditional physics or successful control. The encoded path is conditioned on actual current states; the prior path is not.

Pooled normalized joint records exclude terrain from metric distances; matching context frequencies does not prove conditional fidelity or physical consistency.
Reference records are temporally correlated expert behavior observations, not independent samples or an unrestricted counterfactual dynamics benchmark.
Encoded path receives actual held-out current states; it is not an unconditional prior sample and does not receive held-out actions or successors.
All explicit action labels in the reference are held-out posthoc scoring targets; no new labels enter training or checkpoint selection.
Nearest-neighbor precision/coverage depend on reference density and the inherited radius convention; these are descriptive metrics without confidence intervals.

Zero simulator steps/resets and zero training updates. Hashes and exact definitions are in `prior_sample_diagnostics.json`.
