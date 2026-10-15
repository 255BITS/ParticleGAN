# Selective generator history: measured outcome

**The new transport/history combinations did not work under the complete original gates.** All 16 diagnostic runs finished, including two bounded repair rounds. The unchanged BCAP/DualNorm/direction-blend baseline was the only configuration to pass both diagnostic tasks.

Fresh ordinary validation of that baseline passed **five of six required Tier 1 tasks**, plus the separate clock/state audit. The word model passed its observed quality checks, but its full 20,001-update run hit the 900-second timeout while other training occupied both GPUs. Its certified verdict is **INCOMPLETE**, so this run does not establish full Tier 1 qualification. One exact execution retry is prepared and remains unexecuted pending an isolated GPU; no model, horizon, or gate has been relaxed.

Read the [full training report](TRAINING.md), [machine-readable results](training-summary.json), and [exact executed baseline recipe](training-evidence/executed-baseline-recipe.json). The report explains the negative results, original gates, controlled repairs, resource limitation, and reproduction steps.

| Evidence | Contents |
|---|---|
| [Initial screen](training-evidence/initial/completion.json) | B/T/W/S, two original diagnostic tasks each |
| [First repairs](training-evidence/repairs/completion.json) | Stronger penalty and reduced generator/prior rates |
| [Final repairs](training-evidence/final-critic/completion.json) | Reduced discriminator rate in two measured parents |
| [Ordinary baseline](training-evidence/ordinary-tier1/completion.json) | Six required tasks and separate clock audit, including the word timeout |
| [Independent ordinary audit](training-evidence/ordinary-tier1/independent-result-audit.json) | Original frozen graders, full schedules, hashes and actual qualification status |
| [Training animations](training-media/README.md) | Saved observations, selected frame indices, hashes and explicit media omissions |
| [Original tensor inventory](training-evidence/checkpoints.json) | Local saved states/observations and hashes; task payload governs continuation |
| [Execution repair plan](training-evidence/ordinary-tier1/word-execution-repair-plan.json) | Conditional same-source, same-seed, same-budget word retry |

The [component implementation](../../../particlegan/optim/component_history.py) and [history estimator](../../../particlegan/optim/gradient_signal.py) remain experimental and disabled by default. The unchanged scientific source passed 44 targeted software checks and four CUDA full-trainer resume cases before training. Those software checks are separate from the scientific verdicts above.

The earlier [implementation/preflight readout](IMPLEMENTATION-PREFLIGHT-READOUT.md) is preserved byte for byte. Its [frozen v1 preflight](PREFLIGHT.md) and [draft roster](stage-roster.json) remain unchanged and blocked on the original missing evidence. The later [user-authorized training amendment](TRAINING-AMENDMENT-v2.json) registered fresh experiments; those outcomes do not retroactively satisfy the old preflight.
