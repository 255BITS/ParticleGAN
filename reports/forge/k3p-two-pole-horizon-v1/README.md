# K3P two-pole budget and schedule diagnostic

This bounded diagnostic asks whether the existing word-positive global K3P recipe is merely too slow for the 80-update movement gate, or instead stalls below it. It also measures an unchanged movement-passing global recipe as a control. These four runs supply no ordinary Tier1 qualification, family-selection change, calibration or default adoption.

The [frozen plan](plans.json) binds two unchanged global candidates to two explicitly scoped tasks. Both execute800 updates, with either schedule horizon80 (additional execution at the original schedule) or800 (a stretched schedule bundle). Numerical milestones are80/200/400/800. The existing gate remains mean absolute movement≥.3 and median critic input gradient≤1, with24 scoring observations and five passing terminal checks for each800-update diagnostic. Extra prefix/milestone observations do not enter that diagnostic gate. Both tasks run even if the other fails.

The original `two_pole` task and main `discriminator_stability` view remain intact. Historical word and movement passes motivate the design and do not fill these diagnostic cells. Candidate cards remain byte-identical; only study-specific decision contracts bind the new task conditions. No seed sweep or GAN technique is introduced.

The trace retains actual coordinate gradients, their existing adversarial/L2 components, optimizer rates/gain/displacements, clean critic fields, penalty/payoff gradients, state checkpoints and sign/spread diagnostics. Bulk streams and tensors remain ignored and are archived after execution. All-zero clean particles may remain identical: movement is the declared question, not proof of both-mode fidelity. Stretching the noisy control's horizon also stretches its input-noise window; longer execution can activate the existing critic guard after200 updates. Comparisons do not isolate a single mechanism.

The four task reservations are300 seconds each,600 per candidate and1200 campaign-wide. Execution is sequential on CPU, one Torch thread per arm, fixed named seed0. The [single current family leaderboard](../technique-inventory.md) remains the solution ranking.

Tail `runs/forge/k3p-two-pole-horizon-v1/worker.log` or its queue's `events.jsonl`. Preparation launches no training. The runner requires the exact reviewed scientific commit and uses Forge admission, source freezing, budget accounting and independent grading. Admitted/concluded studies are immutable; do not repeat unchanged training for publication.

```sh
/usr/bin/python reports/forge/k3p-two-pole-horizon-v1/prepare.py
/usr/bin/python -u reports/forge/k3p-two-pole-horizon-v1/run.py --expected-commit EXECUTED_COMMIT
```

Results and actual-training GIFs will be published from these exact saved states when the declared round completes.
