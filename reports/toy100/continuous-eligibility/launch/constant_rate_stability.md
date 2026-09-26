# Lane 3: repair KA2 memory and update stability at constant rates

Keep actual public generator, critic and particle learning rates constant for
this lane. Inspect KA2's critic reference/controller and applied Adam updates
around the retained stationary collapse at update 1750 and later departures.
Use the existing metrics/checkpoints when accessible, rather than repeating the
unchanged 4,600-step run. Develop one coherent memory or update-stability repair
that preserves mobility while preventing the reference/controller/optimizer
interaction from sustaining self-induced excursions.

This is not a smaller fixed-LR grid, an arbitrary hard reset, or the already
fixed floating-buffer EMA bug. Rates alone being constant is insufficient:
replace inherited horizon-based noise and make the public learner continue
without a planned endpoint. An automatic internal damping/memory rule may use
ordinary training signals, with fully logged and checkpointed state. Preserve
meaningful positive updates and sparse prior behavior; do not silently freeze
parameters after first success. Other lanes own adaptive LR policies.

Read `reports/ka2-default-candidate/constant-lr-api/README.md`, its compressed
metrics, actual controller/rate histories and source worker in your API checkout,
plus `particlegan/ka2.py` and `particlegan/training.py`. Use precise numerical
evidence for the proposed repair; a one-time good endpoint is not a solution.
