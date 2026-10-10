# Prepared proximal prior search

Four ordinary arms are prepared and **not admitted or executed**: the exact
current BCAP leader with restoration disabled and global spring strengths
`.1`, `1` and `10`. Every task follows the revision-9 gate ladder, with all six
required Tier 1 gates before all 21 Tier 2 tasks. A failed Tier 1 requirement
blocks that arm's ordinary Tier 2 work after the current-tier peers finish.

The [specification](spec.json) freezes the trainer delta, hypothesis,
falsifier, task contracts, protocol seed 0, budgets and stopping rule. The
[compact registration](registration.json) binds all four ready arms to the
same scientific source. The full reservation is **172,080 seconds**: 43,020
for the separate leader control and 129,060 for the three-strength search.
Per-arm campaign caps are 45,000 seconds, with a combined cap of 180,000.
This retains every original allowance, including the optional clock audit.
There are no scientific retries, automatic refinements or Tier 3 work.

The prediction is that a position spring outside normalized gradients reduces
code drift and converts at least one matched-control Tier 2 FAIL to full
sustained PASS. For `mode_hold`, a passing endpoint is insufficient: modes
must remain at least 8 and high-quality fraction at least .9 across a terminal
suffix of at least 5. Scalar retention requires all 72 stationary checks,
own-state deadline reacquisition and all 24 shifted hold checks. A reduced
failure count or prior drift alone cannot establish a repair. Every lost
passing task remains visible; a global repair must preserve all control passes.

Only `prior_proximal_strength` changes. All BCAP leader trainer settings,
task architecture, target/data law, seen batch sequence, initial prior
distribution/capacity, sampling, updates and evaluation cadence remain fixed.
One complete global configuration is evaluated per strength, with independent
checkpointed RNG streams. The
[mechanism documentation](../../../../docs/prior-proximal.md) records the final
spring's ordering after direction blend and its effect on protected derivatives.
The separate [software evidence](../../prior-proximal-software/README.md)
supplies no Tier 2 credit.

Bulk preflight plans are stored at
`/mnt/ml7tb/ParticleGAN-forge/bcap-prior-search-20261010/proximal/preflight.json`.
Future attempt logs and checkpoints will remain in that archive's `queue/`.
The frozen reproduction source is [workflow.py](../workflow.py). GPU execution
will use one branch worker on GPU 0 plus Forge's automatic CPU worker, with
cross-branch GPU sharing intentionally allowed. Contended wall costs are
accounting, not a speed ranking. Execution remains pending the root's launch
instruction after all five feature implementations are ready.

The [current technique leaderboard](../../technique-inventory.md) remains the
goal's sole generated leaderboard. This prepared comparison claims no measured
repair, independent confirmation or default adoption. Completed publication
will include every task's score/target measurements and all unmeasured cells.
