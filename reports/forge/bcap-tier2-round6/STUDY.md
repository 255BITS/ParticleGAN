# Four bounded BCAP Tier 2 repair ideas

This is one ordinary five-arm comparison: four new global recipes and one fresh
matched control. Five research agents reviewed annealing, game dynamics, critic
geometry, prior density, and independent selection/protocol evidence. The
selected ideas are late annealing, half G/E motion, prior-only smoothing and weak
zero-centered critic damping. Joint Lookahead is deferred: independent wrappers
would change opponent timing, and averaging after direction protection could
undo its repairs.

The reference is the current recipe-owned-prior BCAP leader,
`recipe-prior-refactor-bcap-pure-v1`: non-saturating BCAP, direction blend,
zero-momentum DualNorm, native SVD, per-offset convolution, smoothing .001,
G/E .012, D .018, prior .030, cap/coefficient/interval 1/1/1. Prior locations
remain learned; VICReg weight and prior L2 are zero. The task owns initial
prior kind, capacity, width and sampling. All arms explicitly set prior smoothing
.001 for common force/motion instrumentation, except its .01 candidate. Explicit
.001 is numerically the inherited baseline denominator; software parity checks
verify that this monitoring does not change parameters.

Each task retains current revision-9 architecture, target/data law, batch
sequence, deterministic public initializer, protocol seed 0, consumed named
streams, training allowance, continuation producer/horizon, clean live sampling,
cadence and every numerical gate. No task or evaluator is revised. The scalar
continuation retains horizon 1,000 through update 6,000 and its target shift.
Word hold restores its own earliest confirmed state and retains horizon 20,000;
neither continuation resets annealing. Unequal acquired prefixes are recorded
and cannot establish equal complete-stream consumption.

The exact generated declarations and effective recipe deltas are in
[spec.json](spec.json); immutable admission bindings are in
[registration.json](registration.json). Each arm reserves 43,020 seconds,
including the separate optional clock audit. Five arms reserve 215,100 seconds
within a 220,000-second campaign ceiling; this is a reservation, not predicted
execution cost. Two A6000 slots run one worker each. One configuration per idea,
one round, no scientific retries, seed variants, tuning or Tier 3 work.
`complete_current_tier` finishes independent peers; any required non-PASS then
blocks higher tiers. No diagnostic bypass is requested.

The global success predicate requires all six Tier 1 gates, all 21 eligible
Tier 2 tasks measured to PASS/FAIL, preservation of every fresh control Tier 2
PASS, and at least one additional sustained Tier 2 PASS. Archived 9/21 is
motivation only; a final good point or improved surrogate cannot win. If the
fresh control reproduces nine passes, a winner must reach at least ten without
losing any of the nine. Scientific calibration remains provisional. No named
preset or family selection changes automatically.

1. **Late annealing:** start .6, network floor .2, prior floor .1. Earlier
   searches tested start-0 floors .1/.15/.2/.3/.5 and start-.25 floor .3; their
   complete Tier 1 survivors all had constant rates. This late asymmetric arm
   is new. It predicts a mode-hold suffix of at least five while retaining
   acquisition; reduced adaptation after the scalar shift is a competing risk.
2. **Half network motion:** G/E .006, D multiplier 3, prior multiplier 5.
   Absolute D/prior rates remain .018/.030. Saved native gradients point toward
   useful modes, but network proposals travel roughly seven target sigmas and
   half-step probes improve local correction. This isolates G/E travel; it
   predicts better native precision/accuracy and sustained mode retention.
3. **Prior smoothing:** prior denominator .001 -> .01; networks stay .001.
   The row step is `-eta*g/hypot(norm(g), smoothing)`. At norm .001 its motion
   factor falls .707 -> .100, while norm .1 retains .995. This tests weak-force
   center motion without freezing, clocks or a new density objective. It
   predicts lower late prior RMS travel and improved sustained density. At
   least 25% lower Gaussian continuation row RMS motion is the mechanistic
   forecast; failing it rejects that damping account. Rare/word acquisition
   suppression remains a risk; .01 is a hypothesis, not a calibrated scale.
4. **Weak zero-centered damping:** retain BCAP and add `.05/2` times real
   plus fake squared input slopes, with the same lazy interval. Saved Gaussian
   cap-penalty gradients vanish at 1,000 and 4,000 while fake slopes stay below
   cap 1. This supplies pressure in that inactive band. Native tasks already
   have stronger cap gradients and may regress. This is an explicitly hybrid
   regularizer candidate, not pure R1/R2 replacing BCAP. The
   [zero-centered regularization analysis](https://proceedings.mlr.press/v80/mescheder18a.html)
   supports the hypothesis under its assumptions; it gives no convergence
   guarantee for this normalized finite-step trainer.

Cumulative counters distinguish prior gradient energy, ideal row proposal and
rounded base-optimizer displacement. These precede direction blending; final
motion can be equated only where saved counters verify zero projection. Saved
prefix/final counter differences give scalar continuation RMS per sampled row.
Critic diagnostics separate cap and added damping terms where the host collects
them. Missing force observations remain unavailable. Numerical task gates are
the decision evidence; saved actual-training GIFs illustrate every completed toy
without additional updates or sampling draws.

Motivation retains the original source, initialization and serving contracts:
[current prior-owned readout](../recipe-prior-refactor/publication/README.md),
[direction-blend baseline](../bcap-default-baseline/README.md),
[saved winner failure analysis](../bcap-tier2-search/FAILURE_ANALYSIS.md),
[earlier pacing comparison](../dualnorm-pacing-v2/README.md), and
[five physical repair rounds](../bcap-physics/round5/README.md).
The negative optimism, frozen-prior and transport packages remain stopped.

Reproduction uses the committed source named in the completed receipt:

```sh
.venv/bin/python reports/forge/bcap-tier2-round6/run.py prepare
mkdir -p /mnt/ml7tb/ParticleGAN-forge/bcap-tier2-round6-20261011/logs
.venv/bin/python -u reports/forge/bcap-tier2-round6/run.py run \
  --source-commit EXECUTED_COMMIT --submit --drain \
  > /mnt/ml7tb/ParticleGAN-forge/bcap-tier2-round6-20261011/logs/driver.log 2>&1
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier2-round6-20261011/logs/driver.log
tail -F /mnt/ml7tb/ParticleGAN-forge/bcap-tier2-round6-20261011/queue/events.jsonl
```

Completed originals must never be rerun solely to reproduce reporting. Resume
existing admitted work with `--drain` only. Raw logs, event streams, full receipts,
checkpoints and saved tensors stay outside Git. Commit compact metrics, audit,
provenance, reproduction sources and training GIFs. Update the single current
[technique leaderboard](../technique-inventory.md); no second goal board is made.
