# Finite critic refinement: numerical repair, failed cold acquisition

**Rejected as a complete no-decay replacement.** The guarded source epoch
completes both unchanged cold budgets. Trajectory passes; ring fails with
three modes/HQ .451416 and zero of24 passing observations. No own-acquired
hold, model-error response, residual host or remaining production gate follows.

| Cold host | Budget | Final live metric | Passing observations / suffix | Result |
|---|---:|---|---:|---|
| Trajectory |400|MSE .000909835|22/24;22|PASS|
| Ring |1200|3 modes;HQ .451416|0/24;0|FAIL|

The [complete archive](continuous-evidence/pr84-critic-refinement-finite-cold/manifest.json)
contains both raw results, both final model/Adam/EMA/RNG/noise snapshots,
the launch gates, source snapshot and log. Stored and decompressed hashes are
included. Root independently regrades both verdicts from raw results, verifies
every executed-source hash, checks finite final snapshots, all absolute Adam
counts and noise clocks. The trajectory final snapshot is byte-identical to
the original unguarded pass. The ring advances every player's moments1200
times at G/D .00425 and prior .0085; no LR decay or reset occurs.

Two inner trials, at472 and1037, are rejected. Each fit retains an already
evaluated finite training-loss minimum without retrying. The ring uses69,328
fit gradient attempts on1024 pairs, of which69,326 are finite, plus1200
128-pair first-bank parity queries and the original three native fields per
player/update. This is substantial additional computation, not an equal-cost
comparison. Ring episode time is789.22 seconds on one pinned CPU thread.

The acquisition failure predates either rejected numerical trial. At the
first observation, update50, no samples pass HQ; at100 the model covers modes
0,1,7. Every observation from100 through1200 retains those same three modes
and the same sampled support's nearest-mode assignments (four particles each).
Those support points include one noise draw per particle, so this does not
claim an exact clean trajectory between observations. Missing modes2–6 never
appear in these observations. HQ varies greatly: .00708 at200, .99927 at1000,
then .45142 at1200. A high-HQ endpoint at1000 would have hidden the absent
modes and the later accuracy loss.

This does not negate the original borrowed-state44-check, warm200 and dense
hold1200 passes. It shows that their conditional stability does not establish
acquisition from the unchanged initialization. The [cold472 field diagnostic](pr84-cold472-field-diagnosis.md)
also shows why a toward-missing directional sign is inadequate: most of those
steps simultaneously approach their currently occupied mode. A [single
paired endpoint refit](pr84-profiled-field472.md) finds a stronger restoring
critic response than the frozen-D secant, giving no support for simply
increasing G's gain.

Next tests remain small: capture the first100 updates to isolate the early
three-cluster path, then compare a generator gradient with and without the
derivative through one identical virtual penalized-D update. The cap's
fake-coordinate dependence must be retained and finite-difference checked.
This is distinct from both the current fitted-D partial gradient and earlier
joint implicit-game experiments. [Research and limitations](critic-response-research-followup.md).
No fundamental impossibility has been established.
