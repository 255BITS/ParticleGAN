# Residual-student binding and cold-run handoff

`pr84_residual_refinement.py` binds the frozen PR84 alternating update and
penalized-critic refinement to the 400-update `residual_student` host. It is a
scratch research adapter, not a passed gate or production optimizer. The
source transform has an exact structural inverse and a pinned SHA-256 guard
on the original host training function. The host's G block is replayed whole:
Rp adversarial loss, cover loss, particle/spread terms, and the native
supervised residual on both-land rows are unchanged.

The fit bank contains eight native 12-pair conditional D batches. Each repeats
the actual `slow`/`paired` tensors while using fresh output-noise and four
ordered input-noise draws from *cloned* streams. Its first D gradient is
checked bitwise against the host's actual gradient on every active update.
The fit retains the original D Adam moment, makes no extra Adam update, and
cannot consume the host's RNG or noise clock. G sees the materialized refined
critic; no spatial smoothing applies to this conditional >2D critic.

Validation: `pytest -q tests/test_pr84_residual_refinement.py
tests/test_pr84_critic_refinement_cold.py
tests/test_pr84_critic_refinement.py` passed 23 tests. Two-update cases checked
global and isolated output RNG, nonzero input noise, first-bank gradient,
native conditioning, one Adam moment/player, constant applied rates, and
exact disabled-refinement model/optimizer/RNG/metric parity. With the scratch
recorder disabled, the unchanged original host has the same full-state hash,
noise receipt, and result. No 400-update residual gate was run.

The preceding cold candidate is **ERROR_INCOMPLETE**. Its trajectory gate
passed after 400 updates: identity MSE `0.0009098353` and 22/24 passing
observations. The ring run reported completion through update 460, then a
nonfinite penalized D L-BFGS strong-Wolfe trial raised
`FloatingPointError`. It has no complete 1200-update ring quality result,
no ring PASS, and no promotion credit. The frozen declaration, full source
snapshot, original error log, compressed trajectory receipt, and trajectory
final state are stored in
`continuous-evidence/round5-critic-refinement-cold/`; `manifest.json` records
SHA-256 hashes and the incomplete status. The exact failure-state replay is
being handled separately and is not represented by this archive.

If the cold method is repaired, the gate order remains ring acquisition and
own-acquired stationary continuation before residual_student. This binding
is available for the next cheap legacy host only after those prerequisites.
