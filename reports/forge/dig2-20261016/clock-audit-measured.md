# Dig2: judging the clock-free audit by measurement (clock-audit-measured), 2026-10-16

**Verdict: supported. The support rests on a technicality, and that technicality is the policy question.**

- Under the proposed measured audit rule, the internal-counter anneal (`count6400`) passes ring16 at a 6,400-update cap *and* the clock audit, for both K3P and KA2.
- Its ring16 trajectory is **bit-identical** to the step-schedule control in all 384 observations. A fixed internal counter is the step/horizon clock under another name. It reads neither the step label nor the declared horizon, yet it reproduces the step-scheduled run exactly.
- The one horizon-free anneal (the critic-gated `gated` arm) fails ring16 badly.
- A second finding needs no audit change: **constant-rate K3P passes ring16 at 6,400 and also the unchanged v1 clock audit.**

Branch `task/dig2-clock-audit-measured-20261016`. The preregistration (`reports/forge/dig2-20261016/clock-audit-measured-preregistration.md`, commit `3dd9e49c`) was pushed before any run.

## Terms

- **Clock-free audit.** `clockfree_audit_measurement_v1` runs 4 warmup updates, then 4 more under each of four perturbations:
  - step label +8192
  - declared horizon changed from 128 to 4096
  - extra evaluation sampling
  - save and restore

  It compares hashes of the full learning state across these runs. It then applies a *declaration table*, which fails any recipe whose fields imply a schedule. Examples are `lr_floor≠1`, input-noise anneal, output-noise warmup, the critic-guard minimum count and KA2's call-800 switch.
- **Measured parity.** All four perturbations leave the learning state bit-identical.
- **Power witness (new).** For each schedule the table flags, it checks whether the probe *could* have seen a step- or horizon-driven version of that schedule. It evaluates the recipe's own LR, input-noise and output-noise shapes with step-label and horizon drivers at the probe's settings. If a value would change, a passing measurement is real evidence that the schedule is not step-driven.
- **Internal counter.** A count kept in saved training state. Examples: Adam's step count (used by the critic guard), KA2's penalty call count, `progress_schedule="updates"`. It reads no external clock, but it encodes a fixed duration.
- **Sustained gate (ring16).** A run passes only if at least 5 *terminal* consecutive observations meet every threshold: modes ≥ 16, mass_tv ≤ 0.15, hq ≥ 0.85, component covariance error ≤ 0.85, min eigen ratio ≥ 0.15.

## Part 1: proposed audit revision (needs sign-off; nothing counted was changed)

New task: `clockfree_audit_measured_v2-clockdig`. It uses the same probe and comparisons as v1. v1 is untouched. The rule `measured_parity_v2` is implemented in `experiments/forge/clockfree.py:measured_v2_decision`. The grader only branches when `evaluation.audit_rule == "measured_parity_v2"`.

1. **Measurement first.** If any of the four comparisons differs, the verdict is FAIL. Nothing overrides this.
2. **The declaration table becomes a pre-screen.** It is the v1 table plus one new entry: `progress_schedule="updates"` counts as a fixed internal update count. Entries fall into three groups:
   - **Class S** (LR anneal, input-noise anneal, output-noise warmup). A passing measurement overrides the entry if the power witness shows the probe could have detected that component.
   - **Class C** (critic-guard minimum Adam steps, KA2 call 800, progress `updates`). Overridden only under `internal_counter_policy="measured"`, and only if the counter is found in saved learning state at the warmup.
   - **Never overridden:** `continuous_policy`, extensions and the lazy-penalty `reg_every` counter.
3. **PASS** when no entry still stands. The research task uses `"measured"`. The strict variant (`"declared"`) is regraded offline from the same saved proofs by `summarize.py`.

Unit tests are in `tests/test_forge_clockfree_measured_v2.py` (9 tests). They check that:

- an internal-counter anneal fails v1, passes v2, and fails under the strict policy;
- a state-driven anneal passes under both policies;
- a step-scheduled recipe still fails by measurement;
- the guard counter outcome depends on the policy choice;
- KA2's call counter is witnessed from saved state;
- an unpowered probe keeps its S entry;
- extension and lazy-penalty entries are never overridden;
- an unknown rule is INVALID.

`tests/test_forge_clockfree.py` and `tests/test_progress_schedule.py` pass unchanged (23 tests in total across the three files).

## Part 2: experiment

**Preregistered rule.** An arm *dual-passes* if ring16_6400 is PASS (sustained gate) and v2 with measured counters is PASS. The lead is **supported** if any annealing arm (count6400 or gated) of K3P or KA2 dual-passes.

**Setup.** All arms use seed 0 and one frozen source and runtime, on a CUDA RTX A6000 with one leased GPU. Every arm sets `d_guard_min_steps=0` and `network_lr_horizon_cap=None`.

| Arm | Change |
| --- | --- |
| step | shipped cosine LR, input-noise and output-noise schedules over the task-owned 6,400-update horizon (positive control; reads the clock) |
| const | no annealing: LR floors 1, input noise 0, constant output noise |
| count6400 | `progress_schedule="updates"`, rate 1/6400: an internal counter with the same shape as step |
| gated | `progress_schedule="critic_order"`, rate 0.001, target 0.75, EMA 0.98 (round-1 C settings, not tuned) |

**Task.** `ring16_acquisition_6400-clockdig` is the counted ring16 task with three changes: steps set to 6,400, schedule horizon set to 6,400, and 384 observations. This keeps the original cadence of one observation per 16.7 updates. Gates are unchanged. The id suffix `-clockdig` keeps it separate from the ring16-cap lead's revision.

### Results (ring16 at 6,400 updates, 384 observations; covariance limit 0.85)

| Arm | ring16 6400 | Terminal streak | Passing obs | Last failing update | Final cov | Cov at 1600 / 3200 / 4800 | v1 audit | v2 (measured) | v2 strict | Dual pass |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| K3P step | PASS | 104 | 147 | 4667 | 0.497 | 1.45 / 0.82 / 0.74 | FAIL | FAIL (step_label) | FAIL | no |
| K3P const | **PASS** | 47 | 127 | 5617 | 0.525 | 1.34 / 0.80 / 0.81 | **PASS** | PASS | PASS | **yes (no anneal)** |
| K3P count6400 | PASS | 104 | 147 | 4667 | 0.497 | identical to step | FAIL (decl.) | **PASS** | FAIL (counter) | **yes** |
| K3P gated | FAIL | 0 | 0 | 6400 | 1.993 | 1.79 / 2.59 / 2.16 | FAIL (decl.) | PASS | PASS | no |
| KA2 step | PASS | 54 | 188 | 5500 | 0.435 | 1.27 / 0.60 / 0.57 | FAIL | FAIL (step_label) | FAIL | no |
| KA2 const | FAIL | 3 | 180 | 6350 | 0.785 | 1.29 / 0.63 / **1.99** | FAIL (call 800) | PASS | FAIL (call 800) | no |
| KA2 count6400 | PASS | 54 | 188 | 5500 | 0.435 | identical to step | FAIL (decl.) | **PASS** | FAIL (counters) | **yes** |
| KA2 gated | FAIL | 0 | 0 | 6400 | 1.924 | 1.46 / 1.91 / 1.97 | FAIL | FAIL (restart, see caveats) | FAIL | no |

What the table shows:

1. **count6400 is bit-identical to step.**
   - All 384 sampled observations match exactly for both families.
   - The internal counter p differs from step/6400 by at most 6.5e-14. That difference does not survive float32 parameter updates.
   - In the 4-update probe, step fails `step_label` and count6400 passes every measurement.
   - So the measured audit separates the two runs only by *how the clock is read*, not by what the trainer does.
   - Under v2 with measured counters, an anneal "passes the clock audit" whenever its duration is written as a recipe constant. Here that constant (1/6400) was chosen to equal the task cap.
2. **The horizon-free anneal fails.**
   - gated finishes annealing early: the round-1 K3P replay reached p≈0.94 by update 1,550.
   - After that it sits at the LR floor and never passes (final covariance error 1.99 for K3P, 1.92 for KA2).
   - This matches round 1: annealing early costs covariance on ring16.
3. **K3P needs no annealing at 6,400 updates.**
   - K3P const passes ring16 (streak 47, last failure at update 5,617) and the *unchanged* v1 audit.
   - KA2 const acquires the ring (180 passing observations, covariance error 0.63 at update 3,200). Around update 4,800 it destabilises (covariance error 1.99) and ends with a streak of 3. The annealed KA2 variants hold.
4. **The step control reproduces the earlier 6,400-update study** (ring16-schedule-20261011, where K3P and KA2 passed).
   - Streaks are longer here: 104 and 54, against 66 and 47 in that study. This branch uses `d_guard_min_steps=0` and no network horizon cap.
   - As before, every arm still fails at the 1,600-update cut.

### Answer

- **With v2 and `internal_counter_policy="measured"`: yes.** K3P count6400 and KA2 count6400 pass both ring16 and the clock audit. But count6400 *is* the step schedule. Accepting this result amounts to a policy decision to accept fixed-duration schedules.
- **With strict counters (`"declared"`): no annealing arm passes both.** The strict policy matches the intent of v1's guard and call-800 entries.
- **K3P at constant rate (no annealing)** is the only family that passes both without relabelling a clock, under any rule. It also passes the current v1 audit.

## Commands

```sh
PY=/ml2/codex-storage/workspaces/2026-10-07/task-14/runtime/python312/bin/python; export OMP_NUM_THREADS=1 PYTHONPATH=.
$PY reports/forge/dig2-20261016/clock-audit-measured/make_tasks.py     # 2 research tasks
$PY reports/forge/dig2-20261016/clock-audit-measured/make_arms.py      # 8 cards, view, 8 studies
$PY -m pytest -q tests/test_forge_clockfree_measured_v2.py tests/test_forge_clockfree.py tests/test_progress_schedule.py
Q=/ml2/hypergan/dig2-20261016/clock-audit-measured/queue
for c in {k3p,ka2}-cad-{step,const,count6400,gated}-v1; do
  $PY -m experiments.forge --root . --queue-root $Q enqueue $c --study dig2-cad-$c-study; done
# one drain on one leased GPU; board compile deferred (see caveats):
gpu-lease.sh bash -c '$PY reports/forge/dig2-20261016/clock-audit-measured/drain_nocompile.py --root . --queue-root $Q drain --gpus $LEASED_GPU --workers-per-gpu 1 --campaign dig2-clock-audit-measured-v1'
$PY reports/forge/dig2-20261016/clock-audit-measured/summarize.py $Q/dig2-clock-audit-measured-v1 summary.json
```

`drain_nocompile.py` replaces `knowledge.compile_memory` with a no-op and then calls `experiments.forge.__main__.main`. Attempts, grading and receipts are unchanged.

Logs (all tail-able):

- `/ml2/hypergan/dig2-20261016/clock-audit-measured/queue/events.jsonl`
- `/ml2/hypergan/dig2-20261016/clock-audit-measured/queue/dig2-clock-audit-measured-v1/<attempt>/run.log`
- `/ml2/hypergan/dig2-20261016/clock-audit-measured/logs/drain.log`

The per-arm summary is in `/ml2/hypergan/dig2-20261016/clock-audit-measured/summary.json`. A copy is committed next to this file.

## GPU cost

- **Training:** 680 s for the eight ring16 runs (71 to 105 s each), plus 24 s for the sixteen clock probes. Total 704 s, about **0.20 GPU-hours**, against a budget of 1.5.
- **Lease wall time:** about 37 min. Roughly 21 min of that was the first drain sitting in Forge's post-completion board compile while the GPU was idle (see caveats).

## Caveats

- **Scope.** One seed (0), one source, untuned progress settings. KA2 const's late instability comes from a single trajectory.
- **KA2 gated failed v2 on `restart`.**
  - Exactly one host float differs: `optimizers/1/regularizer/record/last_sur` at update 3, 0.64805317 against 0.64805323 (a difference in the last float32 digit). All later updates match again.
  - The same probe on CPU is fully identical.
  - This looks like a CUDA restore-numerics issue in the KA2 penalty record, not a clock dependence.
  - The rule never overrides a measurement, so the FAIL stands. It does not change the answer, because KA2 gated also fails ring16.
- **Power witness coverage.** The witness tests the shapes Forge knows (cosine LR, linear input-noise anneal, output-noise warmup). Any new scheduled mechanism needs its own witness entry; until it has one, its declaration stands.
- **Counter witness is structural.** It only checks that the counter is present in saved state. A 4-update probe cannot exercise thresholds at 200 or 800 calls.
- **Infrastructure: board compile blocks the GPU lease.**
  - On this tree, `drain` runs `compile_memory` after each completion. That is a full leaderboard rebuild, 56 to 68 MB of JSON per goal.
  - It took more than 20 minutes and held the GPU lease idle.
  - I stopped that drain while the queue state was consistent, and re-ran it with the compile disabled at runtime. No source change was made, and the frozen source digest is unchanged.
  - Other dig2 workers set `FORGE_SKIP_COMPLETION_COMPILE=1`. That variable does not exist in any branch, so it has no effect.
- **Pre-existing test failure inherited from #425.** `tests/test_forge_views.py::test_frozen_thresholds_reference_the_declared_gate_policy_and_current_scorer` fails because #425 changed `particlegan/policy.py`, which word tasks pin by hash. This branch did not cause it.
- **Scoped research task.** `ring16_acquisition_6400-clockdig` is not the ring16-cap lead's counted revision. Its timeout is 600 s.
