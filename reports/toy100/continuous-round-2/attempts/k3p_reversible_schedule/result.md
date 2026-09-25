# k3p_reversible_schedule — in progress

Selected parent K3P remains unchanged; its existing evidence is 22/22 toys, hold 1200/1200, extension 300/300, recovery FAIL 28/81. No parent rerun. No promotions.

One worker, frozen CUDA runtime/fixtures from gap-fill manifest, assigned GPU, FP32/deterministic/TF32 off, one CPU thread. Candidate sources are independent K3P copies under `repo/candidates/`; exact parent hashes are checked against `current-research-base.json`. No agents, seed repeats, pushes, comments, or runtime edits.

C1 loss-envelope shift: **FAIL**, acquisition 0/5, pre-shift hold 0/120, deadline 0/81, no sustained recovery. Its high critic-score difference saturates the controller, leaving the early R1 regime active and rates near full. Both optimizers take 3600 updates. Hold is still running to the canonical endpoint.

C2 motion-hysteresis completed both drivers: FAIL, all 4800 settling checks fail, acquisition 0/5, pre-hold 0/120, recovery 0/81. It was prepared from K3P, not C1/A3. It replaces score magnitude with rate-normalized critic displacement, separates mixing and rate response memories, and uses proportional reopening rather than a full-rate latch. It retains original noise schedules as an explicitly horizon-dependent intermediate.

Logs: `repo/runs/<candidate>/<gate>_mode_hold.log`. Gate ledger: `tests.jsonl` next to this report. Replay: `/tmp/pr38-default-env/bin/python -u repo/tools_local/run_gate.py <candidate> hold|shift`; exact commands/environment are also in each run's `.command.json`.

All full-toy qualification, matched frozen recovery, and repeated-change stress are currently **NOT_RUN**. Final leaderboard and recommendations will replace this progress report.

C3 directed motion: raw canonical shift **FAIL**, despite 81/81 deadline checks (minimum HQ .9370117, delay340). Acquisition0/5 and pre-shift hold0/120 both fail. This is a partial recovery diagnostic, not a solved ring or candidate eligible for all22. Hold is running; a matched frozen diagnostic and 2048-update full-state horizon comparison are next. No promotion.

C3 own hold is now complete: NOT_CONVERGED, all4800 settling checks fail, max7 modes, 0/300 post-failure diagnostics. Its matched frozen diagnostic PASSED 47 identity/sensitivity checks, including exact pre-shift full-state hash42cd092e228244feda2152792c5e8818fa7c62a875c54262cd84b8abf58eea3d, live81/81 versus frozen0/81. Full canonical verdict remains FAIL. A long5600-update horizon pair (declared6000 vs12000; caps1600 vs3200) is running.
