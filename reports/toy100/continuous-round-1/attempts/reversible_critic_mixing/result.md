# Reversible critic mixing

The pinned K3P parent is unchanged: 22/22 toys, ring hold 1200/1200, extension 300/300, target-shift recovery 28/81. This lane replaced only the learning-rate-ratio critic handover. Anchor decay stays 0.999, the particle prior and the sparse-latent rule stay the copied K3P files, and floors stay 0.01/0.05.

**rcm3 keeps the hold and the 300-update extension, reopens the mix at the target shift, and still fails recovery.** Post-shift diagnostics are 0/120, which includes 0/81 from the deadline step onward. The ring is moving (0 modes at the shift, 7 modes and HQ 0.821 at step 3600) while both optimizers continue to take 3600 updates at the floor rates.

## Leaderboard

Rank is hold, then extension, then timely recovery. Toy screens were not started. Parent scores are the published K3P measurements, not re-runs.

| Candidate | Hold | Extension | Recovery | Toys | What the mix did |
|---|---:|---:|---|---|---|
| **rcm3 applied-step ratio** | **1200/1200 PASS**, min HQ 0.990 | **300/300 PASS**, min HQ 0.999 | **FAIL**, 0/120 post-shift, deadline false | NOT_RUN | Closed at critic step 497. Reopened at step 2403, the shift. |
| rcm1 gradient agreement | FAIL, aborted at step 4400, modes stuck at 7, streak 0 | NOT_RUN | NOT_RUN | NOT_RUN | Closed by step 300 on a spiked gradient peak. Reopened near step 2950. Still 7 modes. |
| rcm2 anchor-gradient gap | FAIL, aborted at step 4350, max 6 modes | NOT_RUN | NOT_RUN | NOT_RUN | Batch-mean anchor gap stayed high. `s` stayed 1 for the whole run. |
| K3P parent (not rerun) | 1200/1200, min HQ 0.907 | 300/300, min HQ 0.988 | FAIL 28/81 | 22/22 | LR-ratio clock. One-way. |

rcm1 and rcm2 were stopped before the driver's 4800-check settling budget. Both had already spent more than a thousand dense checks at streak 0. An earlier rcm2 process was an invalid launch: the anchor probe swapped parameter storage while the penalty graph was open. That process is recorded as ERROR and is not a proposal.

## rcm3 rule

Penalties A, B and P, kappa 1, coefficient 1, anchor decay 0.999, and the critic guard (5× after 200 steps) are K3P. `s` starts at 1, so the early penalty is pure A and does not evaluate the anchor.

After each critic Adam step, `disp` is the RMS of the parameter change just applied:

```text
fast <- 0.8 * fast + 0.2 * disp
slow <- 0.995 * slow + 0.005 * disp
ratio = fast / slow
m <- 0.9 * m + 0.1 * ratio
m <= 0.50 -> target 0
m >= 1.25 -> target 1
otherwise keep the previous target
s <- 0.94 * s + 0.06 * target
```

`s` does not read the learning rate, the step index, the horizon, scores, or the target location. A constant critic learning rate would leave `disp` large while gradients stay large, so `s` would stay at 1. That is how this mix leaves the LR-ratio clock.

Inherited schedules, still horizon-dependent, and labeled as such:

- Network learning rate: cosine on `min(total_steps, 1600)` down to floor 0.01, base rate 0.00425, anneal start 0.6.
- Prior learning rate: cosine on `total_steps` down to floor 0.05, multiplier 2.
- Input and output noise: horizon 1200 hardcoded in the unmodified hold and shift drivers.

## rcm3 measurements

Hold, 130.7 s, converged at step 1400 with 0 settling failures. Same convergence step as K3P.

- Hold window: 1200/1200, minimum HQ 0.99023, minimum modes 8.
- Extension, steps 2601–2900: 300/300, minimum HQ 0.99927, minimum modes 8.
- Mix closed at critic step 497, while the critic rate was still 0.00425, because the applied step had already shrunk relative to its slow memory. It did not reopen during the hold (`m` rose back toward 1 and stayed under 1.25).
- Anchor evaluations: 2402 forwards. Pure A for the first 497 calls, then the blend. Anchor weight sum 2387 over 2403 blended calls, so the anchor was essentially fully on after the close.

Shift, 131.1 s. Pre-shift continuation 120/120, minimum HQ 0.99268.

- At step 2400, before the shift: 8 modes, HQ 0.997, `s` = 0, critic displacement RMS 1.49e-5, rates already at the floor (D/G 4.25e-5, prior 4.25e-4).
- The shift drops the live ring to 0 modes. By critic step 2403, `m` crosses 1.25 and the target returns to 1. At step 2500, `s` = 0.998 and critic displacement RMS is 4.65e-5. The mix did reopen.
- From there `s` stays at 1. Checkpoint modes after the shift: 0, 1, 4, 5, 5, 5, 6, 6, 5, 6, 7. Final step 3600 is 7 modes, HQ 0.821. EMA is 6 modes, HQ 0.754.
- Recovery: 0 passing checks out of 120 after the shift. All 81 checks from step 2800 through 3600 fail. `deadline_pass` is false. No stable passing suffix, so delay is undefined.
- Both optimizers record 3600 updates and moment steps 3600. End rates are the floors above. The output is not a frozen snapshot.

No frozen control was run. The recovery did not pass, so a frozen control would not qualify one. Sensitive toy screens, the 22-gate suite, and the repeated-change stress test were not run.

## Why the other two signals failed

rcm1 mixed on agreement between the critic parameter gradient and a slow memory of it, gated by gradient RMS versus a forgetting peak. The direction gap stayed near 0.5 even after HQ reached 1, so agreement never marked a settled critic. The peak was set by a spike, ordinary gradients looked quiet, and the mix closed by step 300. Modes reached 7 at step 1000 and never reached 8, including after the mix reopened near step 2950. Aborted at step 4400, wall time 217 s.

rcm2 mixed on the normalized gap between the live critic's batch-mean input gradient and the EMA anchor's. That gap stayed O(0.3–1) after the rate had reached the floor, so `s` never left 1. The run acquired at most 6 modes, HQ later fell to 0.244, and it was aborted at step 4350.

## Horizon check

Two 40-update prefixes of rcm3, same seed and fixture, network cap 1600 with `total_steps` 7000 versus cap 3200 with `total_steps` 14000. Critic displacement traces matched, `s` stayed 1 in both, and the randomness digest matched (`66e267ec…`). Those 40 updates end before either network anneal (steps 960 and 1920) and before the driver's noise anneal. The match shows the mix does not read the horizon value.

A prefix that crosses an anneal point still changes, because of the inherited schedules:

- Network cosine uses the cap, so cap 1600 and cap 3200 diverge at step 960.
- Prior cosine uses `total_steps`.
- Noise horizon stays 1200 inside `hold.py` and `shift.py` regardless of the config cap.

## Replay

Environment for every command: `CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, one CPU thread, `/tmp/pr38-default-env/bin/python`. Runtime repo and mode-hold fixture are the gap-fill manifest paths. Scripts:

- `repo/candidates/rcm3/run_hold.sh`
- `repo/candidates/rcm3/run_shift.sh`

Sources, with parent hashes in `reports/toy100/gap-fill-20260925/manifest.json`. Only `mechanism.py` differs inside each candidate. Latent and response match the parent (`197df635…`, `7e71d60a…`).

| File | sha256 |
|---|---|
| parent `mechanism.py` | `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391` |
| rcm1 `mechanism.py` | `94fc4c1ce647f35c2ea5a1c8ddc45db75134d2d892dc3aa1f6a30eb2221b65e4` |
| rcm2 `mechanism.py` | `c68310845d1f2ecc1deb297e71d9c064d4ad5c3c2d49630670266e5727457565` |
| rcm3 `mechanism.py` | `acffadaf0ca15bad27e66b5f557554c8061d0f18de29fdba130c2e6a112b10d9` |

Logs and results:

- `repo/runs/rcm3/hold/result.json`, `repo/runs/rcm3/hold.log`
- `repo/runs/rcm3/shift/result.json`, `repo/runs/rcm3/shift.log`
- `repo/runs/rcm3/horizon/compare.json`
- `repo/runs/rcm1/hold.log`, `repo/runs/rcm2/hold.log`, `repo/runs/rcm2/hold-invalid.log`

Gate log: `tests.jsonl` next to this file.

## Next mechanism

Keep the applied-step close. It is what preserved an 8-mode hold after the LR-ratio clock was removed. Reopening all the way to the early R1 penalty, at the floor rate, did not recover the shifted ring inside 1200 further updates. The next change should leave a path for a larger applied step when `ratio` spikes, or should drop the anchor without turning zero-centered R1 back on. Either change is a new candidate. Do not treat a higher learning rate alone as success, and do not start the toy screen until a recovery deadline passes with a matched frozen control.
