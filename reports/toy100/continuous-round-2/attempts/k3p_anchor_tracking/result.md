# K3P anchor tracking (round 2)

K3P stays the selected base. No candidate passed its own hold, the 300-update extension, and the raw shift verdict. `current-research-base.json` was not changed. Parent scores were reused and the pinned sources were not edited.

Parent, not rerun: hold 1200/1200 (min HQ 0.90723), extension 300/300 (min HQ 0.98779), shift deadline 28/81, sustained delay 1130. Mechanism `d2eb08ee932b288cbba25cd1e7be3a9572b129bd1baf0b79718be1eb37ba9391`.

## Leaderboard

| Rank | Candidate | Hold | Extension | Shift verdict | Toys |
|---|---|---|---|---|---|
| — | K3P parent | 1200/1200 | 300/300 | FAIL 28/81, delay 1130; pre-shift held | 22/22 prior evidence |
| 1 | at3 peak rate, slow reference | FAIL 356 good checks, broke at step 5778 | NOT_RUN | FAIL. Stationary 0/5, continued hold 0/120, deadline 38/81, delay 1070 | NOT_RUN |
| 2 | at2 generator-energy rate | FAIL, stopped step 4350, max 7 modes, HQ ~1, streak 0 | NOT_RUN | NOT_RUN | NOT_RUN |
| 3 | at1 cosine / Adam-innovation | FAIL, stopped step 5650, max 7 modes, streak 0 | NOT_RUN | NOT_RUN | NOT_RUN |

at3's 38/81 deadline count is higher than the parent's 28/81 and the run did reach 8 modes with HQ 1.0 at step 3600. The shift verdict still fails: the stationary window and the pre-shift continued hold are 0, and sustained recovery is at step 3470 (delay 1070), past the 2800 deadline. The separate hold also failed. That conjunction ranks below K3P.

## What each candidate changed

All three copy K3P (`latent.py` `197df635…`, `response.py` `7e71d60a…` unchanged). Input noise is the constant 0. Output noise is the constant configured peak 0.029. The critic guard is unchanged (5× Adam RMS after 200 steps). Horizon arguments are accepted and ignored. The critic mix does not read the applied learning rate.

**at1** (`5ef08580…`). Persistence of positive critic-gradient cosine and mean Adam innovation set the mix, the reference step size, and a reversible rate. A single tensor at the guard ratio 5 reopened the full rate even while the mix stayed 0.

Measured: cosine persistence stayed near 0.05 through acquisition, and mean Adam innovation was already below 1.5 by step 100. At step 200 the anchor turned on and the rate began to decay (2 live modes). Later spikes reopened the peak rate with the anchor still on. Step 600 was 5 modes; step 1000 crashed to 0 modes; step 1200 was 1 mode. Stopped at step 5650, live 7 modes, HQ 0.999, streak 0. Driver did not write `result.json`. Log: `runs/at1-hold.log`. About 340s.

**at2** (`6ad74d46…`). Rate and mix follow dense-generator gradient RMS divided by a leaking peak of that RMS. The reference speeds only while that ratio is high, which is also when the anchor is off.

Measured: the generator gradient fell by step 500 while the ring was at 7 modes and HQ 1.0. The rate then sat near the floor with the anchor on. Step 600 was still 7 modes, HQ 1.0. Stopped at step 4350 in the same 7-mode state, streak 0. The missing mode was never acquired. Log: `runs/at2-hold.log`. About 225s.

**at3** (`ae271639…`). Network multiplier fixed at 1 and prior multiplier fixed at 1 (G/D 0.00425, prior 0.0085 on every step). Mix is 1 for the 200-step guard warmup, with reference alpha 0.05, then mix 0 and alpha locked at 0.001. A large prox residual does not speed the reference. This is the decoupling from K3P's handover: a constant peak rate would otherwise leave the mix at 1 and the anchor off.

Hold, driver finished, 304.03s, `runs/at3-hold/result.json`:
- Status `POST_CONVERGENCE_FAIL`
- Step 600: 6 modes, HQ 0.450 (parent has 8 modes by this step)
- Converged at step 5421
- Hold broke at step 5778, HQ 0.6106, 8 modes. Good checks 356 of 1200
- Run ended at step 6078, live 5 modes, HQ 0.662
- Applied critic LR stayed 0.00425. Final mix 0, alpha 0.001. Extra critic forwards 5878
- Extension NOT_RUN

Shift, driver finished, 147.61s, `runs/at3-shift/result.json`:
- Stationary steps 1000–1200: 0/5, min modes 3, min HQ 0.212
- Continued hold 1210–2400: 0/120, min HQ 0.103
- Deadline window: 38/81, `deadline_pass` false, min modes 4, min HQ 0.391
- Sustained recovery at step 3470, delay 1070. Post-shift passing checks 60/120. Suffix 14
- Final live and EMA: 8 modes, HQ 1.0
- Both optimizers: 3600 Adam updates, moment steps 3600, no counter reset
- Rates constant: D 0.00425, G 0.00425, prior 0.0085 (3600 observations each)
- Final mix 0, alpha 0.001, extra critic forwards 3400
- The output moved after the shift (modes fell to 1, then returned). It was not a frozen sample

Matched frozen control, the sensitive four, the 22 toys, the two-horizon prefix, and the delayed/repeated change were NOT_RUN. The shift adapters recorded 14831 ignored horizon arguments. That count is not a paired training prefix.

## What this narrows

Critic-gradient cosine does not mark acquisition. Adam innovation falls quiet by about step 100, while modes are still missing. Generator-gradient energy is already small at 7 modes and HQ 1, so flooring the rate on that signal locks out the eighth mode. Turning the slow anchor on at step 200, including at the constant peak rate, also misses the parent's step-600 eight-mode acquisition. Reopening the peak rate from a one-tensor Adam spike while the anchor stays on knocks modes off.

A later rule has to keep pure `a_r1r2` and the peak rate until a signal that is still absent in the 7-mode, HQ-1 state. This round did not find that signal. Speeding the reference because the prox residual is large remains the innov3 failure (hold passed, recovery 0/81) and was not repeated.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
BASE=reports/toy100/k3p-anchor-tracking-3675716

$PY -u $BASE/at3/hold.py --repo $REPO --config $BASE/at3/config.json \
  --task mode_hold --backend cuda --initial-state $FIX \
  --output <fresh-dir> --network-floor 0.01 --prior-floor 0.05

$PY -u $BASE/at3/shift.py --repo $REPO --config $BASE/at3/config.json \
  --task mode_hold --backend cuda --initial-state $FIX \
  --output <fresh-dir> --network-floor 0.01 --prior-floor 0.05
```

at1 and at2 use the same hold command with their own directories. Wall-clock seconds are not a ranking metric; another process was visible on this GPU UUID during part of the round.

Gate ledger: `tests.jsonl` beside this file. Sources: `repo/reports/toy100/k3p-anchor-tracking-3675716/at{1,2,3}/`.
