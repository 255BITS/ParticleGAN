# Penalty-balance lane: K3P stays selected

No promotion. Two controllers were fully run on their own hold, 300-update extension, and target shift. Neither reached deadline 81/81. Toys, frozen control, delayed/repeated shift, seeds 1235–1237, and the 30000-update continuation are NOT_RUN.

Best partial lead is **pb2_share_square**: hold 1200/1200, extension 300/300, stationary 5/5, pre-hold 120/120, deadline **77/81**, sustained recovery at step 2880 (delay 480). Parent K3P, not rerun, is 1200/1200, 300/300, and 28/81 with delay 1130.

## What the decomposition measured

pb0 copies K3P and adds one `autograd.grad` of the penalty with respect to the critic parameters on every penalty call. That gradient is not written into `.grad` and is not an optimizer step. `g_adv = g_total - g_pen` (backward coefficient stayed 1 on all 3600 steps; 0 decompose errors). Seven of eight critic tensors match; the unused tensor is 1 element out of 20161. Instrumented training reproduced the parent shift exactly: stationary 5/5, pre-hold 120/120, deadline 28/81, delay 1130, stable at 3530.

On the quiet floor (steps 1600–2399) the penalty force is about 7% of the adversarial force (median fight 0.014). In the first 400 updates after the shift, adversarial force rises about 3×, penalty force about 10×, cosine goes to about −0.50, and fight median rises from 0.014 to 0.257, while mixing weight stays 0 and both rates stay on the floor. The anchor is opposing the new data fit, and K3P's learning-rate clock cannot release it. Total-gradient RMS also rises, so the shift is not invisible to a raw RMS rule. That is the signal behind RP1's stationary native false restart: total RMS can move when the penalty share does not.

## Candidates

| Rank | Candidate | Hold | Extension | Stationary | Pre-hold | Deadline | Delay | Toys |
|---|---|---|---|---|---|---|---|---|
| — | K3P parent, not rerun | 1200/1200 | 300/300 | held | held | FAIL 28/81 | 1130 | 22/22 prior |
| 1 | pb2_share_square | PASS 1200/1200, min HQ .913 | PASS 300/300, min HQ .976 | 5/5, min HQ .972 | 120/120, min HQ .914 | FAIL 77/81, min HQ .839, min modes 7 | 480, stable 2880 | NOT_RUN |
| 2 | pb1_fight_reopen | PASS 1200/1200, min HQ .980 | PASS 300/300, min HQ .995 | 5/5, min HQ .992 | FAIL 112/120, steps 1280–1350, min HQ .635, min modes 7 | FAIL 79/81, misses 2910 and 3570, min HQ .871, modes 8 | none | NOT_RUN |
| — | pb0_force_measure | NOT_RUN | NOT_RUN | 5/5 | 120/120 | FAIL 28/81 | 1130 | NOT_RUN |

pb0 is a measurement of unchanged K3P, not a controller. Its canonical hold was not repeated.

**pb1** raises the K3P network and prior multipliers by an EMA (decay 0.9) of the fight statistic `relu(-cos) * 2 * An * Pn / (An² + Pn²)`. Mixing weight stays K3P's handover of the applied critic LR, so a higher rate also releases the anchor. Acquisition matched K3P while the schedule multiplier was 1 (same adversarial and penalty norms as pb0 through step 700). The quiet-floor fight near 0.02 never let the rate return to the floor, and pre-hold checks 1280–1350 failed as the anchor was turning on. Deadline 79/81 kept all eight modes but never locked a sustained suffix.

**pb2** uses the same decomposition and the same handover. The gain is an EMA of `q²`, where `q = Pn / (An + Pn)`. Squaring suppresses the quiet floor (gain about 0.005, applied critic LR about 6e-5 versus the 4.25e-5 floor) and still reopens after the shift (mean gain 0.097 over steps 2400–2800, mean critic LR 0.00045, mean mixing weight 0.20). Pre-hold is restored to 120/120. The output is not stationary: at the shift it falls from 8 modes, HQ .999, to 0 modes, then 4 modes at step 2500 and 8 modes at 2700. Both optimizers run 2400 updates at the shift and 3600 at the end, with no counter reset. Four deadline checks fail, steps 2840–2870, at 7 modes and HQ .839–.870. Sustained recovery starts at 2880, 80 updates after the 2800 deadline. Final live state is 8 modes, HQ .998.

## Rules still in both controllers

- Particle prior, direct response, and bounded sparse-latent damping are byte copies of K3P (`latent.py` `197df635…`, `response.py` `7e71d60a…`).
- Guard remains 5× Adam RMS after 200 steps. Anchor decay remains 0.999. Penalty family remains K3P's R1 blend into cap plus EMA anchor.
- One extra penalty backward per critic step. One Adam update per role.
- No task id, target center, quality score, shift time, or known change.
- **Horizon remains.** K3P's noise schedule uses horizon 1200. The ring drivers also feed that 1200 into the learning-rate cosine (`FixedControl(..., noise_horizon)`), so by the shift both rates are already on their floors even though 3600 updates are executed and the config cap is 1600. The new gain term does not read that horizon. These candidates are labeled intermediates, not horizon-free results.

## What the next mechanism has to change

The squared penalty share is strong enough to keep the pre-shift contract and to move sustained recovery from 3530 to 2880. It is not strong enough to avoid the 7-mode dip at 2840–2870. The linear fight gain was slightly closer on the deadline count and lost the pre-hold. A later rule needs a larger reopen only while the penalty share is elevated, without a quiet-floor bias, and then has to delete the 1200-step noise and LR horizon before it can be a final formulation. Image `img_intensity2` and native grid100 were not opened: recovery is not 81/81, so this is not a ring survivor.

## Replay

```sh
export CUDA_VISIBLE_DEVICES=GPU-72c1b506-891d-b8bc-b353-e020585e1c47
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/tmp/pr38-default-env/bin/python
REPO=/ml2/hypergan/gan-attempts/claude-pool-20260925T041936Z/qualify_a2_bounded_damp/20260925T041936Z-2509885/repo/reports/toy100/qualify-a2-attempt/prepared/repos/cuda
FIX=/ml2/hypergan/ParticleGAN-epsilon-gan-followup/reports/toy100/direct-particle-base/initialization-fixtures/mode_hold/initial-values.pt
BASE=reports/toy100/penalty-balance-3777313/pb2
# mechanism SHA256 62396a5e4058eacd7b3f1356e01db3fa0ec9e3539c2b399e32417f447570bd61

$PY -u $BASE/hold.py --repo $REPO --config $BASE/config.json \
  --task mode_hold --backend cuda --initial-state $FIX \
  --output <fresh> --network-floor 0.01 --prior-floor 0.05

$PY -u $BASE/shift.py --repo $REPO --config $BASE/config.json \
  --task mode_hold --backend cuda --initial-state $FIX \
  --output <fresh> --network-floor 0.01 --prior-floor 0.05
```

pb1 mechanism `267ed8d7…`. pb0 mechanism `6f8808f5…`. Pinned parent mechanism is unchanged, `d2eb08ee…`.

Artifacts: `repo/reports/toy100/penalty-balance-3777313/runs/pb2-hold/result.json`, `runs/pb2-shift/result.json`, `runs/pb2-shift/force-trace.jsonl`, and the same layout for pb0 and pb1. Ledger: `tests.jsonl` (3 PASS hold/extension rows for the controllers, 2 FAIL shifts, 1 diagnostic PASS, 1 diagnostic FAIL).
