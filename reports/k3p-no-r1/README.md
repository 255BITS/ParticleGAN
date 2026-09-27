# K3P without R1: what replaces R1's two jobs?

R1 on reals penalizes the correct slope: it pulls grad D(real) toward 0 even when D is right, which flattens D inside
modes. The suspicion is that this is why k3p_simple fails the native toy100 within-mode covariance (accuracy) gate.
R1 does two jobs, spike control and local settling, so this study removes R1 and replaces each job separately.

Base for every arm: `gs2_c03_lr2_d05` from `../k3p-constant-suite/GRID.md`. That is k3p_simple with reg_coeff .3,
lr .0085 and d_lr_mult .5: RpGAN, one-sided RMS fake cap, AMSGrad (0, .999), constant LR, no instance noise, no EMA
anchor, no direct-particle response, spike guard and A2 on. Every arm removes R1 on reals. The fake cap always stays.
All terms are in RMS units (d = critic input size) and scaled by `reg_coeff/2`, K3P's convention:

    pen = c/2 * [spike + mean relu(||grad D(f)||/sqrt d - kappa)^2 (+ anchor_weight * prox)]

| spike | term |
|---|---|
| none | nothing (the R1-removed control) |
| symcap | mean relu(\|\|grad D(r)\|\|/sqrt d - kappa)^2 |
| pathcap | the same cap at x = r + u (f - r), u ~ U(0,1) per batch-index pair |
| pairsec | secant slope s_i = \|D(r_i) - D(r_j)\| / \|\|r_i - r_j\|\|, r_j = nearest other real; mean relu(s/sqrt d - kappa)^2 (no double backprop) |
| dvalcap | mean relu(\|D(r) - mean_batch D(r)\| - 1)^2 |

| settle | change |
|---|---|
| none | nothing |
| hinge | critic loss mean relu(1 - (D(r) - D(f))), same pairing; the generator keeps RpGAN softplus |
| oadam | critic update = `lib/oadam.py` OptimisticAdam(amsgrad=True), betas (0, .999), base critic LR; the spike guard reads `max_exp_avg_sq` |
| anchor | K3P's EMA-anchor prox term (stock weight 1, decay .999) forced on at constant LR, where it is normally gated off (s == 1) |

That gives 20 arms, `nr_<spike>_<settle>` (`gen_arms.py` -> `arms.json`).

## How it runs

`nr_adapter.py` extends `../k3p-constant-suite/suite_adapter.py` without editing it or `particlegan/`. It patches
classes once per task process, so every host sees the same change: native toy100, the standard-trainer and
custom-loop transfer toys, mode_hold hold/shift and the rings. `run_nr.py` reuses the suite's worker and task
protocols unchanged.

    cd reports/k3p-no-r1
    /home/martyn/dev/ParticleGAN/.venv/bin/python run_nr.py --arms all --tasks screen --device cuda:1 --jobs 6
    tail -f logs/*.log              # one line per task
    tail -f logs/<arm>/<task>.log   # one line per eval
    /home/martyn/dev/ParticleGAN/.venv/bin/python summarize_nr.py [--write]   # leaderboard incl. read-only refs

The screen has 7 tasks: native grid100/rotated100/staggered100, toy-img_bars4, toy-two_pole, ring8-shift and
ring8-multishift. The reference results for gs2_c03_lr2_d05, k3p_simple and k3p_stock are read from the
k3p-constant-fix worktree and are not rerun.

## Receipts

Every task directory gets an `nr_receipt.json`, also merged into `result.json` as `nr_receipt`. It records:
- `r1_weight_applied` (always 0), and proof the stock K3P penalty never ran (`stock_k3p_penalty_calls`)
- the active spike term, with per-term mean and fraction of calls with a nonzero term
- the loss form and d_loss call counts by form
- each critic optimizer's class, update rule, amsgrad flags, betas, LR, state keys and guard buffer
- the anchor's started flag and step, and whether prox was nonzero
- declared and applied noise stds, and declared and applied LR ranges

Unit tests: `python -m pytest -q reports/k3p-no-r1/test_nr_terms.py tests/test_oadam_amsgrad.py`.

## Smoke (300 steps, `runs_smoke/`)

nr_pathcap_oadam, nr_symcap_anchor and nr_pairsec_hinge ran on native-grid100, toy-two_pole (custom-loop, CPU),
toy-img_bars4 (standard trainer) and ring8-shift. The receipts confirm, on every route:
- R1 is 0 and the stock penalty made 0 calls.
- The OAdam critic took one step per critic update, with `prev_step` and `max_exp_avg_sq` in its state.
- The anchor started at step 0 (step 1 on the custom-loop host, after the critic is identified), and prox was
  nonzero on more than 97% of calls.
- The hinge replaced every critic d_loss call.
- Noise was 0 and the LR was constant.

The transfer toys ignore `--steps`, so those smoke cells are full-budget results.

Caveat: on toy-img_bars4 every cap term (symcap, pathcap, pairsec and the fake cap) was 0 on every call, because
image-critic RMS slopes stay below 1. Without R1, any arm whose only active terms are caps (every spike except
dvalcap, with settle none, hinge or oadam) trains img_bars4 with no gradient penalty at all.
