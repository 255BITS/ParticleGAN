# First frozen native grid100 PASS: state-driven prior-rate handoff

This candidate **PASSES** the unchanged native grid100 gate at 7,000 updates: all five required final live noisy checks and the independent 100,000-sample holdout pass. It has 100/100 modes, 21/34 passing full-accuracy observations, first full-accuracy arrival at update 2,000, and zero training-stream deviations. Unchanged rotated100 and staggered100 checks were subsequently run; both fail, so the candidate is **1/3 native** and is not an all-100g solution. The full 22-task board has not been scored for this package.

The candidate starts from the paired birth/death graft and changes only the applied prior stationarity scale. Let `s_p` be the prior tester scale and `s_G` the slowest non-sigma generator tester scale:

```text
if s_G > 1/64:  s_prior_applied = min(s_p, sqrt(s_p * s_G))
else:           s_prior_applied = min(s_p, s_G)
```

The first branch lets the prior table follow an adapting generator. The second bounds prior motion by G once G's **existing** stationarity test reaches its settled 1/64 scale. The prior's own tester is always an upper bound. No task name, frozen score, update horizon or external schedule enters the rule. Optimizer moments and tester windows are preserved at the switch. The recipe, paired BD implementation, QR `batch_feature_zero` initialization, learnable output noise initialized at .02, and noisy sampling-law evaluation are unchanged from the previous native leads. Output sigma stayed .02 and its LR zero throughout this run.

At update 2,088 the G tester cut its scale from 1/32 to 1/64. On update 2,089 the applied G LR changed `.0001328125 → .00006640625` and prior LR changed `.00150260191 → .0001328125` (11.31× smaller); there was no reset or later recross. `handoff-transition.json` and `grid100-rates.jsonl` record the exact applied rates.

| Frozen measure | Final live | Independent holdout | Requirement |
|---|---:|---:|---:|
| Modes | 100 | — | 100 |
| Noisy precision | .97990 | .98030 | ≥ .97000 |
| Centre RMS / data σ | .14005 | .11469 | ≤ .20 |
| Covariance eig ratios | .53741–1.35383 | — | .40–1.70 |
| Mass TV | .03325 | .02009 | ≤ .06 |
| Trace bias abs | .02244 | .02301 | ≤ .10 |
| Radial KS | .02850 | .02472 | ≤ .04 |
| Full verdict | **PASS** | **PASS** | last five + holdout |

The terminal observations at updates 6,000/6,250/6,500/6,750/7,000 all pass the full accuracy and coverage rule. The runner's `21/34` field also equals its full-accuracy count for this run; unlike earlier near misses, there is no ambiguity between the two counters here.

`training.patch` applies directly to the committed paired-BD graft training source and reconstructs the executed SHA256 in `source-sha256.json`; `handoff-only.patch` shows the single additional branch relative to the committed geometric-coupling experiment. The exact overrides, fixture, per-check metrics, applied-rate trace, noisy verdict, smoke receipt and native runner result are archived here. The local saved final state is at `/ml2/hypergan/gan-attempts/combined-h1-h2-20260928/runs/h2-prior-handoff/grid100/final-state.pt` for read-only follow-up.

## Unchanged native transfer checks

Rotated100 and staggered100 used the same package SHA256 `3e37d7690567fba3406851a0ad7fa1d73d533fa0a9a21c8b1d7736103ad14334`, byte-identical overrides, QR initialization, noisy evaluation and 7,000-update frozen budgets. Both cover 100 modes but fail every required terminal accuracy check and their independent 100k holdouts. All three runs have zero stream deviations.

| Native task | Final precision | Centre / data σ | Eig ratios | Trace bias abs | Radial KS | Full accuracy / 34 | Holdout | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| grid100 | .97990 | .14005 | .537–1.354 | .02244 | .02850 | 21 | PASS | **PASS** |
| rotated100 | .94155 | .24464 | .354–1.686 | .26015 | .11513 | 0 | FAIL | **FAIL** |
| staggered100 | .98650 | .17282 | .389–1.510 | .10639 | .04030 | 1 | FAIL | **FAIL** |

Staggered100 narrowly misses the minimum eigenvalue (required ≥.40), absolute trace bias (≤.10) and radial KS (≤.04) at the final observation; its holdout also misses trace and KS. Rotated100 has a larger precision, centre and shape deficit. The exact per-task result, fixture, metric trace and noisy verdict are archived here. No source, recipe or threshold was changed between native tasks.

Two read-only controls narrow the next work. Rotating the **saved passing grid100 samples** by the frozen rotated100 angle preserves all final and holdout pass verdicts; the tiny float32 precision differences are documented in `rotation-evaluator-control.*`. This finds no structural rotation defect in the evaluator. Rescoring the saved grid100 and staggered100 final-five clouds at output sigma `.022` makes **both** pass all five checks and their holdouts, including live and EMA, while `.021` still misses staggered's first terminal minimum-eigenvalue check. These are **post-hoc diagnostics chosen after the frozen results**, not an online candidate or leaderboard pass; see `posthoc-sigma-rescore.*`. Rotated100 remains a separate substantial failure.

A further held-out, data-only local Gaussian MMD probe at trained sigma `.020` favors widening on both saved generators: `d(MMD²)/d(logσ)` is `−1.44e−4` for grid100 and `−3.29e−4` for staggered100 across 12 independent batches each. At `.022`, grid100's derivative is statistically unresolved around zero while staggered100's remains negative, so unconstrained MMD might widen past the diagnostic pass band. See `posthoc-handoff_mmd_sigma.*`; this also made no online model update.
