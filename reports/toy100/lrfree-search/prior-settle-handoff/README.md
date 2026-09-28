# First frozen native grid100 PASS: state-driven prior-rate handoff

This candidate **PASSES** the unchanged native grid100 gate at 7,000 updates: all five required final live noisy checks and the independent 100,000-sample holdout pass. It has 100/100 modes, 21/34 passing full-accuracy observations, first full-accuracy arrival at update 2,000, and zero training-stream deviations. This is a grid100 result only; rotated100, staggered100 and the full 22-task board have not yet been scored for this package.

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
