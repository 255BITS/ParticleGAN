# State-driven critic floor: frozen native grid100

One source mechanism was added to the passing prior-handoff package: before the
existing payoff damping, the applied critic tester scale is floored at the
larger of the non-sigma G tester scale and the *applied* geometrically coupled
prior scale. The frozen recipe, QR initialization, noisy scoring, host, seed,
7,000-update budget, and 100,000-sample holdout were unchanged. There is no
problem name, target geometry, accuracy metric, or step schedule in the rule.

Source: the isolated candidate's `particlegan/training.py`, SHA-256
`b3d472d155702e9d49ea9fc514a145ad5b8a13f80d50bcb16c66dc52de2c8c7b`.
Package SHA-256 `d6f8aedbee8dd191a3a057e2ff3fcd1a6e340b72a09766008e4edfd5f55d7ce8`.
Overrides SHA-256 `bccb5824e4c4e5cf1c39ff202f53fa00c7f2f7e4a8c73ba3ebb125d2ed9ba77c`.
`source-manifest.json` records exact hashes. `critic-floor-only.patch` is the
eight-line source addition relative to the passing handoff package.
`package.patch` reconstructs the full executed package from PR branch commit
`e10bb73d`; it was applied to a clean copy and every Python source file was
byte-compared with the executed package. The only changed Python file from the
passing handoff package is `training.py`.

One-step smoke: `runs/h2-handoff-critic-floor/smoke-grid100/`. Its native
fixture receipt exactly equals the prior passing run's receipt, data stream
deviations are zero, and `checkpoint-smoke.json` confirms schema-4 model and
optimizer round-trip. The one-step frozen gates report INVALID by design.
An independent active-floor smoke on a saved rotated checkpoint confirmed
G scale 1/4, applied prior scale 1/2, D tester scale 1/16 produces D LR
approximately base LR × 1/2 × the existing payoff factor. See
`critic-floor-audit.md`.

Frozen grid100 result: `runs/h2-handoff-critic-floor/grid100/`, RTX A6000 on
`pop-os`, 353.87 seconds. Verdict **PASS**, 21/34 passing observations,
first arrival step 2,000, terminal streak 21, zero data-stream deviations.
At step 7,000: noisy precision .97990, centre RMS .14004 data sigmas,
covariance eigenvalue range .53741–1.35383, absolute trace bias .02244,
radial KS .02850, output sigma .02000. Independent 100k holdout **PASS**:
precision .98030, centre RMS .11469, trace bias .02301, KS .02472.
The native fixture, every scored checkpoint's reported metrics and LRs, and
the holdout fields match the prior passing handoff run exactly. The rule is
dormant on this grid100 trajectory.

The final state is saved locally at
`/ml2/hypergan/gan-attempts/combined-h1-h2-20260928/runs/h2-handoff-critic-floor/grid100/final-state.pt`.
The `native_rate_dynamics.png` plot shows measured G/prior/D scales for the
previous three native runs and a counterfactual critic-floor curve computed
from those saved trajectories; it is **not** a new measured trajectory.
`native_quality_dynamics.png` shows the associated measured accuracy metrics.
`plot_dynamics.py` reproduces both plots. No rotated100 or staggered100
transfer was launched pending commit of this exact candidate.
