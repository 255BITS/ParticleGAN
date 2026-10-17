# E4 handoff

- Frozen package: [source/particlegan](source/particlegan/), digest
  `f69349eeda9679c6db04711be9eeebe9b42ed994bc544dd6df585be61e0f2188`.
  Config: [source/overrides.json](source/overrides.json). Do not edit this
  package in place; copy it to test a new mechanism.
- Native frozen host: `/ml2/hypergan/lrfree-20260926/harness/screen.py`, digest
  `ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c`.
  [Receipts](native-receipts/) show live 3/3 at 7k and 14k, noisy scoring,
  zero stream deviations. Earlier prior-EMA 3/3 used an altered callback host
  and is not comparable.
- E4's added control uses the table's own gradients and stationarity evidence.
  The inherited birth/death reaction still compares model output with real
  data to move mass; no claim of an output-free *whole trainer* is warranted.
- The measured 3/3 remains **diagnostic** under
  `/ml2/hypergan/lrfree-20260926/REQUIREMENTS.md`: `W=50` was examined on a
  rotated100 probe before calibration, `Q=.05` doubles as an uncalibrated hold
  threshold, the p-value null is imperfect, and LR ×1.33 fails 0/3. The
  inherited base ledger is unresolved. Removing the hold leaves a grid miss
  at 7k and a rotated miss at 14k; it is not a compliant drop-in fix.
- Astra's recommended next research step: independently specify and calibrate
  a replacement row-gradient estimator on nonbenchmark synthetic processes
  *before* any native run, keeping only the hot-row action. Address the
  amplitude-blind annealing and anchor evidence in separate changes. Do not
  tune `W` to reproduce the archived native results. An initial disjoint-block
  multiple-test proposal was withdrawn after the
  [oracle feasibility check](oracle-feasibility.md) showed inadequate power
  under strong gradient correlation. A continuous Gaussian scale-mixture gain
  [failed](eb-preflight/RESULTS.md) its prospective synthetic test 119/150;
  a distinct signed-location mixture [passed](location-preflight/RESULTS.md)
  150/150, but only under oracle assumptions. Neither fixes A2 or has a
  native result.
- The user now permits table-position comparisons within the new controller.
  Astra proposed a local-isolation mobility gain. The prospectively specified
  [synthetic preflight](geometry-preflight/RESULTS.md) rejected every
  neighborhood size `{5,10,20,40}` before validation or native use: the 8D
  heavy-tail legitimate mean gain was `.117`–`.122` against `.07`, and larger
  neighborhoods gave the legitimate rare group `.52`–`.75` gain. Do not port
  this LOF mechanism unchanged or select a neighborhood from native results.
- The 21 remaining quick/ring/custom tasks were submitted as
  `e4-noout-fullsuite` to `/ml2/hypergan/lrfree-20260926` with package hash
  `f69349eeda96…` and noisy evaluation; [suite-leaderboard.md](suite-leaderboard.md)
  records the completed statuses. The original 7k native receipts supply the
  other three tasks.
