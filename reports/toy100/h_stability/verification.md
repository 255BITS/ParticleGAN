# Independent selected-H verification (in progress)

Selected candidate: `h_n05r06_mixup_c0p01_lr15`.

**Fresh independent cold replay: 10/10 PASS.** All ten final live metrics reproduce the archived run exactly. Ring: 8 modes, HQ 0.999267578125, passing suffix 5; trajectory MSE 0.002905043074861169. Replay elapsed 77.22 s (one CPU/AVX2 worker; timings are machine-load dependent). Saved-episode/source/config regrade reproduces every verdict; saved ring/trajectory samples and optimizer checkpoint budgets/rates also verify.

All 125 archived source files match the selected cold manifest. Archive SHA256: `ed4a61dcfebc13f1134630db096aad043d60e4846083b4a1384e871ee9ebb334`. The selected declaration matches the original manifest and best-candidate.json. Neither the original attempt nor the PR worktree was edited.

The remaining nine older toys and three native 100-mode problems are **not yet qualified**. A first unipolar extension errored before its first update because the archived mixup helper expects `discriminator.modules()` and that host supplies a conditional lambda. A separately archived plumbing shim will expose the callable's existing critic for paired-noise replay. The auxiliary AE/unused-token hosts also require explicit removal of inherited reconstruction/hold losses to comply with the exclusively adversarial generator requirement. None of these errors or unrun cases is a GAN PASS.

Evidence (relative to this directory):

- `cold-replay/h_n05r06_mixup_c0p01_lr15/status.json`: ten fresh measured passes; per-gate raw episodes, checkpoint tensors, optimizer/noise receipts and source archives are below that directory.
- `regrade-cold.json`: independent saved-episode regrade plus ring/trajectory sample and optimizer verification.
- `archive/`: exact original selected source archive, manifest, declaration and reported metrics.
- `isolation.json`: all 125 archived source comparisons and source/declaration identity.
- `frozen-reference-receipt.json`: exact frozen leading_profile.json dependency copied into the isolated checkout.
- `run-cold.sh`, `run-remaining.sh`: exact commands; `tests.jsonl`: append-only raw events.
- `cold/`: initial setup ERROR, caused by the supplied replay script omitting leading_profile.json; no training occurred. Fixed only by copying the exact frozen dependency.
- `remaining/`: initial unipolar callable-interface ERROR; no optimizer update completed.

No seed sweep, candidate repair, threshold change, host architecture change, or learning-rate schedule was introduced. Actual G/D rates are 0.0015 and particle rate 0.003 throughout; discriminator input sigma is fixed at 0.05. Original own-acquired 1,200-update continuation remains a reported FAIL (5 modes/HQ 0.209228516 final); it is separate from the 22 cold toy cases and has not been retrained by this verification yet.
