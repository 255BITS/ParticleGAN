The fixed public-API convergence test passed: using a clean differentiable G
forward while retaining native DV12 for D improved the terminal learned game
under all four common critics after 512 updates. Both arms keep particles,
identical fresh H/b=0 initialization with sampled C, native optimizers and row
controls. No output-error objective, guard or checkpoint selection is used.

Run from a checkout with the package and PyTorch installed; no external model,
data, local Supra files or protocol edits are needed. Use a fresh output path:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m examples.e22_routed_g_clean --run --out runs/routed-g-clean-fresh
```

The command has a fixed 300-second CPU allowance and returns exit 0 for the
predeclared PASS criterion, exit 1 for a complete scientific failure, and an
exception for incomplete execution. `--preflight` checks shape, initialization
and ownership without updates. The software prerequisites are available with
`python -m pytest -q tests/test_e22_routed_g_clean.py`.

| Common critic | Native clean game | Clean-G clean game | Difference |
|---|---:|---:|---:|
| Native @128 | 3.567862 | 3.567558 | −0.000304 |
| Native @512 | 6.394525 | 6.394137 | −0.000388 |
| Clean-G @128 | 3.567921 | 3.567616 | −0.000304 |
| Clean-G @512 | 6.394352 | 6.393963 | −0.000389 |

The largest difference is −0.000303984, clearing the fixed −0.0001 threshold.
All reference paths calibrated. Removing the learned code worsened the clean-G
terminal game by at least 0.090900; bank and query gradients were nonzero on
511 of 512 updates in both arms. All final learned tensors and optimizer moments
were finite. The complete run took 27.551 seconds including the outer launcher.

Both arms improved monotonically on every fixed clean and noisy curve. Clean-G
led on clean serving at all nonzero checkpoints except 192. The terminal noisy
cohort also improved under all four critics, by 0.000484 to 0.000871. These are
small fixed-panel gains, without a sustained dominance or universal DV12 claim.

This two-site synthetic conditioning-vector-switch test compares two particle
arms. Ordinary LoRA and full Supra are unmeasured. A separately bounded real
caption comparison is required before claiming those targets are beaten.
The [fixed protocol](e22_routed_g_clean_v1.json) and
[compact results and provenance](e22_routed_g_clean_results.json) preserve all
four scores, curve differences, calibration paths and artifact hashes. Raw
states, tensors and update logs remain under the ignored run directory.

The [separate actual caption verification](e22_routed_g_clean_caption_results.md)
completed with scientific FAIL: at 6,400 updates clean-G beat neutral particles
under all four critics, but ordinary native-game LoRA still led under all four.
At 5,120 the comparison with neutral was mixed. This preserves the toy PASS
while rejecting transfer to the fixed actual-task gate; no full-Supra promotion
follows. The executed toy sources, criteria and original receipt are unchanged.
