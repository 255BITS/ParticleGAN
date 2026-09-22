# Host replay of the published rare-mode winner

This checks the Codex linear-skip result. It does not add a formulation or change the leaderboard.

The published archive still scores **9/9 required + 10/10 practical**. `python -m reports.transfer_suite.formulations.build` reports `(rp_logistic_bcap3, 9, 10)`. `python -m reports.transfer_suite.rare_focus.verify` reports one sustained rare-mode winner, two exact winning replays, and unchanged thresholds. That winner is `linear_skip_d96_beta5`: D96×2, Fourier 2, Softplus(beta 5), plus a zero-initialized raw-coordinate linear skip. On the original host (torch 2.13.0+cu126, AVX2, Python 3.12.13) the live curve passes the final six checks and confirms at step 1,150 (minimum normalized spread 0.270, covariance error 0.441, HQ 99.71%).

## This machine

Command, seed 0, original thresholds and budget:

```bash
ATEN_CPU_CAPABILITY=avx2 python3 -u -m benchmarks.transfer_suite.run_linear_skip \
  --tasks vector_unequal_mass --output /tmp/linear-skip-avx2
```

Three runs agreed on every live observation except timing: torch 2.13.0+cpu, torch 2.13.0+cu126, and cu126 with `ATEN_CPU_CAPABILITY=avx2`. All three **fail**. Final live covariance error is **2.624** (bound 0.85) and minimum normalized spread is **0.092** (bound 0.15). HQ 0.990, mass TV 0.054, and minimum mass ratio 0.901 still pass. The 2% component’s own covariance error is 9.49; the other three components are under 0.38. Log: [run.log](run.log). Curve: [episode.json.gz](episode.json.gz).

Step 50 still matches the published HQ and mass TV. Sliced W1 and covariance already differ there, and the gap widens. Forcing the AVX2 kernel dispatch did not recover the published curve. This host reports AVX512 and Python 3.12.3. [comparison.json](comparison.json) · [protocol.json](protocol.json).

The 19/19 score stays the archived one. This replay does not reproduce that live pass, and it does not replace it.
