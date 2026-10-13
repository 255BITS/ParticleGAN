Both fresh recipes pass five ordinary Tier 1 tasks and their own clock, but fail Ring16. Raw energy weight 1 does not repair the Ring covariance gate: it replaces radial compression with broad radial expansion and lowers HQ. This rejects the declared weight-1 contrast on this fixed suite; it does not establish an optimal energy coefficient.

The [collection audit](collection-audit.json) passes all 14 fresh grades/GIFs, 12 ordinary checkpoints, two own-clock proofs (52 saved states), all named streams, actual initialization/ordered batches, native AMS roles, positive R1/R2 gamma .1 every update, and signed energy-state ownership. There are zero cached grades. All 217 historical charge owners and amounts survive unchanged; the new 14 attempts cost 384.751732454024 seconds, giving 231 aggregate attempts and 6970.914169566939 seconds under the original 8000-second ceiling. Nine archives occupy 3614364022 bytes under the 8 GiB cap, with 20250427392 bytes free at audit time.

The [density audit](density-audit.json) recomputes all six gate metrics for 96 saved panels per new recipe. Every value matches its original recorded metric exactly. Both recipes have zero covariance-bound passes, zero full passing panels, and a zero-length terminal passing suffix; the original gate requires five consecutive terminal passes. The fresh control's Ring curve and endpoint also reproduce the historical NS control, while retaining their separate source/attempt identities.

| Ring endpoint / saved-panel result | Fresh NS control, energy 0 | Energy U, weight 1 |
| --- | ---: | ---: |
| Component covariance error, gate ≤ .85 | 1.556546 | 6.371506 |
| Minimum eigenvalue ratio, gate ≥ .15 | .094075 | 1.027582 |
| HQ, gate ≥ .85 | .888672 | .747559 |
| Modes | 16 | 16 |
| Mass TV | .049805 | .030518 |
| Samples | 4096 | 4096 |
| Best saved covariance error | 1.501881 at step 1584 | 5.332378 at step 1500 |
| Covariance failures / 96 panels | 96 | 96 |
| HQ failures / 96 panels | 73 | 96 |

Across the original final five evaluations, both roles have 16 modes, 4096 samples, and mass TV ≤ .15 at every check. The control fails covariance and minimum eigenvalue ratio at all five checks and HQ at one; energy fails covariance and HQ at all five while its minimum eigenvalue ratio passes all five. The [receipt's `last_five_observations`](density-audit.json) retains every exact value for all six metrics.

Endpoint geometry uses the saved actual samples and checkpoint tensors. Mean radial variance in target-variance units grows .297019 → 8.026238 (energy range 1.044083–34.588084); tangential variance grows 3.053394 → 4.393926. The unweighted mapped-center trace ratio grows .341619 → 7.927132, while first-order local noise trace grows 2.738661 → 4.429144. These are diagnostic summaries, not an exact additive covariance partition or a causal attribution to a unique mechanism. Energy preserves mode coverage and improves mass TV, but those metrics do not substitute for density shape or HQ.

Scientific execution is commit `67920d9090727bd74adbd331187f0e6f22533e48`, source digest `2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12` (1216 files). Administrative coordinator `21e207ef6c72d74ecc2f684fba8188851b060a15` only repairs collection topology; it supplies no scientific source identity. Both audits checked the original committed declarations/task pins and ran on stored evidence with zero optimizer updates, sampling draws, model construction, or live evaluation. Python/NumPy/Torch CPU RNG states remain unchanged.

Reproduce from the report worktree using the declared venv, `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-energy-distance-repair/audit_collection.py --execution-commit 67920d9090727bd74adbd331187f0e6f22533e48 --write
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-energy-distance-repair/density_audit.py --execution-commit 67920d9090727bd74adbd331187f0e6f22533e48
```

Both commands exited 0 with empty stderr. The density schema has no top-level status field; its original assertions all passed, and the JSON is preserved byte-for-byte from stdout. Raw logs are retained under `/mnt/ml7tb/ParticleGAN-forge/r1r2-energy-distance-repair-20261012/`.

| Receipt / archive stdout | SHA256 |
| --- | --- |
| `publication/results.json` | `cb3f7b89d1677591a12dd86377d072b55cba1b95e99060b2841cdf42c418b75c` |
| `collection-audit.json` | `10c95598b5d1acbeeb61fdf7844e3f7d12529867cd4288e883c346574030f5e8` |
| `density-audit.json` and `density-audit.stdout.log` | `e5f17a3bcd50cc8c05e2551dc9d634024b23bd3e9f84d4fc6498a3308d8ea31f` |
| `collection-audit.stdout.log` | `6823b43b350ea3372372313c93427205f24b8ed882d68db6a30113f950f81b8f` |
| Both empty stderr logs | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Frozen auditor hashes remain `c94f9d3e78e74c99a5a51c1facde1b0a998b2a56593fb2a333480a2932ecbdb2` (collection) and `6d631030496633f960ecaa3572fd34c465e0adb7e93117c0ac1c5ac3662b534b` (density). No frozen source, declaration, grade, or qualification was edited.
