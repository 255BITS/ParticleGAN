Freezing the initialized prior on the exact energy-U parent preserves **5/6 ordinary Tier 1 passes plus its own clock**, but does not repair Ring16. Endpoint covariance changes only 6.371506 → 6.259248 against the .85 bound, while HQ falls .747559 → .552246 against .85. Reject this single fixed-task prior-freeze repair. The result does not identify an optimal energy strength or authorize another grid.

The [publication](publication/results.json) contains seven exact cached parent cases and seven new frozen-prior cases, with 14 actual-training GIFs. The [collection audit](collection-audit.json) passes all grades, initial-tensor/ordered-batch receipts, named-stream checkpoints, actual optimizer ownership, positive paired R1/R2 .1 every update, and separate finite signed energy state. All 231 original charges remain intact; seven new runs cost **197.33389144105604 seconds**, giving **238 attempts / 7168.248061007995 seconds** under the unchanged 8000-second cap. Full original budgets and gates were retained.

The independent [density audit](density-audit.json) recomputes all six original gate metrics on **192 saved panels**, 96 per role, with **exactly zero difference** from recorded values. Both roles have zero covariance passes, zero full passes and terminal suffix 0/5. Covariance and HQ fail every saved check. The last five checks all retain 16 modes, 4096 samples, mass TV ≤ .15 and minimum eigen ratio ≥ .15; covariance and HQ fail all five. Earlier mode failures occur on 34 parent panels and 67 frozen panels.

| Ring16 metric / saved-panel result | Exact cached energy-U parent | Frozen prior |
| --- | ---: | ---: |
| Component covariance error, gate ≤ .85 | 6.371506 | 6.259248 |
| Minimum eigenvalue ratio, gate ≥ .15 | 1.027582 | .748672 |
| HQ, gate ≥ .85 | .747559 | .552246 |
| Modes, gate ≥ 16 | 16 | 16 |
| Mass TV, gate ≤ .15 | .030518 | .037598 |
| Samples, gate ≥ 4096 | 4096 | 4096 |
| Best saved covariance error | 5.332378 at 1500 | 5.600076 at 1484 |
| Covariance / HQ failures out of 96 | 96 / 96 | 96 / 96 |
| Full passing panels / terminal suffix | 0 / 0 | 0 / 0 |
| Mean radial variance / target variance, diagnostic | 8.026238 | 3.537974 |
| Mean tangential variance / target variance, diagnostic | 4.393926 | 9.201999 |
| Mean mapped-center trace / target variance, diagnostic | 7.927132 | 9.182553 |
| Mean first-order local-noise trace / target variance, diagnostic | 4.429144 | 15.000987 |

The frozen run reduces radial expansion but increases tangential stretch. Its component radial ratios span .779999–7.853580 and tangential ratios 6.640082–11.893056. Generated row centers outside three/four target sigmas increase 8/7 → 106/64. These are **mapped output centers**, not moved latent codes: the actual frozen Ring prior has zero optimizer groups/updates and matches its captured initial state hash `8dcde881cc9941a1520dd63f134a554afc6ca03ea8be8e931406c380b7afca36`, while G/D complete 1600 updates. A fixed latent table therefore does not prevent the observed broad output law. Saved Jacobian geometry is a first-order proxy, not an exact additive covariance partition, historical force reconstruction or unique causal explanation. Core covariance (1.156282 → 2.189326) and normalized sliced-W1 (.038003 → .043007) are diagnostics, not replacement gates.

Acquisition and endpoint behavior remain separate. Both Gaussian recipes pass the original confirmed-acquisition gate: parent first confirms at 209 with 13/24 passing observations; frozen first confirms at 334 with 11/24. **Both Gaussian endpoints fail** the CDF-KS ≤ .05 bound: primary/confirmation values are .068194/.089955 and .051144/.064021. Both words finish all 20001 updates and pass acquisition **and endpoint**: parent first confirms at 2501, 22/24 paired checks; frozen at 834, 24/24. Pole, token and AE pass their original sustained/terminal gates, and each own-clock proof passes four parities. No Tier 2 retention result follows from these Tier 1 results.

The sole global delta is `prior_update: learned → frozen`; removing the false `learned_locations` candidate requirement changes metadata only. Actual latent tables remain initialized identically, become buffers and receive zero prior updates; G/E/D retain their full budgets. Pole coordinates and token embeddings remain trainable under their original direct-generator ownership. Scientific source remains `2b04f2853c511c9bcd417ef41e8fd773e97e8c67c9e057da8abdd4fef2de5f12` (1216 files). New execution is commit `3203c4e419cb7c2a4cd2021d2c48638980579901`; cached parent evidence retains origin `67920d9090727bd74adbd331187f0e6f22533e48` and its original attempts/certificates. See [PLAN](PLAN.md), [registration](registration.json), [saved prerequisite](../r1r2-energy-distance-repair/research/README.md) and [separate software fixtures](software-validation.json). No historical initialization, score pairs or G/D forces were reconstructed.

Both auditors use stored evidence only: zero new updates, draws, generative-model construction or live evaluation, with unchanged Python/NumPy/Torch CPU RNG. Density JSON has no top-level PASS field; the original assertions passed and its file equals successful stdout byte-for-byte. Both stderr logs are empty. Reproduce from the worktree with the declared venv and `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`:

```sh
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-energy-prior-freeze-repair/audit_collection.py --execution-commit 3203c4e419cb7c2a4cd2021d2c48638980579901 --write
/home/martyn/dev/ParticleGAN/.venv/bin/python reports/forge/r1r2-energy-prior-freeze-repair/density_audit.py --execution-commit 3203c4e419cb7c2a4cd2021d2c48638980579901
```

Raw stdout/stderr stay under `/mnt/ml7tb/ParticleGAN-forge/r1r2-energy-prior-freeze-repair-20261012/`.

| Receipt | SHA256 |
| --- | --- |
| `publication/results.json` | `013696f61094eb8c06e9d9c41a9422270315678992f9a1a9680430c56bf7f207` |
| `collection-audit.json` | `3afa7bdd6f96eee72efbbd783c100c2a9639dbf563466cef3db6f8600b28afaa` |
| `density-audit.json` / `density-audit.stdout.log` | `e790df5eb9cbbaa2ebacb66b18af449a43692e894ff937c8c2890912f4048644` |
