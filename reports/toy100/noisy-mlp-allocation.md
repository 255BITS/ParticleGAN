# Noisy MLP allocation screen

Four one-field followups used the retained noisy-MLP probe on `grid100`.
Every run used seed 1234, 7,000 updates, 20,000 generated evaluation draws,
the frozen live gate, and the same public `GANTrainer`. The base architecture
is a three-layer width-128 MLP generator and Fourier-2 discriminator with a
20,000-row normal-initialized particle prior and fixed output noise of standard
deviation 0.026. Its recipe uses LR 0.0006, D multiplier 1.5, prior multiplier
10, Adam β=(0, 0.999), cap coefficient and κ of 1, batch 512, prior regularization
0.05, and cosine annealing from 40% of the budget to a 5% floor. The declared
plan is [noisy_mlp_allocation.json](../../configs/toy100/noisy_mlp_allocation.json).
The output noise is independent of target centers and mode assignments.

| Arm | One changed field | Final live modes | HQ | Mode TV | Min HQ count | Covariance eigenratio | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Retained base | — | 95 | 0.99335 | 0.11395 | 69 | 0.588–1.265 | FAIL |
| prior_reg1 | Prior regularization 1.0 | **100** | 0.98720 | **0.08955** | **103** | 0.524–1.185 | FAIL: 4/5 terminal checks |
| particles40000 | 40,000 particles | 99 | 0.99120 | 0.11025 | 90 | 0.536–1.112 | FAIL |
| batch1024 | Batch size 1,024 | 99 | 0.98825 | 0.09345 | 76 | 0.518–1.643 | FAIL |
| latent16 | Latent dimension 16 | 99 | 0.99235 | 0.10625 | 97 | 0.584–1.119 | FAIL |

`prior_reg1` reached full live coverage at step 5,750. All final live metrics
passed, including mode balance and every per-mode covariance and radial-width
bound. Its first complete quality pass came at step 6,250; steps 6,250,
6,500, 6,750, and 7,000 passed consecutively. The gate requires five terminal
passes, so its recorded verdict is correctly **FAIL**. The EMA copy did sustain
five passes, but EMA does not certify the live gate. The other three changes
kept spread healthy but left one or more modes below the 100-draw mass floor.

The strongest next controlled tests are `prior_reg1` with batch 1,024, or a
slightly longer declared budget at the same recipe to check whether live
quality remains stable for a fifth observation. An earlier 0.25 anneal was
tested separately and hurt allocation (96 modes, TV 0.10885); these results
do not justify changing the gate or scoring frequency. A candidate must pass
grid100 before promotion to the rotated and staggered problems.

Each arm retains its declared config, complete event curve, gate verdict,
source copy/hash, snapshots, and exact final 20,000 live/EMA samples under
`artifacts/toy100/noisy-allocation/`. I independently re-scored every saved
final live sample array: integer and Boolean metrics matched exactly, and
floating metrics matched to `1e-12` despite CPU/GPU reduction order. The
scratch probe is
[noisy_mlp_allocation_probe.py](search/sources/1749806daeec8f114912bb6b4aabf14751f56fe0cc6e838a05874bbb593c5959.py);
production `benchmarks/toy100/train.py` was not modified during these runs.
This is a fixed-seed configuration comparison, not a robustness estimate.
