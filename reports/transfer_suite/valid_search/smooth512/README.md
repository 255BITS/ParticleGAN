# Targeted 512-particle smooth-critic check

**Both declared rare-mass checks fail.** Training stopped after these two episodes; no full-six followup was performed. These results are separate from every 256-particle experiment and do not establish an all-six supported 512-particle profile.

| Critic | Sustained live | Suffix | HQ | Mass TV | Covariance error | Minimum eigenvalue ratio | Min mass ratio | Wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Softplus5 D64×2 / Fourier2 | FAIL | 0/24 | 0.948730 | 0.062432 | 6.549362 | 0.950285 | 0.793457 | 8.661 |
| Softplus5 D96×2 / Fourier2 | FAIL | 0/24 | 0.991455 | 0.078301 | 0.426719 | 0.011894 | 0.329590 | 9.697 |

The 64-wide critic fails covariance error (6.54936 > .85) while its other final bounds pass. The 96-wide critic fails minimum eigenvalue ratio (.011894 < .15) while its other final bounds pass. The latter assigns only .659% of generated mass to the 2% target component; that clears the declared rare-occupancy floor but still has poor within-component geometry. A higher particle count does not automatically fix this trained rare-mode collapse.

Every episode uses 512 particles, batch128, original G64×2/z4, Adam(0,.99), G/D/prior LRs .001/.0015/.01, Rp logistic, b_cap coefficient3/kappa1.25, prior regularization .05, no particle L2, cosine and 1:1 updates. The only changes from the original vector task are the declared support size and smooth-D architecture. There are 1,200 outer steps, 24 fixed live observations and an unchanged requirement for at least five final passing observations. EMA is retained separately. Seed0 only.

Exactly **2 GAN episodes**, **48 live observations**, **18.357107 seconds** summed recorded wall time on a shared CPU host. All failures and full curves remain retained.

Read [index.json](index.json.gz) for original/effective specs, candidate architecture, verdicts, live/EMA summaries and artifact paths. Each episode includes complete curves, actions and actual D/G update counts. The [declaration](declaration.json.gz) was saved before training and requires stopping if neither rare check passes. No other resource profile is pooled into these results.

## Reproduction and archive audit

Source commit `981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45`; the [source archive](source.tar.gz) contains all 58 exact numerical dependency files. [D64 protocol](rare64/protocol.json.gz) and [D96 protocol](rare96/protocol.json.gz) have identical source hashes. [Driver](scripts/run.py) is exact and separately hashed in each protocol. Every JSON preserves original uncompressed bytes, with both compressed and uncompressed hashes in the [inventory](inventory.json.gz).

Original worktree `/ml2/hypergan/ParticleGAN-pr36-valid-recipe`; Python `/home/mikkel/anaconda3/envs/conceptmod/bin/python`; CPU thread1. Commands: `python -u run.py --width 64 --phase rare`, followed by `python -u run.py --width 96 --phase rare`. The driver retains original absolute paths; reproduce in those paths or adapt a separate copy, never overwrite retained evidence. It replaces only the discriminator constructor for each serial episode and restores it afterward.

Run `python verify.py` to check source hashes, gzip byte roundtrips, both full episode invariants and independent numerical verdicts without training. [Verification](verification.json.gz) records the result. No new implementation commit was needed for this experiment.
