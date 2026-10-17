The original rotated100 moving gate failed its second turn: HQ 0.8557 and 99 modes at update 1500, against the unchanged HQ bar 0.86481 and mode bar 95. Two R1 fires occurred; one of two turn periods passed.

| Update | Target turn | Served weights | Original HQ | Fast expected HQ | Average expected HQ | Fast per-mode mean-offset RMS |
|---:|---:|---|---:|---:|---:|---:|
| 500 | 0 degrees | averaged | 0.9609 | 0.949691 | 0.961475 | 0.008457 |
| 1000 | 30 degrees | averaged | 0.9548 | 0.934141 | 0.955826 | 0.015137 |
| 1500 | 60 degrees | fast | 0.8557 | 0.857826 | 0.879270 | 0.025211 |

Expected HQ is the deterministic uniform-table expectation after isotropic output noise sigma 0.029, before bounded latent perturbation. It is not a replacement for the original 20,000-point CUDA observation. All per-mode counts, translations and covariances are in receipt.json.

The final fast and average point clouds have the same absent raw target mode, 99. The second turn broadens local mode clouds: fast within-mode covariance trace increases from 0.001006 to 0.001891; its per-mode mean-offset RMS rises from 0.015137 to 0.025211. Raw mode 76 has a centroid offset 0.074857 and expected noisy HQ 0.559537. Raw mode 88 has expected HQ 0.575250 and unusually broad covariance. These errors occur in different directions across modes.

The generator's polar angle is 2.968, 6.932, and 6.272 degrees across the three checkpoints. Final best global rigid correction is only -0.01182 degrees and improves expected HQ from 0.857826 to 0.857941. This does not support a global rotation-lag explanation or establish that increasing the quarter-calibrated generator rate would fix the failure. Most adaptation is in table distribution and birth/death replacements; row identity across checkpoints cannot certify physical particle motion.

The average improves final expected HQ by 0.02144. Using live G with the averaged prior gives 0.876515; using averaged G with the live prior gives 0.860993. Thus most of this improvement comes from the table average. The actual support guard correctly rejects serving the whole average: coherent rows are 18,415, below the required 19,000. No guard or quality threshold should be relaxed. Deterministic bounds that cover every bounded latent perturbation give final fast expected HQ [0.843152, 0.872108] and average expected HQ [0.865540, 0.892368]. Even the average lower expectation bound exceeds the bar, but no particular finite evaluation draw is certified by that expectation.

The mean-transport counters stop changing during the entire second turn: witness fires remain 43 and mean moves remain 25,245 from update 1000 to 1500. Final metadata says invalid / missing_EMA_group, while the ephemeral critic chart has 95 groups. output_moments.py::freeze_moment rejects the entire witness when any EMA group has zero rows. mean_transport.py::run_mean_phase also rejects all correction if any current EMA group is empty. One unsupported group can therefore block correction in every otherwise occupied group.

The CPU isolation test passes the actual saved real FIFO and actual affine EMA outputs through the frozen output-moment functions with an evaluator-only raw nearest-mode adapter. The original function returns missing_EMA_group. A fixed subset of 99 occupied raw groups, with missing-group direction and weight zero, no weight renormalization, all 10,000 odd observations and unchanged EB cutoff, produces lower bound 0.519596 and a firing witness. The missing mode contributes 112 zero-direction odd observations. This demonstrates that the all-or-nothing veto can discard strong partial moment evidence. It does not identify raw mode 99 with the missing original critic group; the original ephemeral chart was not checkpointed.

The recommended source-backed design is in PROPOSED-PATCH.md. It changes the partial-witness branch only after an existing typed optimizer-surprise detector has fired. Runtime capture of the actual critic chart and independent packet/bound review remain necessary before treating this as the validated cause or a successful repair.

All checkpoint bytes and the 104 guarded source inputs remain unchanged. There was no CUDA initialization, model construction, RNG draw, training update, or process signal. The reproduction finished in seconds and the original FAIL verdict remains retained.
