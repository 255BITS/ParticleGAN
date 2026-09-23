# Isolated output-RNG transfer replay

The fresh, fixed-seed 19-case transfer replay of `configs/toy100/shared_candidate_isolated_rng.json` **failed 11/19**. This config uses the shared affine/H1600/network-floor .01 core with `output_noise_rng="isolated"`. No 100-mode run was made for this candidate.

- Source: clean `1c1a0865fe605c9f212d832c06596c12510f937e`; config SHA-256 `a966d3f3c955b7b19bd4f99705dfebbe9ea862dc3b3a598828364efd49c431e8`.
- Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ionice -c2 -n4 /tmp/pr38-default-env/bin/python -u -m benchmarks.transfer_suite.toy100_compatibility --config configs/toy100/shared_candidate_isolated_rng.json --output /dev/shm/toy100-isolated-rng-full19-1c1a086 --all`.
- Retained evidence: `artifacts/toy100-accuracy/isolated-rng/full19-1c1a086/` (protocol, source archive, config, all 19 compressed episodes, index, summary, and independent regrade).
- Independent `benchmarks.toy_suite._episode_rows(..., candidate=True)` found `FAIL 11/19` with no integrity error. All 19 cases report applied shared noise and preserved output-evaluation training RNG state.

| Failed case | Frozen-gate reason at final check | Passing terminal checks |
| --- | --- | ---: |
| trajectory | identity MSE .25646 > .02 | 0 |
| residual_student | identity MSE .04993 > .02; success .5 < 1; wrong-pad .5 > 0 | 0 |
| mode_hold | 3 modes < 8; HQ .47876 < .9 | 0 |
| vector_unequal_mass | minimum component eigenvalue ratio .04035 < .15 | 0 |
| vector_unequal_width | Final metrics pass, but only 3 consecutive passing checks; 5 required | 3 |
| vector_overlap | Final metrics pass, but only 3 consecutive passing checks; 5 required | 3 |
| img_bars4 | HQ .84375 < .9 | 0 |
| img_blobs4 | Final metrics pass, but only 2 consecutive passing checks; 5 required | 2 |

The other 11 cases passed their saved and independently recomputed gates. This is a quality failure, not missing application of the isolated-noise mechanism or an invalid evidence archive.

The separate [shared-base replay](isolated-rng-shared-base19.md) used the same isolated RNG mechanism with κ=1.25 and the base effective network floor .05. It independently regraded as valid **FAIL 14/19**, missing residual_student, mode_hold, vector_unequal_mass, vector_overlap, and img_bars4. Because κ and the network floor differ from this 11/19 run, the score difference does not isolate an RNG effect. Both archives are retained and neither qualifies for the combined 22-task gate.
