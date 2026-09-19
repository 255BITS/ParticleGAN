# Does overlap explain the plateau?

Reducing Gaussian overlap at inference does not rescue either checkpoint. All six evaluations are certified. Historical FIDs reproduced within 0.00002 and both baseline density/coverage values reproduced exactly.

| Checkpoint | Original FID / coverage | Sigma x0.75 | Sigma x0.5 |
|---|---|---|---|
| 80k | 15.7527 / 64.26% | 15.9090 / 64.03% | 16.2898 / 63.19% |
| 160k | 19.0584 / 58.03% | 19.7838 / 56.97% | 20.9044 / 54.62% |

At 160k, nearest-center confusion falls from 2.67% to 0.29% to zero observed errors, while both FID and coverage worsen. Zero means zero among 32,768 Monte Carlo samples, not mathematically disjoint Gaussian support. Smaller sigma is not a beneficial inference-only fix in this range. This weakens a direct sampling-overlap explanation; it does not rule out damage caused by overlap during training.

All observed confusion errors at both original checkpoints occur within original clone families. Nearest-neighbor spacing decreases, but family-wide center variance increases. The distribution develops tighter local clumps alongside a farther upper tail. The global variance/covariance regularizer does not explicitly enforce local pair separation. See center_variance.json and noise_sweep.png.

Recommendation: prioritize the freeze-centers fork to test whether continued center movement causes deterioration. Also train with sigma x0.75 from the same 80k checkpoint to test prevention during adaptation. It has a smaller initial sampling penalty than x0.5. Neither arm alone will prove an overlap mechanism: freezing constrains all center motion, and sigma changes both training and sampling distributions. Compare to the existing 90k/100k unchanged control, use endpoint coverage, and inspect geometry over time. Do not add repulsion or change architectures before those results.

Both 80k-to-100k forks are queued sequentially on GPU 1 after actual checkpoint verification. Source trainers and historical certificates are preserved. No seed-only repeats and no automatic extension beyond 100k.
