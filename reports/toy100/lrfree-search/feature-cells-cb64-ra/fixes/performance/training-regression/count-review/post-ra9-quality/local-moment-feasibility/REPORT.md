# RA9 fixed local-moment feasibility

Evidence **VALID_FIXED_SAVED_DIAGNOSTIC**; the original RA9 full-grid result remains **FAIL**. All five original terminal accuracy checks fail center RMS (.22250,.21507,.21269,.21111,.21193sigma versus .20); final other original conditions and the full100k holdout pass. This diagnostic produces no repaired output, score, action or quality qualification.

## A persistent anchor mean discrepancy is measurable

The one current-D CPU refit has128 cells/rank8 and100 real-only topology groups. Even and odd references each have100 populated groups; downstream oracle annotation finds no multimode groups and purity1.0 in both halves. Per-group even counts range79–122, odd78–123. This removes the observed coarse-group aliasing in this final snapshot; it does not certify every future fitted chart or generic dataset.

| Fixed comparison | Weighted mean RMS / original target sigma |
|---|---:|
|FAST anchors minus even / odd FIFO|.30210 / .30674|
|EMA anchors minus even / odd FIFO|.24479 / .25942|
|Even minus odd FIFO|.19181|
|EMA anchors minus pooled FIFO|.23327|
|EMA anchors minus existing100k target-real group means|.20397|
|Served clean100k cloud minus existing100k target-real group means|.20488|
|Served clean100k cloud minus EMA anchors|.01883|

All groups contribute; omitted even-reference mass is0. EMA's fixed cross-split directional aggregate is positive: A=4.06945e-5 output units squared, or .0458545 standardized by even within-group variance per coordinate. Local dot products are positive in81/100 groups, representing80.71% of even-reference mass;19 groups disagree. FAST gives91/100 positive groups and standardized A=.0746239.

The shared FIFO's actual split RMS .19181sigma is close to the optimistic iid split-noise RMS .20113sigma. Consequently the aggregate signal does not make every local target precise. D, FAST/EMA and the partition depend on training with this FIFO. These are reproducibility descriptions, not calibrated p-values, equivalence,95% local correctness or transport authority. Existing target-real arrays strengthen the fixed saved-state comparison without new draws, but cannot establish a future adaptive correction law.

## Noise, group selection and covariance budget

Neither the original20k nor100k paired clean/noisy clouds changes its current real-only topology group; downstream oracle mode switches are also0. The saved live/EMA arrays are identical under the positive paired-average lease and are one served cloud. Therefore clean-assigned and noisy-assigned group identities agree in this snapshot, with covariance closure below4.6e-18.

On all rows, weighted covariance per coordinate in original target-sigma squared units is:

| Cloud | Clean | Saved output noise | Symmetric cross term | Noisy |
|---|---:|---:|---:|---:|
|Terminal20k|.11979|.93303|.00138|1.05420|
|Holdout100k|.12122|.92881|.00081|1.05084|

Conditional noise-mean RMS is .09996sigma at20k versus .03813sigma at100k. Such sampling variation matters near the .20 center limit. The clean holdout closely follows the EMA anchor means; output noise does not remove that anchor discrepancy. These identities use all rows and real-only groups. They do not contradict RA8's substantial cross term under the different noisy oracle-HQ selection, and are not rescored gate metrics.

Applied output sigma is .0289999992. The optimistic real-covariance-minus-output-noise proxy has a negative eigenvalue in74 even groups,70 odd groups and67 pooled groups. Its pooled weighted remaining variance per coordinate is6.01048e-5 output units squared (.06678 original target-sigma squared), a small residual relative to local covariance estimation error. Direct local whitening or clipping this subtraction into a covariance target is unsupported. Latent jitter and nonlinear/conditional effects remain additional concerns for a generic law.

## Consequence and scope

A bounded mean-directed proposal is more plausible here than covariance matching, provided a separately declared law resolves reference uncertainty and adaptive dependence. The aggregate does not grant individual movement certificates. In particular, the count owner's separate final reaction audit reports zero ordinary copy authority; this diagnostic does not turn the mean signal into count-authorized birth slots. Any selected new mechanism must retain the5% ordinary budget,95% population certificate survival, rank-at-most8, paired geometry, own-evidence invalidation, typed checkpoint/replay law and unchanged noise/API/gates.

Helper/protocol and187 source/input identities were sealed before any numerical load or forward. One fit uses only the existing projection draw from a private saved-CPU-RNG clone. No second chart, seed change, emission/noise draw, training or model construction. The run used420000 fixed functional D-head rows, took2.4145seconds of measured CPU work, and preserved the loaded trainer state, global Torch/NumPy/trainer RNG and all inputs. Oracle annotation occurred after all chart/moment calculations. The process/log are closed before the final seal.
