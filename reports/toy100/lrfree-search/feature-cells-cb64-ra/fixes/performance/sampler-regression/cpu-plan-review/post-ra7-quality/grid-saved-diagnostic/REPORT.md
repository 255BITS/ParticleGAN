# RA4 saved grid center and width diagnostic

Status: **VALID_SAVED_DIAGNOSTIC**. One fixed final-step7000 state and its original terminal20k/holdout100k clouds; no new draws, emissions, training, checkpoint selection, source/config/gate changes or CUDA context. All stored terminal/holdout fidelity fields reproduce within 1e-11. All 52 input/source bytes and global CPU RNG remain unchanged.

## Center bias persists in the table

| Saved distribution | Terminal center RMS / sigma | Holdout center RMS / sigma | Terminal radial KS | Holdout radial KS |
|---|---:|---:|---:|---:|
| Clean live / EMA | .324831 | .319179 | .536338 | .537811 |
| Noisy live / EMA | .313777 | .302836 | .045224 | .041477 |
| Original target cloud | .087502 | .042047 | .005571 | .001747 |

The unperturbed EMA affine table anchors already have center RMS **.320427 sigma**; FAST anchors have **.360102 sigma**. The holdout clean mode means differ from EMA anchor mode means by only **.016958 sigma RMS**, with vector cosine **.998602**. Clean terminal versus holdout mode means have cosine .989162 and differ by .047741 sigma. This supports a persistent table-centroid error rather than replacement-sampling error alone.

The original serving rule sees table last_decisive=-1, so its source dispatch implies EMA serving at this state. Saved live/EMA cloud fidelity fields are identical. The checkpoint API temporarily releases the served view before saving FAST weights, then reapplies it; the saved FAST and EMA anchors therefore describe distinct training/average tables. The native affine G was initialized as identity and trained: final diagonal weights are approximately .97207/.96022, cross terms .000273/-.012362, bias .025177/-.014134. Its exact saved matrix is included in the receipt. Treating it as a permanently fixed identity would misinterpret the prior coordinates.

## Output noise and radial shape

Saved applied sigma=.029, or **.934444** of target per-axis variance. The learned sigma=.028994 lies just below that applied value. Unperturbed EMA anchors have mean conditional covariance trace ratio .179338; clean clouds have .183692/.181094. Their clean radial distributions are correspondingly narrow. The noisy conditional trace ratios are 1.057788/1.055997, within the original mean covariance-bias limit.

The paired holdout output-noise difference has mean (.002594,.003692) sigma and per-axis variance .929422 sigma squared. It changes nearest-mode assignment for only 2/100000 rows; terminal assignments change for zero rows.

Under exactly the noisy cloud's own mode assignment and HQ mask, the holdout center mean-square decomposition is:

`clean .101460 + selected noise .002025 + cross -.011775 = noisy .091710`.

The corresponding per-axis conditional covariance decomposition is:

`clean .163812 + selected noise .870929 + cross -.032127 = noisy 1.002614`.

Thus output noise is not the source of the persistent centroid offset in this saved case; conditional truncation slightly reduces it. Noise supplies most within-mode width while residual clean spread remains heterogeneous.

Removing empirical mode means from the already selected HQ residuals reduces noisy radial KS to .025758 terminal/.022233 holdout. This is a fixed-mask moment decomposition using the same cloud, with optimistic empirical centering. It does not generate or rescore a repaired candidate and cannot establish a prospective precision or quality pass. It suggests that centroid error contributes materially to radial mismatch.

## Mass, perturbation and row correspondence

All 100 modes remain represented. Mass TV is .04105 terminal/.03570 holdout, versus EMA anchor .03335. Holdout nearest-mode masses range .00798 to .01207. Mass is uneven but remains within the original .06 accuracy limit. Terminal noisy precision .96925 is also just below the original .97 coverage limit; holdout precision .97144 passes that individual condition.

FAST/EMA same-row nearest-mode agreement is **99.985%**, and 99.115% of rows share a mode and both lie within3 sigma. Paired output-anchor motion RMS is .484062 sigma, almost entirely prior-row motion (.483921); affine-map difference contributes only .013369 sigma RMS. These final-state diagnostics do not support widespread cross-mode averaging as the main center-error mechanism.

Clean holdout points have median distance .015772 sigma and p95 .089584 sigma to their nearest EMA anchor. These are lower bounds on latent perturbation: sampled row IDs were not archived, so the actual jitter delta cannot be recovered uniquely. Aggregate clean-versus-anchor centroid differences also include finite replacement sampling. The very close EMA-anchor/holdout centroids make a large systematic perturbation-induced mean shift unlikely here, without identifying individual historical copy effects.

## One prospective direction

A bounded **support-local first-moment calibration** of paired FAST/EMA anchors against held-real references is a justified direction to examine if the fresh candidate repeats this failure. Reference neighborhoods must stay within a certified real-only support component and use existing local-scale/radius checks. A coarse64-cell chart can combine100 components, so a whole-cell centroid shift is not justified. Oracle grid centers used in this report cannot enter that decision. Such a repair would require its own declared law, schema/replay contracts and prospective quality test; no production patch is proposed or installed here.

## Evidence preservation

The numerical helper and input map were frozen before measurement. Attempt1 completed all scoring identities and guards but failed JSON serialization of a saved scalar Tensor in context. The original helper, freeze and failed log are retained unchanged. A separately frozen output-only wrapper converts Tensor context to JSON scalars/lists and runs the same original fixed equations and inputs. Attempt2 exits0; both logs are closed before the final seal. The RA8 metadata watcher and all original queues remain untouched. This diagnostic does not predict RA8 quality.
