# RA8 saved grid centroid diagnostic

Status: **VALID_SAVED_DIAGNOSTIC**. The original frozen RA4 numerical helper is imported unchanged; only the input/output globals and scalar-Tensor JSON serialization are bound to RA8. All57 source/input bytes and CPU RNG remain exact. The fixed final7000 state and original terminal20k/holdout100k paired clouds are used, with zero new draws, emissions, training or CUDA context. Stored fidelity fields reproduce within1e-11. No gate or production law changes.

## Completed original gate

The canonical completed grid receipt is VALID/FAIL. Final failing original conditions are precision .96995<.97, maximum covariance eigenvalue ratio1.819186>1.7, center RMS .203570sigma>.20 and radial KS .041203>.04. Holdout individual fidelity accuracy passes, but the original frozen coverage gate does not; the combined holdout gate and sustained grid gate remain FAIL.

| Terminal step | Failing original coverage conditions | Failing original accuracy conditions |
|---|---|---|
|6000|Precision .96985<.97|Center .200481>.20; radial KS .043212>.04|
|6250|Maximum covariance eigenvalue1.763201>1.7|Radial KS .044028>.04|
|6500|Maximum covariance eigenvalue1.806978>1.7|Radial KS .044450>.04|
|6750|Maximum covariance eigenvalue1.800201>1.7|Center .200316>.20; radial KS .042264>.04|
|7000|Precision .96995<.97; maximum covariance eigenvalue1.819186>1.7|Center .203570>.20; radial KS .041203>.04|

Every terminal check fails. The exact old thresholds remain fixed. Independent covariance diagnostics identify the holdout coverage condition and worst-mode structure separately.

## Center offset is already in the EMA anchors

| Saved object | Center RMS / target sigma |
|---|---:|
|FAST unperturbed affine table anchors|.247404|
|EMA unperturbed affine table anchors|.211553|
|Clean terminal cloud|.212569|
|Clean holdout cloud|.211660|
|Noisy terminal cloud|.203570|
|Noisy holdout cloud|.195885|

EMA anchor versus clean holdout mode means differ by **.020976sigma RMS**, cosine **.995087**. Clean terminal versus holdout mode means differ by .046712sigma, cosine .975761. The clean center offset persists across the independent draw and matches the full-table EMA anchor population closely. Replacement sampling adds error but does not explain that population offset.

FAST/EMA same-row nearest oracle-mode agreement is **100%**, with99.16% of rows both inside3sigma. Their anchor displacement RMS is .326348sigma; the prior-row component is .326342 and the affine component only .011360. This does not show a final-state cross-mode correspondence failure. It does not reconstruct historical copy/EMA paths.

Native G is identity-initialized and trainable, not permanently fixed identity. Final FAST diagonal weights are .982667/.977301, cross terms .002205/-.003858, with bias .024789/-.016657. Direct saved affine anchors are used. Checkpoint models contain FAST training parameters; serving swaps are derived. RA8's actual serving criterion is the typed paired-average stamp, not the immutable helper's legacy table-clock context. The final stamp is eligible19113/19000 at age0 with20000 same-group rows. The saved live/EMA fidelity fields match.

## Saved output noise and perturbation

Applied output sigma remains .029, supplying .934444 of target variance. The holdout saved output-noise difference has mean (.002594,.003692)sigma; there are zero nearest-mode changes in terminal or holdout pairs. Under the exact noisy mode assignment/HQ mask, holdout centroid mean-square decomposes as:

`clean .042498 + selected noise .002075 + cross -.006202 = noisy .038371`.

Output noise and truncation slightly reduce the measured center bias. They are not a large systematic source of that offset. The nearest EMA anchor distance for saved clean holdout points has median .018206sigma and p95 .100230sigma. These are lower bounds on latent perturbation because sampled row IDs were not archived. Aggregate anchor/cloud means also include replacement-sampling error.

The original helper retains covariance/radial fields: noisy center-removed fixed-HQ KS is .035618 terminal/.027937 holdout. This empirical same-cloud decomposition is not a repaired output or new quality verdict. Independent review owns the detailed covariance/tail attribution.

## Current-D/FIFO opportunity and limits

The saved semantic metadata records a rank8,64-cell chart, full real FIFO/calibration, and **35 real-only topology groups**. Its fitted basis and centers are derived and not serialized, so this diagnostic does not claim to reconstruct the exact historical GPU chart. The final reaction makes zero mass/support/global copies or novel births despite two count discoveries. Those phases test categorical mass and support inside/outside counts; their certificates do not directly compare signed first moments within a supported component.

A prospective opportunity is a bounded **support-local first-moment check against held-real references**, with paired current/EMA placement and a movement cap from real-only local scales. Source interfaces already retain current real features, real representative rows and the FIFO. It could expose a persistent signed residual while preserving population counts. Real neighborhoods must remain within one support component:35 groups or64 whole-cell centroids cannot be treated as100 component centers. Neighbor selection must check radius/aliasing and real sample resolution. Reference/test independence, repeated adaptive decisions, serving lease/replay and total work need new declared contracts. The independent reference-feature/neighbor diagnostic examines that feasibility; no moment correction is installed or scored here.

Oracle centers used above are diagnostic scoring only and cannot choose production neighborhoods, directions, thresholds or actions. The saved centroid result alone does not establish a reliable real-only correction, a future strict gate pass or a safe change to the learned toy. This is one fixed negative-result explanation, with no alternate checkpoint, resolution sweep or oracle-driven repair.

The completed canonical grid evidence and own watcher stop are frozen separately in ../ra8-monitor-stop/. Fifteen unrun canonical screens remain PENDING/UNVERIFIED. All old queues, frozen evidence and original RA4 processes remain untouched.
