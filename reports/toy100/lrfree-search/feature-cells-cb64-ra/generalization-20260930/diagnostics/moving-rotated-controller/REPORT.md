# Rotated moving controller audit

The retained original gate failed: HQ at updates500/1000/1500 was0.9609/0.9548/0.8557, with100/100/99 covered modes. The unchanged final HQ bar is0.86481. These CPU inspections perform no training, model evaluation or GPU operations.

## Saved controller facts

R1 records fires at completed-step522 and1021, with ratios3.609 and5.034. A begin-step fire at completed1021 applies to update1022. All four settling ladders record two reopens at1500. The feature generator/noise calibration remains0.25, with G base0.0010625; the table base remains0.0085.

G is at scale1 for updates1022 through1381 and0.5 before/after that span:430 calibrated-base steps over the final500updates. The final stationary reduction occurs at1381. This is an active shock recovery window. Critic internal settling scale reaches0.0000305176, but the existing table-relative floor keeps actual critic LR near0.003182; the critic is not frozen by that internal ladder.

KA2 is active from800. Its actual loss-epoch guard rebases once when anchoring begins. From1000 to1500, EMA updates rise39→527, skipped updates162→174 and reseeds0→5. KA2 alpha is0.0097123 at1500 and the R1 anchor event has fired. Adam AMSGrad denominator inflation is modest: final G medians1.06/1.083 and table median1.107, with table95th percentile1.198. These states do not support a blanket LR boost or a persistently frozen optimizer-memory explanation.

Mean repairs stop: cumulative witness fires43→43 and mean moves25245→25245 from1000 to1500, despite50 more reactions. The final mean witness is invalid with `missing_EMA_group`. Final serving coherence18415/20000 misses the unchanged19000 bar and serves fast rather than EMA. This identifies a concrete all-group mean-repair veto to test.

## Feature state lifecycle

The real FIFO is refilled by2048 critic-real rows per update; the20000row FIFO triggers each10updates. Every reaction clears reference/query evidence, captures current clean outputs and critic features, fits a new chart/projection and freezes a new pre-action EMA witness. At1500 the snapshot serial and reaction step are150 and1500, with zero rows since evaluation. There were50 fresh fits after1000. Mean frames and packets are ephemeral, and local geometry is keyed to parameter identity/version.

The second target turn begins at1001 and the FIFO has turned over before the recorded1021 fire. Existing genealogy tracks actual copies; it is not an old target chart. There is no saved-state evidence for an additional chart/reference cache reset. Changing reaction timing or resetting lineage would introduce a separate mechanism into the test.

## Proposed causal test

Use the original CUDA checkpoint1000 twice: unchanged RA14 control and fresh RA15 partial-recovery source. Reconstruct the original external seed1234 CUDA real stream with exactly2000 real batches of2048 rows, restore the model/optimizer/controller and all private/global RNG states, and run exactly updates1001…1500 with the original60degree target. Preserve the original20k gate draw, inherited pre-turn baseline, mode bar, raw-output rank limit, alpha, action budget and95percent coherence requirement.

The candidate may repair only the initially supported mean groups after an actual typed R1 fire. Inactive weights and directions are zero, original even weights are not renormalized, every odd row is retained, and unsupported groups receive ordinary birth/death actions. A currently empty initially active group still vetoes mean repair. Pair and packet eligibility must reject inactive groups before count divisions.

`PEER-REVIEW.json` independently checks the exported NumPy raw-space score algebra. Its positive LCB is diagnostic evidence, not reconstruction of the original ephemeral critic chart or a prediction of quality success. The paired adapter must capture actual runtime chart counts. Control reproduction is a required gate before interpreting candidate quality.

## Provenance and limits

`CONTROLLER-STATE.json` binds all three checkpoints, the closed original completion, package sources, source freeze and sealed finalizer. Histories retain only the last eight settling decisions and R1 fires. The external data cursor is not in the checkpoint and must be reconstructed. No production, package, frozen-lane or sealed release-preparation file was edited by this audit.
