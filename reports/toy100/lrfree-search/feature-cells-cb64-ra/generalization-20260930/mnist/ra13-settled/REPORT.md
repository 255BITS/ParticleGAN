# Actual RA13 settled guard: original learned fixtures and CUDA replay

Original Toy25 quality gate: **PASS**. Original MNIST is a comparative regression with no added pass threshold. All ten saved MNIST metric records equal corrected E22: **True**. Both original native versus CPU-map CUDA replays **FAIL** exact semantic state. Previous numerical failures remain intact and are included as controls.

All ten MNIST metric and applied-LR records exactly equal corrected E22: **True**. All nine postupdate Toy25 metric and applied-LR records (100→2000) exactly equal frozen original RA11: **True**. Toy checkpoint0 deliberately retains pending-reference selection; its score/base-rate metadata differs from the already-resolved old feature control, while model/prior/private-stream initialization is exactly matched. No synthetic update resolves selection before the first original real update.

| Fixture | Variant | Precision | Modes / recall | TV / active FD |
|---|---|---:|---:|---:|
| toy | CB64-RA11 | 0.965332 | 25.000000 | 0.052114 |
| toy | E22 | 0.715454 | 25.000000 | 0.284546 |
| toy | RA12-auto | 0.668579 | 22.000000 | 0.336016 |
| toy | RA13-settled | 0.965332 | 25.000000 | 0.052114 |
| mnist | CB64-RA11 | 0.281250 | 0.000000 | 40.544410 |
| mnist | E22 | 0.869141 | 0.847168 | 0.544488 |
| mnist | RA12-auto | 0.767578 | 0.719238 | 1.880969 |
| mnist | RA13-settled | 0.869141 | 0.847168 | 0.544488 |

## Source and fixture validity

One reviewed package uses the per-source-bytes digest `9630687b21dcdc72cb63a65dcae58470d47a81307bc06000534df02e11d17c4c` and shared config `a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4`. Only original N1024/z128/batch128 override the configuration. Original public G/D init seeds0/1 reproduce exact saved model/prior hashes; global/private streams, saved real batches, two real batches per update, seed314159, deterministic serial CUDA0, 2000 updates and ten checkpoint steps are unchanged. No extra begin-step, forward, score, oracle input or synthetic update was inserted.

Original scorer/gate ASTs remain exact. Primary sampling explicitly retains original output noise. Image evaluator JSON equals corrected E22 with39 active dimensions. Selection/role-rate/guard validity is checked at every checkpoint. Toy selects feature cells with quarter G/noise bases; MNIST selects reference KNN at original historical bases. Checkpoint0 stays pending. The guard certificate contains only objective phase, explicit contracted-network witnesses, excursion cursor and epoch count.

## Recorded R1 and objective epochs

- toy: R1 fires 0, log [], anchor events 0, loss-epoch rebases 1.
- mnist: R1 fires 0, log [], anchor events 0, loss-epoch rebases 1.

## Original continuation and replay

Both fixtures use saved step1000 with two independent ten-update continuations. Branch0 loads native placement; branch1 CPU-maps the same checkpoint then restores CUDA0 through the public loader. Canonical restoration, every update loss and original noisy-primary sample bytes (Toy8192/MNIST4096, seed314259) match. Sampling preserves training state. All semantic sections except lr_settle match throughout, including the entire guard state. Updates1001–1007 match whole semantic state; updates1008–1010 fail.

CPU-only saved-endpoint inspection isolates one differing semantic leaf: `lr_settle[0][1].last_block`, a131072-coordinate float32 tensor. The original checkpoint aliases this tensor with `blocks[-1]`. Native loading preserves the alias; recursive CPU-to-CUDA conversion creates independent copies. A moved-row rebase at1008 invalidates history withNaN, leaving the CPU-map last_block stale. Toy differs in42rows×128coordinates; MNIST inrow717×128coordinates. All actual blocks, pairs, models, optimizer state and RNG are bit identical. Source consumes last_block for diagnostic row energy, so no training discrepancy is demonstrated within these ten updates; strict semantic-state replay remains a valid failure. Only original birth evaluation duration is excluded. Complete input/restored tensor placement is recorded; RNG buffers stay CPU uint8 and pending R1 tensors restore onto model device.

The failure, source search and CPU-only comparison are sealed separately in `../ra13-replay-diagnosis/DIAGNOSIS.json` (SHA`bd1ed1f2e8c4beeba7b064490669eb897b6ba911e259c1552f53324abd833056`). No frozen package, config, adapter or numerical evidence was changed.

Toy numerical training time 161.575s; MNIST 91.048s. All three phases preserve source/input seals and use the existing owned outer GPU0 slot/shared serial lock with20% cap, without signaling a process. Full checkpoint curves, rates, guard/R1 state and final/minimum-final-training-time comparisons are in comparison.json and the original metrics.jsonl files. This report makes no claim about the separate portability/native/moving gates.
