# RA14: restoration-only correction and original continuation

**Toy and MNIST native versus CPU-map CUDA replays PASS.** Both use the original saved step1000 checkpoint and two ten-update continuations. Every restored semantic field, update loss, per-update semantic fingerprint, endpoint semantic state, and original noisy-primary sample byte matches. Table stationarity state, RNG, optimizer/controller state, backend certificate, R1 detector and settled guard are included. Sampling preserves training state. Only the original observational birth evaluation duration is excluded.

The RA14 native continuation additionally equals the sealed RA13 native continuation at every update and in its endpoint/sample bytes. This verifies the repair in the exact window that exposed the alias defect. The earlier RA13 replay failure remains sealed.

## Source and numerical scope

Reviewed package digest (per source bytes): `68f5706590683a44348cb04b3798917411bfaa54e5d45b17d57c345a4da33c15`. Shared config SHA: `a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4`. Exactly one definition changes: `policy._state_to_device`. A memo preserves repeated tensor identity during device conversion. All other package source bytes, source inventory, checkpoint schemas and configuration bytes are identical to RA13. Every helper call is within checkpoint validation or restoration. The source-aware bridge separately pins the original source/input seals, run receipts, checkpoints and native controls.

No fresh 2000-update training was performed for RA14. The source proof establishes that training construction, updates and evaluation are unchanged, so the closed original RA13 training receipts apply through explicit equivalence. Toy25 original gate is **PASS**; all nine postupdate metric/LR records equal frozen original RA11. All ten MNIST metric/LR records equal corrected E22: final active FD **0.5444880363760234**, precision **0.869140625**, recall **0.84716796875**, all ten classes. The original MNIST fixture has no added numerical quality threshold. Both original full-training fixtures recorded zero R1 fires and one objective epoch rebase.

Zero-update preflight matched original model/prior hashes, private streams, pending selection and guard state, and the complete candidate Recipe with original N1024/z128/batch128 overrides. Saved CPU checkpoint reads preserved the table displacement alias, exact cursor and Recipe. CUDA remained uninitialized. The first bridge preflight’s raw-config normalization error is preserved in `../ra14-replay/CPU-FAILED-CLOSED.json`; correction was made in this fresh r2 adapter without changing package, configuration or prior frozen evidence.

## Runtime and remaining scope

One approved GPU0 launch executed exactly forty continuation updates total, with the original noisy-primary draws (Toy8192, MNIST4096; seed314259). Wall time: **23.966s**. It used the existing owned outer slot, shared serial lock and 20% memory cap, and signaled no processes. GPU lane released at `2026-09-30T22:29:05.241831+00:00`.

This receipt covers original learned training through the restoration-only source proof and the freshly executed original replay. The separate final portability, native and moving validation gates remain owned by the coordinator.
