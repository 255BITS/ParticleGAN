Ring16 full SVD normalization amplifies tiny backward differences into large matrix updates. This experiment applies a fixed float32 numerical-rank cutoff to G/D matrix directions, comparing one intervention at update 401 with every-step truncation while holding architecture, prior, initialization, batches, rates and gates fixed.

Every-step truncation passed independent smoke confirmation at update 684 and all 47 terminal observations through update 1,600 (final covariance error .589678, HQ .963623). Boundary-only had two isolated passing observations but failed independent confirmation and the five-terminal diagnostic. The saved-gradient CUDA probe reduced direction mismatch approximately 1,713-fold, with exact captured-factor parity and repeats.

Validation: both fresh-live CUDA arms completed 1,600 updates; initial contexts, seen batches and the boundary's archived 400-state match; frozen source hashes and all scheduled observations verified. Includes compact metrics/provenance, actual-training GIFs and a verified local raw-artifact archive. Earlier blocked/static receipts remain intact. No retries, production default changes, retention spend or qualification credit.

See [the report](reports/forge/ring16-truncation/README.md) for metrics, reproduction sources and limitations.
