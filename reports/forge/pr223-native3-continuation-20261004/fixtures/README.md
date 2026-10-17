# Closed parent metadata fixture

`closed-parent-metadata.json` is an inert, byte-exact copy of the root-authorized
CLOSED twelve-phase ledger anchor supplied at
`/ml2/hypergan/pg-pr223-native3-parent-boundary-20261004/parent_metadata_ledger.closed12.json`.
It is 2,675 bytes with SHA-256
`ce949407c9a11b2d2873be0d51b2d147261f8b60cee411ea33381c8c5c6f4369`.

The fixture records the original 180-second metadata allowance, twelve completed
phases and 44.37232269323431 seconds charged. It contains no credentials,
checkpoints, model parameters, sample arrays or new numerical verdicts. Portable
controls validate the real preserved phase prefix and allow legitimate additional
phases without reading or replacing the canonical live ledger. New phases in
software controls are synthetic; the root supplies the actual live ledger during
future charged execution.

This fixture is an explicit copied-source control dependency. Runtime preparation
still requires the separate root-supplied immutable anchor and the existing live
ledger identity. No fresh budget or scientific credit follows from this copy.
