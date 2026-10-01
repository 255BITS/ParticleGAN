# RA6 diagnostic serialization source audit

PASS. The package has exactly the same 29 files as RA5; only
`feature_cells.py` changes. Four independent inverse edits reconstruct the
complete RA5 file bytes and AST. Every other module and the config are byte
exact, and the original RA5 freeze remains valid.

The change is confined to `maybe_apply` diagnostics: call the existing
`novel_birth_diagnostics` helper once, retain its result, and use its Python
lists for target cells, children and seed rows. That helper already converts
the same tensors with `detach().cpu().tolist()`. Proposal acceptance, moves,
training arithmetic, noise/RNG code, API, schemas and policy remain unchanged.

Package digest:
`6bb967c405c486d2a5cbd5dcc3bce7d1bd2cb93b3f9c7e916229f1327a8e5fc5`.

This is a CPU stdlib source audit; no numerical contract is repeated and no
Torch/CUDA context or quality job is started. The count owner's separate
saved-input test covers actual JSON serialization. RA5's checkpoint-100 ERROR
and all prior proofs remain preserved. Quality is pending.
