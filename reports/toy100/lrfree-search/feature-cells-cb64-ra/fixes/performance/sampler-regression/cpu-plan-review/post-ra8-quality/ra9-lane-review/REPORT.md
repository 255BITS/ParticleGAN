# RA9 immutable lane — prelaunch PASS

The frozen root READY, lane/source maps, package and config match. Original
learned inputs, scorers and replay source remain exact; native fixture/scorer
maps and candidate noise/stream options remain preserved. All generated
sources compile. The source-freeze has18 local and126 external guards.

The collector differs from the original collector only in the declared
`expected_options.evaluation_generate` literal, plain to indexed, after variant
renaming. Its undo restores the complete original collector AST. This was
declared before numerical execution. All original data/init/stream checks and
quality gates remain present. RA9's frozen collector already declares indexed;
the RA4-only runtime adapter is not needed and is not reused on RA9.

The launcher differs only in `jobs`: toy first, grid second, followed by the
original remainder. All19 original job specifications and16 canonical screens
remain identical. Requested cells128 and all3 optimizer bases survive actual
learned/native recipe merges and CLI paths. The canonical harness and unchanged
CPU monitor hashes match. This is a static prelaunch source/fixture audit; no
runtime initialization, trajectory, score or quality result is claimed.

Use the unchanged `integration/review/monitor_validation.py` with this prepared
lane, writing new canonical receipts under
`integration/review/validation-cb64-ra9-monitor/`. Original collectors, inputs,+gates, packages and earlier strict/adapted receipts remain untouched.

`receipt.json` contains the PASS checks; `FROZEN.json` is the authoritative
post-exit seal. The helper/input binding was frozen before execution.
