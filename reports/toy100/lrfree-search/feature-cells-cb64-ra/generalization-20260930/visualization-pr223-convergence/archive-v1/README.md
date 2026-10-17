# Dense E22 versus ParticleGAN Atlas evidence append

This additive archive records the actual fresh rotated100 runs: current PR155 E22
at `cabe2084284db923d525918cbf3e18de6f20faac` versus ParticleGAN Atlas RA17.
Both used the original fixture, seed 1234, initial models, real streams, 1,500
updates, scorer and 20,000-point acceptance draws. The two original target turns
remain at updates 500 and 1,000. Each formulation retains its own control recipe.

| Original acceptance draw | E22 HQ / modes | Atlas HQ / modes |
| --- | --- | --- |
| 500, before first turn | 88.10% / 100 | 96.09% / 100 |
| 1,000, after first turn | 83.99% / 100 | 96.09% / 100 |
| 1,500, after second turn | 94.73% / 100 | 92.59% / 98 |

Both pass both original turn gates: at least 95 modes and HQ at least 90% of
their own update-500 baseline. Atlas reaches 90% on the separate visualization
sample earlier during initial learning and first-turn recovery; E22 has higher
final original HQ and more final modes. These runs support a scoped comparison,
not a universal final-score advantage.

Each dense recording contains 151 observed 4,096-point float32 clouds at updates
0, 10, ..., 1,500. Two additional target-transition frames reuse the preceding
cloud exactly and rotate only the target centers. There is no interpolation.
The visualization metrics use these 4,096 points, including a mode threshold of
at least 10 HQ points; they are separate from the original 20,000-point gates.
All 302 observation state guards passed. The paired GPU observer diagnostic
passed for E22 fresh updates 1–20 and Atlas restored updates 1,001–1,010. Atlas
full-run checkpoint state and original gate metrics match the prior frozen
Atlas run, excluding only recorded evaluation elapsed seconds from state parity.

## Archive format

`../ARCHIVE-APPEND.json` is schema version 1. Paths in its `files` are relative
to the generalization report root; each entry gives original `source`, `path`,
`bytes`, `sha256` and `role`. Copy absent paths, accept byte-identical existing
paths and reject every conflict. `prepare_archive.py --verify` validates this
local staging tree and the unchanged original closures without importing Torch.

The manifest covers copied payload files. The manifest itself and `FROZEN.json`
are explicitly listed as closure controls outside its own file table to avoid
recursive hashes. `FROZEN.json` hashes the manifest and all payload members.
`PREPARATION-RECEIPT.json` records existing byte-identical report files and the
exact ignored paths needing `git add -f`; trace JSONL filenames are checked
individually. Raw `.pt` checkpoints and `.npz` point clouds stay local and are
listed in `LOCAL-REFERENCES.json` with size and SHA-256. `CLOSURE-INPUTS.json`
retains every source/data/closure input hash and maps it to an archived copy
where available. Original receipt bytes retain their absolute study paths.

## Separate media append

The renderer owns a separate closure for the GIF, MP4, poster, renderer and
render receipt. It must be added with its own manifest after media review.
Rendered per-frame PNGs and all draft/layout attempts remain in the local study.
Earlier four-frame RA14/RA15 artifacts are separate historical evidence and are
not the current E22 versus Atlas comparison.
