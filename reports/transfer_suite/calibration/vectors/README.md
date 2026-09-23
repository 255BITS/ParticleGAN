# Vector transfer calibration handoff

Code commit: `20624589e10da9bc670136cb46409a2da953fe23`. Exactly 8 development tasks plus an unexamined reserved family. All 12 recorded reference executions used seed0, CPU, one Torch thread.

- [Protocol v2 readable leaderboard](support/v2_rescored/README.md)
- [Original v1 source archive](v1_source.tar.gz)
- [Corrected v2 source archive](v2_source.tar.gz)
- [Original-byte hashes and archive inventory](index.json)
- [Tailable original log](support/progress.log)
- [Original calibration script](support/calibrate.py)
- [Additional fixed references](support/extra_references.py)
- [Explicit v2 rescoring script](support/rescore_v2.py)

All JSON files are gzip-compressed without changing their uncompressed bytes. `index.json` maps relative paths to original SHA256 and compressed SHA256. Original v1 source includes the original module/tests/design document; v2 contains the corrected code. Both archives include tracked Python dependencies from the isolated worktree. No reserved samples or results are present.

V2 adds the per-component minimum eigenvalue bound to every separated mixture after a review witness demonstrated a loophole, before controller fitting. Original executions are retained and v2 rows explicitly distinguish execution from scoring protocol. No training was repeated for this correction.

## JSON files

- [results/additional_reference_plan.json.gz](results/additional_reference_plan.json.gz)
- [results/additional_summary.json.gz](results/additional_summary.json.gz)
- [results/frozen_manifest.json.gz](results/frozen_manifest.json.gz)
- [results/summary.json.gz](results/summary.json.gz)
- [results/v2_rescored/frozen_manifest.json.gz](results/v2_rescored/frozen_manifest.json.gz)
- [results/v2_rescored/summary.json.gz](results/v2_rescored/summary.json.gz)
- [results/v2_rescored/target_oracle.json.gz](results/v2_rescored/target_oracle.json.gz)
- [results/v2_rescored/vector_anisotropic__fixed_cosine.json.gz](results/v2_rescored/vector_anisotropic__fixed_cosine.json.gz)
- [results/v2_rescored/vector_narrow__fixed_cosine.json.gz](results/v2_rescored/vector_narrow__fixed_cosine.json.gz)
- [results/v2_rescored/vector_overlap__fixed_cosine.json.gz](results/v2_rescored/vector_overlap__fixed_cosine.json.gz)
- [results/v2_rescored/vector_scale_drift__fixed_cosine.json.gz](results/v2_rescored/vector_scale_drift__fixed_cosine.json.gz)
- [results/v2_rescored/vector_spiral__fixed_cosine.json.gz](results/v2_rescored/vector_spiral__fixed_cosine.json.gz)
- [results/v2_rescored/vector_two_broad__fixed_cosine.json.gz](results/v2_rescored/vector_two_broad__fixed_cosine.json.gz)
- [results/v2_rescored/vector_unequal_mass__fixed_constant.json.gz](results/v2_rescored/vector_unequal_mass__fixed_constant.json.gz)
- [results/v2_rescored/vector_unequal_mass__fixed_cosine.json.gz](results/v2_rescored/vector_unequal_mass__fixed_cosine.json.gz)
- [results/v2_rescored/vector_unequal_mass__fixed_cosine_r1r2_0p1.json.gz](results/v2_rescored/vector_unequal_mass__fixed_cosine_r1r2_0p1.json.gz)
- [results/v2_rescored/vector_unequal_width__fixed_constant.json.gz](results/v2_rescored/vector_unequal_width__fixed_constant.json.gz)
- [results/v2_rescored/vector_unequal_width__fixed_cosine.json.gz](results/v2_rescored/vector_unequal_width__fixed_cosine.json.gz)
- [results/v2_rescored/vector_unequal_width__fixed_cosine_r1r2_0p1.json.gz](results/v2_rescored/vector_unequal_width__fixed_cosine_r1r2_0p1.json.gz)
- [results/vector_anisotropic__fixed_cosine.json.gz](results/vector_anisotropic__fixed_cosine.json.gz)
- [results/vector_narrow__fixed_cosine.json.gz](results/vector_narrow__fixed_cosine.json.gz)
- [results/vector_overlap__fixed_cosine.json.gz](results/vector_overlap__fixed_cosine.json.gz)
- [results/vector_scale_drift__fixed_cosine.json.gz](results/vector_scale_drift__fixed_cosine.json.gz)
- [results/vector_spiral__fixed_cosine.json.gz](results/vector_spiral__fixed_cosine.json.gz)
- [results/vector_two_broad__fixed_cosine.json.gz](results/vector_two_broad__fixed_cosine.json.gz)
- [results/vector_unequal_mass__fixed_constant.json.gz](results/vector_unequal_mass__fixed_constant.json.gz)
- [results/vector_unequal_mass__fixed_cosine.json.gz](results/vector_unequal_mass__fixed_cosine.json.gz)
- [results/vector_unequal_mass__fixed_cosine_r1r2_0p1.json.gz](results/vector_unequal_mass__fixed_cosine_r1r2_0p1.json.gz)
- [results/vector_unequal_width__fixed_constant.json.gz](results/vector_unequal_width__fixed_constant.json.gz)
- [results/vector_unequal_width__fixed_cosine.json.gz](results/vector_unequal_width__fixed_cosine.json.gz)
- [results/vector_unequal_width__fixed_cosine_r1r2_0p1.json.gz](results/vector_unequal_width__fixed_cosine_r1r2_0p1.json.gz)
