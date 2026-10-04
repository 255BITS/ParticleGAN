# PR223 Atlas: repaired three-native continuation

This separate source contains 3 accepted original PASS, 0 original FAIL and 0 unavailable outcomes out of three required native cases. The original catalog remains 19 questions. Its earlier 16 PASS, one INVALID and two NOT_RUN remain separate references; the first native3 pretraining INVALID is a separate cost-only predecessor; combining old 16 and new three does not establish single-source 19/19.

The shared goal is all 100 modes with correct per-mode mass, centers and local shape under the original noisy policy-selected serving law. Grid, rotated and staggered layouts test geometry robustness under the same 13 requirements. Visible mode coverage alone cannot certify PASS. Original coverage and precision/mass bounds, covariance/radial bounds, plus stricter accuracy TV, center, trace and radial KS bounds remain in every row. No rotation angle is inferred.

Each case retains the complete full-winner Recipe: LR 0.00425, nominal prior multiplier 2 and critic multiplier 1; N20k/z2/batch2048; seed 1234; original initialization and public latent perturbation; learned output kernel and actual selected fast/averaged weights. It requires 7,000 updates, all 34 original 20k reads, the final five 20k checks and an independent 100k holdout. Noisy coverage AND accuracy determine the accepted gate. Raw status can report accuracy PASS while joint coverage+accuracy FAIL. Clean and forced-EMA branches are diagnostics.

| Native question | Execution / accepted / raw | Original goal GIF |
|---|---|---|
| grid100 | PASS / PASS / PASS | [9 actual frames](gifs/native-grid100.gif) |
| rotated100 | PASS / PASS / PASS | [9 actual frames](gifs/native-rotated100.gif) |
| staggered100 | PASS / PASS / PASS | [9 actual frames](gifs/native-staggered100.gif) |

This cut contains 3 accepted goal GIFs. The copied GIFs retain byte-original saved 4096-point noisy selected clouds and target references at updates 0, 50, 750, 1750, 2750, 3750, 4750, 5750 and 7000. Fixed comparison axes, the target geometry and local mode-width view illustrate mass and shape recovery. FINAL full-protocol verdict describes the completed case; earlier frames carry their own original read flags. There are no added draws, forwards or scoring reads. Missing attestation, raw ERROR and budget overruns provide no accepted numeric credit or accepted GIF.

Before publication: original19 case debit 3165.841891123 s once; first native3 pretraining-invalid debit 2.049605933 s once; combined predecessor case debit 3167.891497056 s; new case paid 2494.779808400 s; new reserve 0.000000000 s; SAME cumulative metadata charged 113.954378156 s; inclusive 5776.625683612 s / 10,800 s. The shared metadata ceiling remains 180 s. The exact 22-phase predecessor prefix, including its closed ERROR phase, remains in that cumulative ledger; earlier metadata is never added again. Overrun is diagnostic and is already included in measured paid time. Grace and retries are zero.

[FINAL_COST.json](FINAL_COST.json) appends authoritative cost after root closes the publication phase; accepted overall status remains pending until that addendum. [results.json](results.json) preserves full recipes, original gates, raw/accepted distinctions, parent pins and exact source. [input-index.json](input-index.json) records consumed byte identities; [verification.json](verification.json) records passive verification. Bulk arrays/checkpoints/logs, private nonce values and lease descriptors stay outside the public cut.

Source `2557cfc1997a530033e48d3af1fd50b29985af49`; execution digest `a208861906aba4acf0a271c6883a7047980a023cf188580de9c81c9fabd600f3`. This native3 result grants no full19/current26/default/fair-speed credit. The exporter uses only standard-library metadata operations and Pillow to decode existing GIFs.
