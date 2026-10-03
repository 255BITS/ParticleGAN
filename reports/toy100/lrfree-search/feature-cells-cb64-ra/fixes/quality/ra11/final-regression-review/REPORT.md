# RA11 final regression artifact review

Evidence: **VALID**. Source/API/toy/grid proofs are reused; original MNIST and replay artifacts are checked without model execution.

| Record | Original result | Evidence |
|---|---|---|
| MNIST training | COMPLETE | VALID |
| mnist replay | PASS | VALID |
| toy replay | PASS | VALID |
| mode_hold | FAIL | VALID |
| img_intensity2 | FAIL | VALID |
| img_blobs4 | FAIL | VALID |
| img_stripes2 | PASS | VALID |
| img_bars4 | FAIL | VALID |
| vector_two_broad | PASS | VALID |
| vector_unequal_mass | FAIL | VALID |
| vector_unequal_width | PASS | VALID |
| vector_anisotropic | PASS | VALID |
| vector_overlap | PASS | VALID |
| vector_spiral | PASS | VALID |
| ring_shift | PASS | VALID |
| stationary | PASS | VALID |
| grid100 | PASS | VALID |
| rotated100 | PASS | VALID |
| staggered100 | PASS | VALID |

MNIST quality remains a regression: active embedding Fréchet 40.544410, precision 0.281250, recall 0.000000; confident class coverage 1, pixel clipping 0.563290.

Both strict toy and Grid100 passed. Remaining screen failures and MNIST degradation are preserved; this validates the experiment and does not recommend general package replacement.

Ten MNIST checkpoint loads and four replay endpoint loads use CPU storages and original GPU device tags. No constructors, forwards, draws, training updates, scorer runs or CUDA contexts. Only the original eval_seconds replay exclusion applies.

Reset/inheritance/lineage identities are checked only at actual saved reaction boundaries; historical or compressed row IDs and isolation identities are not reconstructed.
