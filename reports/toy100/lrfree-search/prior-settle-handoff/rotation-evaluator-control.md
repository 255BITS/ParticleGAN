# Frozen evaluator rotation control

Frozen sources hash-verified; saved grid100 clouds rotated by native 25° angle. Source clouds unchanged. All pass flags preserved: **True**.

| Cloud | N | Grid precision | Rotated precision | HQ flips | Scalars outside strict tolerance |
|---|---:|---:|---:|---:|---:|
| final_samples.npz / live | 20000 | 0.979900 | 0.979900 | 0 | 1 |
| final_samples.npz / ema | 20000 | 0.979900 | 0.979900 | 0 | 0 |
| final_samples.npz / target | 20000 | 0.988750 | 0.988750 | 0 | 0 |
| holdout_samples.npz / live | 100000 | 0.980310 | 0.980290 | 2 | 2 |
| holdout_samples.npz / ema | 100000 | 0.980230 | 0.980230 | 0 | 0 |
| holdout_samples.npz / target | 100000 | 0.988610 | 0.988610 | 0 | 1 |

The independent saved target was rotated too, preserving both independence and a paired target comparison. This scores final/holdout clouds directly with the unchanged frozen coverage and accuracy evaluators; it does not manufacture a rotated training history.

Full scalar comparisons, declared tolerances, hashes, and direct float64 geometry checks are in result.json.

## Discrepancies

- All live, EMA and target coverage/accuracy pass flags remain true for both final and holdout clouds.
- Live holdout coverage precision changes from 0.98031 to 0.98029: two of 100,000 points cross the float32 `torch.cdist` HQ boundary. Their exact distances are within 8.21e-6 of the radius. Direct float64 distances have no membership changes, even after saving rotated samples as float32.
- The normalized accuracy score changes by 1.149e-5 for final live and 1.148e-5 for holdout target, slightly exceeding the predeclared 1e-5 tolerance. All individual accuracy components remain within tolerance.
- No nearest-mode assignments change. Exact float64 rotation preserves distances to within 1.23e-15.

Conclusion: small floating-point discrepancies, with no changed verdict and no evidence of a structural orientation bug in the evaluator. They cannot explain the training gap. Original sample files and frozen evaluator sources were not modified.
