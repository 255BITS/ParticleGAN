# PR84 repaired cold ring: early acquisition geometry

This read-only analysis uses the [exact first-100 repaired-cold replay](pr84-finite-cold-prefix100.md). It evaluates clean `G(prior.z)` and the sharp critic on fixed inputs from four saved updates; target centers are used **only after training** to describe coverage and score changes. The analysis source is [pr84_early_geometry.py](pr84_early_geometry.py), and its complete [geometry receipt](continuous-evidence/pr84-finite-cold-prefix100/geometry.json) checks the hashes of every captured stage and the archived original PR84 control.

| Update | G output RMS | Common translation energy | Relative deformation RMS | Clean occupied arc before → after | Bounded D parameter move | Subsequent D fit / bounded D move |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | .00890 | 95.7% | .00185 | 40° → 38° | .6032 | 9.47× |
| 25 | .00928 | 94.8% | .00212 | 80° → 77° | .3027 | 2.11× |
| 50 | .06784 | 51.3% | .04732 | 120° → 120° | .0224 | 6.32× |
| 100 | .08921 | 76.2% | .04355 | 98° → 97° | .0347 | 2.41× |

“Common translation energy” is `||mean_i Δx_i||² / mean_i ||Δx_i||²`; the remainder is particle-relative deformation. Network-only functional motion nearly equals full G+prior motion at each checkpoint; prior-only RMS is at most .00418. The clean support remains in a narrow angular sector while its centroid travels from roughly `(0.09, 0)` to `(2.47, 0.01)`. At update 100 the G move changes clean high-quality coverage from two to three modes, but leaves exactly four particles each nearest adjacent target modes 0, 1, and 7. The critic fit moves 2–9× farther in parameter L2 than the ordinary bounded D step. On the same fixed pre-step fake support and eight target centers, its sharp-score RMS change is 1.1–20.5× the D-step score change. These fixed-input scores are diagnostics, not the noisy training loss or a guarantee of better G direction.

The [source-bound original PR84 cold control](continuous-evidence/pr84-independent-audit/manifest.json) has zero high-quality modes at native update 100, versus three for the fitted variant. It later reaches seven modes, whereas the fitted variant remains at three at all native observations after update 100. Thus the extra fit initially **accelerates high-quality acquisition**, yet the resulting cloud remains confined to a three-mode sector; the data do not support saying the fit causes failure *within the first 100 updates*. The archived original control has native 50/100 observations and all 1,200 update records, but no four-stage snapshots, so this report makes no claim about its within-step translation or critic displacement. No extra original-control training run was needed for this bounded comparison.
