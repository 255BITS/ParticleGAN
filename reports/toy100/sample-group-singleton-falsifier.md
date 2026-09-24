# A singleton tail falsely creates a ninth remembered group

The [RMS-overlap memory](sample-group-dispersion-filter.md) passes its first four geometric checks, but one ordinary scarce-group event defeats immediate confirmation. This [falsifier](sample_group_singleton_filter.py) restores the actual passing prestart hold state at update 2400 and reproduces its next native full real128 bank bitwise. It then conditions the following bank to contain exactly one draw from component 0, using the same saved data stream and otherwise drawing from the unchanged ring target. The one-component label event has probability `128(1/8)(7/8)^127 = 6.905e-7`; the explicit tail condition was met on the fourth noise-bank attempt. Labels define this diagnostic event and grade only; the memory sees unlabeled coordinates.

The current-bank MST correctly infers eight groups. Its singleton lies 0.13250 from the cached component centroid, beyond that group's empirical RMS radius 0.10550; the singleton's own RMS is zero. The revised memory therefore appends it as a **false ninth group** while retaining the previously known group as absent. That is a structural false birth, not a target change.

| One free-output MM target from the same passing support | Modes | HQ | Nearest-mode particle counts |
| --- | ---: | ---: | --- |
| Initial support | 8 | 1.00000 | 1,1,2,1,2,1,2,2 |
| Keep unmatched singleton tentative; use prior eight confirmed groups | 8 | 1.00000 | 1,1,2,1,2,1,2,2 |
| Current-bank-only eight groups | 8 | .99976 | 1,1,2,1,2,1,2,2 |
| Immediately confirm singleton as ninth remembered group | 8 | .91577 | 2,1,2,1,2,1,2,1 |

The false birth consumes one of 12 distinct-anchor slots and sends one clean particle through an off-mode intermediate target (its fixed-draw HQ rate is zero). This particular target still clears the `.9` HQ threshold; it is **not** an observed native-training failure. The [source-bound receipt and gzip manifest](continuous-evidence/round6-sample-group-singleton/manifest.json) retain the exact state and data provenance, raw result, and source. No neural or optimizer update was run.

The next data-only rule should keep a newly unmatched group **tentative** until an independent later real minibatch corroborates it. Only a group not already matched to confirmed support should count as corroborating novelty. A tentative singleton in this example would leave the confirmed eight-group anchor map unchanged and preserve HQ 1.0. The smallest version uses two independent bank observations and expires an uncorroborated tentative group when the next bank arrives; that is evidence gating, not learning-rate decay. It can delay acquisition of a rare genuine mode, and two coincident Gaussian tails can still cause false confirmation. A pathwise never-false guarantee would require stronger distribution/separation assumptions or a declared sequential error bound. Until a tentative rule itself survives the same omitted/novel/singleton filters, the RMS-overlap method should not enter a native training gate.
