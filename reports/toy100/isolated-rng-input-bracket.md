# Isolated output-RNG input-noise bracket

The exact β₂=.999/input-noise σ=.25 isolated-RNG variant had passed 9/10 older bottlenecks, missing only `mode_hold` (6 of 8 required modes). A five-point input-noise amplitude bracket was declared before training: σ ∈ {.125, .20, .30, .35, .40}. Each config changed only `name` and `input_noise_std` from the exact control SHA-256 `f933ffc9ad26fd2235c5e69b3d04892dbfa239d2a0017e70b30446a6d7433b3e`; all source was frozen at `1c1a0865fe605c9f212d832c06596c12510f937e`, including output-noise seed offset 1901.

The frozen first-stage gate required `mode_hold` to reach 8 modes with HQ ≥ .9 for five consecutive checks. All five candidates failed and independently regraded as valid `FAIL 0/1` from their relocated archives:

| Input σ | Config SHA-256 prefix | Final modes | Final HQ | Passing suffix |
| ---: | --- | ---: | ---: | ---: |
| .125 | `5e5542e5` | 6 | .66919 | 0 |
| .20 | `c110e91e` | 6 | .67725 | 0 |
| .30 | `780ae4a9` | 6 | .83936 | 0 |
| .35 | `d42b815a` | 7 | 1.00000 | 0 |
| .40 | `0a4c91d4` | 7 | .90991 | 0 |

The manifest and evidence (`artifacts/toy100-accuracy/isolated-rng/input-bracket-1c1a086/`) retain all five configs, source archives, logs, compressed episodes, and independent regrades. No candidate qualified for the preregistered remaining-nine screen or a fresh full-19 replay. The `.25` control was already run and was not repeated.
