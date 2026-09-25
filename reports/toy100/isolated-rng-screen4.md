# Isolated output-RNG bounded transfer screen

Four preregistered variants of `accuracy_shared_policy_isolated_rng.json` were run on the same ten frozen older hosts. The only recipe changes were β₂ ∈ {.99, .999} and input-noise σ ∈ {0, .25}; the `name` differed for provenance. Output-noise σ=.029, warmup=.2, isolated seed offset=1901, κ=1.25, and the effective G/D learning-rate floor=.05 were fixed. The optional `network_lr_floor` field remained absent, as in the base config. Source commit `1c1a0865fe605c9f212d832c06596c12510f937e` was unchanged.

The manifest, all four exact configs with SHA-256, logs, source archives, compressed episodes, and saved plus independent regrades are retained at `artifacts/toy100-accuracy/isolated-rng/screen4-1c1a086/`. Independent regrading from that relocated copy found no provenance or receipt error.

| β₂ | Input σ | Config SHA-256 prefix | Ten-host result | Misses |
| ---: | ---: | --- | ---: | --- |
| .99 | 0 | `3db4540a` | 6/10 | trajectory, mode_hold, vector_unequal_width, img_stripes2 |
| .99 | .25 | `881cd16f` | 7/10 | trajectory, vector_unequal_width, vector_overlap |
| .999 | 0 | `24313d0c` | 7/10 | mode_hold, img_stripes2, img_blobs4 |
| .999 | .25 | `f933ffc9` | 9/10 | mode_hold |

The exact miss reasons were:

| Variant | Case | Frozen-gate reason |
| --- | --- | --- |
| .99 / 0 | trajectory | Final metrics pass; only 1 consecutive passing check, 5 required |
| .99 / 0 | mode_hold | Final metrics pass; only 4 consecutive passing checks, 5 required |
| .99 / 0 | vector_unequal_width | Minimum component eigenvalue ratio .11749 < .15 |
| .99 / 0 | img_stripes2 | Final metrics pass; only 2 consecutive passing checks, 5 required |
| .99 / .25 | trajectory | Identity MSE .03681 > .02 |
| .99 / .25 | vector_unequal_width | Final metrics pass; only 2 consecutive passing checks, 5 required |
| .99 / .25 | vector_overlap | Final metrics pass; only 3 consecutive passing checks, 5 required |
| .999 / 0 | mode_hold | 1 mode < 8 and HQ .00122 < .9 |
| .999 / 0 | img_stripes2 | HQ .875 < .9 |
| .999 / 0 | img_blobs4 | 3 modes < 4 and HQ .78125 < .9 |
| .999 / .25 | mode_hold | 6 modes < 8 |

The preregistered promotion rule required all ten selected hosts to pass before any fresh full-19 replay. No variant qualified; no additional full-19 or 100-mode run was launched from this screen.
