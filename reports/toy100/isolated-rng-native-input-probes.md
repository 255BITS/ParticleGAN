# Isolated-output-RNG native grid probes

The frozen `1c1a086` implementation was evaluated on `grid100` with the generic affine-square generator, seed 1234, 7,000 updates, five terminal checks, and an independent 100,000-sample holdout. All three runs completed but failed both the original coverage gate and the accuracy gate. They are diagnostics, not shared 22-task claims.

| Shared core | Input σ | Config SHA-256 | Final quality modes | Final HQ | Final mass TV | Holdout HQ / TV | Terminal passes |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| β₂=.99 | 0 | `3db4540a…` | 92/100 | .98565 | .10945 | .98549 / .10175 | 0/5 |
| β₂=.99 | .25 | `881cd16f…` | 93/100 | .98535 | .09815 | .98600 / .08876 | 0/5 |
| β₂=.999 | .25 | `f933ffc9…` | 14/100 | .16855 | .18080 | .16934 / .17613 | 0/5 |

The β₂=.99/.25 and β₂=.999/.25 declarations differ only in `betas[1]` and `name`; both use the isolated output RNG and the same source. The latter reached 100 quality modes at step 500, then dropped to 10 at step 750 and finished at 14. The β₂=.99 rows had good final HQ but too much mass imbalance and missing quality modes. For comparison, the older global-output-RNG β₂=.999/inputσ=.5 archived run reached 100 modes at step 750 and finished with TV .0435; its RNG and input amplitude also differ, so that comparison does not isolate a cause.

All saved samples, terminal checks, holdouts, event logs, resolved configs, manifests, and V2 source archives were independently regraded after relocation. The full source SHA manifest matched frozen commit `1c1a086`, and every copied file matched its RAM original by SHA-256. Raw evidence: β₂=.99 pair (`artifacts/toy100-accuracy/isolated-rng-affine-beta99-v1`), β₂=.999/.25 control (`artifacts/toy100-accuracy/isolated-rng-beta999-input025-native-v1`).
