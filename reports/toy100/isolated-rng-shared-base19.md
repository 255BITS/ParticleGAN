# Isolated output-noise RNG: frozen 19-host replay

The declared isolated-RNG affine policy passed **14/19** older hosts and failed five. It used source commit `1c1a0865fe605c9f212d832c06596c12510f937e`, the frozen host seeds, budgets, architectures, and gates. The only new mechanism relative to the earlier shared affine policy is a private, checkpointed output-noise stream; the config retained β₂=.999, input noise 0.5 ending at 10%, output noise 0.029 warmed over 20%, and the 1,600-step G/D horizon.

| Failed host | Final evidence | Frozen requirement | Sustained suffix |
| :--- | :--- | :--- | ---: |
| `residual_student` | Identity MSE 0.04213; success 0.667; wrong-pad 0.417 | MSE ≤ 0.02; success 1; wrong-pad 0 | 0/5 |
| `mode_hold` | 6 modes; HQ 0.99976 | 8 modes; HQ ≥ 0.9 | 0/5 |
| `vector_unequal_mass` | Smallest component covariance eigenvalue ratio 0.0763 | ≥ 0.15 | 0/5 |
| `vector_overlap` | Final SW1 0.07985, mean error 0.10598, covariance error 0.15966 all pass their limits | Five consecutive passing checks | 1/5 |
| `img_bars4` | 3 modes; fourth mode fraction 0.09375 | 4 modes; minimum fraction 0.125 | 0/5 |

The other 14 hosts passed their unchanged gates. `vector_unequal_mass` passed its mass TV, quality, and aggregate covariance limits but failed the rare component's width. `vector_overlap` illustrates why a final-only score is insufficient: it had 12 passing checkpoints overall but only one at the end.

The local artifact at `artifacts/toy100-accuracy/compatibility/isolated-rng-shared-policy-all19-v1/` contains the predeclared `declared_config.json`, source snapshot, full log, 19 compressed episodes, per-step noise and optimizer receipts, and machine-readable `results.json` failure ledger. All 30 files match the RAM originals by SHA-256. Independent regrading of the relocated episodes verified the source, config, frozen specifications, action receipts, and all verdicts, reproducing valid FAIL 14/19. This result is not a shared 22-task pass. These local artifact paths are not GitHub links.
