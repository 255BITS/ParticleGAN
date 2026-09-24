# Noiseless Fourier-5 D-only horizon probes

From the exact noiseless Fourier-5 base (config SHA-256 `51f6bd91…`), these two scratch rows kept G's LR horizon at 1600 updates and the particle prior on its original 7000-update cosine. Only D's horizon changed to 4000 or 7000. This single role-based rule would leave older hosts with budgets at most 1600 unchanged, but it is not integrated into the production common gate; these probes are explicitly ineligible for a 22-task claim.

| D horizon | Final modes | Final HQ | Final mass TV | 100k holdout HQ | Holdout center RMS | Holdout covariance trace bias | Holdout radial KS | Terminal passes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 100/100 | .93985 | .05810 | .94019 | .504σ | −.185 | .0504 | 0/5 |
| 7000 | 100/100 | .93680 | .05785 | .93676 | .573σ | −.247 | .0382 | 0/5 |

Both completed 7000 updates and failed original coverage and accuracy. D4000 slightly reduced the center and covariance errors relative to D7000, but neither met the frozen HQ or fidelity limits. The saved event trace independently verified the actual G, D, and prior LR on **all 7000 updates** in each row; the schedules were identical through update 2400 and diverged only when D4000 began annealing. The probe source is frozen at commit `70f0da9`, separate from production. Full evidence (`artifacts/toy100-accuracy/affine-noiseless-dhorizon-v1`) includes all five checks, 100k holdouts, predeclaration, action receipts, and V2 source archives. All 113 files matched their RAM originals by SHA-256, and both relocated rows regraded as valid failures.
