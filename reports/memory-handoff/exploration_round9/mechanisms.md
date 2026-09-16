# Local prediction and adapter interventions (evaluation only)

All probes use completed checkpoints and the saved evaluation panel. MSE below is an evaluation metric, never this round's training objective.

| Run | Prefix | Next point MSE | Following point after generated / real write | Shuffled memory MSE | Clock zero MSE | Adapter bypass next MSE | Adapter bypass long radial error |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean_full | 8 | 0.00527 | 0.01775 / 0.00202 | 1.40979 | 0.03351 | 0.00527 | not run |
| clean_full | 32 | 0.00505 | 0.01615 / 0.00213 | 1.46888 | 0.03378 | 0.00505 | not run |
| clean_s25 | 8 | 0.00584 | 0.02568 / 0.00373 | 1.40798 | 0.03294 | 0.00584 | not run |
| clean_s25 | 32 | 0.00606 | 0.02323 / 0.00316 | 1.45557 | 0.03224 | 0.00606 | not run |
| clean_s25_detach | 8 | 0.00550 | 0.02377 / 0.00360 | 1.42028 | 0.02657 | 0.00550 | not run |
| clean_s25_detach | 32 | 0.00540 | 0.01944 / 0.00273 | 1.44606 | 0.02814 | 0.00540 | not run |
| clock_control | 8 | 0.00641 | 0.02933 / 0.00489 | 1.41445 | 0.03886 | 0.00641 | not run |
| clock_control | 32 | 0.00664 | 0.02717 / 0.00388 | 1.45982 | 0.03601 | 0.00664 | not run |
| proposal_clean_full | 8 | 0.00618 | 0.02251 / 0.00279 | 1.43682 | 0.02836 | 0.02144 | 3.3894666314511737 |
| proposal_clean_full | 32 | 0.00633 | 0.02247 / 0.00297 | 1.48861 | 0.02233 | 0.01841 | 3.420976259412348 |
| proposal_clean_s25 | 8 | 0.00519 | 0.02197 / 0.00337 | 1.41097 | 0.03337 | 0.07113 | 4.161413958424253 |
| proposal_clean_s25 | 32 | 0.00497 | 0.01860 / 0.00245 | 1.48234 | 0.03171 | 0.06766 | 4.1582082637736955 |
| proposal_control | 8 | 0.00685 | 0.02774 / 0.00414 | 1.40163 | 0.02610 | 0.06729 | 1.9080319504235048 |
| proposal_control | 32 | 0.00609 | 0.02352 / 0.00338 | 1.46881 | 0.02689 | 0.05759 | 1.9054857994145338 |
| residual_clean_s25 | 8 | 0.00462 | 0.01666 / 0.00254 | 1.39204 | 0.03096 | 0.08049 | 0.6987574954162421 |
| residual_clean_s25 | 32 | 0.00442 | 0.01657 / 0.00247 | 1.46339 | 0.03055 | 0.08004 | 0.6996806658570878 |
| shared_full | 8 | 0.02129 | 0.05613 / 0.01643 | 1.34123 | 0.13206 | 0.02129 | not run |
| shared_full | 32 | 0.01005 | 0.02636 / 0.00835 | 1.45688 | 0.11395 | 0.01005 | not run |
| shared_s25 | 8 | 0.00630 | 0.02557 / 0.00457 | 1.39724 | 0.04584 | 0.00630 | not run |
| shared_s25 | 32 | 0.00579 | 0.02038 / 0.00280 | 1.44991 | 0.04354 | 0.00579 | not run |
