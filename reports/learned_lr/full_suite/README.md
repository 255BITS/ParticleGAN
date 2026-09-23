# Learned LR adapter — held-out behavioral comparison

One frozen policy; all nine hosts are held out from policy fitting. Live weights determine PASS. The ten shared checks are reported separately and do not rank controllers. `time_only` keeps the learned bias/progress coefficients and suppresses feedback features.

| Controller | Live toys | Bounds | Sustained toys | Ring modes / HQ | Ring confirmed | Total seconds | Controller seconds | Overall |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| constant | 9/9 | 29/29 | 8/9 | 8/8 / 100.00% | — | 22.20 | 0.26 | PASS |
| cosine | 9/9 | 29/29 | 9/9 | 8/8 / 100.00% | 1050 | 21.27 | 0.29 | PASS |
| learned | 8/9 | 27/29 | 7/9 | 4/8 / 66.94% | — | 21.64 | 0.54 | FAIL |
| time_only | 8/9 | 27/29 | 6/9 | 2/8 / 33.76% | — | 21.25 | 0.53 | FAIL |

Sustained success requires a complete 24-point curve and at least five final passing observations. The ring additionally requires 8/8 modes and HQ≥90%. No incomplete row can pass. Times are single observations, not replicated speed estimates.

| Controller | two_pole | trajectory | residual_student | unipolar | ae_gan_hold | cover_leftover | unused_token_hold | mid_scale_identity | mode_hold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| constant | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| cosine | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| learned | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |
| time_only | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | FAIL |

EMA remains separate:

| Controller / toy | EMA metrics |
| --- | --- |
| constant / cover_leftover | `{"content_kept": 0.9664053090988606, "leak_ratio": 6.082552556363011e-05, "pole_rel_err_minus": 0.04403576999902725, "pole_rel_err_plus": 0.04354649409651756, "same_dir": 0.0003605171515608416, "u_kept": 0.9365585352701874}` |
| constant / mode_hold | `{"hq": 1.0, "modes": 8}` |
| cosine / cover_leftover | `{"content_kept": 0.9638103396566842, "leak_ratio": 8.817618138984354e-05, "pole_rel_err_minus": 0.046218857169151306, "pole_rel_err_plus": 0.044452060014009476, "same_dir": 0.0015068320637131411, "u_kept": 0.9355629588984599}` |
| cosine / mode_hold | `{"hq": 1.0, "modes": 8}` |
| learned / cover_leftover | `{"content_kept": 0.9584371435859993, "leak_ratio": 1.747224523363027e-05, "pole_rel_err_minus": 0.05315807834267616, "pole_rel_err_plus": 0.05224351957440376, "same_dir": 0.000703035990326839, "u_kept": 0.9240931776143845}` |
| learned / mode_hold | `{"hq": 0.66357421875, "modes": 4}` |
| time_only / cover_leftover | `{"content_kept": 0.9604454949621702, "leak_ratio": 8.897241699073526e-05, "pole_rel_err_minus": 0.058265265077352524, "pole_rel_err_plus": 0.0579160638153553, "same_dir": 0.00028401264674672436, "u_kept": 0.9148535636969234}` |
| time_only / mode_hold | `{"hq": 1.0, "modes": 8}` |

Shared checks: 10/10 PASS. Raw bounds, curves, errors, action traces, source hashes and the exact frozen policy are in [results.json](results.json).
