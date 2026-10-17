# RA10 original full-grid artifact audit

Artifact audit: PASS / VALID. Original quality: FAIL.

The frozen sources, fixtures, runtime, schedule and original gates agree. All34 observations, five terminal checks and100k holdout are retained. The original JSON predicates and final conjunction were independently reconstructed. No scoring, training, sample generation or PT interpretation occurred.

- Step6000: coverage False, fidelity False, combined False; failed original thresholds [{'metric': 'max_cov_eig_ratio', 'value': 1.8018687695808284, 'relation': '<=', 'bound': 1.7}, {'metric': 'acc_center_rms_sigma', 'value': 0.21259933766074254, 'relation': '<=', 'bound': 0.2}, {'metric': 'acc_radial_ks', 'value': 0.04091500650906532, 'relation': '<=', 'bound': 0.04}]
- Step6250: coverage False, fidelity False, combined False; failed original thresholds [{'metric': 'max_cov_eig_ratio', 'value': 1.7492155521656332, 'relation': '<=', 'bound': 1.7}, {'metric': 'acc_center_rms_sigma', 'value': 0.2084201572666415, 'relation': '<=', 'bound': 0.2}]
- Step6500: coverage False, fidelity False, combined False; failed original thresholds [{'metric': 'max_cov_eig_ratio', 'value': 1.7153523685975356, 'relation': '<=', 'bound': 1.7}, {'metric': 'acc_center_rms_sigma', 'value': 0.21024576025977268, 'relation': '<=', 'bound': 0.2}]
- Step6750: coverage False, fidelity False, combined False; failed original thresholds [{'metric': 'max_cov_eig_ratio', 'value': 1.8318375103332138, 'relation': '<=', 'bound': 1.7}, {'metric': 'acc_center_rms_sigma', 'value': 0.2049229549568468, 'relation': '<=', 'bound': 0.2}]
- Step7000: coverage False, fidelity True, combined False; failed original thresholds [{'metric': 'max_cov_eig_ratio', 'value': 1.7731222860597085, 'relation': '<=', 'bound': 1.7}]

Holdout coverage/fidelity: False/True. The holdout coverage bit is bound to the original saved-cloud scorer receipt.
