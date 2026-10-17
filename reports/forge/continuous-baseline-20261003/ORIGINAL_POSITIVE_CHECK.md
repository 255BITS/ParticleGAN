# Original Atlas intensity positive: independent check

**PASS — the full 600-update original intensity protocol reproduces its historical trajectory and complete checkpoint exactly.** The primary noisy result first passes at update225, confirms five consecutive checks at325, and finishes with16 passing observations. Clean and EMA observations remain separate diagnostics. The [compact receipt](ORIGINAL_POSITIVE_CHECK.json) records input/source/media identities.

The24 observations at25,50,…,600 are identical in every key, type and value after removing **only each observation's explicit `seconds` field**. This includes primary, clean and EMA metrics, learning rates and complete recorded controller diagnostics. All600 rows of `rates.jsonl` match without excluding anything. The392,235-byte final checkpoint is byte-identical, SHA256 `21b8942b29a1fc2c52980a760bbabc0e226dc6c5e6c993cc6e38eb9a0a523469`; it retains models, optimizer/controller/policy state, prior, named streams, CPU/CUDA RNG and the600-update cursor.

The original config SHA256 `a3ee5c67ac6594014feeb1ec333131abb4b1d86832510b69923100ebd8510ad4`, complete resolved recipe and effective options match. The host uses32 particles, latent width8, batch32 and seed0, with batch-feature-zero initialization, indexed generation and serialized backward execution. Both attempts report zero strict-stream deviations. The mathematical RA15 caller/fixture hashes are unchanged; the new wrapper relocates its dependencies into Forge's verified immutable source snapshot.

The primary image observer reads **`trainer.G` and `trainer.prior`**, rather than `ema=True` or an explicit `served_model()` call. It evaluates every ordered prior row through indexed `_generate`, with latent perturbation disabled and a fixed isolated evaluation generator. The learned output-noise scale starts at.029. Public G/prior references may already hold policy-applied served parameters; this checkpoint records `served_source='fast'`, the reference kNN backend and generator-noise factor1. The role label live does not imply globally unselected weights. This reproduction does not substitute a new public served-sampler or stronger policy-hold question for the original observer.

| Identity | Historical | New baseline |
| --- | --- | --- |
| Source commit | `fa2c378d5ea4dd8f66eabeb17973818a174732ef` | `a0d6d89fb470f551b3f790016a237c40a377e1e8` |
| Package SHA256 | `deab9ad5f04600fd37cd0fbf0a47442cda7fbdd6b73539e49fd8d5880e52662a` | `5a63765e278f9a7c911b946777fffbb6b7b97c2e84700730dae0af4c2df10f3b` |
| Physical GPU |0 |1 |
| Model/runtime | RTX A6000; Torch2.13.0+cu126; CUDA12.6 | Same |
| Paid/execution wall |31.0400621891s historical execution |28.7503506038s current supervised paid cost |

Package changes comprise `k3p.py`, `policy.py`, `recipes.py`, `training.py`, plus added `recipe_schedules.py`;25 package source files are unchanged. The receipt binds both versions of each changed file. Exact agreement here establishes reproducibility for this observed protocol. It makes no claim that all API behavior or algorithms are equivalent, and these two wall measurements are not a speed benchmark.

The actual metric GIF has9 distinct frames at updates25,75,150,225,300,375,450,525,600, 800×550 pixels,80,846 bytes, SHA256 `5e9d0919d530f4881de2b8758be2b4217a29b96e3ab7dd1205b8eb617955b426`. It uses retained training metrics, fixed axes and the exact original bounds `modes>=2` and `HQ>=.9`; there are no generated-image frames or new draws. Frames0/4/8 were inspected. The middle/final frames clearly show crossing and sustained success. Frame0 has a single observation without a point marker, so that initial point is not visible. The original full-test PASS badge is explicitly retrospective and each frame states its actual update.

Raw evidence remains at `/ml2/hypergan/forge-continuous-leaderboard-20261003/baseline-atlas19/portability/img_intensity2`; historical evidence remains at `/ml2/hypergan/gan-attempts/develop-gates-20261001/atlas-original19/portability/img_intensity2`. External QA posters and the verification receipt are at `/ml2/hypergan/toy-original-atlas-positive-check-20261003`. Verification changed none of those original files and ran no training, model forward, scorer or sample draw.

This is one positive among the declared19 original diagnostic tests. It grants no current Forge MoG, ordinary clean-policy, whole-family winner or default-adoption credit. Other rows retain their own pending or measured statuses.
