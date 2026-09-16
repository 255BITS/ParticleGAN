# Frozen-memory information diagnosis

**Result:** original-process information is increasingly difficult to decode from autonomous memory, while matched teacher-forced memory remains informative. Early representation shift is also present. The two saved winners agree on this pattern.

This favors testing an incentive for retaining process identity through generated writes. A pure G-readout failure is insufficient to explain these probe results, but limited probe failure does not prove that information is absent.

## Protocol

- Frozen saved 2k and 5k `match_shuffle25` G/D/prior. No GAN weights, particles, objectives, or checkpoints changed.
- Fresh, independent 2,048 probe-training / 512 validation / 1,024 test episodes. Disjoint histories; the same diagnostic panels for both checkpoints. This is not a GAN seed sweep.
- Prefix32. States recorded after 0, 1, 8, 32, 128 writes. Autonomous branch writes only generated samples after the prefix; teacher control writes the corresponding noisy real observations. Each episode retains one fixed independently sampled saved particle; particles themselves are not held out.
- All centers, radii, phases, speeds, and observation noise follow the training distribution. Teacher histories beyond63 and generated clock times beyond63 exceed original training length/time support; the degradation already appears at8 and32.
- Linear ridge on M and 64–64 SiLU MLP on M or M+z. Training-only feature/target standardization. Ridge alpha and MLP checkpoint selected by validation standardized error; test used only after selection. MLP250 full-batch steps, AdamW lr.003.
- Separate per-depth/domain fits measure accessible information. Applying real-trained probes to generated memory additionally tests representation compatibility. Regression is evaluation-only and is never added to GAN training.

## Held-out nonlinear probe on M

R² uses the test-mean baseline (zero means no improvement). These are process-label predictions from M, not generated trajectory quality.

| Checkpoint | Generated writes | Real radius R² | Generated radius R² | Real signed-speed R² | Generated signed-speed R² | Generated direction accuracy |
|---|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0 | 0.612 | 0.612 | 0.942 | 0.942 | 99.2% |
| match_shuffle25 | 1 | 0.599 | 0.555 | 0.944 | 0.933 | 99.2% |
| match_shuffle25 | 8 | 0.551 | 0.341 | 0.947 | 0.876 | 96.2% |
| match_shuffle25 | 32 | 0.590 | 0.127 | 0.948 | 0.267 | 68.3% |
| match_shuffle25 | 128 | 0.591 | -0.007 | 0.942 | -0.007 | 49.9% |
| match_shuffle25_5k | 0 | 0.599 | 0.599 | 0.954 | 0.954 | 99.7% |
| match_shuffle25_5k | 1 | 0.575 | 0.540 | 0.953 | 0.945 | 99.4% |
| match_shuffle25_5k | 8 | 0.541 | 0.343 | 0.952 | 0.878 | 95.8% |
| match_shuffle25_5k | 32 | 0.569 | 0.126 | 0.953 | 0.436 | 74.6% |
| match_shuffle25_5k | 128 | 0.574 | -0.005 | 0.946 | -0.009 | 47.2% |

## Representation compatibility versus decodability

| Checkpoint | Writes | Real→generated radius R² | Generated-trained radius R² | Real→generated speed R² | Generated-trained speed R² |
|---|---:|---:|---:|---:|---:|
| match_shuffle25 | 8 | 0.038 | 0.341 | 0.569 | 0.876 |
| match_shuffle25 | 32 | -0.818 | 0.127 | 0.015 | 0.267 |
| match_shuffle25 | 128 | -4.014 | -0.007 | -0.411 | -0.007 |
| match_shuffle25_5k | 8 | 0.075 | 0.343 | 0.835 | 0.878 |
| match_shuffle25_5k | 32 | -0.555 | 0.126 | 0.266 | 0.436 |
| match_shuffle25_5k | 128 | -0.393 | -0.005 | -0.174 | -0.009 |

By8–32 writes, domain-specific probes recover more than real-trained probes: part of the change is representational. However, even domain-specific probes lose substantial predictive power, reaching chance-level performance by128.

## Controls and calibration

- Oracle-label ridge calibration gives radius/speed R² > .999999999 and100% direction for both checkpoints.
- z-only and shuffled-M nonlinear controls have negative held-out R² and roughly49–52% direction accuracy. Thus the useful early signal depends on the correct history rather than particle identity or accidental label leakage.
- Teacher-forced M remains strongly informative for signed speed (R²≈.94–.95, direction≈99%) across depths. Radius calibration is weaker (R²≈.54–.61), so these probes are not a complete measure of stored information.
- M+z nonlinear probes do not rescue depth128: both radius and signed-speed R² remain negative, with direction≈48–49%. This checks a modest class of particle-entangled readouts; it does not exhaust possible interactions.
- Linear probes show the same loss pattern. Full MAE, speed-magnitude error, selected hyperparameters, source hashes and split hashes are in `diagnosis.json`.

## Interpretation and recommended scout direction

The current loop appears to progressively lose readily decodable original-process information, alongside a shift in how that information is represented. The near-chance128 result is stronger evidence than output insensitivity alone, because the probes are retrained specifically on generated-state distributions and held-out histories.

Prioritize a local, adversarial objective that makes one generated write retain evidence useful for identifying several future observations from the same real episode. Use bounded independent future queries, not generated rollouts or a memory-coordinate MSE target. Include matched controls for additional horizon conditioning/head capacity and clean versus generated-write exposure. Prefer a shared G-facing scoring route, because a separate predictor can preserve information that G ignores.

This diagnostic does not establish an information-theoretic loss, the exact point where process information disappears, or sufficiency of any proposed objective. Only two existing checkpoints and one diagnostic panel were tested; there are no repeated training seeds or significance claims.

## Validation and cost

- `tests/test_memory_information.py`:3 passed (split/support, ridge calibration and test isolation, causal autonomous states and fixed z).
- Complete diagnostic runtime: 10.15s on cuda:1, one CPU thread. Both models completed without errors.
- Source and checkpoint hashes are recorded; the completed stdout log is `diagnosis.log`.
