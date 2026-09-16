# Frozen-information comparison: first four completed round14 scouts

**The stronger future objective improves clean-state information, but does not preserve it through repeated generated writes.** This is an interim comparison containing only the four completed scouts and the two saved baseline checkpoints. The pending gradient-control scout is excluded.

Identical protocol to `diagnosis.md`: frozen GAN, fresh disjoint2048/512/1024 probe splits, prefix32, validation-selected250-step64–64 SiLU MLP, train-only normalization. Split hashes match exactly. The numbers below use M-only probes fitted separately for each domain and depth. No test-data tuning or training objectives were added.

| Checkpoint | Clean radius R² | Clean speed R² | Generated8 radius R² | Generated8 speed R² | Generated32 radius R² | Generated32 speed R² | Generated128 radius R² | Generated128 speed R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.612 | 0.942 | 0.341 | 0.876 | 0.127 | 0.267 | -0.007 | -0.007 |
| match_shuffle25_5k | 0.599 | 0.954 | 0.343 | 0.878 | 0.126 | 0.436 | -0.005 | -0.009 |
| future_clean10 | 0.615 | 0.949 | 0.391 | 0.859 | 0.170 | 0.371 | -0.003 | 0.013 |
| future_mixed10 | 0.616 | 0.949 | 0.375 | 0.851 | 0.085 | 0.273 | -0.005 | -0.004 |
| future_full10 | 0.613 | 0.949 | 0.367 | 0.856 | 0.122 | 0.292 | -0.011 | 0.002 |
| future_mixed25 | 0.685 | 0.965 | 0.384 | 0.874 | 0.112 | 0.190 | -0.002 | -0.001 |

## What changed

- `future_mixed25` has the strongest clean information: radius R².685 and signed-speed R².964, versus saved2k.612/.942. Teacher-forced depth128 remains.666/.964. Yet autonomous depth32 falls to.112/.190, worse than saved2k.127/.267, and128 returns to approximately chance. The objective can enrich real-history memory without solving persistence.
- `future_clean10` has the strongest generated32 signal among these scouts (.170 radius/.371 speed), compared with2k.127/.267. It still loses nearly all signal by128 (−.003/.013); this small residual is not sustained recovery, nor a significance claim. The saved5k baseline still has higher generated32 signed-speed R² (.436).
- `future_full10`, the best trajectory-Q of these scouts, has generated32 information near the2k baseline (.122/.292). Its generated128 state is approximately chance. Better decodability and better path quality need not select the same model.
- `future_mixed10` is likewise not a retention win. M+z probes do not rescue128 for any of the four scouts.

## Generated32 direction and128 control

| Checkpoint | Generated32 direction | Generated128 direction | Real128 radius R² | Real128 speed R² |
|---|---:|---:|---:|---:|
| match_shuffle25 | 68.3% | 49.9% | 0.591 | 0.942 |
| match_shuffle25_5k | 74.6% | 47.2% | 0.574 | 0.946 |
| future_clean10 | 71.4% | 52.3% | 0.600 | 0.949 |
| future_mixed10 | 63.3% | 48.6% | 0.602 | 0.946 |
| future_full10 | 66.9% | 50.9% | 0.580 | 0.944 |
| future_mixed25 | 60.6% | 51.3% | 0.666 | 0.964 |

## Representation shift persists

| Checkpoint | Real→generated32 radius R² | Generated-trained32 radius R² | Real→generated32 speed R² | Generated-trained32 speed R² |
|---|---:|---:|---:|---:|
| match_shuffle25 | -0.818 | 0.127 | 0.015 | 0.267 |
| match_shuffle25_5k | -0.555 | 0.126 | 0.266 | 0.436 |
| future_clean10 | -4.652 | 0.170 | -0.231 | 0.371 |
| future_mixed10 | -0.823 | 0.085 | 0.045 | 0.273 |
| future_full10 | -0.643 | 0.122 | 0.135 | 0.292 |
| future_mixed25 | -1.467 | 0.112 | -0.012 | 0.190 |

Domain-specific probes substantially outperform transferred real-state probes, especially for radius. Thus drift includes a change in representation as well as a loss of information accessible to these probes. The low128 scores cannot be explained solely by using a clean-trained readout; these probes also train on generated states at exactly128.

## Bounds and recommendation

Do not extend a scout based on these diagnostics alone. Continue using the preregistered trajectory-quality gates. The current results support the narrower diagnosis that preserving future recognizability at a bounded local state is not yet enough to preserve original-process meaning under repeated feedback. Compare the completed gradient-control scout before assigning credit to the new generated-write gradient.

Probe capacity is deliberately modest; failed decoding does not prove information-theoretic erasure. Signed-speed R² includes direction separation. Radius calibration is partial, the saved particles are reused, and all claims apply to the sampled in-support processes. There are no independent training-seed replications.

All four diagnostic runs completed in19.15s on cuda:1. Full errors, ridge/M+z controls, and validation choices: `scout_information_first4.json`; completed log: `scout_information_first4.log`.
