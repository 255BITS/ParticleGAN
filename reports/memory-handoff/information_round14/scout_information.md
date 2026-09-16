# Frozen-information comparison: all five completed round14 scouts

**The future objective can improve information encoded from real histories, but none of these scouts preserves that information through sustained autonomous feedback.** All five completed scouts are compared against the saved2k and5k references. No additional GAN training was performed for this diagnostic.

Protocol matches `diagnosis.md`: frozen GAN; fresh disjoint2048/512/1024 probe splits; prefix32; validation-selected250-step64–64 SiLU MLP; train-only normalization. M-only probes below are fitted separately for each domain and depth. Split hashes match both references exactly. The two scout batches have exactly matching protocol and diagnostic-source hashes. Full results are merged in `scout_information.json`, retaining source batch artifacts.

| Checkpoint | Clean radius R² | Clean speed R² | Generated8 radius R² | Generated8 speed R² | Generated32 radius R² | Generated32 speed R² | Generated128 radius R² | Generated128 speed R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| match_shuffle25 | 0.612 | 0.942 | 0.341 | 0.876 | 0.127 | 0.267 | -0.007 | -0.007 |
| match_shuffle25_5k | 0.599 | 0.954 | 0.343 | 0.878 | 0.126 | 0.436 | -0.005 | -0.009 |
| future_clean10 | 0.615 | 0.949 | 0.391 | 0.859 | 0.170 | 0.371 | -0.003 | 0.013 |
| future_mixed10 | 0.616 | 0.949 | 0.375 | 0.851 | 0.085 | 0.273 | -0.005 | -0.004 |
| future_full10 | 0.613 | 0.949 | 0.367 | 0.856 | 0.122 | 0.292 | -0.011 | 0.002 |
| future_mixed25 | 0.685 | 0.965 | 0.384 | 0.874 | 0.112 | 0.190 | -0.002 | -0.001 |
| future_mixed10_detachwrite | 0.553 | 0.934 | 0.346 | 0.760 | 0.066 | 0.278 | -0.009 | -0.006 |

## Main findings

- `future_mixed25` enriches clean memory: radius/speed R².685/.965 versus saved2k.612/.942. Teacher-forced depth128 remains.666/.964. But generated32 falls to.112/.190 versus saved2k.127/.267, and generated128 is approximately chance. Stronger future recognition does not automatically produce durable process memory.
- `future_clean10` has the best generated32 decodability among new scouts (.170/.371). It loses nearly all signal by128 (−.003/.013), and saved5k retains higher generated32 speed R² (.436). Its small128 residual is not evidence of solved retention or statistical significance.
- `future_full10`, the strongest new scout on trajectory-Q, has generated32 decodability near saved2k (.122/.292). Better probe accuracy and better trajectory quality need not select the same model.
- All new M+z probes remain near chance at128. A simple particle-entangled readout does not recover the missing predictions.

## Generated-write gradient control

`future_mixed10_detachwrite` removes future-ranking writer gradients from the generated context while retaining its head gradients, the clean future-context writer gradients, and the existing next-point mismatch objective. This is the matched generated-future-context gradient control; it does not disable all writer training.

| Metric | mixed10 (connected) | mixed10_detachwrite |
|---|---:|---:|
| Clean radius R² | .616 | .553 |
| Clean signed-speed R² | .949 | .934 |
| Generated8 radius R² | .375 | .346 |
| Generated8 signed-speed R² | .851 | .760 |
| Generated32 radius R² | .085 | .066 |
| Generated32 signed-speed R² | .273 | .278 |
| Generated32 direction | 63.3% | 69.1% |
| Generated128 direction | 48.6% | 47.5% |

The connected gradient improves clean and early generated-state decodability in this matched run. It does not provide a consistent32-write advantage: radius is higher, signed-speed R² is nearly equal, and direction is lower. Both models lose decodable information by128. This supports a useful local effect without establishing sustained memory preservation. Because all weights co-adapt during training, the result does not isolate an immutable writer-only mechanism.

## Generated32 direction and teacher-forced128 calibration

| Checkpoint | Generated32 direction | Generated128 direction | Real128 radius R² | Real128 speed R² |
|---|---:|---:|---:|---:|
| match_shuffle25 | 68.3% | 49.9% | 0.591 | 0.942 |
| match_shuffle25_5k | 74.6% | 47.2% | 0.574 | 0.946 |
| future_clean10 | 71.4% | 52.3% | 0.600 | 0.949 |
| future_mixed10 | 63.3% | 48.6% | 0.602 | 0.946 |
| future_full10 | 66.9% | 50.9% | 0.580 | 0.944 |
| future_mixed25 | 60.6% | 51.3% | 0.666 | 0.964 |
| future_mixed10_detachwrite | 69.1% | 47.5% | 0.542 | 0.932 |

## Representation shift remains

| Checkpoint | Real→generated32 radius R² | Generated-trained32 radius R² | Real→generated32 speed R² | Generated-trained32 speed R² |
|---|---:|---:|---:|---:|
| match_shuffle25 | -0.818 | 0.127 | 0.015 | 0.267 |
| match_shuffle25_5k | -0.555 | 0.126 | 0.266 | 0.436 |
| future_clean10 | -4.652 | 0.170 | -0.231 | 0.371 |
| future_mixed10 | -0.823 | 0.085 | 0.045 | 0.273 |
| future_full10 | -0.643 | 0.122 | 0.135 | 0.292 |
| future_mixed25 | -1.467 | 0.112 | -0.012 | 0.190 |
| future_mixed10_detachwrite | -2.575 | 0.066 | 0.057 | 0.278 |

Domain-specific probes recover more than transferred real-state probes, particularly for radius. Thus representational change remains part of the problem. Yet even probes fitted on the actual generated-state distribution become near-chance at128: a mismatch between clean-trained probe and generated representation is not the full explanation.

## Implication and limits

Keep the existing trajectory-based extension gates. These diagnostics do not justify extending a rejected scout. They narrow the remaining problem: local future recognition can make M informative on real histories, while the repeated G/read→D/write loop still loses accessible information about the original process. A next intervention should address preservation under that repeated mapping or a demonstrably accessible short-memory representation, rather than merely increasing the same future-loss weight.

Modest probe capacity means failed decoding is not proof of information-theoretic erasure. Radius calibration is partial; signed-speed R² includes direction separation. These use held-out histories but reused saved particles, in-support process parameters, and one diagnostic panel. Histories and clocks beyond63 exceed original training support, though degradation is already clear by8 and32. No independent GAN seed repeats or significance claims.

All five new checkpoint diagnoses completed in25.06s total on cuda:1. Full ridge/M+z controls, MAEs, selections, and hashes: `scout_information.json`; source batches/logs: `scout_information_first4.*` and `scout_information_control.*`.
