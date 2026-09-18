# Hypothesis: generator update balance

Existing D learns held-out separation with G fixed, but D warmup worsens joint FID and weaker bcap yields a small, reversing difference. Eight-step traces show most immediate adversarial ranking changes come from G, with little from particles. This is expected adversarial behavior, not proof of the root cause.

Test G-only LR 0.0003 -> 0.00015, unchanged D/E/prior rates, original bcap coefficient 1 every 8, one D update. Two GPUs, same original CNN E-only 10k parent and full Adam/EMA/RNG. Measure FID50k at 15k and 20k with the existing protocol. Compare sustained trajectory and runtime against a concurrent unchanged control. No seed experiments. No automatic long promotion for a small or reversing difference.

A new standalone trainer preserves historical source certificates. Tests check exact unchanged full-state continuation and that the first half-G update changes only G (and EMA G), with identical D/E/prior parameters and all Adam moments. Actual-parent eight-step smokes audit counters, LR groups, source certificates, and matched RNG consumption.

If lower G LR materially improves FID, extend a matched continuation to test persistence. If it fails, distinguish poor feedback direction from support/coverage constraints; D AUC alone cannot adjudicate. Do not freeze E or particles on the current evidence.
