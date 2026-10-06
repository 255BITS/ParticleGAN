The retained neighborhood shows active learning at the failed834 observation. At updates833/834/835, all six generator parameter tensors change after backward;97/106/103 prior rows change, gradients are nonzero, and generator/prior Adam steps advance. These are aggregate live writes after optimizer, hot-row and other normal policy work; they do not isolate each contribution.

| Update | Changed prior rows | G update L2 | Prior update L2 | G gradient L2 | Prior gradient L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
|833|97|.01464693|.04781638|.00639263|.00044123|
|834|106|.01077722|.03071249|.00759533|.00019197|
|835|103|.00711261|.02323029|.00306434|.00021133|

Birth/death evaluates at834 (416→417), with last.step834, zero matched/moves, and cumulative matched/moves unchanged at23. There is no recorded move at833–835. Fixed MoG sigma stays.025; recorded training output sigma stays.029. DV12 bandwidth varies smoothly, from[.068142496,.061603017] before834 to[.068140537,.061602429] after it. Installed serving averages are released: `_fast=None`. Scoring uses live MoG+DV12+G, disables only additive output noise, and retains both latent perturbations.

Across post_update834→observation834, all nine checked live-law tensor leaves are byte-identical (six G tensors, prior table, prior sigma, DV12 bandwidth). The named eval/live/samples CUDA RNG advances. Thus the neighborhood preserves the **pre-read evaluation RNG** at834 even though the scheduled scored snapshot is after the draw. Realized indices and MoG/DV12 noise arrays are absent; they may be recoverable only through a separately authorized verified replay.

| Original scored update | Bank mean | Bank std | Std / target std | CDF KS | Original all-gate result |
| --- | ---: | ---: | ---: | ---: | --- |
|792|2.040351|.495168|.990335|.047275|PASS|
|834|1.980616|.439547|.879095|.051701|FAIL: KS|
|875|2.017471|.529228|1.058456|.041228|PASS|

The834 scored bank is narrower and crosses KS.05 while its mean and width gates pass. Parameter writes, gradients, aliases and normal update counters are present; the monitored radius correction has no difference and the local birth/death event has no move. The evidence supports a **late-fit / sustained-criterion sensitivity finding**, with live network/prior update stability as a mechanism candidate. It does not prove that finite-bank variation alone caused the spike or that a particular parameter update caused the population distribution to narrow. The identical791/844 banks are deterministic paired outputs, not independent repeated evidence. The full population CDF distance and unproven callbacks remain UNKNOWN. The original24-read/final-five FAIL remains valid and unchanged.

The smallest next test is a **PLAN only**:

1. Bind a private frozen replay of the complete original sampling law, including Source, modes/buffers, aliases, fixed MoG sigma, DV12 geometry and backend. Reconstruct the already retained834 bank from the post-update834 pre-read RNG. Require all4096 output values and the final eval RNG to match the existing retained evidence, otherwise stop UNKNOWN. Record the realized indices and both latent base-noise arrays.
2. Reuse those same variates at frozen post-update833 and835. Three4096-output banks total including the exact834 replay, zero training or seed selection. This separates local state effects on matched inputs and could justify a correction to late update stability. It does not by itself decide population KS against.05; a separately bounded estimator or integration would still be needed.

No replay, new model forward, draw, training update or regrade was performed for this diagnosis. No corrective knob is selected from unproved causality, and no gate is relaxed. [NEIGHBORHOOD_DIAGNOSIS.json](NEIGHBORHOOD_DIAGNOSIS.json) contains exact measured tensor summaries, counters, controller values and context byte pins. Source-only reviews supplied interpretation of sampling and update ordering; empirical observations came from ROOT's paid retained analysis.
