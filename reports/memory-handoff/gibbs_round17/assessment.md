# Round17 assessment: inner refinement settles, outer memory still drifts

Six 2000-update scouts completed through the durable queue on both GPUs. All diagnostics completed, zero failures. Queue wall time14.7 minutes; training27.8 GPU-minutes. No scout meets the unchanged extension gates, so no longer runs were launched. All full cold/warm circle passes remain0/128 at256 and1024 steps. Saved round12 remains the winner.

See comparison.md for the full leaderboard, gibbs_comparison.md for same-event diagnostics, design_review.md for formulation limits, and extension_decision.json for exact gates.

## Outcome

Minimum warm Q over prefixes8/32 (not a success probability):

|Model|Q|Change against matched architecture|
|---|---:|---:|
|Saved original2k|.010901|reference|
|One decode, joint weight.25|.007228|+21.7%|
|One decode, architecture only|.005940|control|
|One decode, joint weight.10|.005587|-5.9%|
|Three decodes, joint weight.25|.005056|+16.1%|
|Three decodes, architecture only|.004353|control|
|Three decodes, joint weight.10|.003542|-18.6%|

The stronger joint loss improves its matched architecture at both depths. However, the best new model remains33.7% below the original2k reference. Its prefix32 radial error improves slightly (.891 versus .945) but late Q falls sharply (.005794 versus .010869). This is not a net continuation improvement. More inner decoding costs more and performs worse here.

## What the diagnostics establish

The new implementation compares actual opposing joints: `(E(M,x_real,t),x_real)` and `(h_producer,P(z,M,h_producer,t))`. Real inference receives cooperative gradients; K sees M/clock, not particle. All inner iterations hold M/z/time fixed and perform no writes. The temporary latent resets each physical step. This corrects the post-write/post-write pairing in rounds15/16.

Three-decode models settle rapidly on the tested real-prefix contexts. For joint.10, same-event next-point error is .17351 with one decode, .00555 with two, .00521 with three, and .00521 with seven. Its architecture-only control also settles by three (.00550 versus .00551 with seven). The decoder/encoder architecture therefore learns a useful local refinement internally even without K. This is limited evidence of local settling, not a global contraction or stationary-sampling proof. Extra iterations do not fix temporal continuation.

Memory is still essential to the local read: best new point error .00473 rises to .74787 with zero M and1.45356 with shuffled M. The extra machinery did not simply replace memory with clock. Yet every scout's tested radius/speed information is near chance after128 autonomous writes. Best new clean R2 .603/.940 becomes .108/.321 at32 and -.007/-.002 at128. Finite probe failure does not establish information-theoretic erasure.

One-decode joint models have weak sensitivity to swapping producer latents: output-change error .000091 for weight.25 versus .000941 for its architecture control. Zeroing the latent still changes outputs (.005783); a near-constant learned contribution is possible. Three-decode models are more sensitive to shuffled latent (.023-.026). Do not call all latents collapsed or ignored. The distinct latent contributions still do not preserve the outer process.

The best new model's clean next-point error (.00473) is close to original2k (.00467), despite much worse long continuation. After one generated write, next-read error is also worse (.01596 versus .01496). Thus single-step prediction or fixed-context consistency is not an adequate selection criterion. Process perturbation diagnostics show no new scout overtaking the saved2k across the tested radius/speed/direction variants.

## Recommendation

Keep the original winner. Do not extend these models or increase inner iteration count as the next default sweep. Separate two questions in future experiments:

1. If pursuing faithful Gibbs-style sampling, test a decoder whose particle enters only through the initialized latent, with a matched control. Currently the persistent direct particle path makes `(z,h)` the true inner state; K matches only `(h,x)`. Removing the bypass would test that mismatch but may constrain useful particle capacity. This is not yet implemented or queued, and a deterministic chain still lacks the paper's stochastic mixing assumptions.
2. For the actual circle objective, prioritize the outer one-write transition. The new joint objective sees only clean real-prefix memories, while the runtime repeatedly changes M. First diagnose the same inner refinement under a single generated-write memory versus its paired real-write memory, at identical clock/particle. If repair fails specifically off the real-memory distribution, a matched scout can expose the joint game to those bounded one-write contexts without increasing temporal rollout depth. Prior writer/read-space matching failures remain relevant: this needs a specific support/repair hypothesis, not another undirected weight sweep.

No MSE training, additional temporal rollout, clipping, EMA, private persistent G memory, or expert runtime was introduced. Existing configs remain default-off. 112 focused tests, both GPU smokes, completed-checkpoint diagnostic smokes, exact evaluation-panel audit, source archive audit, and diff whitespace checks passed. Changes are uncommitted and unpushed; no jobs remain running or queued. Stable log remains runs/memory_path/core_round1/train.log.
