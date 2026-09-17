# Round17 design review

Reviewed the frozen implementation and planned configurations without reading
unfinished experiment metrics. No additional implementation defect found.
The nine new focused tests passed before launch. This report concerns what the
experiment can establish, rather than predicting its outcome.

## What is faithful

The real pair is `(E(M,x_real,t), x_real)`; the fake pair is
`(h_producer, P(z,M,h_producer,t))`. The fake latent precedes decoding. Both
branches remain differentiable during the cooperative E/P update. Under the
scouts' default relativistic logistic objective, the minimized loss is
`softplus(K_real-K_fake)`: E's real branch is pushed toward lower real scores,
and the generated branch toward higher fake scores. This is the intended
opposing-joint adversarial direction, not an accidental real-branch detach.

Every inner iteration holds M, physical clock, and particle fixed. The same
refinement generator is used by existing training losses and runtime. The
inner chain performs no D-memory writes; its final sample alone enters the
ordinary runtime writer. Previous inner iterations are sampled without a
gradient graph; the last inference/decode retains input-memory and particle
derivatives. The joint critic trains on cached, detached context and cannot
update the persistent writer.

## Important limits

1. **The inner chain and the physical trajectory are different transitions.**
   This round trains compatibility while history and time are fixed. Runtime
   subsequently changes both through `M_next=W(M,x)` and the next clock tick,
   then resets the ephemeral latent. Even perfect fixed-context joint matching
   would not establish stability of that outer transition or preservation of
   the original process after hundreds of writes. Improvement there would be
   an empirical benefit, not a consequence already supplied by this objective.

2. **The persistent particle is an additional chain state.** P receives z
   directly at every refinement; E does not, and K judges only `(h,x)` given
   M/time. The actual inner transition has state `(z,h)`. Once h depends on z,
   matching the marginal `(h,x)` need not preserve the hidden `(z,h)` coupling.
   Thus one cannot automatically regard the observed h/x alternation as a
   closed Gibbs sampler with decoder conditional `p(x|h,M,t)`. K deliberately
   omits z because real observations were paired independently with particles.
   Simply adding z to K would change the task to per-particle conditional data
   matching and is not an automatic repair.

3. **Determinism and support differ from the stationary-chain argument.**
   Given M/time/z the new chain is deterministic. Tanh bounds h but provides
   neither exploration nor contraction. At fixed context the learned table
   supplies at most 512 deterministic initial candidates, while observed noise
   is continuous. Finite critics can still supply useful gradients, but exact
   support matching and mixing should not be presumed. The paper's formal
   proposition assumes noisy transitions with sufficient reachability and an
   ideally optimized adversarial game. It also reports deterministic-decoder
   experiments, so lack of fresh noise is a limitation of the formal guarantee,
   not evidence that this scout cannot work. [GibbsNet §2.1 and Figure3](https://arxiv.org/html/1712.04120)

4. **The latent can be redundant.** M/time and the direct z path already reach
   P. E can learn mostly context-derived h, or P can largely ignore h, without
   necessarily harming the point GAN. Joint critic confusion alone therefore
   cannot establish meaningful inference. Evaluate latent interventions and
   reconstruction/refinement behavior alongside the existing memory and process
   metrics; intervention sensitivity alone is not proof of benefit.

5. **One versus three decodes changes more than compute.** With one decode,
   the particle-to-h seed projection learns and E appears only in the real
   joint. With three, E participates in real and generated branches, while the
   seed projection remains initialized because warmup is detached. The same
   learned particle still receives direct gradients through final P. Compare
   loss weights against the architecture control at each depth first; a
   cross-depth difference cannot be attributed solely to chain length.

6. **Local support and optimization drift remain.** Joint examples use clean
   real-prefix memories at positions >=4, not cold or generated contexts.
   Existing feedback and pair losses still cover their original bounded
   branches, but the new joint objective does not expand that coverage. K is
   updated before D; the subsequent E/P phase re-encodes with updated D. This
   avoids stale stored memory yet leaves the previously identified moving
   representation question unresolved. It is an optimization choice, not a
   newly discovered autograd bug.

## Interpretation rule

Call this an opposing-joint, conditional ParticleGAN refinement experiment.
The current default loss correctly trains both E branches; a future switch to
the public API's vanilla G loss would omit the real score and would require a
different explicit encoder objective. Do not claim arbitrary recipe modes
preserve the same game.

Promising evidence would combine a gain over the matched architecture control,
better autonomous continuation, and retained process information. Improvement
only with additional fixed-context inference steps would support an inner
refinement mechanism, while leaving outer memory stability unsolved. Failure
would not refute GibbsNet or local adversarial dynamics generally.
