# Round17 diagnostic interpretation

The opposing-joint mechanism was exercised, but settling the inner inference/generation loop did not preserve the outer trajectory. Keep the prior baseline. The best new model, `gibbs1_joint25`, improves minimum warm Q over its architecture control (.00723 versus .00594), but remains below the existing 2k baseline (.01090); all full-circle pass counts remain zero.

## What the inner loop learned

At fixed clean prefix32 memory, particle and clock, the three-decoder models settle rapidly. Extending from three to seven decoder calls changes output by evaluation MSE approximately 6.4e-6 (`arch`), 5.3e-8 (`joint10`), and 4.4e-7 (`joint25`). Point error remains approximately .00550, .00521, and .00544 respectively. Extra iterations therefore offer no material local accuracy improvement on this panel. These are total decoder counts: one means no refinement.

This behavior is already present in the three-decoder architecture control without K. Its inference network is trained through the ordinary G losses, so local settling cannot be credited uniquely to opposing-joint matching. Conversely, stopping a three-decoder-trained model after one decode gives large error (.160–.203); it has learned to depend on refinement. That is an inference ablation of a model trained for three calls, not evidence that extra calls improve a matched one-call model.

Small producer/reencoded-latent discrepancies support approximate fixed-point behavior on these clean contexts. They do **not** establish a stationary distribution, mixing, global attraction, or stability when M and clock advance. We measured finitely many contexts and iterations, without varying latent initialization independently of the fixed particle.

## Latent use depends on the variant

In the one-decoder models, adding K reduces producer-latent variance from .151 (`arch`) to .0050/.0077 (`joint10`/`joint25`). Shuffling h while holding M, z and clock fixed changes output by .000941/.000306/.000091 MSE respectively. For `joint25`, that is about ten times smaller than its architecture control. This is consistent with weaker use of between-particle variation through h. It is **not proof of latent collapse**: zeroing h still changes output by .00578, h variance remains nonzero, and z also reaches the decoder directly. Coordinate scaling and compensating decoder weights complicate cross-model latent-variance comparisons.

The three-decoder models retain appreciable h dependence: shuffling h changes output by .0228–.0260 and worsens point error from roughly .005 to .027–.032. Thus the round does not support a blanket claim that G ignores the added latent. Zeroing/shuffling M is also highly disruptive locally for every model. These interventions establish dependence and local error effects, not long-run benefit over a separately trained model without the corresponding input.

## What K does and does not tell us

K ranks real over generated joints on 43–57% of the prefix32 examples. Near-chance separation is compatible with successful matching, weak discrimination, or an unresolved adversarial game; it cannot decide among them by itself.

The three-decoder K models prefer a valid real pair over another episode's pair only 23–25% of the time, and prefer matched over shuffled h for a real x only about 13%. These are counterfactual compatibility tests outside K's training comparisons. They show that K's score is not a reliable general compatibility energy on those interventions; they do not imply that the trained GAN objective was implemented backwards. Raw margins are uncalibrated across critics. The paired conditioning comparison removes anchor-only offsets but still does not measure process retention.

## The outer-loop failure remains

After128 generated writes, all six models' M-only nonlinear probes have radius/signed-speed R² approximately zero or negative, with direction accuracy48–52%. Adding z does not recover useful radius/speed R². Finite-probe failure is not proof of information-theoretic erasure. Nevertheless, it agrees with the completed process interventions: late output response to the original radius and speed is near zero. Clean-memory decoding remains similar across models, around .60 radius R² and .94 signed-speed R².

The one-write diagnostic also gives no local recovery gain over baseline: prefix32 next-read error after a generated write is .01596 for `gibbs1_joint25`, versus .01496 for baseline and .01461 for its architecture control. A better long-run Q for that scout therefore should not be explained as demonstrated improvement in this particular one-write forecast error.

## Recommended next investigation

1. Keep the baseline and do not extend this round solely because K reaches chance separation or h becomes self-consistent. The completion metrics and information retention do not justify it.
2. Diagnose the same inner loop after one generated write, with the matching next target and fixed next clock. Compare three versus seven calls there. This would test whether the clean-context settling result survives the actual local runtime disturbance, without training a trajectory.
3. Measure E's sensitivity to candidate x while holding M fixed, alongside G's sensitivity to h. Current results permit E to encode mostly context and the decoder to bypass h through direct M/z. A carefully matched bottleneck or shared conditional-read variant could test whether opposing joints become useful when their latent must carry candidate-specific information; stronger K weight alone is not established as the missing ingredient.
4. Prioritize the persistent-memory transition if those diagnostics also settle without recovery. K currently operates at detached real-prefix contexts; it does not train D's writer or guarantee preservation through the next write. Any next local joint experiment should explicitly state how it reaches explored memory and which writer gradients it adds, with a matched control.

This is a deterministic, conditional, fixed-particle adaptation of GibbsNet. It corrects the pairing issue from rounds15–16, but does not reproduce GibbsNet's stochastic unconditional chain. The result limits this adaptation; it does not invalidate the broader GibbsNet mechanism.

Sources: `gibbs.json`, `gibbs_comparison.md`, `comparison.md`, `information.json`, and `process.json` in this directory. All diagnostics were evaluated after training completed; regression errors are evaluation-only.
