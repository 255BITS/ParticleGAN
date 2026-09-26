# API-C5 supervisor audit

Verdict: the frozen implementation has a material declaration mismatch. Its accepted critic reference is a hard copy of the accepted critic, not the declared EMA with decay .99. Preserve the recorded candidate and its results; they cannot qualify a repaired moving-EMA implementation.

This audit only read source and existing artifacts and loaded existing checkpoints onto CPU. It ran no training, GPU work, unit test, model invocation, or additional search lane. Neither the main checkout nor the candidate source was edited. Machine-readable evidence and checkpoint hashes are in `API-C5.json` beside this report.

## Blocking finding: accepted critic reference

`particlegan/training.py:226` temporarily freezes every critic parameter during the generator phase. At line 248 it calls `JointStage.after("g")`; trainability flags are restored only by the `finally` block at line 251. `particlegan/game_update.py:46` calls `resolve()` inside that interval, and line 61 calls `opt_d.anchor.update_()`. `particlegan/k3p.py:72–75` averages trainable parameters but copies frozen parameters directly. Consequently all critic parameters are copied into the reference at the accepted step.

This is confirmed without executing an update: at saved steps **1600, 1740, 1750, 1800 and 2400**, every one of the nine critic state tensors equals the corresponding EMA tensor exactly, with maximum absolute difference zero. These include eight parameter tensors and the fixed Fourier-frequency buffer. Controller counters still report moving-memory EMA updates, so checking counters alone misses the defect.

After initialization, the predictor penalty therefore sees a reference equal to its accepted base critic. The lookahead penalty sees that base reference while its critic parameters are displaced. The measured algorithm is a meaningful secant policy with an immediate base reference, but it is not the declared persistent moving-reference policy.

Action: annotate the immutable C5 run with this deviation. If .99 memory remains the intended design, commit the final reference after restoring original parameter trainability, and verify the actual equation `EMA_new = .99 * EMA_old + .01 * accepted_D` while preserving permanently frozen parameters. That repair is a new candidate and must earn its own evidence. Do not replace the archived source or inherit C5 quality claims.

## Verified scope

- The worker builds the actual public `get_recipe` and `GANTrainer` and calls `GANTrainer.step`. Every source ZIP member matches its declared SHA-256; the reviewed working source also matches. The archive SHA-256 is `e06621731dca330decb5b40e251be382452b349ad690dee0ce931ef05fcbeff3`.
- The evaluator alone knows stopping windows, target-change times and quality scores. `make_recipe()` ignores its evaluator-step argument. `continuous=True` returns learning-rate multipliers `(1, 1)`, bypasses finite-budget stopping, and uses absolute startup-noise counts 360 and 720. No evaluator horizon or target-change signal enters the learner.
- The predictor and correction replay the same real tensors, latent indices and stochastic streams. Optimizer/controller/EMA state, mutable model buffers, completed-step count, CPU RNG, CUDA RNG and trainer streams are restored before correction. The caller's `generator_real` callback is evaluated once outside the two stages.
- Restoring D after its temporary Adam step ensures the G field sees the same joint parameter state as the D field. The corrector calculates gradients at predicted parameters and applies its native Adam step from base parameters.
- The final secant displacement persists native Adam moments and A2 history from the base-gradient predictor. Corrector moment updates are discarded. Saved Adam and controller step counts equal accepted trainer updates. With the declared beta1=0/A2 prior, inactive rows are zero in both displacement fields and remain zero in the final linear combination.
- Generator and prior EMAs are committed after the accepted secant parameters and advance once logically. Critic reference commit has the specific defect above.
- All **4600** recorded optimizer-group rate rows retain G/D/prior rates **.00425/.00425/.0085**. The secant rule deliberately changes realized displacement; constant group rates are not a claim of constant displacement or equal compute.

These conclusions apply to the recorded ring MLPs. A broader public implementation needs explicit contracts for mutable BatchNorm/spectral-normalization buffers and error rollback; the existing small tests do not establish arbitrary stateful-module behavior.

## Qualification status

Fresh-process CUDA exact continuation remains explicitly unpassed. Same-process CPU resume tests and correct checkpoint fields do not establish that gate. The retained supervisor checkpoint note identifies higher-order autograd summation-order variation and correctly prohibits attaching old scores to a changed serial-autograd runtime.

The run completed during this read-only audit: cold arrival 560 with **185/185** checks retained through 2400; shifted arrival 2670, delay 270, with **194/194** checks retained through 4600. These observations belong to the frozen hard-copy-reference implementation. This report does not infer stationary-7500, delayed/repeated-9000, uninterrupted-30000, own-22-task, matched-runtime baseline, or exact-continuation qualification.
