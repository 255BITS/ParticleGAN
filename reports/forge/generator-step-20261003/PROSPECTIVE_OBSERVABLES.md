# Draw-free diagnosis plan for the frozen generator-step pair

No scientific result has been inspected. These are conditional reading criteria, bound to frozen `488b792e` public sources in [the machine record](prospective-observables.json). New tuple: G/noise nominal rate0.00265625, prior0.00796875, D0.011953125. Nominal D/prior rates match the failed critic-balance tuple, while their realized gradients and policy rates can change. No further candidate is declared before this pair's immutable evidence is available.

| Question | Evidence available from the unchanged runner | What it can establish | What stays unknown |
| --- | --- | --- | --- |
| Did foreground saturation change? | Saved draws at0,25,…,600; original HQ/modes/mass bounds; final G-gradient/moment state | Center-patch brightness, finite-template rejection, near-zero sigmoid sensitivity and their retained chronology | The force sequence during updates1–24; attribution to a particular owner |
| Did the prior geometry change? | Final raw prior/EMA rows, controller bandwidth/last perturbations, row/birth-death counters | Endpoint table shape and measured cross-configuration differences; recorded discrete moves/resets | Prior displacement from an unsaved initial parameter state; fixed-G output effect or prior-only cause |
| Was D persistently faster? | Final assigned LRs, initial role rates, stationarity state and smoothed payoff error | Actual endpoint rate fractions and source-defined critic-only damping, with assignment/observation lag | Integrated learning speed or per-update G/D loss balance |
| Did optimizer memory restrict recovery? | Final G-gradient, signed output-bias gradient, Adam moments/step/LR | Nonzero recovery direction and retained denominator scale at the endpoint | Whether moment memory caused entry into the basin or would prevent a future escape |
| Which population was served? | Every image observation's averaged flag; final policy/backend/guard/EMA state | Actual sampled fast/average law and recorded intervention events | An unmeasured alternate EMA grade, noisy grade or full family equivalence |

Source ordering matters. The public trainer begins policy/rate/noise selection, updates D, performs the joint G/prior/output-noise backward pass, observes its payoff/gradient evidence, updates the generator-side optimizer, then averages/restructures/counts. The D rate at a saved endpoint was assigned before that update's final payoff observation. `1/(1+payoff_error²)` is the source damping function, not a reconstruction of the complete D-rate trajectory.

The image cloud is `ParticlePrior`. `Recipe.make_prior` discards its `standardize` option for that class; latent row coordinates are raw. DV12 width follows raw coordinate spread and perturbations are clipped by half the nearest nonzero support distance. Thus `output_noise=False` does not imply an unperturbed finite-atom law. Raw table variance and controller bandwidth should be described separately, with their lag retained.

The actual AMSGrad step denominator uses `max_exp_avg_sq`; KA2 critic surprise deliberately measures decayed `exp_avg_sq`. A large max moment relative to the last gradient describes step scale, without becoming evidence of a depressed surprise controller. Sparse prior A2 history, stationarity correction and row-event counters also need their actual source ownership.

The unchanged API runner saves metric/view records and one terminal checkpoint. It discards the return value of `fixture.step()`, including losses. It does not save intermediate model/rate histories. Missing histories will remain missing; no new observer, model restore, sampler, scorer or hidden counterfactual is introduced during diagnosis.

When root sends each immutable boundary, the next report will preserve original status, full/incomplete resources, added first-window/hold verdict, unreached cells, and separately bound engineering/scientific paid cost. It will compare retained arrays and endpoint owners exactly where supported and identify missing causal evidence. This plan executed zero models, restores, updates, draws or CUDA contexts and did not poll the capacity/scientific output paths.
