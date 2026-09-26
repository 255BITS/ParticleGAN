# API-DV1 single-shift independent audit

The retained single-shift result is supported. No evaluator drift, initialization mismatch, or learner access to evaluator horizons/target-change notifications was found. This is a promising survivor, not a qualified winner.

Source ZIP SHA-256: `6238877c082f2bfcc09b248278686313d8fff3eb6e6c3c35f9e6bdf1f29f202d`.
All 19 declared source hashes and all retained artifact hashes verify. The evaluator, architecture, KA2, optimizer and prior source are unchanged from `fa511ce`; only the continuous controller, recipe support and trainer integration change. Initial model, optimizer and RNG hashes match both archived public API arms. The governing protocol matches byte-for-byte.

The worker calls actual `get_recipe()` and `GANTrainer.step()`. The learner sees real minibatches and previous generator-gradient alignment. It uses no target centers, quality scores, shift notification or ending. Noise remains 0/.029; rate changes are autonomous and reversible.

Recomputed from all 460 raw observations:

- Initial arrival: update 790; 162/162 observations pass through 2400.
- Declared pre-shift hold: 120/120.
- Shifted arrival: update 2830, 430 updates after the change.
- Following arrival: 178/178 pass through 4600, no recorded departures; minimum HQ .900634765625, eight modes throughout.
- Frozen comparator: 0/220 pass.

Concrete caveats:

- Reported LR maxima include un-applied initialization rates; actual logged network/prior maxima are .0042289625/.008459625. Raw logs are complete.
- Standalone LR helper functions do not yet support continuous recipes with `total_steps=None`; `GANTrainer` correctly bypasses those helpers.
- Feature innovation is normalized and not ring-specific, but its EMA variance formula assumes steady fixed-size batches. It is not a calibrated universal statistical detector.
- Stationary, delayed/repeated, long continuation, checkpoint, horizon-prefix, matched-K3P and own-22 evidence must be assessed separately. This audit does not supply those passes.

Read-only source/data audit only; no training, tests, model execution, extra seeds, or worker edits.
