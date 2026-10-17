# Fixed recorded-gradient G-rate probe

Freeze current RA6 table/G diagnosis before this prospective CPU diagnostic. Read frozen RA6 checkpoint1000 and2000, package/config/READY, existing functional forward/evaluator definitions and current saved FIFO. No training loop, new data, new seeds, emissions, birth proposals, CUDA or writes to frozen inputs.

The single hypothesis is the root's fixed1x versus1/4x generator-rate comparison. The saved G optimizer has beta1=0, so exp_avg contains the latest recorded G gradient. Repeat that gradient for one virtual Adam/AMSGrad step at the saved current model and moments, with the saved LR or its quarter. This is neither the actual next gradient nor a historical step reconstruction. Checkpoint GPU streams are not CPU streams; the exact unlogged training batch/noise/gradient cannot be recreated here.

For each saved state, fit one critic-head partition from the saved FIFO with its saved CPU RNG, keeping D and that partition fixed for both candidates. Compare projected head displacement, current p>Q/inside retention, same-cell retention and categorical changes on the fixed clean latent table. Raw oracle labels appear only in separately labeled annotations after learned measurements; they never choose a level, row, acceptance or proposed config.

Virtual candidates change only G weights in detached copies. Prior, D, EMA weights, learned sigma, optimizer/regularizer state and all streams remain unchanged. A plain Adam reference step on private cloned G parameters checks the functional candidate. These virtual optimizer calls are mechanical diagnostics, not new training steps.

Production rate-only config, if selected by root, would change lr=.00425 to.0010625, prior_lr_mult2 to8 and d_lr_mult1 to4. Prior/D bases remain .0085/.00425. Both G and the learned log-sigma optimizer bases quarter; the noise formula, floor, mode and live-noisy evaluator remain unchanged. This G-only probe does not claim to reproduce that coupled G+sigma production step. No production source or config is edited.

Only these two fixed levels and two existing checkpoints are used. No sweep or quality claim follows. Final strict CUDA toy and canonical full-grid gates remain the root's responsibility.
