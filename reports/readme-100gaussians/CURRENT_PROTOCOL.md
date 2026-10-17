# Current-default README animation

This is one bounded illustration of the selected public default, resolved with
`get_recipe()` at execution. For 0.9.0 this is `bispf_optimized`. No recipe field
is tuned for the animation. The full illustration uses the default 7,000 updates,
batch 2,048, a 20,000-row, two-dimensional particle cloud, the example's width-128,
three-layer MLP generator and Fourier-feature critic, and a uniform 10×10 mixture
of Gaussians with σ = 0.03. The particle-cloud exception is explicit: the example
illustrates learnable individual rows, retaining the original demo's prior kind.

Protocol seed is 0, with the public deterministic initializer on whole G/D/prior
modules (seeds 0/1/2). Constructor, data, latent, penalty, noise, model, evaluation
and reference draws have separate recorded streams. Initial and final checkpoints
include the trainer and every consumed stream. Plotting cannot consume training
RNGs. Samples are clean/live; 4,096 fixed row IDs establish visual continuity.

The mode-acquisition gate requires 100/100 covered modes and at least 95% of 20,000
fixed evaluation draws within 3σ of a mode center, at each of the last five of
24 evenly spaced scoring observations. A covered mode contains at least ten such
samples. An independent oracle draw must pass and a collapsed control must fail
before training. The full default budget is required; short software smokes cannot
pass. Additional media observations cannot remove or change a scoring check.

This gate describes mode acquisition, not Gaussian width or covariance fidelity.
Final mode TV, sliced W1, tails and per-mode shape metrics are also reported.
The GIF includes the endpoint and a hold, without best-checkpoint selection.
Its caption and receipt preserve any failed gate. This supplies no Forge
qualification and makes no comparison to the historical K3P/seed-1234 runs.

Budget: one full 7,000-update run; no seed sweep, retuning or extension. Bounded
software smokes may use four updates to verify the public API, RNG isolation,
checkpoint and renderer. Raw logs, observations and checkpoints stay outside Git.
Re-render saved observations for visual iterations rather than retraining.
