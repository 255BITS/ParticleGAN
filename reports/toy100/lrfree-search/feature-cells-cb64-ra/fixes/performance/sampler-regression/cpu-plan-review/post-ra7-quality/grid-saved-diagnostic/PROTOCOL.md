# One saved RA4 grid mechanism diagnostic

Freeze this helper and all inputs before measuring. Read only the completed RA4 grid final-state.pt and its terminal step7000 final and independent holdout clean/noisy live/EMA/target clouds. Do not inspect another checkpoint, draw randomness, emit new samples, train, rerun CUDA or change any model/configuration/gate/watcher/queue. Oracle centers and target sigma are scoring-only geometry in this diagnostic.

For each saved cloud and both unperturbed affine table anchor sets, report nearest-center mass, unconditional and original 3-sigma-conditional centroid/covariance moments, and the original conditional radial CDF error. Verify reproduction of stored terminal/holdout fidelity fields. Report a centered-residual radial CDF as a moment decomposition only, retaining the original HQ mask; it is not a rescored candidate or repaired emission.

For paired clean/noisy clouds, exactly decompose noisy conditional center mean-square and covariance into clean, saved output-noise difference and cross terms under the SAME noisy mode assignment/HQ mask. Also show unconditional noise variance/mean and assignment changes. Compare mode centroids between terminal and holdout, anchors and clean clouds. Use a single-thread exact KD tree for distance from saved clean points to nearest unperturbed affine anchor; this is only a lower bound on actual latent perturbation because sampled row IDs were not saved.

Compute FAST and EMA affine anchors directly from saved weights/bias/prior rows, preserving row identity. Measure same-row nearest-mode correspondence and support, and decompose paired output-anchor displacement into the saved prior-row and affine-map differences. Native G is identity-initialized but trainable, so inspect its actual saved affine matrix. Record controller bandwidth, sigma state and table clocks as context without installing any saved state in a trainer.

One fixed calculation, no tuning/alternate checkpoint selection. Inputs and helper bytes must remain exact before and after. This establishes mechanism evidence only; any generic candidate direction remains prospective and unimplemented.
