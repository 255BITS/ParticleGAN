# Next round: speed before longer quality runs

User selected training throughput as the next target after the host restart.
Do not automatically resume NCSN++ or launch the earlier architecture ideas.
The large G averaged2.932 updates/s; U-Net32 averaged8.389. NCSN++ last checkpoint
is20k (5k-sample FID65.209), last logged step27,200. U-Net finished50k at final
50k-sample FID26.680. See scale_50k/READOUT.md.

1. Establish reproducible short CUDA benchmarks for both architectures with
   full batch64, pretrained D, bcap and particle updates. Separate warmup,
   steady-state training, evaluation, checkpoint I/O and peak GPU memory.
2. Profile G forward/backward, D ordinary passes, bcap double backward, frozen
   feature extraction, FIR resampling, attention and particle optimizer work.
   No hotspot is established yet; the native FIR fallback is only a candidate.
3. Test changes independently on the two GPUs with matching device baselines.
   Candidates: reuse frozen xt features within a batch; equivalent faster FIR;
   channels-last; compilation of supported regions; validated mixed precision.
   Do not silently reduce batch, regularizer frequency, D feature resolution,
   model capacity, or diffusion steps and call it an implementation speedup.
4. Check output/gradient agreement for equivalent transformations, including
   candidate gradients through frozen D and bcap double backward. For numerical
   changes, validate stability and a quality scout as well as throughput.
5. Promote measured speedups through YAML config options, report updates/s,
   memory, estimated50k time, and quality per GPU-hour. Then revisit longer runs.

Keep the established DDGAN/ParticleGAN/joint-UCD equations and hyperparameters.
Do not disable candidate derivatives through frozen features. Reuse existing
hyperparameters; do not launch seed-only repeats. Save config and provenance,
use fresh output directories, retain checkpoints, and provide a combined tail log.

Later quality ideas, not queued: stronger/different pretrained D; pretrained
encoder features alongside G's existing multiscale image path; longer training
once affordable. Clean-image encoder features may help low-noise inputs more
than near-Gaussian terminal states. Changing either architecture can retain the
formulation, but neither is the current speed task.

No recoverable optimization-subagent deliverable was located after restart.
