# Saved-checkpoint toy regression diagnosis

CPU inference only. This directory is the only editable area. The corrected
RA2 package, active quality lane, previous READY sources and original studies
are read only. No new seeds, training updates or CUDA context.

Inspect fixed CUDA toy checkpoints for E22, old CB64-RA and corrected CB64-RA2
at steps1000/2000, plus the corrected initial state. Reuse latent/output noise
already saved in geometry/gpu-inputs.pt; do not generate another noise seed.
Compare each saved table against itself under identical points/noise.

Measure exact nearest-nonidentical distances, inherited reference DV12's
shortlist radius, bounded sorted-coordinate radius, coordinate widths and
duplicate families. Separate radius error from anisotropic bandwidth: compare
the current kernel, exact DV12, exact radius/current local width, and current
radius/reference width. Also score unchanged clean centers and the historical
fixed kernel. Measure the actual saved generator's displacement and original
toy support metrics with identical saved output noise.

Serving is resolved from the saved table stationarity tester. Diagnostics of
the live training table and the served table are labelled separately if they
differ. CPU scores do not replace CUDA acceptance or reproduce a CUDA RNG
continuation. Record source/checkpoint/tensor hashes and no initialized CUDA.

The stability owner reconstructs guard-dependent ordinary allocation on the
same saved model/table/FIFO. Coordinate findings rather than repeating that
planner experiment here. Any sampling patch must follow the observed cause,
preserve the shared training/fake-pool/serving/copy law and checkpoint/cache
contracts, keep bounded large-population work, and pass focused small tests.
