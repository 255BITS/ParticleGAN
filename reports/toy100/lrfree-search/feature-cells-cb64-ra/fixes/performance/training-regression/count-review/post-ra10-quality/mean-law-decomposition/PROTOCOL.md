# One final RA10 mean-law decomposition

This implements the immutable selected design in ../mean-law-source-plan/DESIGN.md.
It is a descriptive saved-state diagnostic. The original full grid is VALID/FAIL;
this helper does not compute or change quality verdicts.

## Fixed inputs and equations

Use only final RA10 Grid100 step7000, the original paired EMA clean/noisy100k
holdout arrays and their identical saved real target. All finalized native
artifacts and frozen sources are raw-byte guarded before Torch import or PT
interpretation. Reuse the immutable native `head_features` function and direct
`F.linear` raw affine function by AST extraction. No network/trainer constructor,
initialization, sampler, scorer, action planner, witness or count test is called.

Fit exactly one requested128/rank8 chart with the original `FeatureCellSnapshot`
implementation, from current saved-D features of the saved real FIFO. Use a
private CPU generator set to a clone of the saved CPU RNG; never seed or advance
the saved/global RNG. The chart's even-fit geometry and original odd support
calibration are retained unchanged. This new CPU chart is descriptive, not the
vanished historical GPU reaction chart or a statistical certificate.

Use unchanged `freeze_moment` and `observe_view` definitions solely to establish
even-real group centers/scales/clipping and current view predicates. Do not call
`odd_witness`, count-family tests, pair proposal, preview, commit or sampling.
L is inside+p>Q in both current FAST/EMA views and their learned groups agree.
U is its complement. No source row, optimizer, controller or lease is changed.

## Measurements

Report one table over the fixed learned groups for even/odd real FIFO, their
inside+p>Q subsets, saved real target, raw EMA anchors A, L, U, saved clean C and
noisy Y. Counts, clipped feature means, raw means, covariance traces and clip
fractions use fixed groups; missing means are null, never renormalized away.
Even-reference group masses weight aggregates. Report covered mass explicitly.

Check the exact A=L+U mixture identity in clipped feature and raw coordinates.
Compare all-even and inside-even target residuals and weighted L/U contributions.
For paired noise, **both** binning and the center/scale/radial-clip frame are g(C)
for C and Y. Compute mean[psi(Y,g(C))-psi(C,g(C))] and the paired raw mean increment.
Natural g(Y) frames are used only for the separate original-style Y population
energy and reassignment report. A-to-C is unpaired and contains row sampling and
latent perturbation; it is never called a paired perturbation effect.

Record natural C-to-Y group transitions, A/C/Y feature energies, physical group
mean gaps and covariance traces, and weighted delta/residual norms, dot products
and cosines. No significance cutoff, variants, oracle labels, rescored cloud,
counterfactual, new data/latent/output-noise draw, optimizer update or quality
sample is introduced. The one chart uses its declared private random projection.

## Neutrality, execution and closure

Record raw inputs, immutable extracted-function ASTs, private chart RNG before/
after, chart geometry digest and saved semantic-state digest. Assert loaded saved
state, global CPU Torch/NumPy RNG and all source/raw file bytes unchanged. No CUDA
context is created. Output is strict JSON; failure attempts remain separate.

Source/input sealing is stdlib only. Numerical invocation is forbidden until
the independent helper review and explicit parent authorization. The runtime
requires the exact SOURCE-FROZEN SHA on its command line. A separate post-exit
sealer requires the run process to have exited, then binds closed outputs/logs
and verifies all pre-execution guards again. No second fit is used as a check.

Finite samples, learned/shared-data D/FIFO, conditional assignment, radial clipping
and the anchor-versus-emitted law limit interpretation. Positive alignment does
not prove historical causality, distribution equality or a production repair.
