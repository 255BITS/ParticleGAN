# One saved RA9 native affine/conditional-mean decomposition

Freeze this protocol, helper, original frozen centroid equations, all actual
RA9 source/config identities, final7000 state and the original paired terminal
20k/holdout100k clean/noisy arrays before importing Torch or loading the PT.
Use one CPU run, threads1, CUDA hidden. No constructors, fresh draws, emissions,
training, chart fit, quality rescoring, actions or parameter/seed sweep.

Reuse only original `assign`, `moments`, `mean_square`, `norm_stats` and
`centroid_comparison` equations. Do not invoke the original full diagnostic's
`summary`, radial-KS, fidelity, covariance decomposition or main runner. Oracle
labels are native-fixture diagnostic annotations only, never fitted geometry,
correction targets or production rules. No points or outputs are changed.

Final native G is trained affine, not a permanently assumed identity. For
both raw FAST and EMA states use the saved actual A,b and prior z. Within the
fixed diagnostic groups of actual x=A z+b, compute conditional means m_z.
Relative to diagnostic target mean t, the exact identity is

`mean(x)-t = A(m_z-t) + (A t+b-t)`.

Report transformed table displacement, affine-map displacement and their
cross term; also express the residual relative to the algebraic inverse-map
mean `(t-b) A^{-T}`. The inverse mean is an interpretation only, not a new
latent proposal. The identity initialization reference is specific to this
native fixture and is not assumed for learned/image generators.

Use oracle component centers and saved raw-real FIFO/independent real-target
cloud conditional means as separate diagnostic references. Report fixed-group
unconditional and existing3sigma-selected anchor means, full raw prior means,
FAST/EMA paired-row affine versus table differences, and only first-moment
comparisons to unchanged saved clean/noisy clouds. No new precision, covariance,
radial or quality verdict is calculated. The original gate remains VALID/FAIL.

A final-state algebraic decomposition is not a historical causal ablation.
Prior means can compensate a changing affine map; cancellation must be
reported. Finite target clouds and sampled output clouds add mean uncertainty.
The checkpoint's raw FAST state may differ from the served live arrays when
the existing paired-average lease selects EMA. Explain that view explicitly.
