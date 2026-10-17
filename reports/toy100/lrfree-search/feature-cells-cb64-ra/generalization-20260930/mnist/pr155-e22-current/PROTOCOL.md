# Fresh current PR155 E22 learned baselines

The numerical source is the exact20-module package from PR155 at
`cabe2084284db923d525918cbf3e18de6f20faac`, raw package SHA
`c174ba0b805cc8e49ca40ebbea11785bb908f3ea32615afdf2ecb43e77221316`.
Its public `e22-noout.json` retains `reopen_signal="optimizer"` and
`reopen_anchor="release"`; it contains neither Atlas feature selection nor
the optional settled guard. Resolved training fields must match the source's
named E22 preset exactly, allowing only the `ka2` versus `e22` report name.

The caller overrides only original learned-fixture dimensions: N1024,
z128, batch128. The seed remains314159, prior initialization314160,
public deterministic orthogonal model initialization G0/D1. Both initial
model hashes and the prior hash must match original frozen fixture values.
Each complete update uses two independent original128-row data batches,
one critic update followed by one generator/table update. Exactly2000
updates per Toy25/MNIST fixture run with serialized autograd backwards.
Checkpoints remain0,100,250,500,750,1000,1250,1500,1750,2000, with exact
data cursor2*step*128 and all original native/private RNG states saved.

Toy evaluates its original noisy primary8192-row draw and original acceptance
law. MNIST evaluates its original4096-row primary draw using the same fixed
classifier/embeddings, active-coordinate normalization, real references and
precision/recall implementation; no numerical MNIST gate is invented.
All original evaluator functions and the whole2000-update loop are AST-pinned.
The original primary sampler explicitly calls `sample(output_noise=True)`.
The configured output-noise initial scale remains.029. No fresh seed,
scorer, threshold, training budget, optimizer rate or data stream is selected.

The parent coordinator alone launches CUDA0 under the existing global serial
mutex, physical GPU UUID,20% process cap, deterministic algorithms/noTF32,
two CPU threads and the original parked-process identity checks. Agents do
not launch GPU jobs or signal other processes. Each run has a fresh immutable
log, result, error if applicable, completion and original checkpoint files.

These fresh baselines are a comparison against the current PR155 recipe.
Previously recorded E22-without-reopening and all failed intermediate
candidates remain historical records. Atlas's fresh learned runs retain their
original execution labels and latest source/replay bridge; they are not
relabelled as fresh training of the newest source. This recipe comparison
does not isolate one mechanism or establish seed robustness, speedups,
scaling laws, or image-scale feature-cell performance.
