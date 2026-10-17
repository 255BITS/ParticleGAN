# Original Toy25/MNIST: current public API auto candidate

Use one root-reviewed frozen `pkg-RA12-auto` and one shared
`configs/RA12-auto.json` for both original learned fixtures. Only original
N1024, z128 and batch128 override that shared configuration. Do not reuse
an old KNN or feature result as this candidate's numerical result.

Construct the original models from the read-only `models_metrics.py` and
explicitly initialize G/D through current public `deterministic_orthogonal_`
at keys 0/1. The original supplied random prior retains seed 314160. Require
exact original G/D/prior hashes and original private stream states. No
`Recipe.initialization`, module-global sampler patch or legacy package hook.

Preserve seed 314159, saved real streams, two real batches per update,
serial backward, CUDA0 deterministic execution, 2000 updates, and original
checkpoints 0/100/250/500/750/1000/1250/1500/1750/2000. Checkpoint zero keeps
auto selection pending. The first ordinary real update resolves selection;
no extra `begin_step` or synthetic model/critic forward is inserted.

The frozen scorer still receives noisy primary samples through a thin
fixture proxy calling the current API with `output_noise=True`. Toy draws
8192, MNIST draws 4096, each original evaluator stream seed 314259. The toy
oracle/scorer and active image evaluator ASTs remain unchanged. Original
Toy25 gate remains precision >= .9, all 25 modes with >= .01 supported mass,
mass TV <= .1. This is an independent learned toy gate. Original MNIST has
no numerical quality pass threshold; preserve its comparative regression.

Validate the entire source-declared `backend_selection` certificate,
including requested/actual/sampling backend, reason, shape, finite conformal
population resolution and role rate mapping, at every original checkpoint.
These fixture validators observe recorded state; they do not select the
training backend. Toy raw width 2 permits feature quarter G/noise bases;
MNIST raw width 784 retains current reference historical bases. R1 remains
enabled in both. No quality-gate result controls training behavior.

Source/CPU preparation only until root reviews and freezes the package and
authorizes numerical execution. Root arbitrates the original owned GPU0
outer slot and shared `quality/.serial-phase.lock`. Never signal parked or
foreign jobs and never start `gpu_slot.py` main. Commands, input/source
hashes, CPU preflight, logs, raw curves and checkpoint replay are separate
receipts. Do not overwrite an earlier numerical attempt.

Original replay retains two independent ten-update continuations of each
saved 1000-step state. Branch zero loads native placement; branch one loads
the same checkpoint with `map_location='cpu'` and restores onto CUDA0.
Both restored states, per-update losses and semantic state fingerprints must
match the canonical native saved state and one another bit for bit. All RNG
buffers remain CPU uint8. Only original observational birth evaluation time
is excluded. After each continuation, compare complete noisy-primary sample
bytes using the original draw count and evaluator stream seed; sampling must
preserve the semantic training state. Record every loaded/restored tensor's
device, dtype and shape, including CPU RNG/Adam step buffers and model-device
R1 pending tensors. This covers R1 pending CPU-map placement in the existing
update budget.
