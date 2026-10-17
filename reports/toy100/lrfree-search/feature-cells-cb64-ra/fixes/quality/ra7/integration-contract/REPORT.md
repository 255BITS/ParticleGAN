# RA7 rate and exact-count integration API proof

PASS on CPU using the actual composed RA7 modules and saved RA6 step2000 model/table tensors. All 29 modules are byte equal to the GROUP-COUNT package already covered by 224 CPU allocation cases, two complete saved reactions and the paired GPU plan proof. No numerical plan suite was repeated. Full config comparison changes only the three declared rate fields.

| Group | RA6 base | RA7 base |
| --- | --- | --- |
| G network | .00425 | .0010625 |
| Prior | .0085 | .0085 |
| Learnable log sigma | .00425 | .0010625 |
| Critic | .00425 | .00425 |

The proof executes the unchanged original contiguous GANTrainer constructor section that creates optimizers, the learnable sigma group, role labels and initial rates. Saved trained parameters are marked already initialized; no weights are redrawn. Non-LR group options, row damping and tester types match. For identical relative scales, applied/base ratios match across 17 scale cases, with 289 critic/prior floor pairs. The original output-noise method produces identical same-state outputs for open, settled and zero-base cases. No gradients, optimizer steps, new seeds or CUDA context were used; CPU RNG and every source/input hash stay unchanged. The first private attempt omitted Torch from its isolated namespace; its retained log identifies the fixture error corrected before PASS.

## Prospective scope

This is a G-plus-sigma rate configuration. It retains the output-noise formula, floor and mode, EMA/serving law, birth/copy budgets, count certificates and original evaluation schedules/gates. The source uses the original optimizer APIs and adds no loss, feature pass, Jacobian solve, serialized key or per-step work. It is a suitable first declared test of the observed generator/table coadaptation before adding more transport machinery.

Same intrinsic ratios at the same scale do not imply the same future stationarity decisions or trajectory. Lower rates may slow adaptation within the unchanged budgets. Fresh strict toy and grid evidence is required; the CPU API proof makes no quality claim. Cross-config checkpoint resume remains rejected by the original recipe equality guard.
