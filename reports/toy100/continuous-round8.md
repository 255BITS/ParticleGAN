# Stationary support memory: remove minibatch disappearance as a force

The current experiment separates estimation of a fixed data distribution from
the ability to repair model error. A minibatch that omits an established group
must not instruct the generator to erase it. More data can refine an estimate
without reducing the generator's response to an error at any training age.
No production replacement for LR decay is qualified yet. The fixed-support
rule passes its warm/hold gates but fails later acquisition after incomplete
bootstrap, so its cold and longer gates are withheld.

The preceding [memoryless candidate](continuous-round7.md) passes acquisition,
continued training and recovery, but loses a mode in one actual neural update
when D's minibatch is conditioned to omit that group. This is a possible event
under the same fixed distribution, not a changed target. The new candidate
uses exactly the existing native D real bank, once per outer update.

Two consecutive real banks must give equal inferred group counts and a unique
reciprocal correspondence with positive separation margin. While unresolved,
G and its prior rest at their pre-update values; D and exactly one Adam update
per player are retained. Compatible banks establish fixed reference centers.
Later raw samples update sufficient statistics inside the fixed disjoint
reference balls; absent identities persist, and new identities cannot be
created after confirmation. The [bootstrap note](anchor-fixed-support-bootstrap.md)
states the assumptions and conditional-mean interpretation explicitly.

| Cheapest filter first | Result |
| --- | --- |
| Two full banks, followed by an omitted group | Fixed eight identities retained |
| Seven groups, then eight, then eight | Waits for the compatible eight/eight pair |
| Noisy singleton observation | No false ninth identity |
| Two banks both omitting the same group | **Can confirm incomplete support**; bootstrap completeness is an assumption |
| Native saved-state failure windows | **44/44 pass**, minimum HQ .968994, eight modes throughout |
| Actual neural omitted-bank and singleton branches | **PASS**: all three updates in all three branches retain eight modes/HQ1 |
| Exact native checkpoint split, including pending memory | **PASS**: full envelope, host and learner hashes identical |
| New warm200 | **200/200 pass**, minimum HQ .999512, controls exact |
| New borrowed-state hold1200 | **1200/1200 pass**, eight modes/HQ1; first200 host and learner parity exact |
| Incomplete bootstrap followed by clear same-target signal | **FAIL**: confirms seven groups, then rejects all22 samples of the missing eighth group |
| New cold/own-state/response/long gates | Withheld after the cheap acquisition counterexample |

The [pure-filter archive](continuous-evidence/round6-sample-group-two-bank/)
retains the incomplete-bootstrap counterexample. The
[actual neural three-update archive](continuous-evidence/round7-two-bank-native-three-step/)
records the ordinary, omitted-group and singleton-tail branches. The
[saved-state archive](continuous-evidence/round7-memory-saved44/) contains
exact sources and every raw comparison. Each saved-state window explicitly
starts a new learner for this diagnostic; it is not presented as a resumed
memory-bearing run.

The [warm/hold archive](continuous-evidence/round7-memory-warm-hold/) preserves
complete host-plus-memory envelopes. The [native split](learner-native-split.md)
verifies exact restoration across the pending-to-confirmed boundary. An
[independent endpoint audit](round7-endpoint-drift-audit.md) finds clean G output
RMS motion .00157 from1200 to2400, while D parameter displacement remains4.206.
That is critic motion, not proof of divergence.

The [later-acquisition falsifier](sample-group-two-bank-audit-and-falsifier.md)
starts from the same passed neural state. Two D banks are conditioned to omit
component0; the second confirms seven identities and the model loses the
eighth mode. The next ordinary bank contains22 samples of that mode, all22
are rejected, and the model remains at seven. The target and G real banks
are unchanged. This directly violates acquisition when evidence appears on
the same dataset; a correct-bootstrap assumption cannot close the task.

The [new native adapter](sample_anchor_memory_candidate.py) retains the
pre-start joint output fit and its rest-on-nonconvergence guard. It applies the
full remembered target with the same numerical budget at every age. The
cumulative data estimate uses sample counts, but counts do not multiply G's
correction. Same-target model-error response will test this distinction after
acquisition. Nominal G/D/prior rates remain .00425/.00425/.0085.

The [complete learner envelope](learner-state-envelope-contract.md) includes
the pending or confirmed support, its ordered statistics and fixed references,
bank identities, confirmation receipt, rejection counts, audit clocks, and
the complete existing model/Adam/EMA/noise/RNG snapshot. Old host-only
checkpoints are insufficient. The new gate driver preserves both states and
checks learner parity independently of the precisely enumerated host
evaluation-counter reconciliation.

This remains an explicit data-coverage objective. It does not yet establish
mass or within-group shape fidelity, arbitrary later support discovery,
stability of all internal parameters, or an objective-preserving GAN cure.
Parallel work examines the actual affine production chart, representability
of emitted distributions under fixed output noise, and variance-reduced game
dynamics. No whole-distribution target shift is required for this task.

The next cheap experiments examine fresh-data evidence for discovery and an
exact finite-bank game operator at a previously failing state. Discovery
confidence must use data independent of candidate selection, allocate error
across repeated candidates, and respect particle mass. Merely waiting for two
outliers or freezing identities forever is insufficient. Variance reduction
also needs a contraction test: the earlier failing game field was largely
coherent across minibatches. Neither route is claimed solved.
