# Settled re-open scalar reconstruction

The exact proposed detector/guard classes reproduce the unchanged detector on all 270 closed observation rows. The default call has exact fast/slow memories, ratios, clocks, log and fire parity with the original implementation and recorded traces. Existing K=12, RISE=2 and CALM=1.25 remain unchanged.

| Saved window | Original fire, completed step | Guarded fire through first action | Mechanism |
| --- | ---: | ---: | --- |
| Static Toy, 750–834 | 820 | none | Actual KA2 `anchor_started` False→True clears reference memory after update800, before its q is queued. |
| Static MNIST, 100–214 | 202 | none | No network ladder was contracted at last calm/onset, so the existing slow horizon follows the excursion and no streak accumulates. |
| Moving window, 500–534 | 514 | 514 | Contracted generator and critic qualify the sustained jump. |
| Moving window, 1000–1034 | 1021 | 1021 | Contracted critic qualifies; generator-only qualification would lose this event. |

The Toy reconstruction performs exactly one loss-epoch rebase. Its guarded ratio is 1 at completed step800, reaches a maximum 2.10498, and drops to 1.83494 with streak0 at the original fire820. A contracted-network witness is present, so the instantaneous reference reset prevents this recorded first fire without an additional readiness rule. MNIST retains an empty onset witness at step180; its guarded ratio reaches 3.00855 but streak stays0, including the original fire202. Both moving fires retain their exact original ratios and completed steps (14.34285 at514; 4.47155 at1021).

The guard is a restart-eligibility rule. Network contraction is evidence that a restart can reopen a reduced ladder; it does not identify external target changes. A static shock after contraction may still qualify. Table/noise roles are excluded from the witness; their original optimizer q values remain in the detector ratio. No group filter or threshold was introduced.

## Scope

The candidate class source is pinned to `0342b0b19a6e5e177ec38840775886576d22e21dc6608e6aa2bb49aa53e1bef4`. Classes are compiled directly from that AST; no package, model or policy constructor is used. CUDA was not initialized and global CPU RNG remained unchanged. There are no PT reads, forwards, draws, updates or scoring calls.

The guarded comparison stops at the first proposed or original optimizer action. Later q values in an original trace were produced after the original moment/tester changes and cannot establish a guarded training counterfactual. These windows initialize legacy scalar history at their left boundaries; they are mechanical reconstructions, not new-law resumable checkpoints. Toy is covered through820, MNIST through202, and the positive moving events through514/1021. This proof does not qualify full Toy/MNIST quality, the complete moving gate, fresh default API runs or resume state.

`result.json` retains every tested prefix row and guard witness. `INPUTS-FROZEN.json`, `LAUNCH.json`, `EXIT.json`, `receipt.json` and `FROZEN.json` bind source, inputs and the closed one-shot CPU invocation.
