# Review before refilling the data-drift lane

The first three public API mechanisms do not qualify. Their scores remain useful:
all three reacquired a changed distribution, but none retained the original
distribution reliably. This review authorizes the next three-proposal attempt
within the existing search, after the predecessor exits. One GPU worker only.

Predecessor:
`/ml2/hypergan/gan-attempts/continuous-api-20260926/20260926T211209Z-1398466/data_drift_mobility/20260926T211209Z-1398477`

Read its final `result.md`, `tests.jsonl`, source archives and declarations in
`repo/reports/data-drift-api/`. Root's independent scores are in
`first-results.json` beside this review. Do not rerun failed variants unchanged.

## Measurements that motivate a successor

- API-DV1 separates real-data drift from gradient coherence. Initial arrival
  790, then 162/162 through 2400. Changed-target arrival after 430 updates, then
  178/178 through 4600. The longer stationary test nevertheless collapses at
  4850–5010, reaching zero HQ and zero modes: 655/672 after initial arrival.
- API-DV2 restricts critic-reference motion to independent real-data evidence.
  It recovers after 400 updates and retains 181/181 afterward, but loses the
  original distribution three times: 159/162 after initial arrival.
- API-DV3 combines accumulated minibatch drift evidence with a slowly moving
  stationary reference. Recovery takes 520 updates, followed by 169/169; initial
  retention is only 153/162. Its temporal detector repair improves the batch32
  regression without changing the threshold, but does not fix game stability.

Real-data change detection alone cannot explain or correct self-induced
instability. Hard-gating or slowly averaging critic memory did not solve it.
Choose a new mechanism for the interaction between these two signals, justified
by the retained update traces. Do not replace the experiment with a learning-rate
or threshold grid. The separate precision lane already owns reference-gap-based
closing/reopening; the constant-rate lane owns implicit coupled game updates.
Keep this lane's distinct real-data evidence useful while addressing stationary
instability. No task IDs, target centers, oracle scores or change notifications.

## Shared execution correction

The predecessor also isolated a cross-process CUDA continuation discrepancy.
Higher-order critic backward creates nodes with thread-local priority counters;
the first fresh-process gradient summation order can differ from warm execution.
Immediate checkpoint state and RNG were equal. A scoped `serial_backward=True`
public trainer option is being tested to make the order reproducible.

Read the predecessor's final diagnostic and the independent audit before using
the correction. Copy only the reviewed standalone patch/dependencies, record
hashes, and explicitly declare execution mode. A corrected-runtime candidate
must earn its own quality evidence. Old multithreaded passes or failures cannot
be reassigned to it. The opt-in must preserve the caller's autograd context and
validate checkpoint configuration; it must not mutate global settings forever.

## Evaluation and source ownership

Keep COMMON.md and the frozen evaluation declarations. Recovery is arrival
followed by stability, not an 81/81 deadline. No seed sweeps or extra final seeds.
Use unique names API-DV4 onward and preserve all failures and incomplete runs.
Reuse unchanged plumbing with a source receipt; never edit the predecessor.

API-RP2 owns the matched public K3P comparison if it survives. Do not duplicate
that work. The frozen22 route map beside this review explains why legacy runners
cannot qualify a current controller: fourteen hosts can adapt GANTrainer, eight
need component-controller integration. A cheap sensitive frozen image gate may
be checked before full22, but its exact model, initialization, evaluation-noise
stream and all quality rules must be retained. Unsupported routes are NOT_RUN.

No winner is selected, no default is promoted and neither PR is merged.
