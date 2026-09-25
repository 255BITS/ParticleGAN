K3P stays the selected base. All three candidates kept its hold and the 300-update extension, and none recovered inside the deadline.

| Candidate | Hold | Extension | Pre-shift | Deadline | Delay |
|---|---|---|---|---|---|
| K3P parent (not rerun) | 1200/1200 | 300/300 | published pass | FAIL 28/81 | 1130 |
| **rr2** prox gap, slow reference | 1200/1200, min HQ 0.91699 | 300/300, min HQ 0.98291 | 5/5 and 120/120 | **FAIL 52/81** | 1140 |
| rr1 parameter-RMS rate | 1200/1200, min HQ 0.92383 | 300/300, min HQ 0.99658 | 5/5 and 120/120 | FAIL 29/81 | 1010 |
| rr3 prox gap, fast baseline leak | 1200/1200, min HQ 0.91699 | 300/300, min HQ 0.98291 | 5/5 and 120/120 | FAIL 0/81 | none |

The prox gap is the shift signal. On a passing hold it stays near 0.0007. After the target moves it jumps to about 0.03. Critic-parameter RMS does not separate those states: rr1's level only reached 0.38, the partial rate never fully opened, and the deadline was 29/81.

rr2 opened generator and critic rates to 0.000884 from that gap. Eight modes were back by step 2500. Slowing the reference to step 0.0002 kept the gap above 10× through step 3600, so the partial rate never released. The deadline window scored 52/81 and sustained recovery waited until step 3540.

rr3 kept the K3P reference step and leaked the quiet baseline at 0.002 per step. The rate closed by step 2499, before eight modes returned, and then stayed on the floor. Deadline 0/81.

Frozen control, the sensitive toys, the full 22, the two-horizon prefix, and a second shift were not run. Input and output noise still follow the driver's horizon of 1200. That is a labeled ablation. The report, hashes, and replay commands are in `result.md` next to `tests.jsonl`.
