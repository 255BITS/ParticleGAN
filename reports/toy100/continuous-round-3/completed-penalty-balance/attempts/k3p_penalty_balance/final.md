K3P stays the selected base. No candidate reached deadline 81/81, so nothing was promoted and the image, native, frozen-control, and long-run gates were not opened.

The critic split is real and non-invasive. An instrumented K3P, with one extra penalty backward and no control change, reproduced the parent shift exactly: stationary 5/5, pre-hold 120/120, deadline 28/81, delay 1130. The backward coefficient stayed 1. On the quiet floor the penalty force is about 7% of the adversarial force. After the shift the penalty force rises about 10× and opposes the adversarial force, while K3P's learning-rate clock keeps the mixing weight at 0 and both rates on the floor. Total-gradient RMS also rises, so the shift is not hidden from a raw RMS rule.

| Candidate | Hold | Extension | Pre-hold | Deadline | Delay |
|---|---|---|---|---|---|
| K3P parent, not rerun | 1200/1200 | 300/300 | held | FAIL 28/81 | 1130 |
| **pb2 share-square** | **1200/1200**, min HQ .913 | **300/300**, min HQ .976 | **120/120** | **FAIL 77/81** | **480, stable at 2880** |
| pb1 fight reopen | 1200/1200, min HQ .980 | 300/300, min HQ .995 | FAIL 112/120, steps 1280–1350 | FAIL 79/81 | none |

pb2 is the best partial lead. It restores the pre-shift contract and moves sustained recovery from step 3530 to 2880. Four deadline checks still fail, at steps 2840–2870, with 7 modes and HQ .839. The output does move: it falls to 0 modes at the shift, returns to 8 by step 2700, and both optimizers continue through 3600 updates. pb1's linear fight gain got 79/81 but never returned to the floor, and that residual rate lost the eight pre-hold checks.

Both controllers still inherit K3P's 1200-step noise horizon, and the ring driver uses that same horizon for the learning-rate cosine. The new gain term does not read it. They are intermediates, not horizon-free results. The next mechanism has to reopen harder only while the penalty share is high, without lifting the quiet-floor rate, and then remove that horizon.

Full scores, traces, hashes, and replay commands are in `result.md`. Ledger: 8 rows in `tests.jsonl`.
