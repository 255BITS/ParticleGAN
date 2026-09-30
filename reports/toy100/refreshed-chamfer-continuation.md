# Refreshed Chamfer targets: warm pass, continuation fail

The candidate does **not** qualify. After a 200/200 warm fork it loses modes
on the same fixed target at updates 2240 and 2250. The final checkpoint is
8 modes / HQ .99976 and hides those two checks. No further host was run
after this failure. The new order is warm, then this continuation, then
cold acquisition. A cold ring run started before that order is recorded
below and is not a promotion.

## Why this rule

The archived one-step Chamfer targets, scored with the host's own evaluation
draws, miss modes at 1000/1050/1100 even if a particle landed on them exactly.
On a local replay of the same adapter, a Gauss-Newton solve reached those
one-step targets to about 1e-5 and reproduced that failing score. Four
nearest-neighbor refreshes on the same real minibatch produced an 8-mode
cloud at the captured steps. One linear Adam-metric pullback did not land
on that cloud (max error up to 6.75) and still failed the gate. Gauss-Newton
onto the frozen four-refresh target did land and passed every captured
checkpoint. That is the rule under test: four output-space assignment
refreshes, then a prior-only nonlinear landing, accepted only when actual
unit-mean Chamfer decreases. No target centers, coefficient sweep, or
generator rest damper.

The local replay did not reproduce the archived cold observation trace
(24/24 mismatches). Step 1 still matched the archived Chamfer objective
(16.70751 vs 16.70751) and latent correction (380.19). Torch was
2.13.0+cu126 with CUDA unavailable.

## Gates

| Gate | Result |
| --- | --- |
| Identity continuation, scheduled, through 2400 | 200/200 then 120/120; exact cold-host parity |
| Constant Adam from the same warm state | Not re-run here; prior warm fork on this machine was 4/200 |
| Refreshed candidate, updates 1001–1200 | **200/200**, minimum HQ .94360 |
| Same branch, every 10 updates from 1210–2400 | **118/120 FAIL** |
| Cold trajectory (earlier, before this order) | PASS, MSE .00094267, suffix 18 |
| Cold ring terminals 1000–1200 (earlier) | 8 modes, HQ .99976 / 1 / .99927 / .98560 / .97485 |

Continuation failures, live weights:

| Update | Modes | HQ |
| ---: | ---: | ---: |
| 2230 | 8 | .94385 |
| **2240** | **6** | .91675 |
| **2250** | **6** | .91162 |
| 2260 | 8 | .98315 |
| 2400 | 8 | .99976 |

Minimum later mode count is 6. The passing suffix from 2260 is 15 checks.
Identity's lowest later HQ is .99829 at 8 modes.

On this machine the correction-disabled PR84 warm fork was 196/200, not the
archived 200/200, and its warm-state hash did not match the published
artifact. Identity parity for the continuation did hold. That archive
mismatch is an environment difference, not a pass for this candidate.

## Stop

The continuation gate fails. Cold acquisition is not repeated, and there is
no hold of an acquired state. PR84's original stencil remains the reference
partial result: it already failed this same continuation (8/120, first
observed fail at 1390). This candidate moves the loss later and does not
clear it.
