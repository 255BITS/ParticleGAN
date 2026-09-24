# Prestart anchor: one missing D bank and nine ordinary updates

The [source-bound paired continuation](prestart_omission_recovery10.py) reuses the qualified own-acquired update-2400 state and the previously archived native update 2401. One arm used the ordinary D real128 bank; the other conditioned **only that one bank** to omit ring component 0. The G bank and fixed target stayed unchanged. Both archived update-2401 states then received nine ordinary native updates, through 2410, with the same frozen prestart anchor method, Adam states, nominal D/G/prior rates `.00425/.00425/.0085`, and original 1200-step noise horizon.

| Arm | Update 2401 | Update 2402 | Updates 2403–2410 | First full recovery |
| --- | --- | --- | --- | --- |
| Ordinary | 8 modes, HQ 1 | 8, HQ 1 | 8, HQ 1 throughout | Already full |
| Conditioned D bank | 7, HQ 1 | 7, HQ .92334 | 8, HQ 1 throughout | 2403 |

The conditioned branch's one-step mode loss is **transient in this measured ten-update continuation**. It is still a real fixed-target coverage failure at 2401 and 2402, so the strict every-update guarantee fails. This one rare conditioned event does not estimate failure frequency or establish indefinite recovery. The suffixes had identical generated host and final RNG, nine D/G Adam advances and 27 optimizer callbacks per role each, and all nine joint fits selected. Wall times were 1.81 s for ordinary and .75 s for conditioned after restoring the update-2401 snapshots; these are local CPU times, not comparable standalone 2400-step training costs.

The [raw receipts, input/final states and source hashes](continuous-evidence/round8-prestart-omission-recovery10/manifest.json) are portable. The recorded 2401 branch remains the original [one-step diagnostic](continuous-evidence/round8-prestart-omission-recovery10/first-step-result.json). No loss, optimizer, architecture, gain, or target distribution was changed in the continuation.
