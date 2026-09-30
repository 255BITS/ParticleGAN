# GAN-4 optimistic Adam — kill

Mechanism: PR84 alternating Adam (smoothed G critic, G curvature 0.25, D curvature 3) with Daskalakis optimistic Adam at α=1 on each committed player step. Previous direction starts at zero. No extra gradient and no added loss.

Purity: GAN dynamics only — no coverage/likelihood term.

Host: this process, PyTorch 2.14.0+cpu, one thread, seed 0. Cold uses constant nominal rates (lr floor 1, no anneal). Warm continues the scheduled prefix from step 1000 at constant rates.

## Gate table

| Gate | PR84 pin | Optimistic Adam α=1 |
| --- | --- | --- |
| Warm checks 1001–1200 | 0/200 eight-mode passes. Stays at **6 modes** (min 6, HQ 1 at step 1200). Step 1000 itself is 6 modes / HQ 0.994, so this torch never enters the historical 8-mode warm cloud. | 0/200 eight-mode passes. **Min 3 modes.** 33 checks below 6 modes (first at 1002). Terminal still 6 / HQ 1. |
| Cold trajectory | **PASS**, identity MSE 0.000942668 (≤ 0.02), suffix 18. 3.53 s. | Not run. Warm regressed. |
| Cold ring 1200 | **FAIL**, live **7 modes**, HQ 0.9927. 23.9 s. | Not run. |
| Stay | Not run. | Not run. |

Warm regression is the stop rule. Optimistic Adam is worse than the same-process PR84 continuation on every mode-count summary that is not the shared terminal checkpoint.

## Rank

Unranked on the GAN-native track. No acquisition and no stay. No merge and no 22/22 claim.

## Keep / kill / next

**Kill** this optimistic-Adam layer. Do not sweep α.

Next single bet, still optimizer geometry: one extragradient step (gradient at the extrapolated point, parameter step from the base point) on the same PR84 alternation, instead of mixing the last two Adam directions with no extra evaluation.
