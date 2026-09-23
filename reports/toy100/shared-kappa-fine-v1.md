# Fine κ bracket for the shared recipe

Sixteen κ values from 1.172 through 1.188 in steps of .001, excluding the previously tested 1.180, were declared before training. Each config changed only `reg_kappa` and `name` from the exact earlier 18/19 base (SHA-256 `52f2971ac493d5d10f0c54666efcb95b8bd31c54544e16cd14f9822f7694e21f`). The source was frozen at `1c1a0865fe605c9f212d832c06596c12510f937e`; fixed seeds, host budgets, gates, and noise/optimizer fields were unchanged. The manifest SHA-256 is `ffdc2ca8ad723d5f8f9fae4f1117571517d6e5416062565b76d3e4e328747b1e`.

**κ=1.176 passed a fresh strict 19/19 older-host replay**, after a separate strict 10/10 staged screen. Its exact config is `configs/k1176.json` (`artifacts/toy100-accuracy/compatibility/shared-kappa-fine-v1/configs/k1176.json`), SHA-256 `6de24743336a3ce7e922deceab91dff0688bdb28ae8f5e6ff2c8b88b9f2bec27`. This search result covers the older 19. A subsequent [fresh production-runner replay](shared22/README.md) verifies the same exact configuration at **22/22**, including all three native accuracy gates.

| Stage | Attempted | Strict passes | Rejected at stage |
| --- | ---: | ---: | ---: |
| Residual-student | 16 | 12 | 4 |
| Trajectory | 12 | 4 | 8 |
| Mode-hold | 4 | 2 | 2 |
| Stripes | 2 | 2 | 0 |
| Bars | 2 | 1 | 1 |
| Remaining five | 1 | 1 | 0 |
| Fresh full 19 | 1 | 1 | 0 |

The adjacent κ=1.177 also passed residual, trajectory, mode-hold, and stripes, but failed bars; κ=1.181 and 1.184 passed residual and trajectory but failed mode-hold. Every later stage on those rows was marked **skipped**, not failed. The 16-row screen took 282 seconds of measured wall time; no attempted host was stopped early within its frozen budget.

Complete evidence (`artifacts/toy100-accuracy/compatibility/shared-kappa-fine-v1`) contains every predeclared config and hash, source archive, compressed attempted episodes, event curves, independent stage grades, a fresh full-19 episode, and explicit skip receipts. All 333 retained files (14,382,171 bytes) matched the RAM originals by SHA-256. Independent regrading after relocation reproduced all 38 attempted stage grades, including 19/19 for κ=1.176.
