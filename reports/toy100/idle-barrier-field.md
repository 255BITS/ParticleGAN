# Idle anti-gradient critic field — KILL

**Host:** neural. **Purity:** GAN dynamics only — no coverage/likelihood term.

One change on PR84: during G's smoothed-critic evaluation, each fake particle looks against its own critic slope (step 0.5, at most 4 steps). A detached push toward that higher critic value is added only when the look beats every current fake logit. Otherwise the term is zero. Constants were not searched. Cold ring was not run.

This harness is PyTorch 2.14.0+cpu, one thread. The scheduled step-1000 prefix is already 6 modes, so the 200-check warm fork is not an 8-mode cloud. Comparison is against PR84 on that same prefix.

| Gate | PR84 pin | Idle barrier field |
| --- | --- | --- |
| Warm, updates 1001–1200 | 0/200, min 6 modes, min HQ .935, final 6 / HQ 1 | **0/200, min 4 modes, min HQ .558, final 5 / HQ .831** |
| Cold trajectory | PASS, MSE .000942668 | not run |
| Cold ring | FAIL, 7 modes, HQ .993, 0/24 checks | not run |
| Stay | not run | not run |

The field fired on 7118 of 76800 fake-particle queries in the 200 warm updates. A 6-mode cloud still has critic peaks above the current fakes, so the "beat every fake logit" gate does not stay idle, and those pushes drop modes. That is a warm regression versus PR84. Stop.

**Rank:** no claim. Below PR84 on the GAN-native track.

**Next single bet (not run):** do not retune the look length. A critic look that can see an empty mode from an occupied particle will also fire on an incomplete warm cloud.
