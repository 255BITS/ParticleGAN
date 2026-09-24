# Idle-when-covered critic response — warm kill

Host: neural. Seed 0. One CPU thread. PyTorch 2.14.0+cpu. GAN dynamics only: no coverage loss, Chamfer, mode assignment, likelihood, or clip ladder.

## Mechanism

G keeps the PR84 five-point stencil unless the current real batch has a sharp-critic peak that beats every clean particle by 0.5, sits more than 1.0 from the particle cloud, and the nearest particle's sharp central difference points away. Those nearby low-critic rows then read the sharp critic at the first of eight segment samples where the critic points toward the peak. The offset is detached. Otherwise every G phase is the PR84 stencil.

## This harness's PR84 pin

The scheduled warm prefix (the fork used by the warm probe) never reaches eight modes. Identity and PR84 share warm-state `d5dd3109e49af00aaef75a47094d85be8681119892f353aa5e54350720c4a260`.

| Gate | PR84 |
| --- | --- |
| Cold trajectory | PASS, identity MSE `.000942668` |
| Cold ring | Terminal 1000–1200: **7 modes**. HQ `.788 / .718 / .905 / .943 / .993`. Verdict FAIL against an 8-mode bar. Wall 27.6 s |
| Warm 1001–1200 | **0/200**. Min **6 modes**, min HQ `.935`, final **6 / 1.0**. Identity fork also 0/200, final 6 / `.999` |

That warm pin is not the board's ~196/200 eight-mode cloud. It is the passing behavior of this build: a stable six-mode continuation. Cold trajectory matches the known PR84 MSE. Cold ring matches the seven-mode Codex pin, not a terminal eight.

## Idle response

| Gate | Result vs this pin |
| --- | --- |
| Warm 1001–1200 | **0/200**. Min **4 modes**, min HQ `.425`, final **5 / .844** |
| Cold trajectory | Not run |
| Cold ring | Not run |

The response armed on 99 of 200 warm steps. Proposal-phase `response_uses` totaled 0. The curvature replay still evaluates the armed field, and the continuation loses a mode. Warm regresses against the PR84 pin, so the candidate stops here. No margin or gap retune.

## Rank

No rank claim. The candidate does not stay on this harness's warm solution, and it was not trained from a cold start.

## Keep / kill / next bet

**Kill.** Next bet, if any, has to keep the curvature replay on the PR84 stencil whenever the proposal itself did not use the response. This channel does not.
