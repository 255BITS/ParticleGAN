# Actual caption verification: G-only clean convergence gate failed

The single fixed G-only clean caption candidate completed 6,400 updates and
returned **scientific FAIL**. At 6,400 it improves neutral particles under
all four common critics, but ordinary native-game LoRA remains better under
all four. At 5,120 its comparison with neutral is mixed and ordinary leads
under all four. The portable synthetic test's small win therefore did not
transfer to the required actual-caption convergence gate.

The gate was frozen before training: at **both** 5,120 and 6,400, the candidate
had to beat **both** controls by more than `1e-4` under **all four** fixed
common native paired RpGAN critics. Zero-code ablation also had to worsen
every game by more than `1e-6`, with calibrated critics and live particles.
Lower is better; positive candidate-minus-control values are worse.

| Updates | Fixed critic | Candidate game | Minus ordinary | Minus neutral | Zero-code minus live |
|---:|---|---:|---:|---:|---:|
| 5120 | Ordinary at 800 | 2.574321 | +0.030604 | -0.000242 | +0.081272 |
| 5120 | Ordinary at 6400 | 2.561327 | +0.015180 | +0.000772 | +0.021879 |
| 5120 | Particle at 800 | 2.860213 | +0.023338 | +0.001459 | +0.033758 |
| 5120 | Particle at 6400 | 2.552848 | +0.013840 | -0.000211 | +0.020109 |
| 6400 | Ordinary at 800 | 2.589291 | +0.014417 | -0.006563 | +0.099086 |
| 6400 | Ordinary at 6400 | 2.570100 | +0.008743 | -0.002286 | +0.029857 |
| 6400 | Particle at 800 | 2.869771 | +0.005694 | -0.010109 | +0.053514 |
| 6400 | Particle at 6400 | 2.562737 | +0.008772 | -0.000498 | +0.026321 |

All four candidate games worsen between the two fixed endpoints. Both are
retained; neither selects a checkpoint. Code helps under every critic at
both endpoints. Bank and query gradients are live on 6,399 of 6,400 updates.
No structural moves were accepted, so this is not evidence for a structural
birth/death improvement.

Only the differentiable G forward changed to clean routed codes. D retains
native DV12. A discarded noisy G forward pays the original draw cadence and
diagnostics on each update. Fresh public initialization exactly reproduces
the original H/b-neutral owners and sampled C, before native policy/EMA
construction and adoption of the retained step-0 global RNG through public
restore. No restored owner is reinitialized. The current-native inherited
800-to-802 state/loss/caller-stream witness, both controls' 6,400-update clean
TEST replay, and candidate 800-to-802 recovery pass exactly. Final learned
models, gradients and optimizer moments are finite; intended native monitor
sentinels remain preserved. Runtime sources and the executed card stay fixed.

Final C norms and gain closure are descriptive, not acceptance criteria. The
closure fraction counts unweighted retained projection coordinates, including
masked token slots, on all 240 TEST contexts and both CFG halves. It measures
`1+tanh(Cz) < 0.05`; it does not measure output quality or causal importance.

| Site | Final C norm | Gain closure fraction |
|---|---:|---:|
| Context projection | 1.034571 | 0.003418 |
| Self-attention QKV | 1.107637 | 0 |
| Self-attention projection | 1.075487 | 0 |
| Cross-attention Q | 1.106665 | 0.036349 |
| Cross-attention KV | 1.034804 | 0.232300 |
| Cross-attention projection | 1.061046 | 0 |

The Python completion clock records 912.899 seconds; the retained parent
launcher receipt records 914.728 seconds from launch to exit, within the fixed
1,750-second allowance. Both receipts and the launcher log are SHA-bound in
the compact JSON readout.
This includes 6,400 quality updates, four replay updates, 6,400 discarded noisy
G forwards plus two replay discards, all fixed endpoint evaluations and final
writes. It is not a matched speed comparison.

The external caption/model/source fixture makes this actual verification a
local adapter; the portable public-API synthetic test remains separate.
The inherited controls are provisional observations with new public replay
and score checks; their original failed CPU review is unchanged. Ordinary's
BF16 branch arithmetic differs from particles' FP32 branch arithmetic. The
saved-state mean-gradient comparison was low-confidence and did not establish
mean-gradient bias. This result supports neither a general DV12 defect nor
full-Supra promotion. Stop this exact fixed candidate; no automatic tuning or
continuation follows the failure.

Exact scores, fixed criteria, scope, hashes and cost are recorded in
[`e22_routed_g_clean_caption_results.json`](e22_routed_g_clean_caption_results.json).
