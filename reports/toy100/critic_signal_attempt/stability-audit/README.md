# Selected H stability preparation

Use only `h_n05r06_mixup_c0p01_lr15`. This directory contains an exact frozen source copy, a read-only audit, and three fixed warm-state probes. It does not publish or change the source attempt.

## Audit result

- All **125** source hashes match `batch-h/manifest.json`; archive SHA256 is `ed4a61dcfebc13f1134630db096aad043d60e4846083b4a1384e871ee9ebb334`.
- Acquired ring checkpoint SHA256 is `799181c2a68df02ee5a7a5963b4a5e4fbf23677623c4fcd9f8757460944155b2`.
- Replaying the retained continuation reproduces **all 1,200 dense metric rows and the complete final checkpoint state exactly**, including live G/D/prior, both Adam states, EMA and all global/data/input-noise RNGs. See `audit-findings.json`, `audit_state.py`, `audit-control-hold/`.
- The first failure is **step 1255: 8 modes, HQ 0.780517578125**. The full hold passes 750/1,200 dense checks and ends at 5 modes/HQ 0.209228515625. No restoration, rate, objective, or noise discontinuity was found that explains it.
- H uses logistic relativistic-pair GAN gradients for G/particles only; D retains R1+R2 coefficient 0.6 and mixup consistency 0.01. Auxiliary coverage, particle L2, VICReg and residual losses are disabled. The discriminator input channel remains fixed sigma 0.05 even when the legacy policy's unused *nominal* input sigma decays to zero. Output noise remains 0.029 after its original 240-update warmup. All Adam states survive: G/D LR 0.0015, prior LR 0.003, beta=(0,0.999), epsilon=1e-8.
- The frozen ring host uses **12 particles, latent dimension 4, batch 128, width 96**, not the native 100-mode dimensions carried in `config.json`. The runner preserves the host resources. Evaluation takes 4,096 samples with the archived step-specific output-noise RNG and fixed latent indices, isolated from every training stream.

The eligible statement is about the update objective. The archived adapter remains explicitly **ineligible for the production common gate** until supported by its admission checks. These are stability probes, not production results.

## Executable probes

Each command restores the exact same H state at step 1200. There is no cold rerun, optimizer reset, seed change, altered noise/penalty, target statistic in an update, LR decay, or freezing. Logs include dense HQ/mode metrics, losses and all optimizer rates; `metrics.jsonl` is easy to tail.

```bash
cd /ml2/hypergan/gan-attempts/selected-h-stability-prep
bash run-probe.sh control outputs/control-screen 200 > outputs-control.log 2>&1
bash run-probe.sh critic_refresh2 outputs/critic-refresh-screen 200 > outputs-critic-refresh.log 2>&1
bash run-probe.sh average2 outputs/average-screen 200 > outputs-average.log 2>&1
bash run-probe.sh extra_adam outputs/game-correction-screen 200 > outputs-game-correction.log 2>&1
```

Only the controls were executed during preparation. Each command refuses to overwrite an output directory. The control's dense prefix is exact and stops at the first failure, update 1255. Full diagnostic control is retained separately in `runner-control-full/`.

| Fresh attempt | Single bounded intervention | Actual work per outer update |
| --- | --- | --- |
| Critic tracking | `critic_refresh2`: two fresh critic steps before one generator/particle step | D2/G1 backward evaluations and committed Adam steps |
| Gradient variance | `average2`: average two independent H minibatch gradients for each player before Adam | D2/G2 backward evaluations, D1/G1 committed Adam steps |
| Game correction | `extra_adam`: predict one alternating H update, calculate fresh D/G gradients at lookahead, restore original model and Adam state, apply corrected gradients once | D2/G2 backward evaluations; D1/G1 provisional and D1/G1 committed steps |

The game correction explicitly discards provisional Adam moments; they are not retained as two committed updates. RNG streams advance through both independent draws. This is an alternating predictor with a joint corrector, not an unlabelled optimizer substitution. `attempt-recipes.json` is the machine-readable declaration.

## Fixed screening order

1. Run the assigned **single 200-update warm probe**, with dense evaluation beginning at step 1201, before H's first failure. Stop immediately on any check with modes != 8 or HQ < 0.90. A complete successful screen is labelled `SHORT_PASS` and is not qualification. A failure ends spending on that fixed proposal.
2. A survivor must be implemented consistently in the frozen cold hosts and pass the same cold gates, using H's exact global recipe and fixed seeds. Preserve the H control and account for the extra D/G work. Do not treat borrowing H's acquired checkpoint as proof that the changed update rule can acquire its own modes.
3. Require the survivor's **own acquired state** to hold for 1,200 additional updates under its same update policy, before later recovery and larger gates. The prepared runner's `1200` option is an optional longer **borrowed-H warm probe**, not a substitute for this own-state qualification.

No broad parameter or seed search is prepared. Full19/full22 or 100-mode evidence belongs to the separate verification work; it is not inferred from the ring audit.
