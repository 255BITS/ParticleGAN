# Completed feature and selective-gradient scouts

All four 50k→70k runs completed and were re-certified against current trainer/config sources. No variant merits a long promotion on these results. Both GPUs are idle after read-only checkpoint probes; no further training is queued.

| Recipe | Final FID50k ↓ | Test MSE ↓ | Train min |
|---|---:|---:|---:|
| Control | **20.3672** | 0.03719 | 14.91 |
| New G branches receive adversarial gradients only | 22.6733 | **0.03518** | 16.91 |
| ResNet34 + selective G growth | 27.8205 | 0.03567 | 19.40 |
| ResNet34 replacement | 86.5698 | 0.04064 | 17.06 |

Selective G growth briefly reached18.9398 at55k, essentially the parent's18.9012, then degraded. The combined run moved22.25→55.81→31.79→27.82, showing a large transient failure followed by partial recovery. ResNet34 alone moved21.60→22.71→23.06→86.57. Final sample grids show many noisy texture patches in the ResNet34-only run; the feature variance trace ratio remains~1.074, so this is not evidence that all diversity vanished.

## Read-only gradient diagnosis

`experiments/probe_cifar_ae_features.py` restores the actual backbone and growth architecture, checks checkpoint source hashes, and measures16 batches of64 examples using live G/D/E/prior weights. Reconstruction gradient measurements respect the selective routing. FID uses EMA and is not measured by these probes. Checkpoint hashes are verified unchanged afterward. Raw results: `GRADIENTS.json` and `GRADIENTS_CONTROL65.json`.

| Checkpoint | G adversarial gradient norm | Applied G reconstruction gradient norm | D fake-input gradient norm | Paired real score > fake score |
|---|---:|---:|---:|---:|
| Control65k | .17870 | .14850 | .65763 | 59.9% |
| ResNet34 65k | .01023 | .05448 | .01496 | 46.9% |
| Control70k | .19732 | .09313 | .72447 | 63.7% |
| Selective G70k | .09681 | .05581 | .26473 | 59.3% |
| ResNet34 70k | .36878 | .07448 | .23991 | 36.6% |
| Both70k | .06116 | .19044 | .04223 | 43.4% |

Before the final failure, ResNet34's65k critic-input gradients were~44× weaker and G adversarial gradients~17.5× weaker than the contemporaneous control. Applied reconstruction gradients were~5.3× adversarial on G, with cosine−.251. Logged adversarial losses hovered near log(2) through much of this interval; the probes confirm weak feedback rather than relying on losses alone.

At70k the ResNet34 G gradient has rebounded, so it is inaccurate to describe the endpoint as uniformly vanishing gradients. Its critic favors generated images over real ones on most sampled pairs despite poor generation. This points to unreliable feedback. The combined70k checkpoint also has weak critic gradients and real-above-fake43.4%, despite its partial FID recovery.

Selective G routing did not fix generation. Removing direct reconstruction gradients from the new branches does not isolate them from coadaptation: old G, encoder and prior still learn reconstruction, changing the branches' inputs and shared outputs. These results weaken a simple direct-new-branch objective-conflict explanation, without proving reconstruction irrelevant.

## Limits and next recommendation

The ResNet34 swap changed feature coordinates while retaining the learned heads and Adam state, and joint G/D training began immediately. Failure tests that swap-and-adapt procedure; it does not establish that a larger pretrained representation is intrinsically worse. The large negative factorial interaction is dominated by the standalone ResNet34 failure and is not grounds to promote the combination.

The unchanged control ended20.3672 versus19.2033 in the prior growth round. Production uses TF32/cuDNN benchmarking rather than the deterministic replay-test settings. The discrepancy is a reproducibility limitation; its precise cause has not been isolated, and these runs do not provide a statistical variance estimate. Use each round's own control and avoid strong claims based on tenths of a point across rounds.

Recommendation: no longer run from these endpoints. A targeted next test would hold G/E/prior fixed briefly after a backbone swap, allowing D's trainable heads to adapt before resuming joint training, with a matched D-only adaptation control. Record discrimination, input gradients and actual GAN learning feedback before releasing G. This is a proposal, not an active or authorized new job. Fresh heads or revised regularization are separate possible interventions and should not be silently combined.

Detailed curves and protocol: [LEADERBOARD.md](LEADERBOARD.md). The original below13 target remains unmet.
