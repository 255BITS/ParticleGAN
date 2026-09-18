# Completed checkpoint capacity scouts

All four 50k→70k scouts completed and passed current source/config certification. Both GPUs are idle. All arms finish with identical training RNG stream states. No new training was launched after completion.

| Recipe | Final FID50k ↓ | Test MSE ↓ | Training cost versus control |
|---|---:|---:|---:|
| Control | **19.2033** | 0.03499 | 1.00× |
| Larger D heads | 19.2402 | 0.03988 | 1.11× |
| Larger G + D heads | 20.1437 | 0.03578 | 1.28× |
| Larger G | 22.8981 | **0.03311** | 1.14× |

No expansion beat unchanged continuation at the final endpoint. None of the final endpoints improved on the common parent's FID18.9012. D-head growth reached18.5407 at65k but rebounded to19.2402 at70k; that selected intermediate minimum does not establish a sustained improvement. All checkpoints remain available.

## Interpretation

Adding G capacity improved reconstruction MSE by5.4% relative to control while worsening generation FID by3.69. Its FID trajectory was19.7458→19.9265→20.2800→22.8981 as reconstruction continued improving. This weakens the simple hypothesis that more G depth alone will resolve the plateau under the current objective and optimizer settings. The added capacity was trainable and function-preserving at insertion, so this was not a reset of the learned generator.

Expanding D alongside G reduced the damage: combined FID20.1437 versus22.8981 for G alone. D-head capacity therefore affected the outcome, although expanding both still lost to control. The factorial interaction is−2.7912 on the FID scale; this single trajectory gives no confidence interval or unique causal diagnosis.

These findings are consistent with extra G capacity benefiting the reconstruction task more than prior-sample generation, or with D feedback failing to constrain the expanded G adequately. They do not prove reconstruction is the sole cause. Earlier encoder-only/detached reconstruction scouts did not improve short continuations, and growth introduces fresh optimizer state on the new parameters. Other architectures, larger pretrained feature backbones, or different adaptation schedules remain untested here.

## Recommendation

Do not promote these four final checkpoints into long training on the basis of this scout. Larger D heads cost11% more and provide no final gain; the generator expansions are worse. Preserve the D-only65k checkpoint as a diagnostic candidate rather than declaring it a winner.

For a next discussion, distinguish changing the pretrained discriminator representation from adding trainable head capacity: this round tested only the latter. Another targeted hypothesis would restrict reconstruction gradients specifically on newly added G branches while retaining the old network's routing; the prior encoder-only experiment removed reconstruction gradients from all of G and did not test this selective version. Both are proposals, not queued jobs.

Detailed curves, timing and protocol are in [LEADERBOARD.md](LEADERBOARD.md). The contemporaneous control is the reference for this round; historical one-D continuations differ slightly and should not substitute for it.
