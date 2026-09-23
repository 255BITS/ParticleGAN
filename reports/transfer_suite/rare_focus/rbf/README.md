# Local Gaussian-feature discriminator handoff

**No sustained rare-mode winner in eight declared architectures.** The best final result passes every bound except minimum component eigenvalue ratio (.131176 < .15). No full-six followup was run. See [findings](FINDINGS.md) and the [parameter-count leaderboard](MATRIX.md).

Exactly 8 GAN episodes, 192 fixed live observations and 75.793955 seconds summed recorded wall time. All cards retain original G, 256 particles, batch128, 1,200 steps, Rp logistic, b_cap3/kappa1.25, prior regularization .05, no particle L2, Adam(0,.99), G/D/prior LRs .001/.0015/.01, cosine and 1:1 updates. EMA is separate. Thresholds and final-five-of24 live requirement are unchanged.

The architecture replaces the original Fourier feature map with raw coordinates plus Gaussian responses at generic standard-normal centers. Declared octave widths are fixed or learned along with centers through only the existing D objective. No target data, labels, centers, moments, normalization, auxiliary loss or seed selection enters construction. Every episode explicitly records research_discriminator metadata and D parameter count.

## Retained evidence

- [Combined index](index.json.gz), [frozen plan](screen/plan.json.gz), [stop decision](selection.json.gz), and [log](screen/run.log).
- [Source archive](source.tar.gz): all 59 numerical dependency files, including `benchmarks/transfer_suite/local_critic_research.py`. [Protocol](screen/protocol.json.gz) records source hashes and exact driver/checker hashes.
- [Driver](scripts/run.py), [static architecture checker](scripts/check_architectures.py), [check results](architecture_checks.json.gz), and [independent validation](validation.json.gz).
- [Inventory](inventory.json.gz) contains stored and original uncompressed hashes. [Verification](verification.json.gz) records the portable audit.

Source base is commit `981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45` plus the exact archived research module. Its SHA256 is `18b1a4f5c7c16bd59599cf1a1a3810cc61dbd63059e897a47bfdd43aab2ce0ff`. No reusable-code/default change is proposed from this unsuccessful screen.

## Reproduction

Restore the exact archive over the base checkout. Original worktree `/ml2/hypergan/ParticleGAN-pr36-valid-recipe`; Python `/home/mikkel/anaconda3/envs/conceptmod/bin/python`; CPU thread1. Original command: `python -u run.py --phase screen`. Driver paths are retained exactly; use those paths or adapt a separate copy without overwriting retained evidence. The driver serially replaces only the discriminator constructor and restores it after each episode. It does not alter the generator, targets, loss or metrics.

All JSON preserves its uncompressed original bytes; original episode gzip bytes are also preserved. Run `python verify.py` without training to verify source hashes, gzip roundtrips, exact architecture-only specs, full curves and independent live/EMA verdicts. `sha256sum -c SHA256SUMS` checks the stored bundle.
