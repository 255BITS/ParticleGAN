# Raw-coordinate and quadratic-skip handoff

**Eight architecture cards; no sustained rare winner.** The smallest linear-skip critic passes all final numerical bounds but only four final observations, so it remains a failure. See [findings and last-five metrics](FINDINGS.md) and the [parameter-count matrix](MATRIX.md). No full-six followup was run in this sealed round.

Exactly 8 GAN episodes, 192 fixed live observations and 63.900755 seconds summed recorded wall time. All original loss, regularization, optimizer, generator, 256-particle/batch128 resource profile,1,200-step budget and numerical/stability thresholds remain fixed. EMA is separate. Each episode records research_discriminator metadata explicitly; only architecture changes.

The cards add raw linear/quadratic skip heads, separate smooth raw/quadratic MLP paths, or residual smooth branches. Fixed monomials feed the ordinary learned critic; no covariance-supervision term, target statistics, normalization or Fourier-amplitude search is introduced. Every added skip output initializes to zero, preserving its base critic initially. The original-axis branch state/output parity and active cap backward are statically checked.

## Evidence and reproduction

[Combined index](index.json.gz), [plan](screen/plan.json.gz), [stop decision](selection.json.gz), [log](screen/run.log), [architecture checks](architecture_checks.json.gz), and [validation](validation.json.gz) retain every outcome. Full episode gzip files include original/effective specs, candidates, live/EMA curves, actions, actual updates and runtime. Original JSON bytes and all failures are preserved.

[Source archive](source.tar.gz) contains all60 exact numerical dependency files. Base commit is `981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45` plus archived research modules. The relevant module is `benchmarks/transfer_suite/skip_critic_research.py`, SHA256 `ac27627990b53d3063bd62fde9823f8b4ccd88df9d48730f6695a0003fe5524e`. [Protocol](screen/protocol.json.gz) hashes all sources and the exact [driver](scripts/run.py)/[checker](scripts/check_architectures.py).

Original worktree `/ml2/hypergan/ParticleGAN-pr36-valid-recipe`; environment `/home/mikkel/anaconda3/envs/conceptmod/bin/python`; CPU thread1. After restoring the archive, run `python -u run.py --phase screen`. The driver retains original absolute paths; reproduce there or adapt a separate copy, never overwrite retained evidence. It serially replaces only the D constructor, then restores it. All nonarchitecture settings stay original.

For direct construction, import `ARCHITECTURES, constructor` from `benchmarks.transfer_suite.skip_critic_research`, select the named card, then call `constructor(card)(2,64,2,2)`. The linear near-miss adds `nn.Linear(2,1,bias=False)` with zero weights to the exact original-axis Softplus5 base.

Run `python verify.py` to validate the bundle without training; [verification](verification.json.gz) records source hashes, gzip roundtrips, exact specs, observations and recomputed live/EMA verdicts. [Inventory](inventory.json.gz) and `SHA256SUMS` cover all stored files. Subsequent four-card refinements, if any, are separate evidence.
