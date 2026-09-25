# Smooth discriminator handoff

**An architecture-only unequal-width PASS for the original recipe.** Axis Fourier + Softplus(beta5) sustains seven final observations at the original1,200 steps. No architecture passes all six data toys; each full finalist profile passes3/6. See [findings](FINDINGS.md) and the [complete matrix](MATRIX.md).

This separate bounded study contains30 GAN episodes,720 fixed live observations and193.454 seconds summed recorded wall time. Eight declared D architectures screen three hard toys, then two finalists receive the other three. Original beta99 optimizer, LRs, loss, penalty, prior regularizer, G, particles, batch, budgets and thresholds are unchanged. EMA is separate. No target-derived features or seed sweeps.

## Evidence

- [Combined index](index.json.gz), [screen plan](screen/plan.json.gz), [cross plan](cross/plan.json.gz), [finalist selection](selection.json.gz). Every episode includes full live/EMA curves, actions, actual updates, candidate architecture, original/effective specs and verdicts.
- [Source archive](source.tar.gz):58 exact numerical files, including the research module. [Screen protocol](screen/protocol.json.gz) and [cross protocol](cross/protocol.json.gz) have identical hashes. Only one identical source copy is retained.
- [Driver](scripts/run.py), [screen log](screen/run.log), [cross log](cross/run.log), [static architecture checks](architecture_checks.json.gz), [validation](validation.json.gz).
- [Reusable module notes](reusable/smooth_critic_research.md), [focused tests](reusable/test_smooth_critic_research.py), [10-test result](tests.log). The numerical module lives in the source archive at `benchmarks/transfer_suite/smooth_critic_research.py`.
- [Inventory](inventory.json.gz) records compressed/file and original uncompressed SHA256 hashes. [Portable verification](verification.json.gz) checks all source hashes and complete numerical verdicts.

## Reproduction

Base checkout: `a11c5304cde01c7fdc96e8a49a5a576b8cb8ebff`, plus the exact archived research module. Original worktree `/ml2/hypergan/ParticleGAN-pr36-valid-recipe`; environment `/home/mikkel/anaconda3/envs/conceptmod/bin/python`; CPU thread1. Research driver temporarily replaces only the vector runner's discriminator constructor, restores it after each serial episode, and records the replacement architecture explicitly. It never patches the GAN loss, targets, metrics or optimizer.

Original commands were `python -u run.py --phase screen` and `python -u run.py --phase cross --names axis_softplus5 axis_tanh`. The driver retains original absolute worktree/output paths; reproduce there or adapt a separate copy, never overwrite retained evidence. Restore exact source bytes from the archive. Runtime/build fingerprint is in each phase protocol.

Run `python verify.py` to verify this bundle without training. `sha256sum -c SHA256SUMS` checks every stored file. Original JSON bytes and all failed attempts are preserved.
