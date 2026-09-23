# Final linear-bypass handoff

**A sustained rare-mode PASS with the original recipe and256-particle profile.** The width96×2/Softplus5/Fourier2 critic plus an initially zero raw linear bypass passes the last6/24 checks. Its full data profile is3/6; see [findings](FINDINGS.md) and the [complete matrix](MATRIX.md). Other task architectures and parent independent replay remain separate evidence.

Four declared refinements produced one rare winner and three failures. The winner then received the other five data toys: exactly9 GAN episodes,216 live observations and64.604173 seconds summed recorded wall time. Every loss/penalty/prior/optimizer/G/resource/budget/metric setting remains original; only D architecture changes. EMA is separate. No further cards were run.

## Evidence

[Combined index](index.json.gz), [screen plan](screen/plan.json.gz), [cross plan](cross/plan.json.gz), [selection](selection.json.gz), [screen log](screen/run.log), and [cross log](cross/run.log) retain all outcomes. Each episode contains explicit candidate architecture and parameter count, original/effective specs, full live/EMA curves, actions, actual updates, runtime and verdicts.

[Source archive](source.tar.gz) contains61 exact numerical files; [screen](screen/protocol.json.gz) and [cross](cross/protocol.json.gz) protocols share all source hashes. Base checkout is `981ccbcd6e7e77a1f41f8aac3cc42d1fa1ceab45` plus archived research modules. The reusable implementation is `benchmarks/transfer_suite/linear_skip_refinement_research.py`, SHA256 `399a6d6dc268bd58952366c5d11e57886faf2cdb7accf882c47fe1e846e4bd1c`.

[Static checks](architecture_checks.json.gz) include exact old64/beta5 state/output/input-gradient/parameter-gradient/active-cap parity and pointwise backward checks for all four new cards. These checks are not training episodes. [Validation](validation.json.gz), [inventory](inventory.json.gz) and [portable verification](verification.json.gz) cover full source, byte and numerical invariants.

## Working reproduction

Restore the archive over its base checkout. Original environment: `/home/mikkel/anaconda3/envs/conceptmod/bin/python`, CPU thread1. To replay only the successful rare case into a new directory:

```sh
python scripts/reproduce_winner.py --source-root /path/to/restored/checkout --output /tmp/new-linear-skip-replay
```

This [standalone replay helper](scripts/reproduce_winner.py) records complete results and a fresh exact source archive. Its CLI was checked with `--help`; it was not executed as an additional GAN run in this bundle. Parent independent replay is counted separately.

For direct model construction:

```python
from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES, constructor
card = next(c for c in ARCHITECTURES if c["name"] == "linear_skip_d96_beta5")
critic = constructor(card)(2, 96, 2, 2)
```

The original [study driver](scripts/run.py) commands were `python -u run.py --phase screen` and `python -u run.py --phase cross --names linear_skip_d96_beta5`. That driver retains original absolute paths (`/ml2/hypergan/ParticleGAN-pr36-valid-recipe` and `/tmp/pr36-valid-linear-final`); adapt a separate copy when rerunning, never overwrite retained evidence. It replaces only the D constructor for serial episodes and restores it afterward.

All original JSON bytes and original episode gzip bytes are preserved. Run `python verify.py` without training and `sha256sum -c SHA256SUMS` to audit this bundle.
