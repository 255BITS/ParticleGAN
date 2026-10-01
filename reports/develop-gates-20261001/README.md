# Post-merge gate diagnosis

The failed develop job did not test E22 or Atlas. It ran the legacy BCap
configuration through the historical 22-host runner. PR #209 had changed the
three native problems to clean scoring, while the 19 transfer hosts retained
their original output-noise sampling law. The combined checker compared the
training noise settings but omitted this evaluation-law difference.

The native clean failures are real under that law. Their within-mode covariance
is about 98% below the target: the generator's narrow points are insufficient
without the .029 output-noise component used in training and in the original
served qualification. These failures do not establish an E22/Atlas regression.

## Paired saved-draw evidence

The [diagnostic receipt](paired-sampling.json) binds the exact original sample
files from [Actions run 36829031177](https://github.com/255BITS/ParticleGAN/actions/runs/36829031177).
It adds only the declared .029 output noise using the existing evaluation seeds.
No model was retrained and no original receipt or acceptance threshold changed.

| Native problem | Clean terminal passes | Original noisy law terminal passes | Clean holdout covariance bias | Noisy holdout covariance bias | Noisy holdout radial KS |
| --- | ---: | ---: | ---: | ---: | ---: |
| grid100 | 0/5 | 5/5 | -.98040 | -.04194 | .01713 |
| rotated100 | 0/5 | 5/5 | -.97536 | -.03826 | .01594 |
| staggered100 | 0/5 | 5/5 | -.97836 | -.03974 | .01621 |

All three 100k-draw noisy holdouts pass the unchanged coverage/fidelity limits.
This is an explanatory paired-draw diagnostic, not a new training qualification.
The repaired checker grades the original mixed-law aggregate **INCOMPLETE**,
while preserving native clean **0/3 FAIL** and transfer **19/19 PASS**.

## Repair and verification

Native toy100 scoring remains clean by default. `--eval-output-noise` explicitly
selects the benchmark wrapper's original served law for both terminal and
holdout draws. The common-22 runner now selects that law before training,
records it, checks it against the transfer law, and rejects clean-only requests
before spending. The CI workflow names the actual legacy BCap cohort and binds
its configuration explicitly. It supplies no clean-sampler or Atlas credit.

New software probes compare complete trainer checkpoints across clean/served
scoring, including models, optimizers and RNG streams, and check the independent
holdout writer. Additional negative controls reject mismatched sampling laws
even when every metric otherwise passes. No `particlegan/` file is changed.

Fresh full-budget common-22 and original Atlas19 replays are running. Their
results must be recorded separately after completion. The Atlas replay uses
the unchanged qualified config, constructors, hosts, scorers, schedules,
sampling law and denominator specified in [the frozen protocol](PROTOCOL.md).
Original evidence remains historical; a pending cell earns no fresh pass.

Raw logs, checkpoints and clouds remain in local artifact directories. The
release branch is untouched.
