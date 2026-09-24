# Constant-rate GAN acquisition leader H

**H passes all ten initial cold screening gates, but the expanded older-toy
audit is 13 PASS / 6 FAIL and its own-state continuation fails. It is an
acquisition reference, not a stable production replacement.**

Candidate: `h_n05r06_mixup_c0p01_lr15`. The generator and learned particles use
only the logistic relativistic-pair discriminator objective. The discriminator
uses fixed input noise 0.05, R1+R2 coefficient 0.6 and interpolation consistency
coefficient 0.01. Actual Adam rates stay at G/D 0.0015 and particles 0.003, with
betas (0, 0.999). Legacy coverage, residual-target fitting and particle
regularization are disabled. No mode labels or evaluation metrics enter updates.

The exact [recipe](best-candidate.json) includes both the base configuration and
the applied experimental overrides. The config alone does not implement H.

| Measured evidence | Result | Seconds |
| --- | --- | ---: |
| Trajectory, 400 updates | PASS; MSE 0.002905043 | 2.15 |
| Cold ring, 1,200 updates | PASS; 8/8 modes, HQ 0.999267578, five terminal passing checks | 11.52 |
| Ten ordered cold toy gates | All PASS | 92.65 total |
| Own-state hold, 1,200 further updates | FAIL; 750/1,200 dense checks pass; final 5 modes, HQ 0.209228516 | 18.08 |
| Independent expanded older-toy audit | 13 PASS / 6 FAIL; all three native100 gates SKIPPED | See per-host receipts |

The ten cold gates are trajectory, mode_hold, residual_student, img_stripes2,
img_bars4, vector_overlap, img_blobs4, img_intensity2, vector_unequal_mass and
vector_unequal_width. [All metrics](best-cold-metrics.json),
[raw cold verdicts](batch-h/h_n05r06_mixup_c0p01_lr15/status.json),
[cold artifact regrade](regrade-h-final.json), and
[dense continuation summary](h-own-hold/diagnostic-summary.json) retain the evidence.
The continued run restores its own acquired live model, Adam moments and RNGs.
Its 57 recovered loss episodes do not erase its failed endpoint. This finite
run does not prove permanent collapse.

| Acquisition ranking | Cold ring | Qualification |
| --- | --- | --- |
| 1. H | 8 modes / HQ 0.999268; all ten cheap gates pass | Own-state continuation FAIL |
| 2. [Pinned local PR84](https://github.com/255BITS/ParticleGAN/blob/e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960/reports/toy100/pr93-cold-independent-audit.md) | 7 modes / HQ 1 | Incomplete acquisition; borrowed-state holds do not establish an own-acquired hold |

H improves verified local acquisition. PR84 retains the better trajectory MSE
(0.000942662). Runtime comparisons are not controlled because the local machine
was shared. **No constant-rate candidate is production-qualified.** The original
attempt skipped recovery, remaining full19 tasks and all three native 100-mode
problems after the hold failure; ten cold passes are not a 22/22 claim.

A fresh independent replay now reproduces all ten cold passes and their live
metrics, including eight ring modes/HQ 0.999267578, in 77.22 seconds total.
The [fresh verdicts](independent-verification/cold-status.json) and
[independent sample/checkpoint regrade](independent-verification/regrade-cold.json)
are retained. The changed elapsed time is not a controlled speed result.

The [complete verification table](independent-verification/verification.md)
records failures on two_pole, unipolar, cover_leftover, mid_scale_identity,
unused_token_hold and ae_gan_hold. Unipolar passes its final observation but
has only one terminal passing check; five are required. The native100 gates
remain SKIPPED after these older failures. The independent audit regrades all
19 source/config/episode and optimizer/noise receipts. Ten original cold metrics
match exactly, with saved or restored samples independently checked wherever
the archived screen retained a checkpoint.

Two auxiliary-feature hosts have an additional scope issue: disabling AE
reconstruction disconnects its encoder from training; disabling unused-token
hold leaves an identical-gradient parameter invariant incompatible with that
host's retention and movement requirements. These are documented with source
links in the verification table. Restoring original auxiliary losses is a
separate hybrid host-compatibility control, never a pure-GAN qualification.

The [continuation audit](stability-audit/audit-findings.json) reproduces all
1,200 observations and complete final model, Adam, EMA and RNG state exactly.
No restoration bug was found. The first failure is update1255/HQ0.780518.
Three bounded probes from H's acquired state also fail: D2:G1 at1254, two-draw
gradient averaging at1228, and the prepared predictive Adam update at1203.
[Raw probe evidence](stability-audit/initial-probes) retains their work counts
and failures; none establishes cold acquisition or own-state stability.

Broad searches were stopped. Three fresh Astra/max attempts now start from this
fixed H source: failing cold toys, discriminator-signal stability, and update
dynamics. Each uses small batches, early rejection, a 16-proposal cap and a
90-minute limit. A survivor still needs the failing older toys, preserved cold
acquisition, its own-state hold, and eventual full production qualification.

Reproduce the selected cold run and its continuation from a fresh output directory:

```bash
bash reports/toy100/critic_signal_attempt/replay-best-h.sh
# Or select an installed Python with this repository's experiment dependencies:
BENCH_PYTHON=/path/to/python bash reports/toy100/critic_signal_attempt/replay-best-h.sh /tmp/h-replay-new
# Also run each remaining older host once, preserving all failures:
REPLAY_OLDER_DIAGNOSTICS=1 bash reports/toy100/critic_signal_attempt/replay-best-h.sh /tmp/h-full-older-new
# Independently regrade copied saved evidence without training:
bash reports/toy100/critic_signal_attempt/regrade-verification.sh /tmp/h-regrade-new
```

The replay extracts the recorded base code and then overlays the exact selected
source archive; it does not use whatever trainer happens to be checked out.
It pins CPU/AVX2 and one compute thread and refuses to overwrite an existing
output directory. The base archive includes the frozen leading-profile data
that was missing from the original local replay script. The ten-gate run and
own-state hold are separate commands with separate evidence. A successful
script exit is not a benchmark PASS; inspect its saved verdicts.

The visible [candidate adapter](critic_signal.py), [screen](critic_signal_screen.py)
and [continuation](critic_signal_continue.py) are the selected implementations,
not the final expanded search adapter. Source base is
`e2f168ddbd2d38f8f94fbb825f8a71d67dc8d960`. The original
[source identity](source-identity.json) distinguishes cold source hashes from
later search code; [publication-manifest.json](publication-manifest.json) inventories
this curated subset. Only H's evidence is included from the original batch;
other rows in `batch-h/manifest.json` are not supplied here. Native trajectory
and ring checkpoints retain raw samples and optimizer state.

The originating discriminator-signal search tested 766 distinct configurations
and 1,820 gates including H's failed hold. Exactly one candidate passed all ten
cold gates. Its final full unit suite passed 1,329 tests and 27 subtests, with
10 skips and one expected failure; [JUnit evidence](full-suite-final.xml) is
retained. Those are original-attempt results, not a claim that the complete
production benchmark or this publication checkout has passed.
