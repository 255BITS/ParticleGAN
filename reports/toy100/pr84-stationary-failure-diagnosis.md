# Exact PR84 fixed-target continuation replay

The selected PR84 smoothed-critic update fails to maintain quality on the
unchanged ring. A read-only observer replayed its scheduled-prefix fork through
update 2,400. Its final full training-state SHA-256, all 420 original host
diagnostics, all 1,400 curvature/width records, optimizer accounting, applied
rates and noise receipts exactly match the archived run. Clean-support forwards,
tensor copies and saved-state snapshots did not alter training.

The original ten-step diagnostic passes all 200 checks at updates 1,001–1,200,
then passes 112/120 checks through update 2,400. Its first sampled failure is
1,390, its worst is seven modes/HQ .78857 at 1,540, and its final checkpoint
recovers to eight modes/HQ .99805. Scoring clean supports with the host's exact
fixed 4,096-draw evaluation indices and output noise at **every** update
1,300–1,600 finds 46 failing updates in three blocks: 1,325–1,327,
1,389–1,390 and 1,531–1,571. The first block is invisible to ten-step
sampling. This does not identify the earliest failure before update 1,300.
Every saved ten-step host observation in this window matches the offline
score bit for bit.

| Update | Same-noise pre-step | Ordinary G proposal | Bounded accepted G | G bound factor |
| --- | --- | --- | --- | ---: |
| 1,325 | 8 / .99658 | 8 / .82422 | 8 / .82422 | 1.000 |
| 1,389 | 8 / .92407 | 8 / .86719 | 8 / .89258 | .618 |
| 1,390 | 8 / .89795 | 8 / .85889 | 8 / .87866 | .477 |
| 1,530 | 8 / .96899 | 8 / .88843 | 8 / .91846 | .602 |
| 1,533 | 8 / .83545 | 7 / .75317 | 7 / .81592 | .489 |
| 1,540 | 7 / .79932 | 7 / .76001 | 7 / .78857 | .519 |

The acute 1,325 drop occurs in one update. G's own-curvature ratio is .23851,
below the frozen .25 bound, so the ordinary proposal is fully accepted.
Network-only application drops HQ from .99658 to .83643; prior-only application
leaves .99585. Four particles on modes 6 and 7 move away from their centers.
At the accepted critic, the smoothed score's local slope **toward** mode 7 is
negative for particles 1 and 7 (−.2676 and −.2311). Their actual moves
increase critic score and center distance by about .103 and .107. For the
two mode-6 particles the score slope toward center is weakly positive, yet
the shared generator network also moves them outward. These are offline
diagnostics using known centers, which the update never reads.

The later failure is accumulated, not a single mode-loss step. Paired
within-step scores show repeated damage from 1,527 to 1,533; mode 2 is
first lost at 1,533. Its two supporting particles are .217/.190 from the
center after update 1,530, .359/.304 after 1,533, and .660/.571 after 1,540.
At 1,530 and 1,540 the accepted critic's smoothed-score slope toward mode 2
is negative for both particles; their outward moves increase its score.
The curvature bound mitigates each ordinary proposal but does not reverse the
drift. At 1,540, network-only functional RMS movement is .03089 versus
prior-only .00592; joint movement is .03534. D's bound remains inactive
through all 1,400 continuation updates. These observations identify a
misaligned local field plus shared-network motion in the saved states; they do
not establish a general cause for all seeds or hosts.

The replay uses the original fixed target, one pinned seed and the scheduled
passing state at update 1,000. It is a conditional stability diagnosis, not
proof of cold acquisition, a new training method or indefinite behavior.
The already failed cold ring gate remains a separate veto. The ten-step
continuation cadence undercounts the observed instability; future candidate
holds should score every update when assessing continuous live quality.

The [hashed evidence manifest](continuous-evidence/pr84-stationary-failure-diagnosis/manifest.json)
contains a compressed 301-row replay, accepted-critic field analysis, exact
source snapshot and compact pre-step/accepted-D state pairs for updates
1,324, 1,325, 1,326, 1,389, 1,390, 1,530, 1,539 and 1,540. The full local
sidecar also retains proposal and bounded states at 13 updates; its hash is
listed in the manifest. The two observer scripts are
`pr84_stationary_failure_replay.py` and `pr84_stationary_field_analysis.py`.

Reproduction uses one CPU thread and the pinned Python/PyTorch environment:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export ATEN_CPU_CAPABILITY=avx2 MKL_ENABLE_INSTRUCTIONS=AVX2 ONEDNN_MAX_CPU_ISA=AVX2 DNNL_MAX_CPU_ISA=AVX2
/tmp/pr38-default-env/bin/python -u reports/toy100/pr84_stationary_failure_replay.py \
  --output NEW_OUTPUT \
  --reference reports/toy100/continuous-evidence/pr84-stationary-hold/forks/original.json.gz
/tmp/pr38-default-env/bin/python -u reports/toy100/pr84_stationary_field_analysis.py \
  --diagnosis NEW_OUTPUT/diagnosis.json --states NEW_OUTPUT/selected-states.pt \
  --output NEW_OUTPUT/field-analysis.json
```
