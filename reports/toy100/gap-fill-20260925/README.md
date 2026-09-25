# Qualification gap fill — September 25, 2026

Status: **COMPLETE — 53/53 jobs, zero execution errors**, 14:43–14:52 UTC.
The initial 51 jobs and two additional P1 ring checks ran across two RTX A6000
GPUs, up to six jobs per GPU with GPU/system-memory headroom. Every job has its
own output directory and log. No training jobs from this batch remain active.

**K3P leads: 22/22 toys, standard hold and 300-update extension PASS; recovery
still FAIL.** K3G and P1 also score 22/22, but both fail 237/300 extension checks
and their recovery gates. RG5+b-cap scores 18/22 and fails hold/recovery.

![K3P converging on grid100, rotated100 and staggered100](k3p-100gaussians-convergence.gif)

Actual live-generator snapshots from the declared seed-1234 runs: axis-aligned
grid, 25° rotation, and staggered rows. All 34 saved checkpoints from updates
0–7,000 are shown, with synchronized step labels and fixed axes. Each panel
shows 4,096 generated and target samples; displayed metrics use the full 20,000
evaluation samples. All three runs pass the full coverage and accuracy gates.
Playback is not real time. No retraining, sample interpolation, or EMA substitution.
[`Rendering source`](render_k3p_convergence.py) ·
[`Frame/source provenance`](k3p-convergence-provenance.json) ·
[`Final frame`](k3p-100gaussians-final.png).

```sh
tail -f /ml2/hypergan/gan-attempts/gap-fill-20260925/progress.jsonl
```

`/ml2/hypergan/gan-attempts/gap-fill-20260925/status.json` lists active PIDs and
queued jobs; `completed.json` records the initial 51 finished jobs and their raw
verdicts. `supplemental-completed.json` records the two added P1 ring checks.

## Fixed protocol

[`manifest.json`](manifest.json) declares every command, source file and hash,
initialization fixture and hash, output path, and frozen runtime path. Candidate
and harness copies are under `sources/`. Training is CUDA FP32 with deterministic
algorithms, TF32 disabled, one CPU thread per worker, and the existing gate
thresholds. Source formulations and saved configurations are unchanged.

The 19 transfer toys reuse the original CPU initialization fixtures; the three
additional vector fixtures come from the earlier A2 qualification and were
captured before any optimizer update. All actual training is on CUDA. Native
problems use native CUDA initialization, seed 1234, and the full 7,000-update
coverage **and** accuracy gate. No seed sweeps are run, following `AGENTS.md`.
New single-seed results must be distinguished from the existing four-seed rates.
Assignment across identical A6000 devices is recorded per job; elapsed runtime
under concurrency is not used to rank formulations.

| Formulation | New jobs |
|---|---:|
| K3P | 16 missing transfer toys + rotated100 + staggered100 = 18 |
| K3G | Three vector toys + grid100 + rotated100 + hold + shift = 7 |
| RG5+b-cap, no A2, .01/.05 floors | 19 transfer toys + rotated100 + hold + shift = 22 |
| P1 implementation of K3 | Three vector toys + own hold + own shift = 5 |
| RG5+A2, a_r1r2, .1/.1 floors | Matched frozen control = 1 |

The ring hold includes the existing 300-update extension after its 1,200-update
hold. Extended stability is assessed separately from the harness's standard
hold verdict. Target-shift recovery requires timely sustained live recovery
and a matched frozen control; a failed frozen run is the expected negative
control, not a candidate failure. The RG5+A2 control fills evidence for the
already measured .1/.1 variant; it does not transfer that variant's score to
RG5+b-cap or to base-floor RG5+A2.

The source audit found that the broadly tested P1 and the old K3 ring control
implement the same formula with different source code. Two supplemental jobs
measure P1's own ring outcomes. Both reproduce the old K3 outcome. The final P1
score uses P1's exact saved source throughout, rather than inheriting that control.

## Results and evidence

| Formulation | Declared 22 | Standard hold | 300-update extension | Deadline recovery |
|---|---:|---|---|---|
| K3P | 22 PASS | 1,200/1,200 PASS, existing | 300/300 PASS, existing | 28/81 FAIL, existing |
| K3G | 22 PASS | 1,200/1,200 PASS | 63/300 passing, FAIL | 22/81 FAIL |
| P1 (K3 formula) | 22 PASS | 1,200/1,200 PASS | 63/300 passing, FAIL | 22/81 FAIL |
| RG5+b-cap, no A2, base floors | 18 PASS / 4 FAIL | NOT_CONVERGED | Not reached | 0/81 FAIL |
| RG5+A2+a_r1r2, .1/.1 floors | Not qualified as a 22-toy configuration | 120/120 checks in shift protocol, existing | Not run | **81/81 live vs 0/81 frozen: paired PASS** |

RG5+b-cap fails `mode_hold`, `vector_unequal_mass`, `vector_unequal_width`, and
`img_stripes2`. The last has only three terminal passing checks against a five-check
requirement. Each of the other three additional vectors passes for all four
tested formulations. P1's historical staggered100 seed 1235 remains a failure;
22/22 describes the declared native seed 1234, not all historically tested seeds.

- [`qualification-summary.json`](qualification-summary.json): all 22 gates per
  formulation, exact source consistency checks, inherited/new evidence links,
  and separate native seed denominators; generated by [`score.py`](score.py).
- [`results-summary.json`](results-summary.json): all 53 new raw verdicts and
  metrics, including explicitly derived extension checks; generated by
  [`collect.py`](collect.py). Raw results are under `results/*.json.gz`, with
  both compressed and original SHA-256 hashes in the summary.
- [`rg5-recovery-pair.json`](rg5-recovery-pair.json): verifies identical pre-shift
  diagnostics/configuration/fixture/counters, no frozen optimizer updates after
  2400, live recovery after 110 updates, and 0/81 frozen recovery. The original
  raw statuses are unchanged; the paired verdict is now PASS.
- [`runtime-hashes.json`](runtime-hashes.json): hashes of 614 Python/config/data
  files in each frozen runtime. Candidate and harness source hashes are in the
  manifest. Checkpoints/logs and the referenced runtimes/fixtures remain local;
  these report files do not package a portable Python/CUDA environment.
- [`progress.jsonl`](progress.jsonl): completed launch/verdict timeline, GPU
  assignments and PIDs. The runner reached 12 simultaneous jobs. The batch
  completed in about nine minutes; concurrent runtime is not a model metric.
- [`audit.json`](audit.json): all 53 exit codes and result snapshots checked,
  55 source files and 19 fixtures verified, all five new native gates complete,
  and zero active jobs remaining.

Raw counts are 44 PASS, eight FAIL, and one NOT_CONVERGED. One FAIL is the
expected frozen negative control; the two standard-hold PASS statuses do not
include their failing extensions. These raw counts therefore are not a single
overall formulation success rate.

The [leaderboard](../continuous-practical-leaderboard.md) is updated. K3P's next
research question is recovery; no new floor schedule or combined mechanism was
introduced by this batch. No defaults were promoted.
