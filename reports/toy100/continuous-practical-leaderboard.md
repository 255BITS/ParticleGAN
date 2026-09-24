# Current GPU leaderboard

The completed CUDA audit is now authoritative for GPU research. All supported
runs trained on an RTX A6000 with the same pinned FP32/CUDA profile: **97 toy
runs plus eight convergence runs**. All saved verdicts were independently
recomputed. Unsupported adapters receive no credit.

| Fully tested candidate | Toy PASS | Good post-convergence hold / 1200 |
|---|---:|---:|
| Shared column RMS | 11/22 | 131, then FAIL |
| Shared RMS | 11/22 | 35, then FAIL |
| H | 11/22 | Not confirmed by update 6000 |
| Epsilon base | 9/22 | 19, then FAIL |

**Shared column RMS is the next GPU research reference. No release winner is
qualified.** All twelve native 100-mode runs fail the coverage/accuracy gates.
The averaging and PR107/140/143 adapters have limited toy coverage; their
separate results and explicit unsupported cells are in the full matrix.

[Full GPU leaderboard](gpu-leaderboard/LEADERBOARD.md) ·
[Protocol, raw evidence, and portable replay](gpu-leaderboard/README.md) ·
[GPU research reference](gpu-leaderboard/current-gpu-reference.json)

The hold begins after the first 200 consecutive full-ring/HQ >= .90 checks and
scores the next 1,200 updates. Learning-time dips do not themselves fail it.
PR107 and PR140 never confirm within the GPU budget; PR143 confirms at 1400
then has 11 good hold checks before a miss. These dense results do not share
the old sparse CPU stay denominator.

[Historical CPU leaderboard](continuous-practical-leaderboard-cpu-history.md)
remains available, including the MKL reproduction audit. CPU scores and GPU
scores are not pooled. The old CPU epsilon selection is retained as provenance,
not the current GPU ranking.
