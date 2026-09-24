# Forward-KL cumulative free-output screen

The frozen-width, all-history donor rule retained external coverage for every
tested update, but failed its predeclared independent quadrature consistency
check. This remains a pure-output diagnostic; it makes no neural, critic,
prior, Adam, or learning-rate update.

| Arm | Completed updates | Fixed late-noise coverage | GH9 accepted-step reversals |
| --- | ---: | --- | ---: |
| warm1324 → 1339 | 16 | all 8 modes, minimum HQ .999756 | 0 |
| cold1 → 16 | 16 | all 8 modes/HQ 1 | 9 |

The cold reversals were one EM step at update 7 (`+1.61e-7` in GH9), then a
donor at update 14 (`+4.13e-5`) and EM steps 14–20 there
(`+1.67e-7` to `+4.67e-7`). GH5 decreased at every accepted step. The
**whole update** also decreased GH9 at both cold steps (`−.00734073` and
`−.00189047`), so this is a per-step numerical selection failure, not an
observed coverage loss or evidence that the true integral rose overall.
The original no-reversal condition still fails. External mode-mass TV at
both final endpoints is `1/6`; eight-mode HQ does not imply exact law match.

The first-bank h remained `.031286240422040236`, and the native output-noise
clock was used: cold sigma starts at zero and rises on the original 1200-step
schedule; warm sigma is `.029`. Each update appended its real128 bank to
the target and searched every observed real point as a donor, at most twelve
moves, then took at most twenty fixed-weight EM steps. The measured wall time
was about 344 seconds for warm16 and 253 seconds for cold16, each on one CPU
thread. This all-history donor search scales poorly with bank history and is
not the proposed bounded neural search.

The warm-first script completed warm16 and entered its automatically
duplicated cold arm. It was deliberately terminated after one redundant
cold update (exit 143); its output is explicitly `STOPPED_REDUNDANT_COLD`,
not a complete result. A separately declared cold companion called the
same frozen `run_case` and optimizer and completed cold16. The source bytes,
declarations, transcript, warm partial rows, cold result, and SHA-256
manifest are in
[`round8-forward-kl-cumulative`](continuous-evidence/round8-forward-kl-cumulative/manifest.json).
Both arms matched the archived v2 first-bank clean outputs, objective, and
native real-bank SHA exactly. No cold gate credit was taken until warm16
finished. The paired evidence supports the quality observations above while
failing the predeclared GH9 numerical-consistency gate.

The next bounded check is a fixed-state quadrature accuracy audit of the
reversed update 7/14 proposals. It will measure whether a deterministic
refinement can certify a proposed decrease or cause the update to rest;
there will be no bandwidth, optimizer, or seed sweep.
