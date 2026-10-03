# Retained Atlas acquisition gaps

Both original transfer runs pass their terminal five checks. Their first passing read precedes stable acquisition; the original grades already account for the intervening losses. This read-only audit compares the saved numeric fields against the exact frozen inequalities and agrees with the independent original convergence records.

| Original task | First passing read | First five consecutive passing reads | Later losses after first pass | Final uninterrupted passing suffix | Final five |
|---|---:|---|---|---|---|
| vector_overlap | 150 | 650, 700, 750, 800, **850** | 350, 450, 550, 600 | 650–1200; 12 checks | 1000–1200; PASS |
| img_intensity2 | 350 | 425, 450, 475, 500, **525** | 400 | 425–600; 8 checks | 500–600; PASS |

Neither run loses a gate after its first five-read confirmation. Each completed its original budget and all 24 declared observations: 1200 updates/every 50 for overlap, 600/every 25 for intensity. These are diagnostic task results; the full 26-slot cohort remains 7 PASS, 11 FAIL and 8 BLOCKED, with no ordinary qualification or default adoption.

For overlap, every nonpassing read fails only `mean_error <= .15`: steps 50, 100, 350, 450, 550 and 600. At step 350 the recorded value is .15003348886966705, only .00003348886966705 over the unchanged limit. The initial good interval 150–300 has four checks, so it does not confirm acquisition. All 12 checks from 650 onward pass `sw1_normalized <= .18`, `mean_error <= .15` and `covariance_error <= .45`. The final values are .032883850801143796, .03997652605175972 and .06019920855760574 respectively. No additional CDF/KS or component gate is imported.

For intensity, the actual binary conditions are `modes >= 2` and `hq >= .9`. Per-image nearest-template RMSE `.06` and minimum quality mass `.25` define those metrics; overall mean RMSE and nearest-template TV are diagnostics. At step 400, HQ falls to .53125 and quality-qualified modes to one. Both nearest-template classes remain present, with fractions [.46875, .53125], while their quality-qualified fractions are [.03125, .5]. This is a loss of image fidelity and quality-qualified coverage, not disappearance of a nearest-template class. Its mean RMSE .058188870549201965 does not rescue the failed gates. The final read has two modes, HQ 1 and mean RMSE .02307453751564026.

The separately bound ring hold result is a **precision loss**. All 200 confirmation checks at steps 1201–1400 pass; six disjoint hold checks at 1401–1406 then pass. At 1407, HQ .89453125 crosses below .9 while all eight modes and cover 1 remain. The previous HQ is .937255859375. The original status remains `POST_CONVERGENCE_FAIL`; the full 1200-check hold is incomplete and no restart or reacquisition is credited. Hold and extension share one physical run and 207 dense observations, rather than the transfer tasks’ 24-check clock.

All three use the original selected public policy, including actual DV12 perturbation, with additive output noise disabled. Overlap’s public sampled population is 256 rows; intensity enumerates the original 32 rows. Their final selected source is fast and their resolved backend is kNN/controller-reference. “Clean” here does not mean latent perturbation is disabled. The ring also uses its original 12-row kNN/controller-reference law.

The frozen scientific source is commit `9563dea57bb150f2a0275bbe8d785bf76210fca3`, digest `db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037`. [gaps.json](gaps.json) pins the exact original raw/resolved/graded receipts, evaluator/adapter/sampling/public-policy source bytes, whole-cohort study and prior full-completion forensic report. [audit.py](audit.py) recomputes chronology using only stdlib scalar comparisons and asserts agreement with the original convergence metadata. It reads no arrays or checkpoints and invokes no model, sampler, official scorer, optimizer, GPU or queue. All 19 consumed files are checked against their pinned hashes; originals remain unchanged.

These observations establish when the saved gates pass and fail. They do not resolve behavior between the evaluation boundaries or establish why generator/prior dynamics produced a transient failure. The separate prepared grid100 output-kernel endpoint diagnostic remains one unexecuted width-law question; this chronology audit adds no experiment or gate.
