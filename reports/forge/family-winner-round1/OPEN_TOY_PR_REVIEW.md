# Recent toy PR review

This compact review covers only PRs [245](https://github.com/255BITS/ParticleGAN/pull/245), [246](https://github.com/255BITS/ParticleGAN/pull/246) and [253](https://github.com/255BITS/ParticleGAN/pull/253), plus the observation-only [PR254 follow-up](https://github.com/255BITS/ParticleGAN/pull/254). Exact heads, source/card/media hashes, checks and numerical results are in [the receipt](open-toy-pr-review.json). This is a review record, not another configuration board. The integration base is `8021a1c50c4aff90ddea5010d368cffdc857b2f6`; original scientific cohorts keep their own identities.

All three questions have bounded diagnostic usefulness rated **4/5** here. That definition rating is separate from learned convergence, receipt validation and release qualification. None fills a required family-search or Forge cell, demonstrates sustained convergence, or promotes defaults. Older toy PRs are not re-audited here.

## PR245: retain the merged positive benchmark

**Question:** does the existing generated caption fixture still satisfy accuracy and beneficial-code gates after extending all three arms through 2,048 native updates, beyond the earlier 512-update horizon?

The reviewed head is `3c3a366475d7e82947844302ed3a4deffa9354ed`, merged as `848a70621bc9008e0b14b9fe3e1122f16758799e`. All four exact-head CI/build checks succeeded. Current-base integration controls passed **15 benchmark cases in 2.04s plus three renderer cases in 0.28s: 18 total**. These software controls do not replace the original full training evidence. The CI correction keeps the three unchanged renderer controls required in a separate fresh process with a 60-second cap.

The original 6,144-update campaign completed in 610.634/900 seconds and passed its fixed terminal gate: untied particle TEST48 RMSE **0.484234982**, ordinary **0.546611431**, shared-Up **0.518284612**. Untied improved every source against both controls; removing codes worsened RMSE by **44.215%** and harmed each source. All 15 source-card pins matched; the inherited caller/profile closure and package matched the inspected current base. Seven retained media inputs and the shipped eight-frame GIF/PNG matched their receipts. The read-only merge assessment against `0335ecf0` found no overlapping changed paths or conflicts.

The observed call-800 penalty blend is descriptive. More updates, controller adaptation and phase exposure changed together, so this does not isolate a penalty-phase cause. It also does not establish actual-caption/Supra recovery or a fair equal-capacity advantage for the additional untied heads. Retain it as a distinct positive **full-budget, terminal-only** toy.

## PR246 and PR254: retain the conditional negative and its goal media

**Question:** with D16 unchanged, does G64 learn the observed remote marker sufficiently better than G16, while both reconstruct the deterministic held target beyond a marker-blind optimum?

Original head `b35dee597943c95499ca4a94d3cd141bd19f8fda` has four successful CI/build checks. Its sole 400-update campaign completed in **16.410747/60 seconds**, with finite native state and matched caller/source prerequisites. It retained four scientific failures: G64 missed the 10% advantage at both 100 and 200, and both arms missed the terminal total-error bound. At 200, MSE was **1.598696470** G16 and **1.594727516** G64; G64 gained only **0.2483%**. The four failures remain unchanged.

The primary bound is not proved impossible from the input or noise law. Labels and primary predictions use the same **clean** `routed_generate(..., sigma=0, perturb=False)` semantics. Content, time and marker are observed; there is no hidden nuisance. The student has the teacher's parameter topology, identical fixed BF16 host and the same fit calibration. Matching teacher generator/encoder/router/table values is an **analytic zero-error clean representation witness**, not an executed capacity capture or a proof of optimizer reachability in 200 updates. Native DV12 and output-noise effects remain relevant to training; they do not create an irreducible floor in this primary clean score. No noisy-serving capacity or calibration acceptance is established.

For paired labels `y± = m ± h` and predictions `p± = pm ± ph`,

`pair MSE = mean((pm-m)²) + mean((ph-h)²)`.

The held patch begins at coordinate 12, outside the marker at 0..1 and the direct generator's radius-four receptive field. A marker-blind prediction has `ph=0`, so its best possible total MSE is `V = mean(h²)`. **V is a restricted marker-blind bound, not irreducible error given the complete observed inputs.** Here `V=0.0005061657866463065`; the original `total ≤ .5V` gate requires MSE≤`0.00025308289332315326`, or RMSE≤`0.0159086`. It is sufficient for conditional fidelity but can fail because of midpoint error even with perfect contrast.

The separate, already executed saved-200 diagnostic isolates contrast using `C/V = target_power/V - 2α + predicted_power/V`. It found **1.000345118** G16 and **0.996276603** G64, both failing its own `.5V` contrast criterion. Alignment was only `.000300642` and `.004269537`; midpoint error dominated total error. This verifies little target-contrast recovery at that endpoint, without identifying an optimizer defect, a native-noise impossibility or a real-task failure mechanism. Earlier contrast observations were not retained. The relative 10% baseline inequality also accepts zero-versus-zero; that boundary supplies no percentage-gain claim. The actual baseline errors were positive.

[PR254](https://github.com/255BITS/ParticleGAN/pull/254), tip `491756abc88d04ae3afe40893c5a7d6d819b5035`, adds only six observation/report files. Exporter `7afd6e3ca021f5b3cf69daf28513986cf6092ef8` has **26 passing software controls**, including independent missing/changed recorded-source-identity negatives. It changes no caller, package, gate or original result. Its three-frame goal GIF uses the actual 0/100/200 metric observations; there were no retained intermediate prediction images. The separate contrast appears only at 200. Independent checks matched both exporter/dependency Git files, all nine consumed files, the GIF and committed copy; original-public and review-time-only trace attestations remain distinct. The exporter performed **zero forwards, updates or draws and no model rescoring**. FAIL and all four failed checks are readable.

Retain this useful negative. A future variant can predeclare separate contrast and midpoint fidelity questions, or a separately witnessed noisy-serving law; neither would replace these original gates or convert this result to PASS. Four-times G rows are an unequal-compute comparison, not a default-speed qualification.

## PR253: retain the terminal-precision counterexample

**Question:** in a constructed frozen `[W,-W]` terminal cancellation, do routed edits provide beneficial convergence under BF16 versus FP32 arithmetic, and what does the final native update reveal about local precision?

Exact head `a31ce8c1ddac831ae032ecaf4b43d84919cd6173` has four successful CI/build checks. Four arms completed 512 updates each, **2,048 total**, in **65.875590/300 seconds**. The original convergence gate **failed**: particle-FP32 RMSE **0.003563903** versus ordinary-FP32 **0.003457183**, about **3.087% worse**. Zero-code RMSE **0.003422292** improved on live particles. Aggregate improvement, source non-harm and signed aggregate/per-source code-benefit checks all failed; live gradients did not make codes useful.

The separate final-update precision diagnostic passed its predeclared ratio bounds: FP32/BF16 quadratic term ratio `.0195477` and slope-discrepancy ratio `.3595714` were each≤`.75`. Neither parameter tangent was beneficial. That local arithmetic observation does not rescue convergence or demonstrate a full-Supra remedy. The terminal matrix is deliberately constructed; it is not a measurement of real final-layer geometry.

Six source-card pins, original report/completion/observed-media hashes and the six-frame GIF/PNG matched. The archived software receipt records four controls and exactly four tiny replay updates. The saved-data reader records 565 checks without model/API execution; its separate v2 metadata fix only canonicalizes tuple/list recipe equality. Producer ownership, gradients and model-replay claims retain their stated source-bound limits. No reader or scientific run was repeated for this review.

Keep PR253's useful counterexample available rather than treating the precision PASS as a winning learned configuration. It remains convergence-FAIL and receives no winner/default merge credit. Its training GIF already preserves that failure.
