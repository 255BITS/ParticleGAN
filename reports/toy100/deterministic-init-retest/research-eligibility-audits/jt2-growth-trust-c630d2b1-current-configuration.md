JT2 retains its new-init **PASS: 15/24**, first arrival 450, miss at 700, and final passing streak of 10. Its exact tested configuration is **disqualified for the horizon-independent default**.

The trust mechanism does not use the horizon, but its host does. Actual G/D rates decay from .00425 to 4.25450588241e-05; the prior ends at 0.000425086476531. Decay starts after 720 completed updates and approaches the .01/.05 floors at 1200. All 2400 per-role action rows match that formula. Mechanism prose claiming floors of 1 is stale relative to this executed config.

Input noise .5 reaches zero at completed update 120 (training update 121); output noise reaches .029 at completed update 240 (training update 241). Changing the declared host end changes actual learner dynamics. Quality and historical eligibility labels remain intact; no repair or follow-up was run. [Exact sources and receipts](jt2-growth-trust-c630d2b1-current-configuration.json).
