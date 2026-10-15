# Final discriminator-rate screen — independent audit

All four runs completed their original update budgets and failed their full scientific gates. Neither candidate is eligible for ordinary Tier1.

| Candidate | Task | Full gate | Terminal passing suffix | Charged execution seconds |
|---|---|---|---:|---:|
|bcap-transport-halfcritic-v1|ring16_acquisition|FAIL|0 / 96|115.718548|
|bcap-transport-halfcritic-v1|two_pole|FAIL|0 / 24|7.037536|
|bcap-selective-history-halfjoint-halfcritic-v1|ring16_acquisition|FAIL|0 / 96|45.613728|
|bcap-selective-history-halfjoint-halfcritic-v1|two_pole|FAIL|0 / 24|6.689930|

Total actual task execution: 175.059741495992 s against a 1200 s campaign cap; zero outstanding reservations. Four unique attempts, no scientific retries. All full 80/1600 updates and 24/96 observations verified. Independent curve grading, raw-to-grade hashes, scored-output hashes, and provenance checkpoint hashes pass. Source digest39fad3b03e5178bcdc68cc17ba012ac4083d5887147bef91f0bbeee1205baea6 verifies1224/1224 members. Runtime/protocol/seed0/named RNG settings and task definitions are unchanged. Ring initial model/prior hashes and final named-stream states match across arms. Detailed metrics, costs, receipt hashes and actual checkpoint optimizer group rates are in independent-result-audit.json.

Both final discriminator-step candidates have zero final passing observations on both tasks. Exact parent deltas are only d_lr_mult1.5→0.75 for T and3→1.5 for S-halfjoint. Saved optimizer states verify actual discriminator rate0.009 on all four; generator/prior rates are retained. This falsifies the bounded proposal at the declared settings and gates. The preregistered stop rule applies: stop tuning this transport/history family. Fresh ordinary validation is baseline B only, under its original six required gates plus the clock diagnostic, cap2520 seconds.
