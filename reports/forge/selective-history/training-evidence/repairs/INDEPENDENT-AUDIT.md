# Penalty and generator-rate repairs — independent audit

All four runs completed their original update budgets and failed their full scientific gates. Neither candidate is eligible for ordinary Tier1.

| Candidate | Task | Full gate | Terminal passing suffix | Charged execution seconds |
|---|---|---|---:|---:|
|bcap-transport-penalty2-v1|ring16_acquisition|FAIL|0 / 96|40.930496|
|bcap-transport-penalty2-v1|two_pole|FAIL|4 / 24|7.166970|
|bcap-selective-history-halfjoint-v1|ring16_acquisition|FAIL|4 / 96|45.133822|
|bcap-selective-history-halfjoint-v1|two_pole|FAIL|0 / 24|6.837281|

Total actual task execution: 100.068569712021 s against a 1200 s campaign cap; zero outstanding reservations. Four unique attempts, no scientific retries. All full 80/1600 updates and 24/96 observations verified. Independent curve grading, raw-to-grade hashes, scored-output hashes, and provenance checkpoint hashes pass. Source digest39fad3b03e5178bcdc68cc17ba012ac4083d5887147bef91f0bbeee1205baea6 verifies1224/1224 members. Runtime/protocol/seed0/named RNG settings and task definitions are unchanged. Ring initial model/prior hashes and final named-stream states match across arms. Detailed metrics, costs, receipt hashes and actual checkpoint optimizer group rates are in independent-result-audit.json.

Penalty2 ends pole with four passing observations, short of five; its Ring covariance error2.357043 exceeds0.85 and fails every observation. Halfjoint ends Ring with all final bounds passing but only four consecutive passing observations; its pole has no passing observations and ends at gradient median1.101885. The full failure labels must be retained despite these individually encouraging endpoints.

Halfjoint changed lr0.012→0.006 and d_lr_mult1.5→3, retaining discriminator0.018 and transport weights1/1. Saved Ring optimizer rates are generator0.006, prior0.015, discriminator0.018; pole direct coordinates use generator role0.006. There is no encoder model in these two tasks; its changed rate remains untested by this screen. Penalty2 changed only T reg_coeff1→2, history off.
