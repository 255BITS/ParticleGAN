# RP5 completed long and remaining vectors audit

The unchanged public RP5 learner completed the uninterrupted 30,000-update protocol. Initial arrival 570 retained 544/544 observations through 6000. Change arrivals were 6350 (+350), 8250 (+450), and 27250 (+250), followed by 146/146, 1876/1876, and 276/276 passing observations. No post-arrival misses occurred. Minimum HQ after these arrivals was .931152, .901367 and .909912 respectively. Each pre-change final120 window passed; frozen controls passed 0/180, 0/1920 and 0/300.

Precision closed at834, reopened at6012, then closed at6733. Applied rates switch on the next update. The later two shifts recover at the closed nonzero floors: network .0000425 and prior .000425. No requirement to reopen was invented. Every accepted/precision clock and all30,000 rates agree with source behavior; the fixed360/720 noise initialization is horizon-independent.

All immutable source entries and completed artifact hashes verify. All15 package files match the original RP5 single candidate; recipe and initial complete state match single/stationary. Metrics, rates and full receipts match the single prefix through2400 and stationary through6000. Independent raw checkpoint hashing reproduces all saved full receipts at0,6000,7800,9000,27000,30000, including precision/reference, optimizer/model/EMA/private/global RNG and caller data stream. Factory-owned eager CUDA Adam counters are part of this declared RP5 candidate.

The main learner never loads a checkpoint; only a separate frozen control does. The old9000 import warning was erroneous and already retracted: the earlier import precedes the loop. No restart or invalidation is justified. Legacy single-shift convenience summaries are ignored in favor of actual long segment boundaries.

Existing fresh-process2400→2500 and interior2450 proof does not substitute for the declared long-change replays. Saved states support separate6000→6100,7800→7900 and27000→27100 comparisons with correct absolute target offsets; those executions remain pending in the audited artifacts. Raw-byte verification is not a replay.

Remaining vector results archived under `reports/toy100/continuous-api-search/rp5-vector-wave2-audit.json`:
- api-rp5-vector_anisotropic: PASS, 20/24, final suffix 20.
- api-rp5-vector_overlap: PASS, 24/24, final suffix 24.
- api-rp5-vector_spiral: PASS, 23/24, final suffix 23.

Each completed vector has independent canonical ordered-parameter and EMA proof, unchanged frozen task/card/scorer/gates, same RP5 package and explicit isolated output2303 measurement stream. Parameter comparisons exclude buffers; prior false two_broad audit stays preserved. Checkpoint bytes and caller data RNG are archived, but no vector replay was executed. JSONL compression round-trips exactly. No training, GPU, tests, worker edits, shared manifests, README, state or PR edits were performed.
