# EMA generator idle against the BCAP direction control

**Verdict: no Tier 1 winner.** The frozen seed-0 paired comparison completed all
twelve starts under the original six required task contracts. The control has
five PASS and one INCOMPLETE; the opt-in EMA-idle candidate has four PASS and two
INCOMPLETE. These are whole-recipe counts. No Tier 2/3, held-out transfer,
robustness or release qualification follows.

| Required task | Control | EMA-idle | Candidate skipped generator steps |
| --- | --- | --- | ---: |
| `ae_gan_hold` | PASS | PASS | 34 |
| `five_word_joint_smoke` | INCOMPLETE | INCOMPLETE | 7,388 |
| `gaussian1d_smoke` | PASS | INCOMPLETE | 54 |
| `ring16_acquisition` | PASS | PASS | 0 |
| `two_pole` | PASS | PASS | 1 |
| `unused_token_hold` | PASS | PASS | 0 |

The candidate completes the Gaussian 1,000-step lifecycle with only 946 actual
generator/prior optimizer updates, and the word 20,001-step lifecycle with only
12,613 generator/encoder/prior updates. The original optimizer-history gates
therefore remain **INCOMPLETE**, even where saved output curves are promising.
The control word reaches 23 scored observations through step 19,168, then
times out at the unchanged 900-second task limit without a final scored
archive. Its exact terminal update count is unknown. Ring and unused-token
have zero idle skips and identical candidate/control numerical metrics; these
cells do not establish an EMA advantage.

The [pinned paired publication](../test-performance-evidence/pr399-paired-tier1-12.json)
preserves the original available grades, complete recipe and task contracts,
terminal states, saved-output and file hashes without importing a PASS from a
partial observation. The local score-ready packet is SHA-256
`f6dcd3487d013faf82d6d85e569a79af7a86841fbd075ef375ab760f7b33351c`.
The [separate dynamics publication](../test-performance-evidence/atlas-dynamics-6.json)
has one PASS, one FAIL and four INCOMPLETE attempts. The 66-attempt cumulative
packet is SHA-256
`2ebd65d9b0c210a6f4d24365f5feb2cc7dd19c9a9daf3536fbab4ed0f678e29b`;
48 historical rows are unchanged, with six dynamics and twelve paired attempts
added once by attempt ID. Those are distinct scientific cohorts, not a pooled
score or an addition to the 249-run non-Atlas ledger.

Owner source digest `706726813d2693d506a2ede3f2471852adc5f8a6b3a8057cc50bf7dc644b0966`
and executed closure `27ab4cf9c9de35487f5dca60dc7c0329ddf89977d37f91e547e517de97a7bf06`
are separate identities. The closure changes only existing-lease FD forwarding;
all other frozen source members, including model, trainer and evaluator, match
the owner bytes. Both arms ran on the A6000 with Python 3.11.15 and Torch
2.11.0+cu128. Shared-GPU co-tenancy makes their runtime unsuitable for a
controlled speed claim. Charges are 2,052.616064 queue seconds and
2,072.815230 inclusive supervisor seconds, with zero paid retries. The clocks
are not added. Exact cumulative optimizer updates are unavailable because the
timed-out word control has only a verified lower bound.

This is a scored research record and a reason to keep the candidate opt-in. It
does not change original gates, ordinary qualification, selected defaults or
the release decision. GPU host/replay compatibility fixes in PR399 can be
reviewed separately from adoption of the EMA-idle formulation.
