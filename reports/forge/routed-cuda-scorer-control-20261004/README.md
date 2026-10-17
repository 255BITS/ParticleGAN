# CUDA scorer engineering control

Two fixed CUDA scorer/owner structural controls; no scientific convergence or named training-budget credit.

Current status: **COMPLETE_ENGINEERING_CONTROL**. Strict requested controls: **2 PASS, 0 SKIP, 0 FAIL, 0 ERROR**.

The unused-token control checks selected models remain on CUDA while the original CPU scorer stays pure. The cover control checks selected models and source reference tensors remain on CUDA. Both use the exact frozen opt-in test nodes. Collect-only readiness executes no fixtures or tests and grants no CUDA PASS.

| Attempt | Source digest | Actual status | Paid seconds | Conservative reserve |
|---|---|---|---:|---:|
| v1 | `2354ab1ffba4faeb33c77461f4c0732948a2306157529148e8c9df545f287dc7` | INVALID; neither requested test ran | 4.234133972 | 0 |
| v2 | `f4355e10df69baf079dfb5d4e25ae2024d9dac2be7c3cc91f305a9f8f1153cb3` | INVALID; neither requested test ran | 5.197659394 | 0 |
| v4 | `ace203b5cc2f1a4ffbed3c15c701c2429a736abe1f23150bf99976e135affb49` | COMPLETE_ENGINEERING_CONTROL | 7.592017245 | 0.000000000 |

Prior engineering debit **9.431793366093s**, current paid **7.592017245013s**, current reserve **0.000000000000s**; inclusive charged **17.023810611106/120s**. Each immutable earlier cost is counted once. These controls are separate from the 10,500-second named training campaign and supply no learned-quality, ordinary-tier, default, convergence-time or speed credit.

Scientific source `fb7acc775b3a1a6184d36b55e035b9da04531492` / `f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed`. The separately derived test-wrapper snapshot, actual runtime, complete source/input hashes, nonce SHA256 and durable terminal/cost joins are retained in [results.json](results.json) and [input-index.json](input-index.json). No internal nonce values, lease descriptors, raw logs or checkpoints are copied.

[Byte-original JUnit](cuda-scorer-junit.xml), [strict engineering receipt](control-receipt.json), [collect-only prerequisite](collection-receipt.json). Two tiny optimizer/observer controls establish this source's engineering behavior; they are not full scientific task runs.

This publisher reads retained bytes only. It imports no Torch/Forge/producer/scorer, constructs or restores no model, draws no samples and submits no jobs. Root separately attests the immutable input card. The compact receipts are portable; independent revalidation requires the local source snapshots and raw proof paths in the input index, which are not hydrated automatically.
