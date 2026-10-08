# Five-word acquisition and continuation hold

The word question now separates reaching the complete joint goal from retaining
it during learning. The new `five_word_joint_smoke` is required in Tier 1 and
`five_word_joint_hold` is required in Tier 2. Both retain the existing generator,
encoder, joint critic, five-row learned particle cloud, batch size 256, seed 0,
deterministic initializer and clean live sampling.

Tier 1 completes all 20,001 updates and evaluates 24 scheduled states. One state
must satisfy the complete generation and paired reconstruction bounds, and an
independent generation draw must confirm those bounds at the same training
state. Reconstruction is deterministic under clean sampling; it still checks
every correctly paired word, including padding, with token probabilities ≥.90.
The earliest confirmed state is certified with models, both optimizers, policy,
data generator and every named RNG stream. Later loss of the goal is reported
and does not revoke acquisition.

Tier 2 verifies and restores that candidate's own earliest confirmed checkpoint.
It checks the restored state and 24 evenly spaced states during exactly 4,000
additional updates. Every primary and confirmation check must satisfy the full
generation and inverse bounds. Optimizer histories and streams are preserved;
the original 20,000-update schedule horizon and recipe remain unchanged. The
selected BCAP learner keeps its constant rates. No target shift is part of this
word hold question.

The original `five_word_joint_acquisition` declaration retains its five-terminal
check gate and remains available for historical reproduction. Its declaration
bytes and saved verdicts are unchanged. Old passing snapshots are not new smoke
passes because they lack the new independent confirmation and certified earliest
checkpoint. New cells remain unmeasured in the ordinary technique inventory.

The full task-only CUDA verification is declared in [protocol.json](protocol.json)
and executed by [run_probe.py](run_probe.py). Its result, actual-training GIFs,
artifact receipt and software validation are published here after completion.
This probe grants no full-tier or family qualification.

Reservations are 900 seconds for word smoke and 300 seconds for word hold. The
revision-8 required ladder reserves 2,220 seconds through Tier 1, 42,720 through
Tier 2 and 46,320 through Tier 3. Including the optional clock diagnostic, the
new inventory campaign reserves 46,620 seconds per selected family and 326,340
seconds for the current seven-family roster. Historical campaigns are unchanged.

```sh
python reports/forge/five-word-tier-split/run_probe.py \
  --output runs/forge/five-word-tier-split-v1/full --device cuda:0 \
  > runs/forge/five-word-tier-split-v1/full.log 2>&1
tail -F runs/forge/five-word-tier-split-v1/full.log
python -m pytest -q tests/test_forge_word_smoke_hold.py
python reports/forge/regenerate_technique_inventory.py --refresh-publication
```
