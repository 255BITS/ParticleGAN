# Five-word acquisition and continuation hold

**The selected BCAP DualNorm learner acquires the complete word goal, but fails
continued retention.** The full CUDA probe confirms acquisition at update 1,667,
finishes all 20,001 smoke updates, and restores that exact checkpoint for the
separate 4,000-update hold. Both endpoints pass; failures inside the hold window
still reject Tier 2.

| Question | Result | Full joint checks | Endpoint |
| --- | --- | --- | --- |
| [Tier 1 acquisition](../../toy_audit/api_contract/five_word_smoke_hold/media/five_word_joint_smoke.gif) | PASS; first confirmed update 1,667 | 13/24 independently confirmed states | PASS |
| [Tier 2 continued hold](../../toy_audit/api_contract/five_word_smoke_hold/media/five_word_joint_hold.gif) | FAIL; all 4,000 additional updates completed | 20/25 independently confirmed states | PASS |

The first hold failure occurs at cumulative update 3,001, after 1,334 additional
updates: generation still produces all five confident words with mass TV .03574,
but paired reconstruction is wrong and its minimum correct-token probability is
.0001255. Updates 5,001, 5,167, 5,334 and 5,501 also fail. At 5,167 generation
itself falls to three modes. The final 5,667 checkpoint recovers all five modes,
mass TV .03457 and exact reconstruction with minimum token probability 1.0.
This separates temporary acquisition from continuous retention without hiding
later failures or dropping the inverse requirement.

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
and executed by [run_probe.py](run_probe.py). [Final metrics and provenance](readout.json),
[artifact receipt](archive.json), [software validation](validation.json) and the
[API publication](../../toy_audit/api_contract/five_word_smoke_hold/publication.json)
retain the exact recipe, source and runtime. This probe grants no full-tier or
family qualification. The single current technique inventory is refreshed to
show the new ordinary contracts as unknown, preserving every scientific row.

There are exactly 24,001 new research updates, with no seed alternatives, rate
annealing, scientific retry or optimizer reset. Smoke costs 806.111 adapter-loop
seconds and hold 124.320; total wall time including GIF export is 947.754 seconds,
within the 1,200-second combined reservation. Training ran on physical GPU 0,
with project-default disabled multithreaded autograd and spectral truncation.
One automatic cuSOLVER SVD fallback warning is retained in raw stdout; all states
and outputs remain finite. Its causal role was not tested and no driver changed.

[publish.py](publish.py) recomputes all 100 saved primary/confirmation metric sets
from their actual arrays, verifies both final state certificates and the earliest
acquisition certificate, and copies the two genuine training GIFs. Bulk states,
arrays and stdout remain outside Git in the local artifact archive, SHA-256
`e2944bc031f96e9bbdd09fe8c8be24dff95a9d92fd8dd07c45a5ad9b220e0d0b`.
The frozen executed source is `0270a501b7d90f4d6c34b4c3e9ef4166f877b301`.

The public checkpoint validator now treats `recipe.total_steps` as a schedule
horizon, consistent with its caller-owned `execution_limit`. A valid state after
that horizon can be restored; the host still checks its actual execution cap.
CUDA checks cover preserved optimizer history, rates and rejection past the cap.
This matters for an acquisition state at 20,001 or continuation past 20,000.
There are 372 passing software checks, including nine CUDA numerical/provenance
checks. Existing word execution/model fixtures now use explicit CUDA devices;
schema and planning checks remain read-only. One pre-existing source-pin check
for archived shallow Gaussian declarations is excluded and documented in the
validation receipt. Separate combined-branch verification adds four CUDA and
two schema checks for the split alongside optional smoothing, with no repeat
of research training.

Recommendation: adopt the task split and retain the measured holding failure.
Use the [separate saved-state diagnosis (PR346)](https://github.com/255BITS/ParticleGAN/pull/346)
to choose a mechanism experiment. [Optional smoothing (PR345)](https://github.com/255BITS/ParticleGAN/pull/345)
remains off by default and available to declared configuration searches. This
probe demonstrates acquisition and transient inverse/generation loss, and does
not identify a causal optimizer defect or establish long-term stability.

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
