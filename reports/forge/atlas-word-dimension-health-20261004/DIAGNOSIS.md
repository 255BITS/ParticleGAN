# Word min11: retained undefined-dimension audit diagnosis

The original family result stays **INVALID**, with **558.5739127129782 paid
seconds and zero reserve**. This inspection creates no new numerical grade,
model, training restoration, sampler call or experiment. It reads the saved
checkpoint as tensor dictionaries on CPU and verifies its original byte and
typed-state hashes. All retained inputs were hashed again after inspection.

The portable machine report is [diagnosis.json](diagnosis.json). Its reproducer
is [inspect_word_health.py](inspect_word_health.py); both accept relocated raw
and frozen-source roots while enforcing the pinned identities. They never
import a historical training owner or invoke the numerical scorer.

## Exact rejected leaf and source-defined behavior

The frozen checkpoint contains eight nonfinite leaves. Seven are the already
recognized public LR diagnostic sentinels. The **only rejected leaf** is the
Python float NaN at `policy.birth_death.last.d_R`.

The saved public birth/death record is:

```text
step=20001; k=5; d_R=NaN; d_F=.926260882020859
skip="dimension undefined"; moved_rows=None
dim_skips=5363; evals=20001
```

The frozen `particlegan/birth_death.py:206` statistic excludes reference points
whose nearest non-self distance is zero. If none remain, it explicitly returns
`float("nan"), 0`. At lines 511–524, `maybe_apply` records the dimensions,
increments `dim_skips`, sets the exact skip marker and returns before the
density-evidence update or row-move branch. This is an intentional safely
handled eligibility sentinel, not a NaN learned parameter. It still represents
an unavailable reference dimension; this diagnosis grants no claim that the
birth/death mechanism successfully balanced the target.

The terminal 11×170 real-reference reservoir has **four unique rows**, with
multiplicities **2,2,3,4**. Every row therefore has an exact duplicate. The
source's deterministic critic-feature callback preserves equal inputs, which
explains zero reference-neighbor distances and the stored skip. No new critic
forward was needed or executed to inspect this fact.

The generic audit helper `finite_policy_state` at
`experiments/forge/policy_adapters.py:59` recognizes the narrow LR sentinels but
rejects this birth/death diagnostic. The producer records that false health
flag in `WordJointPolicyFixture.guards` (line 313), and the grader rejects it in
`views.py:711`. The scientific supervisor then correctly preserves an INVALID
ownership result instead of granting ordinary numerical-failure credit.

## Saved state, observation and goal limits

The terminal checkpoint has 192 tensor leaves. No nonfinite learned G/E/D,
averaged model, particle table, output-noise, Adam/AMSGrad or active birth/death
tensor was found. The nonfinite tensor belongs to an allowed LR masked-history
leaf. This excludes a saved-terminal learned-state corruption; it does not
prove every earlier tensor was finite or diagnose the training dynamics.

All 24 recorded metric dictionaries are finite. All 24 complete-state/global
purity records pass, all named-stream deviations are zero, and the actual
G/E/prior/D optimizer counters and public/caller clocks reach 20,001. The
machine report binds the original Recipe, initialization, task law, every
observation step and selected-snapshot identity.

The numerical goal remains missed. Every recorded
`reconstruction_exact` value is zero, against the original required value one.
The recorded endpoint has:

| Metric | Recorded endpoint | Original bound |
| --- | --- | --- |
| Sample count | 1,024 | ≥ 1,024 |
| Quality fraction | .9248046875 | ≥ .95 |
| Modes | 2 | = 5 |
| Mass TV | .6 | ≤ .1 |
| Paired reconstruction exact | 0 | = 1 |
| Minimum reconstruction token probability | 0 | ≥ .9 |

These are original retained values and declared bounds, not a fresh scoring or
regraded result. A diagnostic-marker repair cannot turn this run into a pass.
The target has five words, while the independently named resource adaptation
has eleven actual rows. No original N5, independent-Atlas, historical,
default or speed credit follows.

Only the terminal dimension record is retained. The counter establishes 5,363
skip events, but not their exact first time or alignment with each numerical
observation. The first exact observable dimension record is at update 20,001;
there is no per-update dimension trace or earlier checkpoint. The available
state explains the audit rejection, not why coverage or inverse reconstruction
failed. An optimizer/controller causal explanation is unresolved.

## Narrow patch and future binding

The separately prepared audit patch recognizes only an exact public
`birth_death.last` record with float-NaN `d_R`, the marker `dimension undefined`,
finite float `d_F`, original table-derived `k`, valid last/completed/evaluation/
skip clocks and `moved_rows=None`. It keeps the raw NaN and typed digest
unchanged. Missing/different markers, unknown diagnostic paths, forged current
move records and nonfinite learned state remain errors. No birth/death
algorithm, loss, target, optimizer, initialization, sampling law or numerical
gate changes.

Commit and source-pin the audit patch before any future use. Current
implementation/task source hashes must be updated under the actual new source
identity; the old raw health flag, grade, source snapshot and cost remain
unchanged. No unchanged C6 retry is proposed. Retained coverage/reconstruction
failures need their own evidence-backed question before another training run.

The audit-only patch is committed locally as
`6690feb5cd7d48511be6321f058d272c47e8318e`, based on
`9345c4a69b176e79ed0cc46533a2199935c0ae47`. It changes only
`experiments/forge/policy_adapters.py` and adds
`tests/test_forge_policy_state_health.py`. The focused model-free checks passed:
61 passed, 11 model-constructing controls deselected, in .28 seconds. Missing
markers, wrong clocks or table-derived `k`, active move records, and nonfinite
learned/optimizer/unknown state are covered by rejection controls. A separate
read-only tensor-dictionary check accepts this exact saved diagnostic record
without changing checkpoint bytes or its typed state digest. This is health
classification evidence; the old scientific grade remains unchanged.

## Immutable provenance

Frozen scientific source commit:
`fb7acc775b3a1a6184d36b55e035b9da04531492`.
Source digest:
`f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed`.

Checkpoint: `word-joint-policy/state.pt`, 4,544,387 bytes,
SHA256 `16388c4e59a657a154a4730e285191acfd18fbc23d41102d0c3ba82227efb8a7`.
Raw result SHA256:
`e26ec506c81e8cdd74447c466d516ef30785de501da28dd50e698c115ebe7515`.
Original grade SHA256:
`5eb697cf6678ef764204ca5cd9e755a9d2f6f1fe0530c58a608b18ee9e3d766a`.
Exact source, resolved request, run-log and study hashes are in the machine
report. The inspection was CPU-only with CUDA invisible and uninitialized.

To reproduce just this read-only inspection after relocating the inputs:

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/pr155-e22-venv/bin/python inspect_word_health.py \
  --run RAW_FAMILY/attempts/five_word_joint_acquisition_word_joint_policy_min11_v1 \
  --source-root FROZEN_SNAPSHOT --output NEW_DIAGNOSIS.json
```
