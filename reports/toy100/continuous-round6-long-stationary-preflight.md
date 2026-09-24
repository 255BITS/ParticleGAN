# Source-bound long stationary continuation: preflight only

`sample_anchor_long_stationary_probe.py` is a reusable mode-hold diagnostic for an
explicit `reports.toy100.module:context_factory`. It resumes the factory's
own-acquired, post-update 2400 snapshot only after that same factory has passed
the source-bound cold trajectory and ring, every-update 1201–2400 hold, and
paired same-target 2401–2450 model-error response. The response starts from
the 2400 snapshot; the long arm separately starts from that same unperturbed
snapshot. It keeps nominal D/G/prior rates `.00425/.00425/.0085`, Adam states,
the original 1200-step noise horizon, and the fixed target. The generated
host only changes its loop to `range(2400, 12000)` and inserts the previously
audited before-`set_step` restore/observer. Its source is archived and hashed.

The long arm would grade all 9600 completed updates with the existing
8-mode/HQ ≥ .9 threshold and stop on the first miss. It prints progress every
100 updates and passively summarizes model/Adam finiteness and norms plus
*existing* thin-SVD correction receipts every 200. A quality miss saves full
before-step and after-checkpoint states. A numerical exception saves the last
complete before-step state and, if capturable, a clearly nonrestartable partial
state. Raw SVD max/min is a diagnostic for the recorded 24-row Jacobian; it
does not certify full parameter-space conditioning. No extra game gradients,
real samples, or noise draws are used by the observer.

No long training has been run. Three no-training tests passed, including a
real qualified cold/hold/response preflight, exact 2400→12000 AST generation,
and rejection of a failed response before an output directory is created.
The qualified preflight returned the prestart factory's update-2400 snapshot
and `PASS_LOCAL_RESPONSE`; it does not overrule the later structural concern.

The separate same-target conditional-bank audit explains why this driver must
not yet be launched on the current memoryless sample-anchor factory. The
ring sampler draws each of 128 component indices uniformly, so one bank with
no component 0 has probability `(7/8)^128 = 3.775989577235679e-8`.
`anchor_missing_batch_group.py` used a **borrowed passing** update-2400 state
and showed the free-output target lose one mode when that bank has seven
groups. The stronger `anchor_missing_batch_neural.py` started from the
**own-acquired** update-2400 state and conditioned only the replayed D real
bank; G real draws and the eight-mode target were unchanged. Ordinary next
update was 8/HQ 1; the conditioned neural update was 7/HQ 1 after a converged
joint fit. Both branches restored the same snapshot, had one D and G Adam
update, three callbacks per player, and the same final RNG hash. The
diagnostic deliberately conditions the batch; it does not show that this
rare event occurred in an ordinary run, but it exposes a possible
same-target, one-step coverage loss that a perpetual guarantee must address.

Source and raw evidence for that separate audit live under
`artifacts/continuous-learning/round6/anchor-missing-batch-neural` in the
integration workspace. All 151 archived source hashes matched current source
at audit time. This preflight commit contains no new candidate or training
result.
