# Physical two-GPU pilot readout

The registered v3 pilot **passed its operational checks** on two RTX A6000s.
Two completed vector tasks passed scientifically; one deliberately cancelled
attempt and its single repair remain recorded. Total paid time was
**45.000919218 seconds**, with zero final reservations and no live pilot worker.
This establishes queue behavior, not accepted scientific calibration.

| Candidate / attempt | Physical GPU | Outcome | Paid seconds |
| --- | ---: | --- | ---: |
| K3P | 1 | PASS | 19.400400857 |
| Critic-penalty ablation, original | 0 | CANCELLED | 8.452047890 |
| Critic-penalty ablation, one repair | 0 | PASS | 17.148470471 |

Both completed `vector_two_broad` runs used the learned MoG prior, fixed seed 0,
matched initialization law, clean live sampling and all 1,200 registered updates.
Each recorded 24 observations, 23 passing checks, a confirmed pass at step 300,
and zero unintended RNG deviations. K3P ended at HQ 0.988281, mass TV 0.000244
and normalized SW1 0.027492; the ablation ended at HQ 0.972656, mass TV 0.007568
and normalized SW1 0.078604. K3P had lower distribution errors on this task.
Both measured 69,537,280 peak allocated / 71,303,168 reserved PyTorch bytes.
These are single-seed, single-task results, not general algorithm rankings.

## Operational evidence

The [saved proof](pilots/develop-20260929-gpu-pilot-v3/summary.json) and
[process/event witnesses](pilots/develop-20260929-gpu-pilot-v3/orchestration.jsonl)
record these checks:

- The feature checkout and an independent fresh clone at `cd5308ec` submitted
  identical request IDs and shared exactly one execution per cell.
- Both runtime PIDs were observed simultaneously on different physical GPU
  UUIDs after real training observations appeared in their logs.
- Only the coordinator PID was interrupted. Its replacement recovered the same
  live attempt IDs, tokens, PIDs, process start identities and execution leases;
  training progress continued without duplicate launch.
- Cancelling the ablation stopped its owned process group and released its lease
  and reservation. One explicit repair reused the frozen request. No scientific
  failure or completed pass was retried.
- Exactly three attempts were paid once each, within the 5,400-second registered
  maximum. Final reservations were zero and all pilot process groups exited.
- The two pre-existing desktop clients retained their original process identities
  and command lines. No external training process occupied either device during
  the preflight ownership check.

Source digest:
`fc012b47b3714f8b7f67e21ce4d1bf3ff2809b3ff5aa001cff00b36663d1d39d`.
The [registration](calibration-lanes/develop-20260929-gpu-pilot-v3/registration.json)
and [proof file hashes](pilots/develop-20260929-gpu-pilot-v3/files.json) bind the
scope, orchestration source, process receipts and logs. Scientific attempts are
[`e572cb95`](attempts/e572cb95deeb49a18e712fa80f13e9c6/result.json),
[`661bd983`](attempts/661bd9830be94225891954f5fb690342/result.json), and
[`f04fd435`](attempts/f04fd435083a46b68ff8bf278a8eb26d/result.json).
Both exact-revision readouts are concluded:
[K3P](records/readout-c3df7159c90725fdef1cb700.json) and
[ablation](records/readout-060e7be73e2832e314b9573f.json).

An [independent review](pilots/develop-20260929-gpu-pilot-v3/independent-review.md)
verified the saved process and receipt certificates. The
[retiering proof](pilots/develop-20260929-gpu-pilot-v3/retiering-proof.json)
changes an ordinary view's task tier in memory: policy identity changes,
scientific/ordinary execution identity stays fixed, and the real pilot receipts
remain excluded from ordinary qualification. Direct diagnostic reduction remains
tier 0 and ineligible. Queue bytes, logs and paid cost were unchanged.

```sh
tail -F /home/martyn/dev/ParticleGAN/runs/forge/calibration-develop-20260929-gpu-pilot-v3/progress.jsonl
```

## Remaining acceptance

Diagnostic receipts remain ineligible for ordinary qualification. The
[v3 matrix](calibration/develop-20260929-quick-v3.md) now has two reference-cell
passes and one corrected intensity failure, with the rest unknown; there is no
complete positive/negative reference pairing or accepted gate profile. The
[host-provenance review](HOST_PROVENANCE_AUDIT.md) must guide any further
bounded science. Legacy reconciliation and default cutover remain conditional on
accepted calibration; a successful scheduler pilot does not waive that gate.
