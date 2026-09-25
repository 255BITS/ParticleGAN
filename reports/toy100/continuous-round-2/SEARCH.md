# Round 2: solve, verify, then promote

The user explicitly defines the research algorithm:
**Try to solve the failing problem -> identify a promising winner -> verify its
frozen gates -> only then use a verified winner as the next shared search base.**

K3P remains the selected base. This round launched three Codex and five Grok
attempts, with one Codex lane verifying unchanged A3. That verification is now
complete: A3 passes only 4/22 toys and is rejected. The user subsequently reduced
the live allocation to one Codex plus seven Grok; one finished-proposal Codex
attempt was stopped with results retained. Use the round-3 rolling launcher for
future launches. No Claude. Do not change current-research-base.json or promote
a partial result.

Read reports/toy100/continuous-round-1/README.md and relevant prior lane reports.
Do not repeat unchanged baselines or failed mechanisms. Every proposal must address
a measured failure. No coefficient grids, seed sweeps, metric feedback, target
centers, known change times or task-specific controllers. Preserve the GAN and
existing auxiliary losses. Search time budgets never enter the formulation.

For search lanes (the A3 verification exception is assigned explicitly):
1. Start from copied, hashed K3P files. Preserve successful acquisition mechanisms
   unless a precise declaration changes them. A3 and other first-round candidates
   are diagnostic references only, not shared new baselines.
2. Finish BOTH a meaningful candidate's canonical hold+300 extension AND shift
   driver, even if one fails. Do not terminate a plateau and call it a complete
   benchmark. Hard time limits and execution failures remain explicit incomplete
   evidence. Retain failures, applied-rate traces and exact source hashes.
3. A provisional winner must pass own hold/extension and every live shift quality
   flag: stationary.pass_all, continued_hold.pass_all and recovery.deadline_pass
   with all 81 deadline checks. The frozen driver then returns UNCONFIRMED pending
   its matched control. This is a trigger for verification, not a failed search.
   An 81/81 deadline with failed pre-shift hold is still a failure.
4. Immediately verify that provisional winner: matched frozen recovery control,
   sensitive four gates, then ALL 22 frozen toys at original seeds, budgets and
   thresholds. Native runs are 7000 updates, coverage AND accuracy. The exact same
   frozen formulation must earn every pass. Confirmed shift status must be PASS;
   compare exact pre-shift state/RNG/optimizer history and require the no-update
   control to fail all 81 deadline checks. A failed gate means no promotion.
5. Audit a substantial identical prefix under two declared horizons, including
   model/optimizer/controller/EMA/RNG state and actual rates/noise. A short prefix
   before either schedule changes is insufficient. Confirm native and transfer
   adapters install the same formulation. Fully passing survivors additionally
   face separately declared delayed and repeated changes on uninterrupted state.

The supervisor alone promotes after auditing complete evidence. If a candidate
solves the ring near the time cap, leave exact frozen hashes and ready-to-run
qualification commands so verification can continue immediately.

At most three mechanism proposals per search lane, one benchmark worker, no nested
agents. Start training within five minutes. No edits to old attempts, no pushes or
comments. Preserve every failure and return FAIL / ERROR / NOT_RUN explicitly.
