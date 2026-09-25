# Current task: improve the selected K3P GAN

Read AGENTS.md, reports/toy100/current-research-base.json,
reports/toy100/k3p-base/README.md, and the current leaderboard. K3P is the
user-selected research and launcher base. Use the exact pinned files under
reports/toy100/gap-fill-20260925/sources/k3p: config, mechanism, latent, response,
and the corresponding gate driver. Config alone is not this formulation.

The user's current direction is continuous learning without a training-horizon
definition of "later". Read reports/toy100/k3p-base/continuous-search.md for that
objective, the constant-LR/disabled-anchor trap, and the hold/recovery-first
qualification sequence. The dedicated launch-gan-k3p-continuous.py configures
three bounded lanes using the existing search launcher.

K3P passes all 22 declared GPU toys, plus its 1,200-update ring hold and all 300
extension checks. The measured blocker is target-shift recovery at the selected
.01/.05 floors: 28/81 deadline checks pass, against the 81/81 requirement. Start
with that failure and the matched hold/extension. Do not rerun the unchanged
parent merely to rediscover known results. Preserve all 22 passes in a proposal
that improves recovery; early screens do not qualify the full suite.

The formulation remains Rp logistic GAN. A critic-LR-driven handover moves from
normalized real R1/fake RMS cap toward real/fake L2 caps plus an EMA-critic
input-gradient anchor (parameter EMA decay .999). A mature critic-gradient
spike guard caps the ratio to Adam's RMS history at 5. Inherited A2 sparse-latent
damping and direct-particle response remain part of the complete formulation.
The anchor adds critic forward/input-gradient work; no extra optimizer steps.

Use the frozen runtime, fixtures and commands in
reports/toy100/gap-fill-20260925/manifest.json. Source/runtime hashes and raw
results are retained with that report. Use probe.py for transfer tasks,
native100.py for full native coverage AND accuracy, and hold.py/shift.py plus
shift_frozen.py for ring stability and the matched frozen recovery control.
Fresh output directories, separate tail-able logs, and a declared source hash
are required. Keep every FAIL/ERROR and explicit NOT_RUN in the ledger.

Keep architecture, data, existing declared seeds, evaluation thresholds and step
budgets fixed. No seed sweeps, coefficient grids, task-name switches, target
centers/statistics, mode labels or metric feedback in training. Any proposed
loss, optimizer, regularizer or schedule change must be declared as a new
formulation and earn its own results. A .1/.1-floor variant and K3P+RG5 are
untested candidates; another formulation's recovery cannot be inherited.
Keep the original auxiliary AE/token host losses and selected schedules as
starting defaults. All training, gradients, Adam and mechanism-history tensors
stay CUDA FP32, deterministic and TF32-off, with one CPU thread per worker.
Transfer initialization uses the retained zero-update CPU fixtures; native
problems use CUDA initialization. Preserve the qualified Adam arithmetic.

A useful recovery proposal must also protect mode_hold, unequal mass, stripes,
unequal width, all three native gates, and then the full 22. Native runs require
all 7,000 updates and canonical coverage AND accuracy. Frozen transfer verdicts
require sustained terminal checks. No pooling of scores from different variants.
The selected parent has grid 4/4 historical seed passes but only one run each
on rotated/staggered; neither that nor 22/22 is a broad robustness estimate.

Hold and recovery were measured continuously. Do not claim fresh-process
checkpoint continuation without preserving critic EMA and hook/LR history,
latent observation counts, response history, model, optimizer and RNG state;
the current checkpoint helper does not serialize all module-global history.
Prefer the existing uninterrupted hold/recovery drivers for qualification.

Work within the invoking launcher's time, proposal and worker caps; no extra
agent sessions, pushes or comments. Selection itself does not authorize a new
experiment batch. For an explicitly launched attempt, start real training
promptly, adapt from measured failures, summarize the leaderboard and recommend
next experiments. Avoid new benchmark infrastructure or completed controls.

[Previous selected-base brief](../formulation-search-before-k3p.md).
