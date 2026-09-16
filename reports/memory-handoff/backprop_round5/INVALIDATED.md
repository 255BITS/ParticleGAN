# INVALIDATED: use backprop_round5_corrected

This queue used an incomplete G gradient: its real score was detached despite
its dependence on the generated memory. Two jobs completed but their results
must not enter model selection or valid leaderboards. Two active jobs were
intentionally terminated, including an unaffected extension that will restart
from its original2k checkpoint. One pending job was cancelled. No conclusions
are drawn from these outputs. The earlier detached-feedback and recent-memory
rounds are unaffected. The corrected implementation retains both branches of
the paired loss and tests cancellation of shared context-only score offsets.
