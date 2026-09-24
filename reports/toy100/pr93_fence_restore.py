"""PR #93 fence restore: the published exit clip armed on the PR84 stencil.

``exit_aware_candidate`` leaves ``clip_enabled`` false so a caller can score
the stencil alone. The fence-restore receipt turns the clip on.
"""

from contextlib import contextmanager

from reports.toy100.exit_aware_step_clip import exit_aware_candidate


@contextmanager
def pr93_fence_restore(*, task="mode_hold", start_step=0):
    with exit_aware_candidate(task=task, start_step=start_step) as (recorder, source):
        recorder.clip_enabled = True
        yield recorder, source
