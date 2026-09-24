"""Three-update exact split-resume contract for a future memory adapter.

Callers supply source-bound bounded host runners after the adapter exists.
This module neither creates an optimizer nor changes a learning algorithm.
It rejects a resumed path that silently initializes fresh learner memory,
changes a source, skips a bank, or merely looks similar in its quality score.
"""

from reports.toy100.learner_state_envelope import assert_exact_split, validate
from reports.toy100.pr84_critic_refinement_capture import _sha


def run_short_split(run_uninterrupted, run_prefix, run_resumed, *, method,
                    source_sha256, initial_host_sha256, start_step=0,
                    split_step=1, target_step=3):
    """Run caller-provided short paths and compare exact final states/records.

    Each callback must use the same frozen factory and host inputs. The
    uninterrupted and prefix callbacks receive a final absolute step. The
    resume callback receives the exact prefix learner envelope and a final
    absolute step. Every returned branch must contain its complete envelope,
    source and starting host hashes, absolute observed-step list, and passive
    per-update observations. The resumed branch additionally reports the
    envelope it loaded *before* the next data bank.
    """
    if not (type(start_step) is type(split_step) is type(target_step) is int
            and 0 <= start_step < split_step < target_step
            and target_step - start_step <= 3):
        raise ValueError('exact split harness is limited to at most three updates')
    full = run_uninterrupted(target_step)
    prefix = run_prefix(split_step)
    resumed = run_resumed(prefix['envelope'], target_step)
    for name, branch, end, steps in (
            ('uninterrupted', full, target_step, range(start_step+1, target_step+1)),
            ('prefix', prefix, split_step, range(start_step+1, split_step+1)),
            ('resumed', resumed, target_step, range(split_step+1, target_step+1))):
        if (branch.get('source_sha256') != source_sha256
                or branch.get('steps') != list(steps)
                or len(branch.get('observations', ())) != len(list(steps))):
            raise RuntimeError(f'{name} source or exact update sequence changed')
        validate(branch['envelope'], method=method,
                 source_sha256=source_sha256, completed_step=end)
    if (full.get('initial_host_sha256') != initial_host_sha256
            or prefix.get('initial_host_sha256') != initial_host_sha256
            or full.get('initial_learner_sha256') != prefix.get('initial_learner_sha256')):
        raise RuntimeError('full and prefix did not start from the same host and learner state')
    if (resumed.get('restored_host_sha256') != prefix['envelope']['host_sha256']
            or resumed.get('loaded_envelope_sha256') != prefix['envelope']['envelope_sha256']
            or resumed.get('learner_loaded_before_next_bank') is not True):
        raise RuntimeError('resume did not restore the complete prefix before its next bank')
    if _sha(prefix['observations'] + resumed['observations']) != _sha(full['observations']):
        raise AssertionError('uninterrupted and split per-update observations differ')
    assert_exact_split(full['envelope'], resumed['envelope'], method=method,
                       source_sha256=source_sha256, completed_step=target_step)
    return dict(status='EXACT_SHORT_SPLIT_PASS', method=method,
                source_sha256=source_sha256, start_step=start_step,
                split_step=split_step, target_step=target_step,
                full_envelope_sha256=full['envelope']['envelope_sha256'],
                prefix_envelope_sha256=prefix['envelope']['envelope_sha256'],
                resumed_envelope_sha256=resumed['envelope']['envelope_sha256'],
                observation_sha256=_sha(full['observations']),
                training_scope='caller-provided frozen host only; at most three updates')
