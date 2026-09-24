"""Complete host-plus-learner checkpoints for future support-memory adapters.

The existing mode-hold snapshots contain model/Adam/EMA/noise/RNG only. A
memory-bearing recorder must save and restore its ordered sufficient
statistics as well. This module does not construct groups or change updates.
The two audit counters never enter the matching algorithm.
"""

from contextlib import contextmanager
import math
from unittest.mock import patch

import torch

from reports.toy100.pr84_critic_refinement_capture import _clone, _sha, snapshot
from reports.toy100.pr84_critic_refinement_resume import restore_snapshot


SCHEMA = 'toy100.complete-host-learner.v1'
LEARNER_SCHEMA = 'two-bank-fixed-support-v1'
LEARNER_KEYS = frozenset(('schema', 'last_bank_id', 'last_bank_sha256',
    'pending', 'confirmed', 'confirmed_sums', 'confirmed_counts',
    'confirmed_squared_norm_sums', 'reference_centers',
    'fixed_half_separation', 'accepted_samples_total',
    'rejected_samples_total', 'confirmation'))
PENDING_KEYS = frozenset(('bank_id', 'bank_sha256', 'sums', 'counts',
                         'squared_norm_sums'))
CONFIRMATION_KEYS = frozenset(('first_bank_id', 'second_bank_id',
    'first_bank_sha256', 'second_bank_sha256', 'reason', 'first_groups',
    'second_groups', 'max_pair_distance', 'min_separation', 'margin', 'pairing'))
ENVELOPE_KEYS = frozenset(('schema', 'learner_schema', 'method', 'source_sha256', 'completed_step',
    'bank_count', 'last_observed_step', 'host', 'learner', 'host_sha256',
    'learner_sha256', 'envelope_sha256'))


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _sha256_hex(value, name):
    if (type(value) is not str or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{name} is missing or malformed')
    return value


def source_fingerprint(bound_files):
    """Digest the complete frozen filename→SHA-256 declaration, in order."""
    if type(bound_files) is not dict or not bound_files:
        raise ValueError('source binding requires a nonempty file-hash mapping')
    for name, digest in bound_files.items():
        if type(name) is not str or not name or name.startswith('/') or '..' in name.split('/'):
            raise ValueError('source binding has an unsafe relative filename')
        _sha256_hex(digest, f'source digest for {name}')
    return _sha(bound_files)


def _statistics(sums, counts, squared, *, name, allow_empty=False):
    """Validate one ordered group table without sorting or aggregating it."""
    if any(type(value) is not list for value in (sums, counts, squared)):
        raise ValueError(f'{name} statistics must be ordered lists')
    if (not sums and not allow_empty) or len(sums) != len(counts) or len(sums) != len(squared):
        raise ValueError(f'{name} statistic lengths differ or are unexpectedly empty')
    for index, (total, count, second) in enumerate(zip(sums, counts, squared)):
        if (not isinstance(total, torch.Tensor) or total.dtype != torch.float64
                or total.device.type != 'cpu' or total.shape != (2,)
                or not bool(torch.isfinite(total).all())):
            raise ValueError(f'{name} group {index} sum must be a finite CPU float64 2-vector')
        _integer(count, f'{name} group {index} count', minimum=1)
        if (not isinstance(second, torch.Tensor) or second.dtype != torch.float64
                or second.device.type != 'cpu' or second.shape != ()
                or not bool(torch.isfinite(second)) or float(second) < 0):
            raise ValueError(f'{name} group {index} squared norm sum must be a finite nonnegative CPU float64 scalar')
        lower = float(total.square().sum()) / count
        if float(second) + 1e-8 * max(1., lower) < lower:
            raise ValueError(f'{name} group {index} second moment is inconsistent with its sum/count')


def _confirmation(value, group_count, last_id):
    if type(value) is not dict or set(value) != CONFIRMATION_KEYS:
        raise ValueError('confirmation receipt is missing or has changed fields')
    first = _integer(value['first_bank_id'], 'first confirmation bank ID', minimum=1)
    second = _integer(value['second_bank_id'], 'second confirmation bank ID', minimum=1)
    if first + 1 != second or second > last_id:
        raise ValueError('confirmation bank IDs are not distinct increasing observations')
    if (_sha256_hex(value['first_bank_sha256'], 'first confirmation bank SHA-256')
            == _sha256_hex(value['second_bank_sha256'], 'second confirmation bank SHA-256')):
        raise ValueError('confirmation reused the same raw real bank')
    if value['reason'] != 'CONFIRMED':
        raise ValueError('confirmation receipt lacks confirmed reason')
    if (_integer(value['first_groups'], 'first_groups', minimum=1) != group_count
            or _integer(value['second_groups'], 'second_groups', minimum=1) != group_count):
        raise ValueError('confirmation group count differs from frozen identities')
    pairing = value['pairing']
    if (type(pairing) is not list or len(pairing) != group_count
            or any(type(item) is not int for item in pairing)
            or sorted(pairing) != list(range(group_count))):
        raise ValueError('confirmation pairing is not a complete ordered permutation')
    for name, strictly_positive in (('max_pair_distance', False),
                                    ('min_separation', True), ('margin', True)):
        number = value[name]
        if type(number) is not float or not math.isfinite(number) or (
                number <= 0 if strictly_positive else number < 0):
            raise ValueError(f'confirmation {name} is invalid')
    if value['margin'] != value['min_separation'] - 2 * value['max_pair_distance']:
        raise ValueError('confirmation margin differs from its geometric certificate')


def _memory(state):
    """Validate and clone every field of the versioned two-bank state.

    Unexpected fields fail closed; none are filtered out of the hashed copy.
    A future learner schema needs an explicit validator rather than silently
    dropping covariance, pending-bank, or other statistics.
    """
    if type(state) is not dict or set(state) != LEARNER_KEYS:
        raise ValueError('learner state is missing or has changed required statistics')
    if state['schema'] != LEARNER_SCHEMA:
        raise ValueError('learner state schema changed')
    last_id = state['last_bank_id']
    last_hash = state['last_bank_sha256']
    if (last_id is None) != (last_hash is None):
        raise ValueError('last observed bank ID and hash must be present together')
    if last_id is not None:
        _integer(last_id, 'last_bank_id', minimum=1)
        _sha256_hex(last_hash, 'last_bank_sha256')
    if type(state['confirmed']) is not bool:
        raise ValueError('confirmed flag must be a boolean')
    separation = state['fixed_half_separation']
    if state['confirmed']:
        if type(separation) is not float or not math.isfinite(separation) or separation <= 0:
            raise ValueError('confirmed identities lack positive fixed half separation')
    elif separation is not None:
        raise ValueError('unconfirmed identities cannot have fixed separation')
    for name in ('accepted_samples_total', 'rejected_samples_total'):
        _integer(state[name], name)
    if state['confirmed']:
        if state['accepted_samples_total'] != sum(state['confirmed_counts']):
            raise ValueError('accepted sample count differs from confirmed statistics')
    elif state['accepted_samples_total'] or state['rejected_samples_total']:
        raise ValueError('unconfirmed memory has sample acceptance counters')
    confirmation = state['confirmation']
    if state['confirmed'] != (confirmation is not None):
        raise ValueError('confirmed state and confirmation receipt disagree')
    _statistics(state['confirmed_sums'], state['confirmed_counts'],
                state['confirmed_squared_norm_sums'], name='confirmed',
                allow_empty=not state['confirmed'])
    reference = state['reference_centers']
    if type(reference) is not list or len(reference) != len(state['confirmed_sums']):
        raise ValueError('ordered reference centers differ from confirmed identities')
    for index, center in enumerate(reference):
        if (not isinstance(center, torch.Tensor) or center.dtype != torch.float64
                or center.device.type != 'cpu' or center.shape != (2,)
                or not bool(torch.isfinite(center).all())):
            raise ValueError(f'reference center {index} is not a finite CPU float64 2-vector')
    if state['confirmed']:
        _confirmation(confirmation, len(reference), last_id)
        if (last_id == confirmation['second_bank_id']
                and last_hash != confirmation['second_bank_sha256']):
            raise ValueError('confirmation second bank hash differs from last observed bank')
        if len(reference) < 2:
            raise ValueError('confirmed reference support needs at least two identities')
        distances = torch.cdist(torch.stack(reference), torch.stack(reference))
        distances.fill_diagonal_(float('inf'))
        half = float(distances.min()) / 2.
        if not math.isclose(separation, half, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('fixed half separation differs from frozen reference centers')
    if not state['confirmed'] and state['confirmed_sums']:
        raise ValueError('unconfirmed state cannot contain confirmed statistics')
    pending = state['pending']
    if pending is not None:
        if type(pending) is not dict or set(pending) != PENDING_KEYS:
            raise ValueError('pending bank state is missing required statistics')
        _integer(pending['bank_id'], 'pending bank_id', minimum=1)
        _sha256_hex(pending['bank_sha256'], 'pending bank_sha256')
        _statistics(pending['sums'], pending['counts'],
                    pending['squared_norm_sums'], name='pending')
        if (state['confirmed'] or pending['bank_id'] != last_id
                or pending['bank_sha256'] != last_hash):
            raise ValueError('pending bank is inconsistent with confirmation or last bank')
    if last_id is None and (pending is not None or state['confirmed']):
        raise ValueError('unobserved learner cannot have pending or confirmed support')
    if last_id is not None and not state['confirmed'] and pending is None:
        raise ValueError('unconfirmed observed learner is missing its pending bank')
    return _clone(state)


def _host(host, completed_step):
    if type(host) is not dict or 'learner' in host:
        raise ValueError('expected the existing complete host snapshot without learner state')
    try:
        scope = host['snapshot_scope']
        noise = host['noise']['step_calls']
        policy = host['noise_policy']['_step_calls']
    except (KeyError, TypeError) as error:
        raise ValueError('incomplete host snapshot') from error
    if (scope.get('version') != 1 or noise != completed_step
            or policy != completed_step
            or scope.get('host_loop_step') not in (completed_step - 1, completed_step)):
        raise ValueError('host snapshot is not a complete post-update boundary')
    for key in ('generator', 'critic', 'prior', 'optimizer_d', 'optimizer_g',
                'ema_g', 'ema_z', 'rng', 'noise_policy'):
        if key not in host:
            raise ValueError(f'host snapshot omits {key}')
    return _clone(host)


def validate(envelope, *, method=None, source_sha256=None, completed_step=None):
    """Reject altered, missing, or schema-incompatible learner memory."""
    if type(envelope) is not dict or set(envelope) != ENVELOPE_KEYS:
        raise ValueError('complete learner envelope keys changed or memory missing')
    if envelope['schema'] != SCHEMA or envelope['learner_schema'] != LEARNER_SCHEMA:
        raise ValueError('learner envelope schema changed')
    if type(envelope['method']) is not str or not envelope['method']:
        raise ValueError('learner method is missing')
    if method is not None and envelope['method'] != method:
        raise ValueError('learner method differs from bound candidate')
    source = _sha256_hex(envelope['source_sha256'], 'learner source SHA-256')
    if source_sha256 is not None and source != source_sha256:
        raise ValueError('learner source differs from bound candidate')
    step = _integer(envelope['completed_step'], 'completed_step', minimum=1)
    if completed_step is not None and step != completed_step:
        raise ValueError('completed update differs from bound resume boundary')
    banks = _integer(envelope['bank_count'], 'bank_count', minimum=1)
    if banks > step:
        raise ValueError('more distinct observed banks than completed updates')
    if _integer(envelope['last_observed_step'], 'last_observed_step', minimum=1) != step:
        raise ValueError('learner memory was not observed on the completed update')
    host = _host(envelope['host'], step)
    learner = _memory(envelope['learner'])
    if learner['last_bank_id'] != envelope['last_observed_step']:
        raise ValueError('learner last bank ID differs from observed absolute step')
    if _sha(host) != envelope['host_sha256'] or _sha(learner) != envelope['learner_sha256']:
        raise ValueError('host or learner state hash changed')
    unsigned = {key: value for key, value in envelope.items() if key != 'envelope_sha256'}
    if _sha(unsigned) != envelope['envelope_sha256']:
        raise ValueError('complete host-plus-learner envelope hash changed')
    return host, learner


def pack(host, recorder, *, method, source_sha256, completed_step):
    """Copy complete host state and recorder memory without a random draw."""
    if not callable(getattr(recorder, 'learner_state_dict', None)):
        raise ValueError('recorder cannot serialize its learner memory')
    host = _host(host, completed_step)
    learner = _memory(recorder.learner_state_dict())
    count = _integer(getattr(recorder, 'learner_bank_count', None),
                     'learner_bank_count', minimum=1)
    last = _integer(getattr(recorder, 'learner_last_observed_step', None),
                    'learner_last_observed_step', minimum=1)
    if count > completed_step or last != completed_step or learner['last_bank_id'] != last:
        raise ValueError('learner bank audit counters differ from completed host update')
    value = dict(schema=SCHEMA, learner_schema=LEARNER_SCHEMA, method=method,
        source_sha256=source_sha256,
        completed_step=completed_step, bank_count=count, last_observed_step=last,
        host=host, learner=learner, host_sha256=_sha(host),
        learner_sha256=_sha(learner))
    value['envelope_sha256'] = _sha(value)
    validate(value, method=method, source_sha256=source_sha256,
             completed_step=completed_step)
    return value


def capture(local, recorder, *, method, source_sha256, completed_step):
    """Take a passive live checkpoint after an update and before the next bank."""
    before = snapshot(local)
    value = pack(before, recorder, method=method,
                 source_sha256=source_sha256, completed_step=completed_step)
    if _sha(snapshot(local)) != _sha(before):
        raise RuntimeError('learner checkpoint changed host state or training RNG')
    if _sha(_memory(recorder.learner_state_dict())) != value['learner_sha256']:
        raise RuntimeError('learner checkpoint changed recorder statistics')
    return value


def _same_host(local, host):
    actual = snapshot(local)
    # Existing source-bound resume starts Python's next loop at completed_step,
    # whereas a post-checkpoint host may have recorded completed_step - 1.
    actual['snapshot_scope']['host_loop_step'] = host['snapshot_scope']['host_loop_step']
    return _sha(actual) == _sha(host)


def load_after_host_restore(local, recorder, envelope, *, method, source_sha256,
                            completed_step):
    """Restore memory once, after host restore and before the next set_step."""
    host, learner = validate(envelope, method=method,
                             source_sha256=source_sha256,
                             completed_step=completed_step)
    if not _same_host(local, host):
        raise RuntimeError('host was not exactly restored before learner memory')
    load = getattr(recorder, 'load_learner_state_dict', None)
    if not callable(load):
        raise ValueError('recorder cannot restore its learner memory')
    load(_clone(learner))
    recorder.learner_bank_count = envelope['bank_count']
    recorder.learner_last_observed_step = envelope['last_observed_step']
    if (_sha(_memory(recorder.learner_state_dict())) != envelope['learner_sha256']
            or recorder.learner_bank_count != envelope['bank_count']
            or recorder.learner_last_observed_step != envelope['last_observed_step']):
        raise RuntimeError('recorder did not restore exact ordered learner state')
    if not _same_host(local, host):
        raise RuntimeError('learner restore changed host model, Adam, noise, or RNG')


def restore(local, recorder, envelope, *, method, source_sha256, completed_step):
    """Restore both states before the next bank; no old host-only fallback."""
    host, _ = validate(envelope, method=method, source_sha256=source_sha256,
                       completed_step=completed_step)
    restore_snapshot(local, host, completed_steps=completed_step)
    load_after_host_restore(local, recorder, envelope,
                            method=method, source_sha256=source_sha256,
                            completed_step=completed_step)


@contextmanager
def attach_before_next_bank(resume_state, recorder, envelope, *, method,
                            source_sha256):
    """Insert memory restore into the audited pre-set_step host resume hook.

    Enter this context after ``resume_mode_hold`` is constructed but before
    calling the host. The host restore remains in its original hook. This
    wrapper restores learner memory immediately afterward and before the
    next noise clock, D bank, or G bank. It performs no update itself.
    """
    host, _ = validate(envelope, method=method,
                       source_sha256=source_sha256,
                       completed_step=resume_state.completed_steps)
    if _sha(resume_state.saved) != _sha(host):
        raise ValueError('resume host snapshot differs from memory envelope')
    original = resume_state.before_step
    restored = False

    def before_step(step, local):
        nonlocal restored
        if not restored and step != envelope['completed_step']:
            raise RuntimeError('memory resume did not start at its exact outer update')
        original(step, local)
        if not restored:
            if not resume_state.restored or local['noise_policy']._step_calls != step:
                raise RuntimeError('learner restore is not before next set_step')
            load_after_host_restore(local, recorder, envelope,
                                    method=method, source_sha256=source_sha256,
                                    completed_step=step)
            restored = True

    with patch.object(resume_state, 'before_step', before_step):
        yield
    if not restored:
        raise RuntimeError('memory-bearing continuation never restored learner state')


def assert_exact_split(uninterrupted, resumed, *, method, source_sha256,
                       completed_step):
    """Final no-tolerance comparison for a short source-bound split harness."""
    validate(uninterrupted, method=method, source_sha256=source_sha256,
             completed_step=completed_step)
    validate(resumed, method=method, source_sha256=source_sha256,
             completed_step=completed_step)
    if (uninterrupted['host_sha256'] != resumed['host_sha256']
            or uninterrupted['learner_sha256'] != resumed['learner_sha256']
            or uninterrupted['bank_count'] != resumed['bank_count']
            or uninterrupted['last_observed_step'] != resumed['last_observed_step']
            or uninterrupted['envelope_sha256'] != resumed['envelope_sha256']):
        raise AssertionError('uninterrupted and split-resume learner states differ')
