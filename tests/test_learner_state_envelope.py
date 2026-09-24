"""No-training exact checkpoint and missing-memory rejection tests."""

from copy import deepcopy
from io import BytesIO
import math
from types import SimpleNamespace

import pytest
import torch

from reports.toy100 import learner_state_envelope as envelope
from reports.toy100.learner_state_split_harness import run_short_split
from reports.toy100.pr84_critic_refinement_capture import _sha


METHOD = 'test_ordered_support_memory'
SOURCE = 'a' * 64


def host(step=3):
    return dict(generator={'weight': torch.tensor([1.])}, critic={'weight': torch.tensor([2.])},
        prior={'z': torch.tensor([3.])}, optimizer_d={'step': step},
        optimizer_g={'step': step}, ema_g=[torch.tensor([4.])],
        ema_z=torch.tensor([5.]), rng={'torch': torch.tensor([6], dtype=torch.uint8)},
        noise={'step_calls': step}, noise_policy={'_step_calls': step},
        snapshot_scope={'version': 1, 'host_loop_step': step-1})


def test_source_fingerprint_is_complete_and_mapping_order_independent():
    first = {'b.py': 'b'*64, 'a.py': 'a'*64}
    assert envelope.source_fingerprint(first) == envelope.source_fingerprint(
        {'a.py': 'a'*64, 'b.py': 'b'*64})
    with pytest.raises(ValueError, match='unsafe relative filename'):
        envelope.source_fingerprint({'../outside.py': 'a'*64})


def memory(offset=0.):
    return dict(schema=envelope.LEARNER_SCHEMA, last_bank_id=3,
        last_bank_sha256='b'*64, pending=None, confirmed=True,
        fixed_half_separation=math.sqrt(17.)/2., accepted_samples_total=3,
        rejected_samples_total=0, confirmation=dict(first_bank_id=1,
            second_bank_id=2, first_bank_sha256='a'*64,
            second_bank_sha256='b'*64, reason='CONFIRMED',
            first_groups=2, second_groups=2, max_pair_distance=.1,
            min_separation=math.sqrt(17.),
            margin=math.sqrt(17.)-.2, pairing=[0, 1]),
        confirmed_sums=[torch.tensor([2.+offset, 4.], dtype=torch.float64),
                        torch.tensor([-3., 1.], dtype=torch.float64)],
        confirmed_counts=[2, 1],
        reference_centers=[torch.tensor([1., 2.], dtype=torch.float64),
                           torch.tensor([-3., 1.], dtype=torch.float64)],
        confirmed_squared_norm_sums=[
            torch.tensor(10.+offset*offset+2*offset, dtype=torch.float64),
            torch.tensor(10., dtype=torch.float64)])


class Recorder:
    def __init__(self, state, bank_count=3, last=3):
        self.state = deepcopy(state)
        self.learner_bank_count = bank_count
        self.learner_last_observed_step = last

    def learner_state_dict(self):
        return deepcopy(self.state)

    def load_learner_state_dict(self, state):
        self.state = deepcopy(state)


def test_complete_host_and_ordered_memory_roundtrip_restores_exactly(monkeypatch):
    original = host()
    source = Recorder(memory())
    value = envelope.pack(original, source, method=METHOD,
                          source_sha256=SOURCE, completed_step=3)
    raw = BytesIO()
    torch.save(value, raw)
    raw.seek(0)
    loaded = torch.load(raw, weights_only=True, map_location='cpu')
    assert value['envelope_sha256'] == loaded['envelope_sha256']
    assert value['host_sha256'] == _sha(original)
    assert value['learner_sha256'] == _sha(memory())

    local = dict(host=host(1))
    recorder = Recorder(memory(offset=1.), bank_count=1, last=1)
    events = []

    def restore_host(target, saved, *, completed_steps):
        events.append('host')
        assert completed_steps == 3
        target['host'] = deepcopy(saved)

    ordinary_load = recorder.load_learner_state_dict
    def load_learner(state):
        events.append('learner')
        assert _sha(local['host']) == value['host_sha256']
        ordinary_load(state)
    recorder.load_learner_state_dict = load_learner
    monkeypatch.setattr(envelope, 'restore_snapshot', restore_host)
    monkeypatch.setattr(envelope, 'snapshot', lambda target: deepcopy(target['host']))
    envelope.restore(local, recorder, loaded, method=METHOD,
                     source_sha256=SOURCE, completed_step=3)
    assert events == ['host', 'learner']
    assert _sha(recorder.learner_state_dict()) == value['learner_sha256']
    assert recorder.learner_bank_count == recorder.learner_last_observed_step == 3
    assert envelope.pack(local['host'], recorder, method=METHOD,
                         source_sha256=SOURCE,
                         completed_step=3)['envelope_sha256'] == value['envelope_sha256']


def test_missing_changed_or_reordered_memory_is_rejected():
    value = envelope.pack(host(), Recorder(memory()), method=METHOD,
                          source_sha256=SOURCE, completed_step=3)
    missing = deepcopy(value)
    del missing['learner']
    with pytest.raises(ValueError, match='memory missing'):
        envelope.validate(missing, method=METHOD, completed_step=3)
    changed = deepcopy(value)
    changed['learner']['confirmed_sums'][0][0] -= 1.
    with pytest.raises(ValueError, match='state hash changed'):
        envelope.validate(changed, method=METHOD, completed_step=3)
    reordered = deepcopy(value)
    for name in ('confirmed_sums', 'confirmed_counts', 'confirmed_squared_norm_sums'):
        reordered['learner'][name].reverse()
    with pytest.raises(ValueError, match='state hash changed'):
        envelope.validate(reordered, method=METHOD, completed_step=3)
    wrong_count = deepcopy(value)
    wrong_count['bank_count'] = 2
    with pytest.raises(ValueError, match='envelope hash changed'):
        envelope.validate(wrong_count, method=METHOD, completed_step=3)
    with pytest.raises(ValueError, match='learner method differs'):
        envelope.validate(value, method='other_candidate', completed_step=3)
    with pytest.raises(ValueError, match='learner source differs'):
        envelope.validate(value, method=METHOD, source_sha256='b' * 64,
                          completed_step=3)
    with pytest.raises(ValueError, match='changed required statistics'):
        envelope.pack(host(), Recorder(dict(confirmed_sums=memory()['confirmed_sums'])),
                      method=METHOD, source_sha256=SOURCE, completed_step=3)
    missing_pending = deepcopy(value)
    missing_pending['learner']['pending'] = {'bank_id': 3}
    with pytest.raises(ValueError, match='pending bank state is missing'):
        envelope.validate(missing_pending, method=METHOD, completed_step=3)


def test_resume_hook_loads_memory_before_next_clock_or_bank(monkeypatch):
    initial = host()
    value = envelope.pack(initial, Recorder(memory()), method=METHOD,
                          source_sha256=SOURCE, completed_step=3)
    local = dict(host=host(1), noise_policy=SimpleNamespace(_step_calls=1))
    recorder = Recorder(memory(offset=1.), bank_count=1, last=1)
    events = []

    class Resume:
        completed_steps = 3
        saved = initial
        restored = False

        def before_step(self, step, current):
            events.append('host')
            current['host'] = deepcopy(self.saved)
            current['noise_policy']._step_calls = step
            self.restored = True

    ordinary = recorder.load_learner_state_dict
    def load(state):
        events.append('learner')
        assert local['noise_policy']._step_calls == 3
        ordinary(state)
    recorder.load_learner_state_dict = load
    monkeypatch.setattr(envelope, 'snapshot', lambda current: deepcopy(current['host']))
    resume = Resume()
    with envelope.attach_before_next_bank(resume, recorder, value,
                                          method=METHOD, source_sha256=SOURCE):
        resume.before_step(3, local)
        assert events == ['host', 'learner']
        assert recorder.learner_bank_count == 3
        assert _sha(recorder.learner_state_dict()) == value['learner_sha256']
    assert resume.before_step.__func__ is Resume.before_step


def test_pending_bank_and_its_raw_hash_survive_exact_restore(monkeypatch):
    pending = dict(schema=envelope.LEARNER_SCHEMA, last_bank_id=1,
        last_bank_sha256='c'*64,
        pending=dict(bank_id=1, bank_sha256='c'*64,
            sums=[torch.tensor([3., 0.], dtype=torch.float64)],
            counts=[1],
            squared_norm_sums=[torch.tensor(9., dtype=torch.float64)]),
        confirmed=False, fixed_half_separation=None,
        accepted_samples_total=0, rejected_samples_total=0,
        confirmation=None, confirmed_sums=[], confirmed_counts=[],
        confirmed_squared_norm_sums=[], reference_centers=[])
    value = envelope.pack(host(1), Recorder(pending, 1, 1), method=METHOD,
                          source_sha256=SOURCE, completed_step=1)
    local = dict(host=host(1))
    other = Recorder(memory(), 3, 3)
    monkeypatch.setattr(envelope, 'snapshot', lambda current: deepcopy(current['host']))
    monkeypatch.setattr(envelope, 'restore_snapshot',
                        lambda current, saved, *, completed_steps:
                            current.update(host=deepcopy(saved)))
    envelope.restore(local, other, value, method=METHOD,
                     source_sha256=SOURCE, completed_step=1)
    assert other.learner_state_dict()['confirmed'] is False
    assert other.learner_state_dict()['pending']['bank_sha256'] == 'c'*64
    assert _sha(other.learner_state_dict()) == value['learner_sha256']
    altered = deepcopy(value)
    altered['learner']['pending']['bank_sha256'] = 'd'*64
    altered['learner']['last_bank_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='state hash changed'):
        envelope.validate(altered, method=METHOD, completed_step=1)


def test_capture_is_passive_and_unknown_state_is_not_dropped(monkeypatch):
    local = dict(host=host())
    recorder = Recorder(memory())
    original_host = _sha(local['host'])
    original_memory = _sha(recorder.learner_state_dict())
    monkeypatch.setattr(envelope, 'snapshot', lambda current: deepcopy(current['host']))
    value = envelope.capture(local, recorder, method=METHOD,
                             source_sha256=SOURCE, completed_step=3)
    assert value['host_sha256'] == original_host
    assert value['learner_sha256'] == original_memory
    assert _sha(local['host']) == original_host
    assert _sha(recorder.learner_state_dict()) == original_memory
    expanded = recorder.learner_state_dict()
    expanded['new_covariance_sums'] = [torch.eye(2, dtype=torch.float64)]
    with pytest.raises(ValueError, match='changed required statistics'):
        envelope.pack(local['host'], Recorder(expanded), method=METHOD,
                      source_sha256=SOURCE, completed_step=3)


def test_exact_split_harness_rejects_memory_only_divergence():
    first = envelope.pack(host(), Recorder(memory()), method=METHOD,
                          source_sha256=SOURCE, completed_step=3)
    second = deepcopy(first)
    envelope.assert_exact_split(first, second, method=METHOD,
                                source_sha256=SOURCE, completed_step=3)
    second = envelope.pack(host(), Recorder(memory(offset=1.)),
                           method=METHOD, source_sha256=SOURCE, completed_step=3)
    with pytest.raises(AssertionError, match='learner states differ'):
        envelope.assert_exact_split(first, second, method=METHOD,
                                    source_sha256=SOURCE, completed_step=3)


def test_source_bound_three_update_harness_rejects_fresh_memory():
    initial_host_sha = '0' * 64
    initial_memory_sha = '1' * 64

    def branch(end, first, *, changed_memory=False):
        observed = list(range(first, end+1))
        sums = [torch.tensor([float(end), 0.], dtype=torch.float64),
                torch.tensor([0., float(end)], dtype=torch.float64)]
        counts = [end, end]
        squared = [torch.tensor(float(end), dtype=torch.float64),
                   torch.tensor(float(end), dtype=torch.float64)]
        pending = (dict(bank_id=end, bank_sha256='b'*64, sums=sums,
                        counts=counts, squared_norm_sums=squared) if end == 1 else None)
        state = dict(schema=envelope.LEARNER_SCHEMA, last_bank_id=end,
            last_bank_sha256='b'*64, pending=pending, confirmed=end>1,
            fixed_half_separation=None if pending else math.sqrt(2.)/2.,
            accepted_samples_total=0 if pending else 2*end,
            rejected_samples_total=0,
            confirmation=None if pending else dict(first_bank_id=1,
                second_bank_id=2, first_bank_sha256='a'*64,
                second_bank_sha256='b'*64, reason='CONFIRMED',
                first_groups=2, second_groups=2, max_pair_distance=.1,
                min_separation=math.sqrt(2.),
                margin=math.sqrt(2.)-.2, pairing=[0, 1]),
            confirmed_sums=[] if pending else sums,
            confirmed_counts=[] if pending else counts,
            confirmed_squared_norm_sums=[] if pending else squared,
            reference_centers=[] if pending else
                [torch.tensor([1., 0.], dtype=torch.float64),
                 torch.tensor([0., 1.], dtype=torch.float64)])
        if changed_memory:
            state['confirmed_sums'][0][1] = .5
            state['confirmed_squared_norm_sums'][0] += 1.
        checkpoint = envelope.pack(host(end), Recorder(state, end, end),
                                   method=METHOD, source_sha256=SOURCE,
                                   completed_step=end)
        return dict(envelope=checkpoint, source_sha256=SOURCE, steps=observed,
            observations=[dict(step=step, modes=8, hq=1.) for step in observed],
            initial_host_sha256=initial_host_sha,
            initial_learner_sha256=initial_memory_sha)

    def full(end):
        return branch(end, 1)

    def prefix(end):
        return branch(end, 1)

    def resumed(checkpoint, end):
        result = branch(end, 2)
        result.update(restored_host_sha256=checkpoint['host_sha256'],
            loaded_envelope_sha256=checkpoint['envelope_sha256'],
            learner_loaded_before_next_bank=True)
        return result

    report = run_short_split(full, prefix, resumed, method=METHOD,
        source_sha256=SOURCE, initial_host_sha256=initial_host_sha)
    assert report['status'] == 'EXACT_SHORT_SPLIT_PASS'

    def fresh_memory(checkpoint, end):
        result = resumed(checkpoint, end)
        result['envelope'] = branch(end, 2, changed_memory=True)['envelope']
        return result

    with pytest.raises(AssertionError, match='learner states differ'):
        run_short_split(full, prefix, fresh_memory, method=METHOD,
            source_sha256=SOURCE, initial_host_sha256=initial_host_sha)
