"""Instance-local, CPU-only passive observation of the original direct owner.

Importing this file constructs no torch objects. Original calls are made once;
their return objects and exceptions are passed through. Observer faults only
make the diagnostic incomplete. No state getter is added. Existing checkpoint calls suspend only verified
observer method bookkeeping; no served-state getter is added.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType

SCHEMA = 'forge_atlas889_two_pole_passive_trace_v1'
MAX_STEPS = 80
MAX_TRACE_BYTES = 4 * 1024 * 1024
MAX_SCOPE_BYTES = 8 * 1024 * 1024
MAX_SCOPE_ITEMS = 50000
_ABSENT = object()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class PassiveTwoPoleRecorder:
    """The numerical owner is supplied; this object never constructs one."""

    def __init__(self, owner):
        self.owner, self.torch = owner, owner.torch
        self.records, self.unknown, self.scope_unknown = [], [], []
        self.hooks, self.current = [], None
        self.serialized_hooks = []
        self.bytes, self.capture_calls, self.purity_checks = 0, 0, 0
        self.purity_failures, self.disabled = 0, False
        self.calls = {name: 0 for name in ('begin_step', 'after_generator_backward',
            'optimizer_step', 'direct_begin', 'direct_end', 'after_generator_step',
            'table_tester_observe', 'table_group_q', 'surprise_decide', 'finish_step')}

    def _unknown(self, reason):
        text = str(reason)[:240]
        if text not in self.unknown and len(self.unknown) < 32:
            self.unknown.append(text)

    def _tensor(self, value):
        if value is None:
            return None
        if not isinstance(value, self.torch.Tensor) or str(value.device) != 'cpu':
            raise ValueError('passive tensor must be an existing CPU tensor')
        if value.numel() > 4096:
            raise ValueError('passive tensor payload exceeds its bounded direct-host scope')
        copied = value.detach().clone()
        if copied.is_floating_point() and not bool(self.torch.isfinite(copied).all()):
            raise ValueError('nonfinite passive tensor payload')
        return dict(dtype=str(value.dtype), shape=list(value.shape), values=copied.tolist())

    def _value(self, value, depth=0):
        if depth > 12:
            raise ValueError('passive payload nesting exceeded')
        if value is None or type(value) in (bool, int, str):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError('nonfinite passive scalar')
            return value
        if isinstance(value, self.torch.Tensor):
            return self._tensor(value)
        if isinstance(value, (list, tuple)):
            if len(value) > 128:
                raise ValueError('passive payload list exceeded')
            return [self._value(v, depth + 1) for v in value]
        if isinstance(value, dict):
            if len(value) > 128 or any(type(k) not in (str, int) for k in value):
                raise ValueError('passive payload dictionary exceeded')
            return {str(k): self._value(v, depth + 1) for k, v in value.items()}
        raise ValueError('unknown passive payload type: ' + type(value).__name__)

    def _scope(self):
        """Read explicit live roots, storage/version/grad and existing RNG states.

        This is a declared-root check, not proof of every ambient Python leaf.
        It reads __dict__ directly and never calls checkpoint/serving getters.
        """
        owner, torch = self.owner, self.torch
        roots = dict(table=owner.table, generator=owner.generator, critic=owner.critic,
            optimizer_g=owner.opt_g, optimizer_d=owner.opt_d, policy=owner.policy,
            named_streams=owner.streams, calls=owner.calls,
            real=getattr(owner, 'real', None), row_indices=owner.row_indices)
        digest, seen, unknown = hashlib.sha256(), {}, []
        size, items = 0, 0

        def emit(path, value):
            nonlocal size, items
            raw = _canonical([path, value]).encode()
            size += len(raw)
            items += 1
            if size > MAX_SCOPE_BYTES or items > MAX_SCOPE_ITEMS:
                raise ValueError('live-scope observation overflow')
            digest.update(raw)

        def walk(value, path, depth=0):
            if depth > 40:
                raise ValueError('live-scope nesting overflow')
            if value is None or type(value) in (bool, int, str):
                emit(path, value)
                return
            if type(value) is float:
                emit(path, value.hex())
                return
            if isinstance(value, (FunctionType, MethodType, BuiltinFunctionType, BuiltinMethodType)):
                # Installation bookkeeping and existing callbacks are not invoked.
                emit(path, ['callable', type(value).__name__])
                return
            oid = id(value)
            if oid in seen:
                emit(path, ['alias', seen[oid], oid])
                return
            seen[oid] = path
            if isinstance(value, torch.Tensor):
                if str(value.device) != 'cpu':
                    raise ValueError('foreign device in direct live scope')
                raw = value.detach().contiguous().numpy().tobytes()
                emit(path, ['tensor', oid, value.untyped_storage().data_ptr(),
                    value._version, str(value.dtype), list(value.shape), list(value.stride()),
                    value.storage_offset(), bool(value.requires_grad), hashlib.sha256(raw).hexdigest()])
                if (value.is_leaf or value.retains_grad) and value.grad is not None:
                    walk(value.grad, path + '.grad', depth + 1)
                return
            if isinstance(value, (torch.dtype, torch.device)):
                emit(path, ['immutable_torch_descriptor', str(value)])
                return
            if isinstance(value, torch.Generator):
                if str(value.device) != 'cpu':
                    raise ValueError('foreign generator in direct live scope')
                emit(path, ['generator', oid, hashlib.sha256(value.get_state().numpy().tobytes()).hexdigest()])
                return
            numpy = sys.modules.get('numpy')
            if numpy is not None and isinstance(value, numpy.ndarray):
                emit(path, ['numpy', oid, str(value.dtype), list(value.shape),
                    hashlib.sha256(value.tobytes()).hexdigest()])
                return
            if isinstance(value, dict):
                emit(path, ['dict', oid, len(value)])
                def key(k):
                    return str(k) if type(k) in (str, int, bool) else 'object:' + str(id(k))
                for k in sorted(value, key=key):
                    walk(value[k], path + '/' + key(k), depth + 1)
                return
            if isinstance(value, (list, tuple)):
                emit(path, [type(value).__name__, oid, len(value)])
                for index, item in enumerate(value):
                    walk(item, path + '/' + str(index), depth + 1)
                return
            if isinstance(value, (set, frozenset)):
                emit(path, ['set', oid, sorted(str(x) for x in value)])
                return
            if hasattr(value, '__dict__'):
                emit(path, ['object', oid, type(value).__module__, type(value).__qualname__])
                walk(vars(value), path + '.__dict__', depth + 1)
                return
            kind = type(value).__module__ + '.' + type(value).__qualname__
            unknown.append(path + ':' + kind)
            emit(path, ['unmapped', oid, kind])

        for name, value in roots.items():
            walk(value, name)
        emit('global.cpu_rng', hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest())
        emit('global.python_rng', repr(random.getstate()))
        numpy = sys.modules.get('numpy')
        if numpy is None:
            unknown.append('global.numpy_rng:module_not_loaded')
        else:
            state = numpy.random.get_state()
            emit('global.numpy_rng', [state[0], hashlib.sha256(state[1].tobytes()).hexdigest(),
                int(state[2]), int(state[3]), float(state[4]).hex()])
        return digest.hexdigest(), sorted(set(unknown))

    def _capture(self, name, build):
        """Only this boundary is version-pure; original hooks may legitimately write."""
        if self.disabled or self.current is None:
            return
        self.capture_calls += 1
        try:
            before, unknown_before = self._scope()
            payload = self._value(build())
            after, unknown_after = self._scope()
            self.purity_checks += 1
            if before != after or unknown_before != unknown_after:
                self.purity_failures += 1
                self._unknown('observer changed declared live scope: ' + name)
            self.scope_unknown = sorted(set(self.scope_unknown + unknown_before + unknown_after))[:64]
            event = dict(name=name, payload=payload, before_sha256=before,
                after_sha256=after, pure=before == after and unknown_before == unknown_after)
            count = len(_canonical(event).encode())
            if self.bytes + count > MAX_TRACE_BYTES or len(self.current['events']) >= 32:
                self.disabled = True
                self._unknown('trace overflow; no truncation counted as complete')
                return
            self.bytes += count
            self.current['events'].append(event)
        except Exception as error:
            self._unknown('observer capture failed at ' + name + ': ' + type(error).__name__)

    def _group(self):
        matches = [g for g in self.owner.opt_g.param_groups
                   if len(g['params']) == 1 and g['params'][0] is self.owner.table]
        if len(matches) != 1:
            raise ValueError('exact direct table group is required')
        return matches[0]

    def _point(self):
        owner, group = self.owner, self._group()
        state = owner.opt_g.state.get(owner.table, {})
        policy = owner.policy
        tester = policy.lr_settle.testers[0][owner.opt_g.param_groups.index(group)]
        def scalars(obj):
            if obj is None:
                return None
            return {k: v for k, v in vars(obj).items()
                    if v is None or type(v) in (bool, int, float, str)}
        return dict(table=self._tensor(owner.table), training_gradient=self._tensor(owner.table.grad),
            optimizer_step=self._value(state.get('step')), exp_avg=self._tensor(state.get('exp_avg')),
            exp_avg_sq=self._tensor(state.get('exp_avg_sq')),
            max_exp_avg_sq=self._tensor(state.get('max_exp_avg_sq')),
            group=dict(lr=group['lr'], betas=list(group['betas']), eps=group['eps'],
                amsgrad=group.get('amsgrad', False), base_lr=policy.initial_lrs[0][owner.opt_g.param_groups.index(group)]),
            direct_response=dict(gain=owner.opt_g.direct_response.last_gain,
                started=owner.opt_g.direct_response.started,
                history=self._tensor(owner.opt_g.direct_history)),
            table_tester=scalars(tester), controller=scalars(policy.controller),
            surprise=scalars(policy.surprise), policy_phase=policy._phase,
            completed_steps=policy.completed_steps, hot_rows=self._tensor(policy._hot),
            averaged_table=self._tensor(policy.averaged_table),
            log_output_sigma=self._tensor(policy.log_output_sigma),
            fixed_data_cursor='NO_CURSOR_FIXED_ALL_BANK12')

    def _count(self, name):
        self.calls[name] += 1

    def _hook(self, obj, name, make, *, serialized=False):
        original = getattr(obj, name)
        previous = vars(obj).get(name, _ABSENT)
        wrapped = make(original)
        setattr(obj, name, wrapped)
        self.hooks.append((obj, name, previous))
        if serialized:
            self.serialized_hooks.append((vars(obj), name, previous, wrapped))

    def _checkpoint_state(self, original, *args, **kwargs):
        """Pass one existing getter through its exact original instance state.

        Only the three installed hooks on vars-based controller serializers
        are suspended. Arbitrary callable state remains visible to the owner's
        unchanged strict typed digest. This boundary performs no state getter
        or serving call beyond the single original caller's getter.
        """
        entries = tuple(self.serialized_hooks)
        if len(entries) != 3 or any(
                attributes.get(name, _ABSENT) is not wrapped
                for attributes, name, previous, wrapped in entries):
            self._unknown('checkpoint hook identity mismatch; original state retained')
            return original(*args, **kwargs)
        # Prevalidation precedes every mutation. These are the exact plain
        # instance dictionaries of SettleTest and OptimizerSurprise.
        suspended = []
        try:
            for attributes, name, previous, wrapped in entries:
                if previous is _ABSENT:
                    del attributes[name]
                else:
                    attributes[name] = previous
                suspended.append((attributes, name, previous, wrapped))
            return original(*args, **kwargs)
        finally:
            for attributes, name, previous, wrapped in reversed(suspended):
                if attributes.get(name, _ABSENT) is previous:
                    attributes[name] = wrapped
                else:
                    # Never silently overwrite an unexpected original write.
                    self._unknown('checkpoint hook changed during original getter: ' + name)

    def install(self):
        """Only this owner instance is wrapped; no class/module patches."""
        owner, policy = self.owner, self.owner.policy
        try:
            if (str(owner.table.device) != 'cpu' or tuple(owner.table.shape) != (12, 1)
                    or owner.prior is not None or owner.policy.completed_steps != 0):
                raise ValueError('fresh unsampled direct CPU12x1 owner required')

            def begin(original):
                def call(*args, **kwargs):
                    self._count('begin_step')
                    if len(self.records) < MAX_STEPS:
                        self.current = dict(step=policy.completed_steps + 1, events=[])
                        self.records.append(self.current)
                    else:
                        self.disabled = True
                        self._unknown('more than80 begin calls')
                    self._capture('before_begin', self._point)
                    value = original(*args, **kwargs)
                    self._capture('after_begin', self._point)
                    return value
                return call

            def after_backward(original):
                def call(*args, **kwargs):
                    self._count('after_generator_backward')
                    value = original(*args, **kwargs)
                    self._capture('after_original_generator_backward', lambda: dict(point=self._point(),
                        loss_gan=kwargs.get('loss_gan'), loss_critic=kwargs.get('loss_critic')))
                    return value
                return call

            def optimizer(original):
                def call(*args, **kwargs):
                    self._count('optimizer_step')
                    self._capture('before_optimizer_step', self._point)
                    try:
                        value = original(*args, **kwargs)
                    except BaseException as error:
                        self._capture('optimizer_raised', lambda: dict(exception_type=type(error).__name__, point=self._point()))
                        raise
                    self._capture('optimizer_returned', lambda: dict(return_type=type(value).__name__, point=self._point()))
                    return value
                return call

            def direct_begin(original):
                def call(*args, **kwargs):
                    self._count('direct_begin')
                    value = original(*args, **kwargs)
                    self._capture('actual_adam_inputs', lambda: dict(point=self._point(),
                        restore_token=None if value is None else dict(lr=value[1], betas=list(value[2]))))
                    return value
                return call

            def direct_end(original):
                def call(*args, **kwargs):
                    self._count('direct_end')
                    # Finally-entry is not proof that the attempted Adam call returned.
                    self._capture('after_adam_attempt_before_restore', self._point)
                    value = original(*args, **kwargs)
                    self._capture('after_direct_group_restore', self._point)
                    return value
                return call

            def after_step(original):
                def call(*args, **kwargs):
                    self._count('after_generator_step')
                    self._capture('before_hot_and_settle', self._point)
                    value = original(*args, **kwargs)
                    self._capture('after_hot_and_settle', self._point)
                    return value
                return call

            def table_observe(original):
                def call(*args, **kwargs):
                    self._count('table_tester_observe')
                    # ratio is the original caller's restored-group LR/base LR.
                    self._capture('table_tester_actual_input', lambda: dict(ratio=args[1] if len(args) > 1 else kwargs['ratio'],
                        step=kwargs.get('step'), point=self._point()))
                    value = original(*args, **kwargs)
                    self._capture('table_tester_actual_return', lambda: dict(returned=value, point=self._point()))
                    return value
                return call

            def group_q(original):
                def call(*args, **kwargs):
                    value = original(*args, **kwargs)
                    try:
                        optimizer = args[0] if args else kwargs['optimizer']
                        group = args[1] if len(args) > 1 else kwargs['group']
                        if optimizer is owner.opt_g and group is self._group():
                            self._count('table_group_q')
                            self._capture('table_group_q_actual_return', lambda: dict(q=value,
                                consumer_group=dict(lr=group['lr'], betas=list(group['betas']), eps=group['eps']),
                                point=self._point()))
                    except Exception as error:
                        self._unknown('observer group_q postprocessing failed: ' + type(error).__name__)
                    return value
                return call

            def decide(original):
                def call(*args, **kwargs):
                    self._count('surprise_decide')
                    value = original(*args, **kwargs)
                    self._capture('surprise_actual_decision', lambda: dict(returned=value, point=self._point()))
                    return value
                return call

            def finish(original):
                def call(*args, **kwargs):
                    self._count('finish_step')
                    self._capture('before_average_and_birth_death', self._point)
                    value = original(*args, **kwargs)
                    self._capture('after_average_and_birth_death', lambda: dict(event=value,
                        point=self._point(), moved_rows=None if policy.birth_death is None
                        else self._tensor(policy.birth_death.moved_rows)))
                    return value
                return call

            self._hook(policy, 'begin_step', begin)
            self._hook(policy, 'after_generator_backward', after_backward)
            self._hook(owner.opt_g, 'step', optimizer)
            self._hook(owner.opt_g.direct_response, 'begin', direct_begin)
            self._hook(owner.opt_g.direct_response, 'end', direct_end)
            self._hook(policy, 'after_generator_step', after_step)
            group = self._group()
            tester = policy.lr_settle.testers[0][owner.opt_g.param_groups.index(group)]
            self._hook(tester, 'observe', table_observe, serialized=True)
            self._hook(policy.surprise, 'group_q', group_q, serialized=True)
            self._hook(policy.surprise, 'decide', decide, serialized=True)
            self._hook(policy, 'finish_step', finish)
            self._hook(policy, 'state_dict', lambda original:
                lambda *args, **kwargs: self._checkpoint_state(original, *args, **kwargs))
        except Exception as error:
            self._unknown('observer install failed: ' + type(error).__name__)
            self.uninstall()
        return self

    def uninstall(self):
        for obj, name, previous in reversed(self.hooks):
            try:
                if previous is _ABSENT:
                    delattr(obj, name)
                else:
                    setattr(obj, name, previous)
            except Exception as error:
                self._unknown('observer bookkeeping removal failed: ' + type(error).__name__)
        self.hooks.clear()
        self.serialized_hooks.clear()

    def receipt(self, *, completed_steps, scored_clocks):
        expected = {name: 80 for name in self.calls}
        # Surprise decides once in every begin_step under the fixed Atlas recipe.
        coverage = (self.calls == expected and [row['step'] for row in self.records] == list(range(1, 81))
            and completed_steps == 80 and len(scored_clocks) == 24
            and scored_clocks == [math.ceil(i * 80 / 24) for i in range(1, 25)])
        complete = (coverage and not self.unknown and not self.scope_unknown
            and not self.disabled and self.purity_failures == 0)
        return dict(schema=SCHEMA, status='COMPLETE_DECLARED_SCOPE' if complete else 'UNKNOWN',
            complete=complete, completed_steps=completed_steps, scored_clocks=list(scored_clocks),
            call_counts=dict(self.calls), records=self.records, typed_json_bytes=self.bytes,
            capture_calls=self.capture_calls, purity_checks=self.purity_checks,
            purity_failures=self.purity_failures, unknown=list(self.unknown),
            ambient_leaf_scope='DECLARED_OWNER_ROOTS_ONLY; EXTERNAL_CALLBACK_CLOSURES_UNKNOWN',
            unmapped_declared_leaves=list(self.scope_unknown),
            training_gradient='12x1 direct table gradient consumed by original Adam',
            metric_gradient='separate existing24-bank combined real12+particle12 critic input derivatives',
            consumer_q='original returned tensor only; exp_avg_sq/current restored beta; not AMSGrad denominator',
            observer_state_getter_calls=0, observer_extra_algorithm_calls=0,
            fixed_data_cursor='NO_CURSOR_FIXED_ALL_BANK12', private884_patch_installed=False)
