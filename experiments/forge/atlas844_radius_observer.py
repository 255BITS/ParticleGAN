"""Passive, instance-local Atlas844 Gaussian radius observation.

No sampler/model/update is called by this module. Original bound methods are
called once and their return objects are returned unchanged. Checkpoint getters
are deliberately absent. Raw witnesses and snapshots belong to the owned
attempt's private artifact directory, never to a public result.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import hashlib
import io
import json
import os
import struct
import types

import torch


CANDIDATE_ID = "atlas-existing-mog-radius-observer844-v1"
VIEW_ID = "atlas_existing_mog_radius_observer844_v1"
TRACK_ID = "atlas844_existing_mog_radius_observer"
STUDY_ID = "atlas-existing-mog-radius-observer844-study-v1"
TASK_ID = "gaussian1d_acquisition"
SCHEMA = "forge_atlas844_passive_radius_evidence_v1"
PROOF_SCHEMA = "forge_atlas844_radius_observer_software_proof_v1"
WITNESS_BYTES = 65536
SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_ROWS, MAX_WIDTH, MAX_NODES = 256, 4, 24000
CLOCKS = tuple((i * 1000 + 23) // 24 for i in range(1, 25))
NEIGHBORHOOD = frozenset((833, 834, 835))
EXTRA_OPERATIONS = ("extra_prior_samples", "extra_rng_draws", "extra_model_forwards",
    "extra_backward_calls", "extra_optimizer_steps", "extra_decision_evaluations",
    "state_getter_calls", "state_mutations", "foreign_device_initializations")
SOFTWARE_CHECKS = ("duplicate_positive_radius", "distinct_support", "all_zero_support",
    "radius_without_effect", "clipped_and_rounded_effects", "fake_pool_coverage",
    "primary_and_isolation_coverage", "same_pair_commit", "on_off_original_calls_and_rng",
    "populated_fast_purity", "mutation_is_detected", "forbidden_getters_and_models",
    "witness_overflow", "missing_coverage_is_unknown", "scheduled_snapshot_bounds")
_HOOKS = ("maybe_apply", "_nearest_other", "_jitter", "_move", "_isolation_pick")


class CaptureLimit(Exception):
    """Diagnostic-only bounded capture refusal; never a production decision."""


class ObserverPurityError(RuntimeError):
    """Live state changed during observation; state is never repaired/restored."""


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tensor_bytes(value):
    # Every subsequent view/mask/operation belongs to disjoint scratch storage.
    cpu = value.detach().to(device="cpu").clone().contiguous()
    return cpu, cpu.reshape(-1).view(torch.uint8).numpy().tobytes()


def _row_bytes_differ(left, right):
    """IEEE byte differences on disjoint scratch, including signed zero."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise CaptureLimit("shadow typed shape differs")
    a = left.detach().clone().contiguous().reshape(len(left), -1).view(torch.uint8)
    b = right.detach().clone().contiguous().reshape(len(right), -1).view(torch.uint8)
    return (a != b).any(1)


def _same_bytes(left, right):
    return (left.shape == right.shape and left.dtype == right.dtype
            and _tensor_bytes(left)[1] == _tensor_bytes(right)[1])


class LiveReader:
    """Read explicit live roots directly, without a module/optimizer getter.

    Unknown leaves remain UNKNOWN; the reader never invokes them. Identity and
    version fields prove local observer purity, not cross-run initial identity.
    Byte-only partitions permit a later retained comparison without conflating
    learned values with addresses, counters or global RNG.
    """
    def __init__(self, context, trainer):
        self.context, self.trainer = context, trainer
        self.external = None
        self.excluded_fields = set()

    def capture(self, *, retain=False):
        trainer, context = self.trainer, self.context
        self.seen, self.records, self.payload, self.unknown = {}, {}, {}, []
        self.used, self.nodes = 0, 0
        roots = (("generator", trainer.G), ("critic", trainer.D), ("prior", trainer.prior),
            ("optimizer_g", trainer.opt_g), ("optimizer_d", trainer.opt_d),
            ("policy", trainer.policy), ("trainer", trainer), ("named_streams", context.streams))
        self.parameter_names = {}
        def parameters(module, prefix, visited):
            if id(module) in visited:
                return
            visited.add(id(module))
            fields = vars(module)
            for name, value in fields.get("_parameters", {}).items():
                if value is not None:
                    self.parameter_names.setdefault(id(value), prefix + "." + name)
            for name, child in fields.get("_modules", {}).items():
                if child is not None:
                    parameters(child, prefix + "." + name, visited)
        for name, module in roots[:3]:
            parameters(module, name, set())
        for name, value in roots:
            self._walk(value, name, retain)
        self._rng(torch.get_rng_state(), "global_rng.cpu", retain)
        if trainer.device.type == "cuda":
            if not torch.cuda.is_initialized():
                self.unknown.append("global_rng.cuda: owned device not initialized")
            else:
                self._rng(torch.cuda.get_rng_state(trainer.device), "global_rng.cuda", retain)
        ambient = dict(grad_enabled=torch.is_grad_enabled(),
            inference_enabled=torch.is_inference_mode_enabled(), default_dtype=str(torch.get_default_dtype()),
            deterministic=torch.are_deterministic_algorithms_enabled(),
            matmul_precision=torch.get_float32_matmul_precision(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark)
        self.records["ambient"] = ambient
        partitions = {}
        for group in ("learned_values", "gradients", "optimizer", "control", "rng", "identity"):
            subset = {}
            for path, record in self.records.items():
                if group == "identity":
                    subset[path] = record
                elif group == "rng" and record.get("kind") == "rng":
                    subset[path] = {k: v for k, v in record.items() if k != "identity"}
                elif group == "gradients" and ".grad" in path:
                    subset[path] = {k: v for k, v in record.items() if k not in ("identity", "storage", "version")}
                elif (group == "learned_values" and record.get("kind") == "tensor"
                      and ".grad" not in path and path.startswith(("generator", "critic", "prior"))):
                    subset[path] = {k: v for k, v in record.items() if k not in ("identity", "storage", "version")}
                elif group == "optimizer" and path.startswith("optimizer_"):
                    subset[path] = {k: v for k, v in record.items() if k not in ("identity", "storage", "version")}
                elif group == "control" and path.startswith(("policy", "trainer")):
                    subset[path] = {k: v for k, v in record.items() if k not in ("identity", "storage", "version")}
            partitions[group] = _hash(subset)
        return dict(records=self.records, partitions=partitions,
            fingerprint=_hash(self.records), complete=not self.unknown,
            unknown=list(self.unknown), typed_bytes=self.used,
            tensors=self.payload if retain else {})

    def _rng(self, state, path, retain, identity=None):
        cpu, raw = _tensor_bytes(state)
        self.used += len(raw)
        if self.used > SNAPSHOT_BYTES:
            raise CaptureLimit("live typed-byte limit")
        self.records[path] = dict(kind="rng", identity=identity,
            dtype=str(cpu.dtype), shape=list(cpu.shape), sha256=hashlib.sha256(raw).hexdigest())
        if retain:
            self.payload[path] = cpu

    def _key(self, key, ordinal):
        if type(key) in (str, int, bool):
            return str(key)
        if isinstance(key, torch.Tensor):
            name = self.parameter_names.get(id(key))
            if name is None:
                self.unknown.append("unmapped tensor dictionary key")
            return "parameter:" + (name or str(ordinal))
        self.unknown.append("unsupported dictionary key: " + type(key).__qualname__)
        return "unknown_key:" + str(ordinal)

    def _walk(self, value, path, retain):
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise CaptureLimit("live leaf-count limit")
        if value is None or type(value) in (str, int, bool):
            self.records[path] = dict(kind=type(value).__name__, value=value)
            return
        if type(value) is float:
            self.records[path] = dict(kind="float", ieee754=struct.pack("!d", value).hex())
            return
        if isinstance(value, (torch.dtype, torch.device)):
            self.records[path] = dict(kind=type(value).__name__, value=str(value))
            return
        if id(value) in self.seen:
            self.records[path] = dict(kind="alias", target=self.seen[id(value)])
            return
        self.seen[id(value)] = path
        if isinstance(value, torch.Tensor):
            if value.device.type == "cuda" and value.device != self.trainer.device:
                self.unknown.append(path + ": foreign device")
                self.records[path] = dict(kind="unknown", identity=id(value))
                return
            size = value.numel() * value.element_size()
            if self.used + size > SNAPSHOT_BYTES:
                raise CaptureLimit("live typed-byte limit")
            cpu, raw = _tensor_bytes(value)
            self.used += len(raw)
            self.records[path] = dict(kind="tensor", identity=id(value),
                storage=value.untyped_storage()._cdata, storage_offset=value.storage_offset(),
                version=value._version, dtype=str(value.dtype), device=str(value.device),
                shape=list(value.shape), stride=list(value.stride()), requires_grad=value.requires_grad,
                sha256=hashlib.sha256(raw).hexdigest())
            if retain:
                self.payload[path] = cpu
            if value.is_leaf:
                self._walk(value.grad, path + ".grad", retain)
            elif value.requires_grad:
                self.unknown.append(path + ": nonleaf gradient scope unproven")
            return
        if isinstance(value, torch.Generator):
            if value.device.type == 'cuda' and value.device != self.trainer.device:
                self.records[path] = dict(kind='unknown', identity=id(value),
                    device=str(value.device))
                self.unknown.append(path + ': foreign generator device; state not read')
                return
            self._rng(value.get_state(), path, retain, id(value))
            return
        if isinstance(value, dict):
            self.records[path] = dict(kind="dict", identity=id(value), length=len(value))
            for i, (key, child) in enumerate(value.items()):
                self._walk(child, path + "/" + self._key(key, i), retain)
            return
        if isinstance(value, (tuple, list)):
            self.records[path] = dict(kind=type(value).__name__, identity=id(value), length=len(value))
            for i, child in enumerate(value):
                self._walk(child, path + "/" + str(i), retain)
            return
        if isinstance(value, (set, frozenset)) and all(type(v) in (str, int, bool) for v in value):
            self.records[path] = dict(kind=type(value).__name__, identity=id(value),
                values=sorted(value, key=lambda v: (type(v).__name__, str(v))))
            return
        if isinstance(value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType, type)):
            code = getattr(getattr(value, "__func__", value), "__code__", None)
            self.records[path] = dict(kind="callable_identity", identity=id(value),
                code=None if code is None else hashlib.sha256(code.co_code).hexdigest())
            # Identity is covered; arbitrary callable globals are not a full state proof.
            self.unknown.append(path + ": callable closure/global scope unproven")
            return
        fields = getattr(value, "__dict__", None)
        if isinstance(fields, dict):
            self.records[path] = dict(kind="object", identity=id(value),
                type=type(value).__module__ + "." + type(value).__qualname__)
            for key, child in fields.items():
                if (self.external is None or child is not self.external) and (id(value), key) not in self.excluded_fields:
                    self._walk(child, path + "." + key, retain)
            return
        self.records[path] = dict(kind="unknown", identity=id(value),
            type=type(value).__module__ + "." + type(value).__qualname__)
        self.unknown.append(path)


def legacy_radius(query, support):
    """Exact legacy shortlist on owned scratch, not a live owner/cache call."""
    from particlegan.birth_death import _knn
    distances, _ = _knn(query, support, min(4, len(support) - 1) + 1)
    distances = distances.masked_fill(distances == 0, float("inf"))
    nearest = distances.min(1).values
    return torch.where(torch.isfinite(nearest), nearest * .5, torch.zeros_like(nearest))


class RadiusObserver:
    def __init__(self, context, trainer, output, *, enabled=True):
        from particlegan.birth_death import ParticleBirthDeath
        self.context, self.trainer, self.reader = context, trainer, LiveReader(context, trainer)
        self.birth, self.policy = trainer.policy.birth_death, trainer.policy
        if (type(self.birth) is not ParticleBirthDeath or self.birth.rows.mog_prior is not trainer.prior
                or self.birth.rows.table is not trainer.prior.z or tuple(trainer.prior.z.shape) != (256, 2)):
            raise ValueError("844 requires the actual original Gaussian raw-MoG reference owner")
        self.originals = {name: getattr(self.birth, name) for name in _HOOKS}
        if any(method.__func__ is not getattr(ParticleBirthDeath, name)
               for name, method in self.originals.items()):
            raise ValueError("844 hooks must start from unmodified bound core methods")
        output = Path(output)
        if output.resolve() != output:
            raise ValueError("observer output alias forbidden")
        self.output = output / "radius-observer844"
        self.output.mkdir(exist_ok=False)
        self.enabled = enabled
        self.counts = {name: Counter() for name in ("gates", "fake_pool", "primary_move", "isolation_move")}
        self.site, self.move, self.radius = "fake_pool", None, None
        self.witness, self.first_positive, self.first_affected, self.unknown = None, None, None, []
        self.snapshots, self.neighborhood = [], []
        self.purity_checks, self.purity_failures = 0, 0
        self._installed = False
        self.installed_hooks = {}
        self.scheduled_update = None
        self.reader.external = self

    def install(self):
        if self._installed:
            raise ValueError("one observer per fresh reaction owner")
        birth = self.birth
        for name, function in (("maybe_apply", self._maybe_apply),
                ("_nearest_other", self._nearest), ("_jitter", self._jitter),
                ("_move", self._move), ("_isolation_pick", self._isolation_pick)):
            # Explicit receiver is ignored: original stored methods remain bound.
            def wrapper(receiver, *args, _function=function, **kwargs):
                return _function(*args, **kwargs)
            setattr(birth, name, types.MethodType(wrapper, birth))
            self.installed_hooks[name] = vars(birth)[name]
            self.reader.excluded_fields.add((id(birth), name))
        birth._radius_observer844 = self
        original_backward = self.policy.after_generator_backward
        def after_generator_backward(*args, **kwargs):
            result = original_backward(*args, **kwargs)
            if self.scheduled_update in NEIGHBORHOOD:
                self.update_boundary("after_generator_backward", self.scheduled_update)
            return result
        self.policy.after_generator_backward = after_generator_backward
        self.installed_backward = after_generator_backward
        self.reader.excluded_fields.add((id(self.policy), "after_generator_backward"))
        self._installed = True
        return self

    @contextmanager
    def guarded(self):
        before = self.reader.capture()
        try:
            yield
        finally:
            after = self.reader.capture()
            self.purity_checks += 1
            if before["fingerprint"] != after["fingerprint"]:
                self.purity_failures += 1
                raise ObserverPurityError("844 observer changed a live value/identity/version/grad/mode/RNG/flag")

    def _observe(self, callback):
        if not self.enabled:
            return None
        try:
            with self.guarded():
                return callback()
        except CaptureLimit as error:
            self._unknown(str(error))
            return None

    def _unknown(self, reason):
        if reason not in self.unknown:
            if len(self.unknown) < 32:
                self.unknown.append(reason)
            elif "unknown_reason_overflow" not in self.unknown:
                self.unknown[-1] = "unknown_reason_overflow"

    def verify_live_owner(self):
        selection = self.policy._feature_selection
        actual = "knn" if selection is None else selection.state["actual_backend"]
        good = (self.policy.birth_death is self.birth
            and (selection is None or selection.reference_birth_death is self.birth)
            and self.birth.rows.mog_prior is self.trainer.prior
            and self.birth.rows.table is self.trainer.prior.z
            and self.policy.table_optimizer is self.trainer.opt_g
            and (not self._installed or all(vars(self.birth).get(name) is wrapper
                for name, wrapper in self.installed_hooks.items()))
            and (not self._installed or vars(self.policy).get('after_generator_backward')
                is self.installed_backward))
        self.counts["gates"]["live_owner_checks"] += 1
        if not good or actual not in ("pending", "knn"):
            self._unknown("actual reaction owner/backend differs; no forced replacement")
        if self.policy.completed_steps > 0 and actual != "knn":
            self._unknown("actual completed owner is not knn")
        return good

    def _maybe_apply(self, *args, **kwargs):
        # Core exceptions are never caught by a diagnostic-only fallback.
        self.verify_live_owner()
        c = self.counts["gates"]
        c["maybe_apply_entries"] += 1
        ready = self.birth.fill == self.birth.N and self.birth.rows_since_eval >= self.birth.N
        c["ready" if ready else "not_ready"] += 1
        self.site = "fake_pool"
        result = self.originals["maybe_apply"](*args, **kwargs)
        if result is not None:
            c["evaluations"] += 1
            c["dimension_skip"] += int(result.get("skip") == "dimension undefined")
            c["dry_run"] += int(self.birth.dry_run)
            c["reported_pairs"] += int(result.get("matched", 0))
            c["reported_move_rows"] += int(result.get("moves", 0))
        return result

    def _isolation_pick(self, *args, **kwargs):
        self.counts["gates"]["isolation_pick_calls"] += 1
        self.site = "isolation_move"
        result = self.originals["_isolation_pick"](*args, **kwargs)
        self.counts["gates"]["isolation_returned_rows"] += len(result[0])
        self.counts["gates"]["isolation_empty"] += int(len(result[0]) == 0)
        return result

    def _nearest(self, *args, **kwargs):
        result = self.originals["_nearest_other"](*args, **kwargs)
        # Save the exact result object; no tensor operation or decision here.
        self.radius = result
        self.counts[self.site]["radius_calls"] += 1
        return result

    def _jitter(self, latent, noise):
        self.radius = None
        result = self.originals["_jitter"](latent, noise)
        if self.enabled:
            self._observe(lambda: self._analyse(latent, noise, result, self.radius))
        return result

    def _move(self, child, parent):
        self.site = "isolation_move" if self.site == "isolation_move" else "primary_move"
        self.counts[self.site]["move_calls"] += 1
        self.counts[self.site]["move_query_rows"] += len(child)
        self.move = None
        def before():
            self.move = dict(child=child.detach().clone(), parent=parent.detach().clone(),
                ema=self.birth.rows.averaged_table.detach()[parent].clone(), candidate=None)
        self._observe(before)
        result = self.originals["_move"](child, parent)
        if self.move is not None and self.move.get("candidate") is not None:
            def after():
                actual = self.birth.rows.table.detach()[child]
                ema = self.birth.rows.averaged_table.detach()[child]
                table_ok = _same_bytes(actual, self.move["candidate"])
                ema_ok = _same_bytes(ema, self.move["ema_candidate"])
                self.counts[self.site]["commit_joins"] += len(child)
                if not table_ok or not ema_ok:
                    self._unknown("actual corrected table/EMA commit differs from same-pair candidate")
                if self.witness is not None and self.witness["metadata"]["event"] == self.move["event"]:
                    self.witness["metadata"]["actual_commit_matches"] = bool(table_ok and ema_ok)
            self._observe(after)
        self.move = None
        return result

    def _analyse(self, latent, noise, returned, radius):
        self.verify_live_owner()
        counts = self.counts[self.site]
        counts["jitter_calls"] += 1
        counts["query_rows"] += len(latent)
        controller = self.birth.rows.controller
        if controller is None or controller.variant not in ("dv10", "dv11", "dv12"):
            counts["controller_zero_exit"] += 1
            return
        if controller.variant != "dv12":
            counts["non_dv12"] += 1
            self._unknown("original controller does not execute dv12 radius")
            return
        counts["dv12_calls"] += 1
        if radius is None:
            self._unknown("missing actual production radius result")
            return
        support = self.birth.rows.table.detach().clone(memory_format=torch.preserve_format)
        query = latent.detach().clone(memory_format=torch.preserve_format)
        original_noise = noise.detach().clone(memory_format=torch.preserve_format)
        if len(support) > MAX_ROWS or len(query) > MAX_ROWS or query.shape[1] > MAX_WIDTH:
            raise CaptureLimit("query/support witness geometry unsupported")
        old = legacy_radius(query, support)
        new = radius.detach().clone()
        displacement = controller.latent_bandwidth * 1. * original_noise
        norm = displacement.norm(dim=1)
        old_cap = (old / norm.clamp_min(1e-20)).clamp_max(1.)
        new_cap = (new / norm.clamp_min(1e-20)).clamp_max(1.)
        old_delta = displacement * old_cap.unsqueeze(1)
        new_delta = returned.detach().clone()
        old_input, new_input = query + old_delta, query + new_delta
        radius_diff = _row_bytes_differ(old, new)
        positive = (old == 0) & (new > 0)
        delta_diff = _row_bytes_differ(old_delta, new_delta)
        input_diff = _row_bytes_differ(old_input, new_input)
        ema_diff = torch.zeros_like(input_diff)
        old_ema = new_ema = None
        if self.move is not None:
            old_ema, new_ema = self.move["ema"] + old_delta, self.move["ema"] + new_delta
            ema_diff = _row_bytes_differ(old_ema, new_ema)
            self.move.update(candidate=new_input, ema_candidate=new_ema)
        for name, mask in (("old_zero_new_positive", positive), ("radius_differences", radius_diff),
                ("old_cap_active", old_cap < 1), ("new_cap_active", new_cap < 1),
                ("cap_differences", _row_bytes_differ(old_cap, new_cap)), ("delta_differences", delta_diff),
                ("rounded_input_differences", input_diff), ("rounded_ema_differences", ema_diff)):
            counts[name] += int(mask.sum())
        event = dict(completed_steps=self.policy.completed_steps,
            evaluation_ordinal=self.birth.counters["evals"],
            site=self.site, call_ordinal=counts["jitter_calls"])
        if self.move is not None:
            self.move["event"] = event
        if self.first_positive is None and bool(positive.any()):
            self.first_positive = dict(event=event,
                query_ordinal=int(positive.nonzero()[0]), old_zero_new_positive=True)
        affected = delta_diff | input_diff | ema_diff
        if self.first_affected is not None or not bool(affected.any()):
            return
        index = int(affected.nonzero()[0])
        self.first_affected = dict(event=event, query_ordinal=index,
            delta_changed=bool(delta_diff[index]), rounded_input_changed=bool(input_diff[index]),
            rounded_ema_changed=bool(ema_diff[index]))
        arrays = dict(support=support, query=query, original_noise=original_noise,
            old_radius=old, new_radius=new, norm=norm, old_cap=old_cap, new_cap=new_cap,
            old_delta=old_delta, new_delta=new_delta, old_input=old_input, new_input=new_input)
        if self.move is not None:
            arrays.update(parent=self.move["parent"], child=self.move["child"],
                ema_parent=self.move["ema"], old_ema=old_ema, new_ema=new_ema)
        owned = {name: value.detach().cpu().clone() for name, value in arrays.items()}
        typed_bytes = sum(value.numel() * value.element_size() for value in owned.values())
        if typed_bytes > WITNESS_BYTES:
            raise CaptureLimit("first affected full-batch witness exceeds typed-byte limit")
        zeros = ((support - query[index]).square().sum(1) == 0).sum()
        self.witness = dict(metadata=dict(event=event, query_ordinal=index,
            source_marker_original_noise_draw=True, full_query_noise_batch=True,
            old_zero_new_positive=bool(positive[index]), exact_zero_support_rows=int(zeros),
            bandwidth=float(controller.latent_bandwidth), trust=1.,
            dtype=str(query.dtype), device=str(query.device), query_stride=list(query.stride()),
            support_stride=list(support.stride()), delta_changed=bool(delta_diff[index]),
            rounded_input_changed=bool(input_diff[index]), rounded_ema_changed=bool(ema_diff[index]),
            actual_commit_matches=None if self.move is None else "PENDING",
            backend_flags=self.reader.capture()["records"]["ambient"],
            downstream_legacy_decision="UNKNOWN", typed_bytes=typed_bytes), tensors=owned)

    def _write(self, filename, value, maximum):
        stream = io.BytesIO()
        torch.save(value, stream)
        raw = stream.getvalue()
        if len(raw) > maximum:
            raise CaptureLimit(filename + ": serialized-byte limit")
        path = self.output / filename
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        return dict(path="radius-observer844/" + filename, bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest())

    def _snapshot(self, label, destination, *, scheduled_update=None):
        def capture():
            snapshot = self.reader.capture(retain=True)
            pin = self._write(label + ".pt", snapshot, SNAPSHOT_BYTES)
            destination.append(dict(label=label, completed_steps=self.policy.completed_steps,
                scheduled_update=scheduled_update,
                phase=self.policy._phase, complete_live_leaf_scope=snapshot["complete"],
                unknown_leaf_count=len(snapshot["unknown"]), partitions=snapshot["partitions"], pin=pin))
        self._observe(capture)

    def scheduled_observation(self, step):
        if step not in CLOCKS or step != self.policy.completed_steps:
            raise ValueError("844 snapshot must use one exact original scored clock")
        self.verify_live_owner()
        self._snapshot("observation-" + str(step), self.snapshots)

    def update_boundary(self, stage, scheduled_update):
        if scheduled_update not in NEIGHBORHOOD:
            return
        if stage not in ("pre_update", "post_update", "after_generator_backward"):
            raise ValueError("unknown observational update boundary")
        self._snapshot(stage + "-" + str(scheduled_update), self.neighborhood,
            scheduled_update=scheduled_update)

    def finish(self):
        self.verify_live_owner()
        witness_pin = None
        if self.witness is not None:
            def write():
                nonlocal witness_pin
                witness_pin = self._write("first-affected.pt", self.witness, WITNESS_BYTES)
            self._observe(write)
        gates, fake = self.counts['gates'], self.counts['fake_pool']
        site_joins = all(values['jitter_calls'] == values['move_calls']
            and values['radius_calls'] == values['jitter_calls']
            and values['dv12_calls'] == values['jitter_calls']
            and values['query_rows'] == values['move_query_rows']
            and values['commit_joins'] == values['move_query_rows']
            for values in (self.counts['primary_move'], self.counts['isolation_move']))
        event_joins = (gates['ready'] == gates['evaluations'] == self.birth.counters['evals']
            and gates['maybe_apply_entries'] == gates['ready'] + gates['not_ready']
            and fake['jitter_calls'] == fake['radius_calls'] == fake['dv12_calls'] == gates['evaluations']
            and fake['query_rows'] == self.birth.N * gates['evaluations']
            and gates['reported_move_rows'] == sum(self.counts[s]['move_query_rows']
                for s in ('primary_move', 'isolation_move')) and site_joins)
        if not event_joins:
            self._unknown('ready/fake-pool/radius/move/query/commit coverage does not join')
        complete = (self.policy.completed_steps == 1000
            and event_joins
            and self.counts["gates"]["maybe_apply_entries"] == 1000
            and [s["completed_steps"] for s in self.snapshots] == list(CLOCKS)
            and len(self.neighborhood) == 9 and not self.unknown and not self.purity_failures
            and all(s["complete_live_leaf_scope"] for s in self.snapshots + self.neighborhood))
        return dict(schema=SCHEMA, candidate_id=CANDIDATE_ID, task_id=TASK_ID,
            numerical_credit="DIAGNOSTIC_ONLY", extra_model_or_rng_operations=0,
            completed_steps=self.policy.completed_steps, complete_event_coverage=complete,
            complete_call_count_joins=event_joins,
            coverage_status="COMPLETE" if complete else "UNKNOWN",
            purity_checks=self.purity_checks, purity_failures=self.purity_failures,
            counts={site: dict(values) for site, values in self.counts.items()},
            first_old_zero_new_positive=self.first_positive, first_affected=self.first_affected,
            first_affected_witness=witness_pin, comparison="typed_IEEE_bytes",
            snapshots=self.snapshots, neighborhood=self.neighborhood,
            unknown=list(self.unknown), historical_exposure="UNKNOWN",
            downstream_legacy_decision="UNKNOWN", full_live_leaf_scope=("COMPLETE" if
                self.snapshots and all(s["complete_live_leaf_scope"] for s in self.snapshots) else "UNKNOWN"))
