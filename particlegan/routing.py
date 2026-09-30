"""Paired conditional adaptation of E22 for densely blended particle banks.

Unlike independent-particle E22, a row has no standalone output law here.
Evidence and proposals replay the complete conditional bank.  A move retires
one row and splits another row's represented mass; protected, separately
supplied contexts must certify the resulting fast AND averaged functions.
These are empirical paired guards, not the independent-particle BH/kNN null.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
import math

import torch
from torch import nn


@dataclass(frozen=True)
class RoutedBatch:
    """Fit observations and a separate protected acceptance pool.

    Each context and target is a batch-first tensor.  Source, time and other
    conditions can be packed in ``context``; targets remain explicit.  Guard
    observations are never used for gradients, evidence or candidate choice.
    Keep a third, untouched test set for measuring generalization.
    """
    context: torch.Tensor
    targets: torch.Tensor
    guard_context: torch.Tensor
    guard_targets: torch.Tensor


@dataclass(frozen=True)
class RoutedCandidate:
    """Functional bank state; ``codes`` is the actual dense decoder input.

    Callbacks must read per-row values from this bundle, not captured live
    parameters.  ``generate`` must decode ``codes``: it equals weights@table
    for clean evaluations, and includes mixed-code DV12 jitter in training.
    """
    table: torch.Tensor
    log_mass: torch.Tensor
    row_state: dict
    averaged: bool = False
    codes: torch.Tensor | None = None


class RoutedExecution:
    """Ephemeral helper supplied to ``model_forward`` by :class:`RoutedRows`.

    Call :meth:`mix` exactly once at every declared site, in order. The helper
    and its candidate expire when that full model call ends; retaining it does
    not provide access to past activations or a reusable routing state.
    """

    def __init__(self, sites, candidate, batch_size, perturb_fn):
        self._sites, self._candidate = sites, candidate
        self._batch_size, self._perturb_fn = batch_size, perturb_fn
        self._next, self._usage, self._active = 0, None, True

    def mix(self, site_name, logits):
        """Mix a shared bank at one declared site using [B,*tokens,N] logits."""
        if not self._active:
            raise ValueError("routing.mix is valid only during its complete model forward")
        if self._next >= len(self._sites) or site_name != self._sites[self._next]:
            raise ValueError("routing sites must occur exactly once in their declared order")
        candidate = self._candidate
        if (not isinstance(logits, torch.Tensor) or logits.ndim < 2
                or logits.shape[0] != self._batch_size or logits.shape[-1] != len(candidate.table)
                or any(size == 0 for size in logits.shape) or logits.layout != torch.strided
                or not logits.is_floating_point() or logits.device != candidate.table.device
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("routing site logits must be finite floating [contexts, *tokens, rows] tensors on the table device")
        weights = (logits.to(candidate.table.dtype) + candidate.log_mass).softmax(-1)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError("routing site softmax must retain finite mass after row deletion")
        codes = weights @ candidate.table
        if self._perturb_fn is not None:
            mixed = codes.reshape(-1, candidate.table.shape[1])
            perturbed = self._perturb_fn(mixed)
            if (not isinstance(perturbed, torch.Tensor) or perturbed.shape != mixed.shape
                    or perturbed.device != mixed.device or perturbed.dtype != mixed.dtype):
                raise ValueError("routed latent perturbation must preserve the mixed-code shape, device and dtype")
            codes = perturbed.reshape(codes.shape)
        # Tokens are correlated uses of one context. Sites receive equal weight
        # regardless of their token counts; neither dimension inflates ESS.
        usage = weights.reshape(self._batch_size, -1, len(candidate.table)).mean(1)
        self._usage = usage if self._usage is None else self._usage + usage
        self._next += 1
        return codes

    def finish(self):
        if self._next != len(self._sites):
            raise ValueError("the complete model forward must call every declared routing site exactly once")
        return self._usage / len(self._sites)

    def close(self):
        # A callback retaining this object cannot reuse a past candidate or any
        # activations. The returned output/usage keep their normal autograd graph.
        self._active = False
        self._candidate = self._perturb_fn = self._usage = None


class RoutedRows:
    """Owner-free specification of a conditional dense key/value bank.

    ``route(models, context, candidate)`` returns normalized [B,N] weights;
    add ``candidate.log_mass`` to logits before softmax.  Tied key/value
    routing, e.g. query(context)@candidate.table.T, is supported.
    ``generate(models, context, candidate, weights)`` decodes candidate.codes.
    ``features(models, context, samples, targets)`` returns learned [B,F]
    critic features; evaluate targets with the same context to obtain the
    paired real baseline.  Callbacks must be deterministic in evaluation.

    Alternatively, ``model_forward(models, context, candidate, routing)``
    reruns the complete conditioned model. Declare an ordered ``sites`` tuple;
    call ``routing.mix(name, logits[B,*tokens,N])`` once at each site. The mixer
    adds log mass, normalizes, mixes the shared table, and applies DV12 to the
    flattened mixed codes independently at each site. Downstream logits can
    depend on preceding mixed codes. Every counterfactual reruns all sites.
    Evidence uses a token mean per context and an equal mean across sites.

    The router owns a [N] ``log_mass`` buffer or parameter.  Declare other
    independently cloneable router rows by their named parameter/buffer paths.
    Shared router/encoder parameters remain shared and are never cloned.
    Defaults require every protected context to avoid additional feature
    error, and a strictly positive mean improvement of the fast function.
    Optional ``output_error_guard`` also bounds clean final paired-output MSE
    increases for both fast and averaged models on those protected contexts.
    Output tolerances measure raw per-context MSE averaged across tokens and
    channels; they do not affect proposal selection or the training objective.
    """
    def __init__(self, *, route=None, generate=None, features, model_forward=None,
                 sites=(), log_mass_key="log_mass",
                 row_parameters=(), row_buffers=(), probe_budget=8, probe_interval=1,
                 reservoir_size=64, min_observations=8, min_effect=1e-6,
                 improvement_margin=1e-8, max_context_harm=0.0,
                 persistence_threshold=.75, split_scale=.1, candidate_budget=4,
                 output_error_guard=False, max_output_error_increase=0.0,
                 max_output_context_harm=0.0, routed_geometry="mass_atoms_v1"):
        if type(routed_geometry) is not str or routed_geometry != "mass_atoms_v1":
            raise ValueError("unsupported routed_geometry; routed DV12 law changed; "
                             "restore with prior release or explicit migration")
        if model_forward is None:
            if not all(callable(fn) for fn in (route, generate, features)):
                raise TypeError("routed route, generate and features must be callbacks")
            if sites:
                raise ValueError("named routing sites require the model_forward callback")
        elif not callable(model_forward) or not callable(features) or route is not None or generate is not None:
            raise TypeError("model_forward and features must be callbacks, as an alternative to route/generate")
        if isinstance(sites, (str, set, frozenset, dict)):
            raise ValueError("routing sites must be an ordered collection of distinct names")
        sites = tuple(sites)
        if model_forward is not None and (not sites or any(not isinstance(name, str) or not name for name in sites)
                                          or len(set(sites)) != len(sites)):
            raise ValueError("model_forward requires ordered distinct routing site names")
        for name, value in (("probe_budget", probe_budget), ("probe_interval", probe_interval), ("reservoir_size", reservoir_size),
                            ("min_observations", min_observations), ("candidate_budget", candidate_budget)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"routed {name} must be a positive integer")
        if min_observations > reservoir_size:
            raise ValueError("routed min_observations must fit in the context reservoir")
        for name, value in (("min_effect", min_effect), ("improvement_margin", improvement_margin),
                            ("max_context_harm", max_context_harm), ("split_scale", split_scale)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"routed {name} must be finite and nonnegative")
        if not 0 < persistence_threshold <= 1 or not 0 <= split_scale <= 1:
            raise ValueError("routed persistence_threshold and split_scale must be within their unit bounds")
        if type(output_error_guard) is not bool:
            raise ValueError("routed output_error_guard must be a boolean")
        for name, value in (("max_output_error_increase", max_output_error_increase),
                            ("max_output_context_harm", max_output_context_harm)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"routed {name} must be finite and nonnegative")
        if not output_error_guard and (max_output_error_increase or max_output_context_harm):
            raise ValueError("routed output error tolerances require output_error_guard=True")
        if not isinstance(log_mass_key, str) or not log_mass_key:
            raise ValueError("routed log_mass_key must name a router tensor")
        row_parameters, row_buffers = tuple(row_parameters), tuple(row_buffers)
        names = (*row_parameters, *row_buffers)
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("routed row tensor names must be distinct paths")
        self.route, self.generate, self.features = route, generate, features
        self.model_forward, self.sites = model_forward, sites
        self.log_mass_key, self.row_parameters, self.row_buffers = log_mass_key, row_parameters, row_buffers
        self.probe_budget, self.reservoir_size, self.min_observations = probe_budget, reservoir_size, min_observations
        self.probe_interval = probe_interval
        self.min_effect, self.improvement_margin, self.max_context_harm = float(min_effect), float(improvement_margin), float(max_context_harm)
        self.persistence_threshold, self.split_scale, self.candidate_budget = float(persistence_threshold), float(split_scale), candidate_budget
        self.output_error_guard = output_error_guard
        self.max_output_error_increase = float(max_output_error_increase)
        self.max_output_context_harm = float(max_output_context_harm)

    def to_dict(self):
        config = {name: getattr(self, name) for name in (
            "log_mass_key", "row_parameters", "row_buffers", "probe_budget", "reservoir_size",
            "min_observations", "min_effect", "improvement_margin", "max_context_harm",
            "persistence_threshold", "split_scale", "candidate_budget")}
        config["routed_geometry"] = "mass_atoms_v1"
        if self.model_forward is not None:
            config.update(model_forward=True, sites=self.sites)
        if self.output_error_guard:
            config.update(output_error_guard=True, max_output_error_increase=self.max_output_error_increase,
                          max_output_context_harm=self.max_output_context_harm)
        if self.probe_interval != 1:
            config["probe_interval"] = self.probe_interval
        return config

    def candidate_for(self, models, table, *, averaged=False, copy=False):
        router = models.get("router")
        if not isinstance(router, nn.Module):
            raise ValueError("routed rows require an explicit router module")
        parameters = dict(router.named_parameters(remove_duplicate=False))
        buffers = dict(router.named_buffers(remove_duplicate=False))
        tensors = {**buffers, **parameters}
        if self.log_mass_key not in tensors:
            raise ValueError("router must expose the declared per-row log_mass tensor")
        for names, available in ((self.row_parameters, parameters), (self.row_buffers, buffers)):
            if any(name not in available for name in names):
                raise ValueError("declared routed row tensors do not match their router owners")
        names = tuple(dict.fromkeys((self.log_mass_key, *self.row_parameters, *self.row_buffers)))
        rows = {name: tensors[name] for name in names}
        if (not isinstance(table, torch.Tensor) or table.ndim != 2 or not all(table.shape)
                or table.layout != torch.strided or not table.is_floating_point()):
            raise ValueError("routed table must be a dense floating [rows, latent_dim] tensor")
        for name, value in rows.items():
            if (value.ndim < 1 or value.shape[0] != len(table) or value.device != table.device
                    or value.layout != torch.strided):
                raise ValueError(f"routed row tensor {name} must match table rows and device")
        if rows[self.log_mass_key].shape != (len(table),) or rows[self.log_mass_key].dtype != table.dtype:
            raise ValueError("routed log_mass must have shape [rows] and use the floating table dtype")
        if copy:
            table = table.detach().clone()
            rows = {name: value.detach().clone() for name, value in rows.items()}
        return RoutedCandidate(table, rows[self.log_mass_key], rows, bool(averaged))

    def weights_for(self, models, context, candidate):
        if self.model_forward is not None:
            return self.forward_with_usage(models, context, candidate)[1]
        if (not isinstance(context, torch.Tensor) or context.ndim < 2 or not len(context)
                or context.device != candidate.table.device):
            raise ValueError("routed context must be a nonempty batch on the table device")
        weights = self.route(models, context, candidate)
        if (not isinstance(weights, torch.Tensor) or weights.shape != (len(context), len(candidate.table))
                or not weights.is_floating_point() or weights.device != candidate.table.device
                or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any())):
            raise ValueError("routed callback must return finite nonnegative [contexts, rows] weights")
        tolerance = max(1e-6, 8 * torch.finfo(weights.dtype).eps)
        if not torch.allclose(weights.sum(1), torch.ones(len(context), device=weights.device, dtype=weights.dtype),
                              rtol=tolerance, atol=tolerance):
            raise ValueError("routed weights must sum to one for every context")
        return weights.to(candidate.table.dtype)

    def forward(self, models, context, candidate, *, perturb_fn=None):
        return self.forward_with_usage(models, context, candidate, perturb_fn=perturb_fn)[0]

    def forward_with_usage(self, models, context, candidate, *, perturb_fn=None):
        """Return final samples and per-context mean bank usage from one rerun.

        Multi-site usage averages tokens within a site and then averages sites.
        No intermediate activations or route weights survive on this object.
        """
        if self.model_forward is None:
            weights = self.weights_for(models, context, candidate)
            codes = weights @ candidate.table
            if perturb_fn is not None:
                codes = perturb_fn(codes)
                if not isinstance(codes, torch.Tensor) or codes.shape != (len(context), candidate.table.shape[1]):
                    raise ValueError("routed latent perturbation must preserve the mixed-code shape")
            output = self.generate(models, context, replace(candidate, codes=codes), weights)
        else:
            if (not isinstance(context, torch.Tensor) or context.ndim < 2 or not len(context)
                    or context.device != candidate.table.device):
                raise ValueError("routed context must be a nonempty batch on the table device")
            routing = RoutedExecution(self.sites, candidate, len(context), perturb_fn)
            try:
                output = self.model_forward(models, context, candidate, routing)
                weights = routing.finish()
            finally:
                routing.close()
        if (not isinstance(output, torch.Tensor) or output.ndim < 2 or len(output) != len(context)
                or not output.is_floating_point() or output.device != candidate.table.device):
            raise ValueError("routed generation must return one floating sample per context")
        return output, weights

    def bind(self, *, models, averaged_models, table, averaged_table, optimizers,
             table_optimizer, seed=0, completed_steps=None, controller=None,
             allow_frozen_table=False):
        return RoutedRowControl(self, models=models, averaged_models=averaged_models,
                                table=table, averaged_table=averaged_table, optimizers=optimizers,
                                table_optimizer=table_optimizer, seed=seed,
                                completed_steps=completed_steps, controller=controller,
                                allow_frozen_table=allow_frozen_table)


class RoutedEvidence:
    """Dense gradient persistence plus paired counterfactual row diagnostics.

    Exposure is NOT evidence of support.  We accumulate learned feature
    residuals and actual deletion effects per context, with routing-mass
    weighted effective sample sizes.  Gradient persistence is a diagnostic
    gate, without an iid-particle p-value or false-discovery-rate claim.
    """
    _TENSORS = ("M", "Qs", "W", "S", "touches", "mass_sum", "mass_sq",
                "residual_sum", "effect_sum", "effect_sq", "effect_weight",
                "effect_weight_sq", "effect_contexts", "last_probe", "flag")

    def __init__(self, table, spec):
        self.spec, self.n, self.d, self.Q = spec, *table.shape, .05
        kw = {"device": table.device, "dtype": torch.float64}
        self.M = torch.zeros(table.shape, **kw)
        for name in ("Qs", "W", "S", "mass_sum", "mass_sq", "residual_sum", "effect_sum",
                     "effect_sq", "effect_weight", "effect_weight_sq"):
            setattr(self, name, torch.zeros(self.n, **kw))
        self.touches = torch.zeros(self.n, device=table.device, dtype=torch.long)
        self.effect_contexts = torch.zeros_like(self.touches)
        self.last_probe = torch.full_like(self.touches, -1)
        self.flag = torch.zeros(self.n, device=table.device, dtype=torch.bool)
        self.fraction, self.valid = 0., False
        self.counters = {"updates": 0, "resets": 0, "global_resets": 0, "flagged_rows_sum": 0}

    @torch.no_grad()
    def update(self, grad):
        if (not isinstance(grad, torch.Tensor) or grad.shape != self.M.shape or grad.layout != torch.strided
                or not grad.is_floating_point() or grad.device != self.M.device):
            raise ValueError("routed evidence requires the full dense table gradient")
        grad = grad.double()
        touched = grad.ne(0).any(1)
        keep = .98
        self.M[touched] = keep * self.M[touched] + grad[touched]
        self.Qs[touched] = keep * self.Qs[touched] + grad[touched].square().sum(1)
        self.W[touched] = keep * self.W[touched] + 1
        self.S[touched] = keep ** 2 * self.S[touched] + 1
        self.touches[touched] += 1
        self.counters["updates"] += 1
        self.refresh()

    @torch.no_grad()
    def observe_fit(self, weights, loss):
        w = weights.double()
        # Replaying a reservoir is not another independent context observation.
        self.mass_sum.copy_(w.sum(0))
        self.mass_sq.copy_(w.square().sum(0))
        self.residual_sum.copy_((w * loss[:, None]).sum(0))

    @torch.no_grad()
    def observe_effect(self, row, weights, base, deleted, evaluation):
        # A bounded effect distinguishes harm from support; exact unnormalized
        # paired error is used separately by the acceptance guard.
        effect = (base - deleted) / (base + deleted).clamp_min(1e-30)
        w = weights[:, row].double()
        self.effect_sum[row] = (w * effect).sum()
        self.effect_sq[row] = (w * effect.square()).sum()
        self.effect_weight[row] = w.sum()
        self.effect_weight_sq[row] = w.square().sum()
        self.effect_contexts[row] = int((w > 0).sum())
        self.last_probe[row] = evaluation
        self.refresh()

    @property
    def effect(self):
        return self.effect_sum / self.effect_weight.clamp_min(1e-30)

    @property
    def effective_contexts(self):
        return self.effect_weight.square() / self.effect_weight_sq.clamp_min(1e-300)

    @property
    def gradient_persistence(self):
        return self.M.norm(dim=1) / (self.W * self.Qs).clamp_min(1e-300).sqrt()

    def refresh(self):
        enough = self.effective_contexts >= self.spec.min_observations
        observed_effect = self.effect.abs() > self.spec.min_effect
        persistent = ((self.touches >= self.spec.min_observations)
                      & (self.gradient_persistence >= self.spec.persistence_threshold))
        self.flag = enough & observed_effect & persistent
        self.fraction = float(self.flag.double().mean())
        self.valid = bool(enough.any())
        self.counters["flagged_rows_sum"] += int(self.flag.sum())

    @torch.no_grad()
    def reset(self, rows=None):
        # Every contribution changes after a normalized dense split.  Retaining
        # unaffected rows' old counterfactuals would silently mix two functions.
        nonempty = self.valid or bool(self.W.any()) or bool(self.effect_weight.any())
        for name in self._TENSORS:
            value = getattr(self, name)
            value.fill_(-1 if name == "last_probe" else 0)
        self.fraction, self.valid = 0., False
        if nonempty:
            self.counters["resets"] += self.n
            self.counters["global_resets"] += 1

    def diagnostics(self):
        return {"law": "conditional_paired_diagnostic", "fraction": self.fraction,
                "mass": self.mass_sum / self.mass_sum.sum().clamp_min(1e-30),
                "paired_residual": self.residual_sum / self.mass_sum.clamp_min(1e-30),
                "deletion_effect": self.effect, "effective_contexts": self.effective_contexts,
                "support": (-self.effect).clamp_min(0), "gradient_persistence": self.gradient_persistence,
                "counters": dict(self.counters)}

    def state_dict(self):
        return {"tensors": {name: getattr(self, name).clone() for name in self._TENSORS},
                "fraction": self.fraction, "valid": self.valid, "counters": dict(self.counters)}

    def check_state(self, state):
        if not isinstance(state, dict) or set(state) != {"tensors", "fraction", "valid", "counters"}:
            raise ValueError("invalid routed evidence state")
        tensors = state["tensors"]
        if not isinstance(tensors, dict) or set(tensors) != set(self._TENSORS):
            raise ValueError("invalid routed evidence tensor state")
        for name in self._TENSORS:
            value, current = tensors[name], getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != current.shape or value.dtype != current.dtype:
                raise ValueError(f"routed evidence {name} does not match the bank")
        if type(state["valid"]) is not bool or not isinstance(state["counters"], dict) or not 0 <= state["fraction"] <= 1:
            raise ValueError("invalid routed evidence diagnostics")

    @torch.no_grad()
    def load_state_dict(self, state):
        self.check_state(state)
        for name, value in state["tensors"].items():
            getattr(self, name).copy_(value)
        self.fraction, self.valid, self.counters = float(state["fraction"]), state["valid"], dict(state["counters"])


class RoutedRowControl:
    """Bound conditional evidence and guarded structural row transport.

    Both split rows inherit half the parent's first Adam moment and one quarter
    of its second/AMSGrad-max moments, retaining the parent's optimizer age.
    Latent directional history is cleared at changed rows. All conditional
    evidence is invalidated; callers rebase affected row testers and restart
    shared router/encoder testers.  Models and optimizers retain caller
    ownership; the enclosing policy checkpoints their full state.
    """
    _POOLS = ("fit_context", "fit_targets", "guard_context", "guard_targets")

    def __init__(self, spec, *, models, averaged_models, table, averaged_table,
                 optimizers, table_optimizer, seed, completed_steps, controller,
                 allow_frozen_table=False):
        self.spec, self.models = spec, dict(models)
        self.averaged_models = {**self.models, **averaged_models}
        self.table, self.averaged_table, self.optimizers = table, averaged_table, tuple(optimizers)
        self.table_optimizer, self.controller = table_optimizer, controller
        self.completed_steps = (lambda: 0) if completed_steps is None else completed_steps
        if type(allow_frozen_table) is not bool:
            raise ValueError("allow_frozen_table must be a boolean")
        if (not table.is_leaf or len(table) < (1 if allow_frozen_table and not table.requires_grad else 2)
                or (not table.requires_grad and not allow_frozen_table)):
            raise ValueError("routed restructuring requires at least two trainable table rows")
        table_owners = [optimizer for optimizer in self.optimizers for group in optimizer.param_groups
                        for parameter in group["params"] if parameter is table]
        if (table.requires_grad and (table_optimizer not in self.optimizers or table_owners != [table_optimizer])):
            raise ValueError("the declared routed table optimizer must own the table exactly once")
        if not table.requires_grad and (len(table_owners) > 1 or (table_owners and table_owners != [table_optimizer])):
            raise ValueError("a frozen routed table must have at most its declared optimizer owner")
        table_owner = table_owners[0] if table_owners else None
        if (averaged_table.shape != table.shape or averaged_table.dtype != table.dtype
                or averaged_table.device != table.device or averaged_table.requires_grad
                or averaged_table.untyped_storage().data_ptr() == table.untyped_storage().data_ptr()):
            raise ValueError("routed averaged table must be separate, frozen and match the fast table")
        fast, average = self.candidate(), self.candidate(averaged=True)
        self._bindings = {"table": (table, averaged_table, table_owner)}
        self.row_parameters = {"table": table}
        for name, tensor in fast.row_state.items():
            averaged = average.row_state[name]
            owners = [optimizer for optimizer in self.optimizers
                      if any(parameter is tensor for group in optimizer.param_groups for parameter in group["params"])]
            if tensor.requires_grad and len(owners) != 1:
                raise ValueError("trainable routed row tensors require exactly one optimizer owner")
            if averaged.requires_grad or averaged.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr():
                raise ValueError("averaged router row state must be frozen and separate")
            self._bindings["router." + name] = (tensor, averaged, owners[0] if owners else None)
            if isinstance(tensor, nn.Parameter):
                self.row_parameters["router." + name] = tensor
        storages = set()
        for tensor, averaged, _ in self._bindings.values():
            if tensor.shape != averaged.shape or tensor.dtype != averaged.dtype or tensor.device != averaged.device:
                raise ValueError("averaged routed row state must match its fast owner's shape, dtype and device")
            for value in (tensor, averaged):
                address = value.untyped_storage().data_ptr()
                if address in storages:
                    raise ValueError("distinct routed row owners must not share storage")
                storages.add(address)
        history = getattr(table_owner, "latent_history", None)
        if history is not None and (not isinstance(history, torch.Tensor) or history.shape != table.shape
                                    or history.device != table.device or history.dtype != table.dtype):
            raise ValueError("routed table optimizer latent_history must match its table")
        if table.requires_grad and len(table_owners) != 1:
            raise ValueError("routed table must have exactly one optimizer owner")
        self.evidence = RoutedEvidence(table, spec)
        self.stream = torch.Generator(device=table.device).manual_seed(seed)
        for name in self._POOLS:
            setattr(self, name, None)
        self.fit_fill = self.guard_fill = self.fit_cursor = self.guard_cursor = self.rows_since_eval = 0
        self.probe_clock = {"observed_updates": 0, "last_probe_update": 0}
        self.probe_cursor = 0
        self.latest_gradient = None
        self.moved_rows, self.moved_parameters, self.restart_router = None, {}, False
        self.counters = {name: 0 for name in ("evals", "probes", "proposals", "guard_rejections", "moves", "splits")}
        self.last = {}

    def candidate(self, averaged=False, copy=False):
        return self.spec.candidate_for(self.averaged_models if averaged else self.models,
                                       self.averaged_table if averaged else self.table,
                                       averaged=averaged, copy=copy)

    def generate(self, context, *, averaged=False, candidate=None, perturb_fn=None):
        candidate = self.candidate(averaged) if candidate is None else candidate
        return self.spec.forward(self.averaged_models if candidate.averaged else self.models,
                                 context, candidate, perturb_fn=perturb_fn)

    def weights(self, context, *, averaged=False, candidate=None):
        candidate = self.candidate(averaged) if candidate is None else candidate
        return self.spec.weights_for(self.averaged_models if candidate.averaged else self.models, context, candidate)

    @torch.no_grad()
    def begin(self, batch):
        self.check_batch(batch)
        pairs = (("fit", batch.context, batch.targets), ("guard", batch.guard_context, batch.guard_targets))
        for prefix, context, targets in pairs:
            if getattr(self, prefix + "_context") is None:
                setattr(self, prefix + "_context", torch.zeros((self.spec.reservoir_size, *context.shape[1:]), device=context.device, dtype=context.dtype))
                setattr(self, prefix + "_targets", torch.zeros((self.spec.reservoir_size, *targets.shape[1:]), device=targets.device, dtype=targets.dtype))
            pool_context, pool_targets = getattr(self, prefix + "_context"), getattr(self, prefix + "_targets")
            values = (context[-self.spec.reservoir_size:].detach(), targets[-self.spec.reservoir_size:].detach())
            cursor = getattr(self, prefix + "_cursor")
            indices = (torch.arange(len(values[0]), device=context.device) + cursor) % self.spec.reservoir_size
            pool_context[indices], pool_targets[indices] = values
            setattr(self, prefix + "_cursor", (cursor + len(values[0])) % self.spec.reservoir_size)
            setattr(self, prefix + "_fill", min(self.spec.reservoir_size, getattr(self, prefix + "_fill") + len(values[0])))
        self.rows_since_eval += len(batch.context)
        self.probe_clock["observed_updates"] += 1

    @torch.no_grad()
    def check_batch(self, batch):
        """Validate paired observations without allocating or filling pools."""
        if not isinstance(batch, RoutedBatch):
            raise TypeError("routed observations require RoutedBatch with separate guard contexts")
        if batch.context is batch.guard_context:
            raise ValueError("guard contexts must be supplied separately from proposal contexts")
        pairs = (("fit", batch.context, batch.targets), ("guard", batch.guard_context, batch.guard_targets))
        for prefix, context, targets in pairs:
            if (not isinstance(context, torch.Tensor) or context.ndim < 2 or not len(context)
                    or not isinstance(targets, torch.Tensor) or targets.ndim < 2 or len(targets) != len(context)
                    or context.device != self.table.device or targets.device != self.table.device
                    or not targets.is_floating_point() or not bool(torch.isfinite(context).all())
                    or not bool(torch.isfinite(targets).all())):
                raise ValueError("routed contexts and targets must be paired nonempty batches on the table device")
            pool_context, pool_targets = getattr(self, prefix + "_context"), getattr(self, prefix + "_targets")
            if pool_context is not None and (context.shape[1:] != pool_context.shape[1:] or targets.shape[1:] != pool_targets.shape[1:]
                                             or context.dtype != pool_context.dtype or targets.dtype != pool_targets.dtype):
                raise ValueError("routed reservoir observation shapes and dtypes must remain unchanged")
        if batch.context.shape[1:] != batch.guard_context.shape[1:] or batch.targets.shape[1:] != batch.guard_targets.shape[1:]:
            raise ValueError("fit and guard observation shapes must describe the same conditional task")
        fit = [batch.context.flatten(1)]
        guard = [batch.guard_context.flatten(1)]
        if self.fit_context is not None:
            fit.append(self.fit_context[:self.fit_fill].flatten(1))
        if self.guard_context is not None:
            guard.append(self.guard_context[:self.guard_fill].flatten(1))
        self._check_disjoint(torch.cat(fit), torch.cat(guard))

    @staticmethod
    def _check_disjoint(fit, guard):
        # Equality is an information-boundary check, not data-space geometry.
        # Unique rows allocate [fit+guard,width], never [fit,guard,width].
        fit, guard = torch.unique(fit, dim=0), torch.unique(guard, dim=0)
        if len(torch.unique(torch.cat((fit, guard)), dim=0)) != len(fit) + len(guard):
            raise ValueError("protected guard contexts must not overlap fit/proposal contexts")

    def observe_backward(self, grad=None):
        if not self.table.requires_grad:
            raise ValueError("frozen routed tables support forward/serving only; disable row evidence and birth/death")
        grad = self.table.grad if grad is None else grad
        self.evidence.update(grad)
        self.latest_gradient = grad.detach().clone()

    @contextmanager
    def _evaluating(self):
        modules = dict.fromkeys(module for root in (*self.models.values(), *self.averaged_models.values())
                               for module in root.modules())
        flags = [(module, module.training) for module in modules]
        for module in modules:
            module.training = False
        devices = [self.table.device.index] if self.table.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                yield
        finally:
            for module, flag in flags:
                module.training = flag

    def _measure(self, context, targets, candidate, *, with_usage=False, with_output_error=False):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.table.device) if self.table.device.type == "cuda" else None
        models = self.averaged_models if candidate.averaged else self.models
        if with_usage:
            output, usage = self.spec.forward_with_usage(models, context, candidate)
        else:
            output = self.spec.forward(models, context, candidate)
        if output.shape != targets.shape:
            raise ValueError("routed generated samples must match paired targets")
        fake = self.spec.features(models, context, output, targets)
        real = self.spec.features(models, context, targets, targets)
        if (not isinstance(fake, torch.Tensor) or fake.ndim != 2 or not len(fake[0]) or fake.shape[0] != len(context)
                or not isinstance(real, torch.Tensor) or real.shape != fake.shape
                or fake.device != self.table.device or real.device != self.table.device
                or not fake.is_floating_point() or not real.is_floating_point()):
            raise ValueError("paired routed features must return matching floating [contexts, features] matrices")
        loss = (fake.double() - real.double()).square().mean(1)
        if not bool(torch.isfinite(loss).all()):
            raise ValueError("paired routed feature errors must be finite")
        if with_output_error:
            output_error = (output.double() - targets.double()).square().flatten(1).mean(1)
            if not bool(torch.isfinite(output_error).all()):
                raise ValueError("paired routed output errors must be finite")
        if (not torch.equal(cpu_rng, torch.get_rng_state())
                or (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(self.table.device)))):
            raise ValueError("routed evaluation callbacks must be deterministic; supply shared noise in the observation context")
        result = (loss,)
        if with_usage:
            result += (usage,)
        if with_output_error:
            result += (output_error,)
        return result if with_usage or with_output_error else loss

    def _delete(self, base, row):
        mass = base.log_mass.clone()
        mass[row] = -torch.inf
        state = {**base.row_state, self.spec.log_mass_key: mass}
        return replace(base, log_mass=mass, row_state=state)

    def _split(self, base, child, parent, delta):
        table = base.table.clone()
        parent_table = table[parent].clone()
        table[parent], table[child] = parent_table - delta, parent_table + delta
        state = {name: value.clone() for name, value in base.row_state.items()}
        for value in state.values():
            value[child] = value[parent]
        mass = state[self.spec.log_mass_key]
        half = base.log_mass[parent] - math.log(2.)
        mass[parent], mass[child] = half, half
        return RoutedCandidate(table, mass, state, base.averaged)

    def _delta(self, parent, fast, average):
        def radius(table):
            distance = (table - table[parent]).norm(dim=1)
            positive = distance[distance > 0]
            return positive.min() * .5 if len(positive) else table.new_zeros(())
        norm_bound = torch.minimum(radius(fast.table), radius(average.table)) * self.spec.split_scale
        direction = None if self.latest_gradient is None else -self.latest_gradient[parent]
        if direction is None or not bool(direction.norm() > 0):
            direction = torch.randn(fast.table.shape[1], generator=self.stream,
                                    device=fast.table.device, dtype=fast.table.dtype)
        return direction / direction.norm().clamp_min(1e-30) * norm_bound

    @torch.no_grad()
    def maybe_apply(self, mutate=True):
        """Refresh paired row evidence, and optionally propose guarded moves.

        ``mutate=False`` stops after the fit-context deletion probes. It never
        chooses proposals, reads protected guard targets, draws proposal RNG,
        or changes the bank, row state, optimizer or serving averages.
        """
        if type(mutate) is not bool:
            raise ValueError("routed mutate must be a boolean")
        if not self.table.requires_grad:
            raise ValueError("frozen routed tables support forward/serving only; disable row evidence and birth/death")
        self.moved_rows, self.moved_parameters, self.restart_router = None, {}, False
        minimum = self.spec.min_observations
        fills = (self.fit_fill, self.rows_since_eval, self.guard_fill) if mutate else (self.fit_fill, self.rows_since_eval)
        if min(fills) < minimum:
            return None
        # Count successful begin() calls, not contexts or eligibility checks.
        # Default 1 preserves the historical every-eligible-call behavior.
        if (self.spec.probe_interval > 1
                and self.probe_clock["observed_updates"] - self.probe_clock["last_probe_update"] < self.spec.probe_interval):
            return None
        self.rows_since_eval = 0
        self.probe_clock["last_probe_update"] = self.probe_clock["observed_updates"]
        self.counters["evals"] += 1
        evaluation = self.counters["evals"]
        context, targets = self.fit_context[:self.fit_fill], self.fit_targets[:self.fit_fill]
        fast, average = self.candidate(copy=True), self.candidate(averaged=True, copy=True)
        with self._evaluating():
            if self.spec.model_forward is None:
                baseline = self._measure(context, targets, fast)
                weights = self.weights(context, candidate=fast)
            else:
                baseline, weights = self._measure(context, targets, fast, with_usage=True)
            self.evidence.observe_fit(weights, baseline)
            budget = min(len(self.table), self.spec.probe_budget)
            probes = (torch.arange(budget, device=self.table.device) + self.probe_cursor) % len(self.table)
            self.probe_cursor = (self.probe_cursor + budget) % len(self.table)
            for row in probes.tolist():
                deletion = self._delete(fast, row)
                if self.spec.model_forward is None:
                    deleted_weights = self.weights(context, candidate=deletion)
                    expected_weights = weights.clone()
                    expected_weights[:, row] = 0
                    remainder = expected_weights.sum(1, keepdim=True)
                    if bool((remainder == 0).any()):
                        raise ValueError("routed deletion cannot certify a numerically exclusive row; increase routing temperature")
                    expected_weights = expected_weights / remainder
                    if (bool((deleted_weights[:, row] != 0).any())
                            or not torch.allclose(deleted_weights, expected_weights, rtol=5e-3, atol=5e-5)):
                        raise ValueError("routing callback must apply candidate.log_mass as additive logits before softmax")
                # Multi-site deletion reruns the full function. Its upstream
                # changes alter later queries, so same-query renormalization
                # is invalid there; routing.mix enforces additive mass itself.
                deleted = self._measure(context, targets, deletion)
                self.evidence.observe_effect(row, weights, baseline, deleted, evaluation)
                self.counters["probes"] += 1
            effect, enough = self.evidence.effect, self.evidence.effective_contexts >= minimum
            fresh = self.evidence.last_probe >= max(1, evaluation - math.ceil(len(self.table) / budget))
            children = (enough & fresh & (effect > self.spec.min_effect)).nonzero().flatten()
            parents = (enough & fresh & (effect < -self.spec.min_effect)).nonzero().flatten()
            self.last = {"step": self.completed_steps() + 1, "moves": 0, "law": "routed_paired",
                         "fit_error": float(baseline.mean()), "eligible_deaths": len(children), "eligible_births": len(parents)}
            if not mutate:
                self.last["evidence_only"] = True
                return dict(self.last)
            if not len(children) or not len(parents):
                return dict(self.last)
            children = children[effect[children].argsort(descending=True, stable=True)]
            parents = parents[effect[parents].argsort(stable=True)]
            proposals = []
            for child in children.tolist():
                for parent in parents.tolist():
                    delta = self._delta(parent, fast, average)
                    for variant, displacement in (("duplicate", torch.zeros_like(delta)), ("antisymmetric", delta)):
                        proposal = self._split(fast, child, parent, displacement)
                        loss = self._measure(context, targets, proposal)
                        proposals.append((float(loss.mean()), child, parent, variant, proposal,
                                          self._split(average, child, parent, displacement)))
                        self.counters["proposals"] += 1
                    if len(proposals) >= 2 * self.spec.candidate_budget:
                        break
                if len(proposals) >= 2 * self.spec.candidate_budget:
                    break
            # Candidate selection NEVER reads protected contexts.
            proposals.sort(key=lambda item: (item[0], item[3] != "antisymmetric"))
            selected = proposals[0]
            if float(baseline.mean()) - selected[0] < self.spec.improvement_margin:
                self.last["skip"] = "proposal did not improve fit paired error"
                return dict(self.last)
            _, child, parent, variant, proposed, proposed_average = selected
            guard_context, guard_targets = self.guard_context[:self.guard_fill], self.guard_targets[:self.guard_fill]
            guard_options = {"with_output_error": True} if self.spec.output_error_guard else {}
            before = self._measure(guard_context, guard_targets, fast, **guard_options)
            after = self._measure(guard_context, guard_targets, proposed, **guard_options)
            average_before = self._measure(guard_context, guard_targets, average, **guard_options)
            average_after = self._measure(guard_context, guard_targets, proposed_average, **guard_options)
            if self.spec.output_error_guard:
                before, output_before = before
                after, output_after = after
                average_before, average_output_before = average_before
                average_after, average_output_after = average_after
            gain = float((before - after).mean())
            average_gain = float((average_before - average_after).mean())
            harm, average_harm = float((after - before).max()), float((average_after - average_before).max())
            accepted = (gain >= self.spec.improvement_margin and average_gain >= -1e-12
                        and max(harm, average_harm) <= self.spec.max_context_harm + 1e-12)
            self.last.update(child=child, parent=parent, variant=variant, guard_error_before=float(before.mean()),
                             guard_error_after=float(after.mean()), average_guard_error_before=float(average_before.mean()),
                             average_guard_error_after=float(average_after.mean()), guard_gain=gain,
                             average_guard_gain=average_gain, max_context_harm=harm,
                             average_max_context_harm=average_harm, guard_contexts=self.guard_fill,
                             accepted=accepted)
            if self.spec.output_error_guard:
                output_increase = float((output_after - output_before).mean())
                average_output_increase = float((average_output_after - average_output_before).mean())
                output_harm = float((output_after - output_before).max())
                average_output_harm = float((average_output_after - average_output_before).max())
                output_accepted = (max(output_increase, average_output_increase)
                                   <= self.spec.max_output_error_increase + 1e-12
                                   and max(output_harm, average_output_harm)
                                   <= self.spec.max_output_context_harm + 1e-12)
                self.last.update(feature_guard_accepted=accepted, output_guard_accepted=output_accepted,
                                 guard_output_mse_before=float(output_before.mean()),
                                 guard_output_mse_after=float(output_after.mean()),
                                 average_guard_output_mse_before=float(average_output_before.mean()),
                                 average_guard_output_mse_after=float(average_output_after.mean()),
                                 guard_output_mse_increase=output_increase,
                                 average_guard_output_mse_increase=average_output_increase,
                                 max_output_context_harm=output_harm,
                                 average_max_output_context_harm=average_output_harm)
                accepted = accepted and output_accepted
                self.last["accepted"] = accepted
            if not accepted:
                self.counters["guard_rejections"] += 1
                return dict(self.last)
            self._commit(child, parent, proposed, proposed_average)
            return dict(self.last)

    def refresh_evidence(self):
        """Refresh evidence without proposing or mutating particle rows."""
        return self.maybe_apply(mutate=False)

    @staticmethod
    def _adam_transport_state(optimizer, tensor):
        """Validate the explicitly supported row-local Adam state layout."""
        from .k3p import K3PGeneratorAdam

        if type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW, K3PGeneratorAdam):
            raise ValueError("routed birth/death optimizer transport supports Adam, AdamW and K3PGeneratorAdam only")
        groups = [group for group in optimizer.param_groups
                  for parameter in group["params"] if parameter is tensor]
        if len(groups) != 1 or groups[0].get("differentiable", False):
            raise ValueError("routed split transport requires one nondifferentiable Adam owner per row tensor")
        response = getattr(optimizer, "direct_response", None)
        if response is not None and any(parameter is tensor for parameter in response.params):
            raise ValueError("routed split transport does not support direct-particle response on a routed row tensor")
        state = optimizer.state.get(tensor, {})
        if not state:
            return state
        expected = {"step", "exp_avg", "exp_avg_sq"}
        if groups[0].get("amsgrad", False):
            expected.add("max_exp_avg_sq")
        if set(state) != expected:
            raise ValueError("routed split transport requires the standard Adam moment state layout")
        step = state["step"]
        if (not isinstance(step, (int, float, torch.Tensor))
                or (isinstance(step, torch.Tensor) and (step.ndim != 0 or step.layout != torch.strided))
                or not math.isfinite(float(step)) or float(step) < 0):
            raise ValueError("routed split transport requires a valid scalar Adam age")
        for name in expected - {"step"}:
            moment = state[name]
            if (not isinstance(moment, torch.Tensor) or moment.shape != tensor.shape
                    or moment.dtype != tensor.dtype or moment.device != tensor.device
                    or moment.layout != torch.strided or not moment.is_floating_point()
                    or not bool(torch.isfinite(moment).all())
                    or (name != "exp_avg" and bool((moment < 0).any()))):
                raise ValueError(f"routed split transport requires a valid row-local Adam {name}")
        return state

    def validate_optimizer_transport(self):
        """Check structural owners before enabling routed birth/death.

        Adam/AdamW and native K3PGeneratorAdam use the supported moment layout.
        Coupled weight decay is allowed: the split supplies an explicit history
        prior, whose exact half-mass gradient interpretation assumes zero
        coupled regularization and an unperturbed duplicate. Unknown optimizer
        layouts or additional coupled row histories require a separate contract.
        This validation never changes owner weights, state, or optimizer clocks.
        """
        for tensor, _, optimizer in self._bindings.values():
            if optimizer is not None:
                self._adam_transport_state(optimizer, tensor)
        history = getattr(self.table_optimizer, "latent_history", None)
        if history is not None and (not isinstance(history, torch.Tensor) or history.shape != self.table.shape
                                    or history.dtype != self.table.dtype or history.device != self.table.device
                                    or history.layout != torch.strided):
            raise ValueError("routed split transport requires table-shaped latent history")

    @torch.no_grad()
    def _commit(self, child, parent, proposed, proposed_average):
        # Preflight every owner before any weights are written. Snapshot parent
        # history before changing either row, including when the child comes
        # first in storage. Preserve the shared optimizer age: resetting only
        # moments at a late split produces an inconsistent bias correction.
        self.validate_optimizer_transport()
        transported = {}
        for name, (tensor, _, optimizer) in self._bindings.items():
            if optimizer is not None:
                state = optimizer.state.get(tensor, {})
                transported[name] = {key: state[key][parent].clone() * factor
                                     for key, factor in (("exp_avg", .5), ("exp_avg_sq", .25),
                                                         ("max_exp_avg_sq", .25)) if key in state}
        rows = torch.tensor([parent, child], device=self.table.device, dtype=torch.long)
        for name, (tensor, averaged, optimizer) in self._bindings.items():
            values = proposed.table if name == "table" else proposed.row_state[name.removeprefix("router.")]
            average_values = proposed_average.table if name == "table" else proposed_average.row_state[name.removeprefix("router.")]
            tensor[rows], averaged[rows] = values[rows], average_values[rows]
            if optimizer is not None:
                for key, value in transported[name].items():
                    optimizer.state[tensor][key][rows] = value
                if tensor is self.table:
                    history = getattr(optimizer, "latent_history", None)
                    if history is not None:
                        # Restart A2 directional comparisons after a possible
                        # antisymmetric key/value perturbation. Damping's global
                        # counters and the transported Adam moments stay intact.
                        history[rows] = 0
                self.moved_parameters[tensor] = rows
        self.moved_rows, self.restart_router = rows, True
        self.evidence.reset()
        self.latest_gradient = None
        if self.controller is not None:
            self.controller.previous_gradient = None
            self.controller.alignment = self.controller.last_cosine = 0.
        self.counters["moves"] += len(rows)
        self.counters["splits"] += 1
        self.last.update(moves=len(rows), splits=1)

    def diagnostics(self):
        return {"counters": dict(self.counters), "last": deepcopy(self.last), "rows": self.evidence.diagnostics()}

    def state_dict(self):
        return deepcopy({"schema": 1, "config": self.spec.to_dict(), "table_shape": tuple(self.table.shape),
                         "pools": {name: getattr(self, name) for name in self._POOLS},
                         "fit_fill": self.fit_fill, "guard_fill": self.guard_fill,
                         "fit_cursor": self.fit_cursor, "guard_cursor": self.guard_cursor,
                         "rows_since_eval": self.rows_since_eval, "probe_cursor": self.probe_cursor,
                         "probe_clock": self.probe_clock,
                         "latest_gradient": self.latest_gradient, "evidence": self.evidence.state_dict(),
                         "row_ownership": {name: {"shape": tuple(tensor.shape), "dtype": str(tensor.dtype),
                                                  "parameter": isinstance(tensor, nn.Parameter),
                                                  "optimizer": None if optimizer is None else self.optimizers.index(optimizer)}
                                           for name, (tensor, _, optimizer) in self._bindings.items()},
                         "stream": self.stream.get_state(), "counters": self.counters, "last": self.last,
                         "moved_rows": self.moved_rows, "restart_router": self.restart_router})

    def check_state(self, state):
        expected = self.state_dict()
        allowed_keys = (set(expected), set(expected) - {"probe_clock"}) if self.spec.probe_interval == 1 else (set(expected),)
        if not isinstance(state, dict) or set(state) not in allowed_keys or state.get("schema") != 1:
            raise ValueError("invalid routed-control checkpoint schema")
        if (not isinstance(state["config"], dict)
                or state["config"].get("routed_geometry") != "mass_atoms_v1"):
            raise ValueError("routed DV12 law changed; restore with prior release or explicit migration")
        for name in ("config", "table_shape"):
            if state[name] != expected[name]:
                raise ValueError("routed-control checkpoint configuration does not match")
        if "probe_clock" in state:
            clock = state["probe_clock"]
            if (not isinstance(clock, dict) or set(clock) != {"observed_updates", "last_probe_update"}
                    or any(type(value) is not int for value in clock.values())
                    or not 0 <= clock["last_probe_update"] <= clock["observed_updates"]):
                raise ValueError("invalid routed probe clock")
        if not isinstance(state["pools"], dict) or set(state["pools"]) != set(self._POOLS):
            raise ValueError("invalid routed context reservoirs")
        for name, value in state["pools"].items():
            if value is not None and (not isinstance(value, torch.Tensor) or value.ndim < 2
                                       or len(value) != self.spec.reservoir_size or not bool(torch.isfinite(value).all())):
                raise ValueError("invalid routed context reservoir tensor")
            current = expected["pools"][name]
            if current is not None and value is not None and (current.shape != value.shape or current.dtype != value.dtype):
                raise ValueError("routed context reservoir shapes or dtypes do not match")
        for prefix in ("fit", "guard"):
            context, targets = state["pools"][prefix + "_context"], state["pools"][prefix + "_targets"]
            if (context is None) != (targets is None) or (targets is not None and not targets.is_floating_point()):
                raise ValueError("routed context and target reservoirs must remain paired")
            fill, cursor = state[prefix + "_fill"], state[prefix + "_cursor"]
            if (type(fill) is not int or not 0 <= fill <= self.spec.reservoir_size or type(cursor) is not int
                    or not 0 <= cursor < self.spec.reservoir_size or (context is None and fill)):
                raise ValueError("invalid routed reservoir counters")
        fit, guard = state["pools"]["fit_context"], state["pools"]["guard_context"]
        if fit is not None and guard is not None:
            if (fit.shape[1:] != guard.shape[1:]
                    or state["pools"]["fit_targets"].shape[1:] != state["pools"]["guard_targets"].shape[1:]):
                raise ValueError("routed checkpoint reservoirs describe different conditional tasks")
            self._check_disjoint(fit[:state["fit_fill"]].flatten(1), guard[:state["guard_fill"]].flatten(1))
        for name in ("rows_since_eval", "probe_cursor"):
            if type(state[name]) is not int or state[name] < 0:
                raise ValueError("invalid routed observation counter")
        if state["probe_cursor"] >= len(self.table):
            raise ValueError("invalid routed probe cursor")
        grad = state["latest_gradient"]
        if grad is not None and (not isinstance(grad, torch.Tensor) or grad.shape != self.table.shape or grad.dtype != self.table.dtype):
            raise ValueError("routed stored gradient does not match the table")
        self.evidence.check_state(state["evidence"])
        if state["row_ownership"] != expected["row_ownership"]:
            raise ValueError("routed row ownership checkpoint does not match")
        try:
            torch.Generator(device=self.stream.device).set_state(state["stream"].cpu())
        except (AttributeError, TypeError, RuntimeError) as error:
            raise ValueError("invalid routed private RNG state") from error
        moved = state["moved_rows"]
        if moved is not None and (not isinstance(moved, torch.Tensor) or moved.ndim != 1 or moved.dtype != torch.long
                                  or bool((moved < 0).any()) or bool((moved >= len(self.table)).any())):
            raise ValueError("invalid routed moved rows")
        if type(state["restart_router"]) is not bool or not isinstance(state["counters"], dict) or not isinstance(state["last"], dict):
            raise ValueError("invalid routed diagnostics")

    @torch.no_grad()
    def load_state_dict(self, state):
        self.check_state(state)
        for name, value in state["pools"].items():
            setattr(self, name, None if value is None else value.clone().to(self.table.device))
        for name in ("fit_fill", "guard_fill", "fit_cursor", "guard_cursor", "rows_since_eval", "probe_cursor"):
            setattr(self, name, state[name])
        self.probe_clock = deepcopy(state.get("probe_clock", {"observed_updates": 0, "last_probe_update": 0}))
        self.latest_gradient = None if state["latest_gradient"] is None else state["latest_gradient"].clone().to(self.table.device)
        self.evidence.load_state_dict(state["evidence"])
        self.stream.set_state(state["stream"].cpu())
        self.counters, self.last = deepcopy(state["counters"]), deepcopy(state["last"])
        self.moved_rows = None if state["moved_rows"] is None else state["moved_rows"].clone().to(self.table.device)
        self.restart_router = state["restart_router"]
        self.moved_parameters = {} if self.moved_rows is None else {tensor: self.moved_rows for tensor in self.row_parameters.values()}
