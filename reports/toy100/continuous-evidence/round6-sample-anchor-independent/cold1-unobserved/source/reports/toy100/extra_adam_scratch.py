"""Joint ExtraAdam, Gidel et al. Algorithm 4 option 1, for two frozen hosts.

Each outer update evaluates the complete game gradient twice. Both D and G
gradients are captured before either player moves. Adam moments advance at
both evaluations. The second parameter step starts from the saved original
weights, while retaining the lookahead moments. There is no iterate averaging,
rate decay, or additional lookahead coefficient. The one-evaluation control
is simultaneous Adam, not the original alternating Adam host.

The AST adapter wraps only the existing training-gradient block. Noise clocks,
EMA updates, budgets and evaluations remain outside it. Existing host samplers
run at each evaluation; stochastic minibatches/noise are freshly drawn, while
deterministic full-batch data stay fixed. Gradient cost is explicitly doubled.
The source transform is archived and checked by an exact structural inverse.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import inspect
import math
from unittest.mock import patch

import torch

PAPER = "https://arxiv.org/html/1802.10551v5"
METHODS = {"extra_adam": 2, "sim_adam": 1}
HOSTS = {"mode_hold": "train_mode_hold", "trajectory": "train"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _call(node, name: str) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func) == name


def _contains(node, name: str) -> bool:
    return any(_call(item, name) for item in ast.walk(node))


def transformed_function(module, task: str) -> tuple[ast.Module, str, str]:
    """Return executable AST, readable source, and original function hash."""
    original = inspect.getsource(getattr(module, HOSTS[task]))
    tree = ast.parse(original)
    function = tree.body[0]
    loops = [node for node in function.body if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "step"
             and _contains(node, "opt_d.step") and _contains(node, "opt_g.step")]
    if len(loops) != 1:
        raise ValueError("frozen host must have one recognizable game-update loop")
    loop = loops[0]
    if task == "mode_hold":
        start = next(i for i, node in enumerate(loop.body)
                     if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "real")
        end = next(i + 1 for i, node in enumerate(loop.body)
                   if isinstance(node, ast.Expr) and _call(node.value, "opt_g.step"))
    else:
        start = next(i for i, node in enumerate(loop.body)
                     if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "context")
        end = next(i + 1 for i, node in enumerate(loop.body)
                   if isinstance(node, ast.Try) and _contains(node, "opt_g.step"))
    block = loop.body[start:end]
    for name in ("opt_d.step", "opt_g.step", "d_loss.backward", "g_loss.backward"):
        if sum(_call(node, name) for statement in block for node in ast.walk(statement)) != 1:
            raise ValueError(f"frozen gradient block changed: {name}")
    if any(_contains(node, "checkpoint") or _contains(node, "noise_policy.set_step") for node in block):
        raise ValueError("measurement or noise clock entered gradient block")

    class ScheduleOnce(ast.NodeTransformer):
        def visit_Expr(self, node):
            if _call(node.value, "schedule_optimizer"):
                return ast.If(test=ast.Compare(left=ast.Name(id="_extra_phase", ctx=ast.Load()),
                                              ops=[ast.Eq()], comparators=[ast.Constant(0)]),
                              body=[node], orelse=[])
            return self.generic_visit(node)

    wrapped = ast.For(
        target=ast.Name(id="_extra_phase", ctx=ast.Store()),
        iter=ast.Call(func=ast.Attribute(value=ast.Name(id="_extra_state", ctx=ast.Load()),
                                        attr="phases", ctx=ast.Load()),
                      args=[ast.Name(id=name, ctx=ast.Load()) for name in ("step", "opt_d", "opt_g")],
                      keywords=[]),
        body=[ScheduleOnce().visit(node) for node in block], orelse=[],
    )
    loop.body[start:end] = [wrapped]
    ast.fix_missing_locations(tree)

    # Inverting precisely these two edits must recover the original function.
    restored = deepcopy(tree)
    restored_loop = next(node for node in restored.body[0].body if isinstance(node, ast.For)
                         and isinstance(node.target, ast.Name) and node.target.id == "step")
    extra = restored_loop.body[start]

    class UnwrapSchedule(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.unparse(node.test) == "_extra_phase == 0":
                if len(node.body) != 1 or node.orelse or not _call(node.body[0].value, "schedule_optimizer"):
                    raise ValueError("unrecognized schedule transformation")
                return node.body[0]
            return self.generic_visit(node)

    restored_loop.body[start:start + 1] = [UnwrapSchedule().visit(node) for node in extra.body]
    if ast.dump(restored, include_attributes=False) != ast.dump(ast.parse(original), include_attributes=False):
        raise ValueError("source transformation changed more than the declared update block")
    return tree, ast.unparse(tree) + "\n", sha(original.encode())


class ExtraAdamRecorder:
    def __init__(self, method: str):
        if method not in METHODS:
            raise ValueError("unsupported game optimizer")
        self.method, self.evaluations = method, METHODS[method]
        self.phase = None
        self.optimizers = None
        self.pending = {}
        self.base = {}
        self.point = {}
        self.rows = {}
        self.outer_steps = 0
        self.joint_points_verified = 0
        self.base_restores_verified = 0
        self.host_source = None

    @torch.no_grad()
    def _copy_parameters(self):
        return {p: p.detach().clone() for optimizer in self.optimizers
                for group in optimizer.param_groups for p in group["params"]}

    def phases(self, step, opt_d, opt_g):
        if self.phase is not None or opt_d is opt_g:
            raise RuntimeError("invalid nested game update")
        if self.optimizers is None:
            self.optimizers = (opt_d, opt_g)
            params = [p for optimizer in self.optimizers for group in optimizer.param_groups for p in group["params"]]
            if len(params) != len(set(params)):
                raise RuntimeError("game players share parameters")
            self.rows = {optimizer: dict(role=role, calls=0, rates=[], diagnostics=[])
                         for role, optimizer in zip(("d", "g"), self.optimizers)}
        if self.optimizers != (opt_d, opt_g):
            raise RuntimeError("game optimizer changed during episode")
        self.base = self._copy_parameters()
        for phase in range(self.evaluations):
            self.phase = phase
            self.pending = {}
            self.point = self._copy_parameters()
            yield phase
            if self.pending:
                raise RuntimeError("incomplete joint game gradient")
            if any(self.rows[optimizer]["calls"] != (self.outer_steps * self.evaluations + phase + 1)
                   for optimizer in self.optimizers):
                raise RuntimeError("missing optimizer halfstep")
        self.phase = None
        self.base, self.point = {}, {}
        self.outer_steps += 1

    @torch.no_grad()
    def step(self, optimizer, ordinary_step, closure=None):
        if self.phase is None or optimizer not in self.rows or closure is not None:
            raise RuntimeError("optimizer call is outside the declared game gradient")
        if optimizer in self.pending:
            raise RuntimeError("duplicate optimizer gradient")
        expected = self.optimizers[len(self.pending)]
        if optimizer is not expected:
            raise RuntimeError("host gradient order changed")
        gradients = {}
        for group in optimizer.param_groups:
            if (group.get("weight_decay", 0) or any(group.get(key, False) for key in
                    ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"))):
                raise ValueError("ExtraAdam requires ordinary CPU Adam groups")
            if not isinstance(group["lr"], (float, int)) or not math.isfinite(group["lr"]) or group["lr"] <= 0:
                raise ValueError("invalid group rate")
            for parameter in group["params"]:
                if parameter.grad is None:
                    raise RuntimeError("frozen host parameter missed a gradient")
                gradients[parameter] = parameter.grad.detach().clone()
        self.pending[optimizer] = gradients
        if optimizer is self.optimizers[0]:
            return None
        if any(not torch.equal(parameter, saved) for parameter, saved in self.point.items()):
            raise RuntimeError("a player moved before both game gradients were captured")
        self.joint_points_verified += 1
        if self.phase == 1:
            for parameter, original in self.base.items():
                parameter.copy_(original)
            if any(not torch.equal(parameter, saved) for parameter, saved in self.base.items()):
                raise RuntimeError("failed to restore original parameters for final step")
            self.base_restores_verified += 1
        for target in self.optimizers:
            for parameter, gradient in self.pending[target].items():
                parameter.grad = gradient
            ordinary_step(target)
            row = self.rows[target]
            rates = [float(group["lr"]) for group in target.param_groups]
            if row["rates"] and rates != row["rates"][0]:
                raise RuntimeError("declared constant group rates changed")
            row["rates"].append(rates)
            row["calls"] += 1
            diagnostic = []
            for group in target.param_groups:
                parameters = group["params"]
                count = sum(p.numel() for p in parameters)
                gradient = sum(float(self.pending[target][p].double().square().sum()) for p in parameters)
                move = sum(float((p - self.base[p]).double().square().sum()) for p in parameters)
                diagnostic.append(dict(parameters=count, gradient_rms=math.sqrt(gradient / count),
                                       move_from_outer_base_rms=math.sqrt(move / count)))
            row["diagnostics"].append(diagnostic)
        self.pending = {}
        return None

    def receipt(self):
        records = []
        for optimizer, row in self.rows.items():
            groups = []
            for index, group in enumerate(optimizer.param_groups):
                role = "d" if row["role"] == "d" else "prior" if group.get("_comparison_prior") else "g"
                groups.append(dict(role=role, parameters=sum(p.numel() for p in group["params"]),
                                   lr=float(group["lr"]), betas=list(group["betas"]),
                                   moment_steps=[int(optimizer.state[p]["step"]) for p in group["params"]]))
            records.append(dict(role=row["role"], calls=row["calls"], groups=groups,
                                rates=row["rates"], diagnostics=row["diagnostics"]))
        return dict(method=self.method, paper=PAPER, outer_steps=self.outer_steps,
                    gradient_evaluations_per_player_per_outer_step=self.evaluations,
                    extra_gradient_evaluations_per_player_per_outer_step=self.evaluations - 1,
                    host_sampler_block_reexecuted_per_gradient_evaluation=True,
                    moments_updated_at_every_gradient_evaluation=True,
                    output_weights="final base-point update, no iterate averaging",
                    joint_points_verified=self.joint_points_verified,
                    base_restores_verified=self.base_restores_verified,
                    common_gate_eligible=False, host_source=self.host_source,
                    optimizers=records)


@contextmanager
def extra_adam(task: str, method: str):
    from benchmarks.locked_shared import mode_hold, trajectory

    module = {"mode_hold": mode_hold, "trajectory": trajectory}[task]
    tree, source, original_sha = transformed_function(module, task)
    recorder = ExtraAdamRecorder(method)
    recorder.host_source = dict(task=task, original_function_sha256=original_sha,
                                generated_function_sha256=sha(source.encode()),
                                source_transform_inverse_verified=True)
    original_step = torch.optim.Adam.step

    def patched_step(optimizer, closure=None):
        return recorder.step(optimizer, original_step, closure)

    with ExitStack() as stack:
        stack.enter_context(patch.dict(module.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, f"<extra-adam-{task}>", "exec"), module.__dict__, namespace)
        stack.enter_context(patch.object(module, HOSTS[task], namespace[HOSTS[task]]))
        stack.enter_context(patch.object(torch.optim.Adam, "step", patched_step))
        yield recorder, source
