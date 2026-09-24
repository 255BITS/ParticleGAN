"""Scratch binding of the frozen PR84 critic refinement to residual_student.

The host's complete D-then-G block is structurally transformed, leaving its
native supervised residual and cover losses untouched. Each fit closure sees
eight repeated native 12-pair batches with the four *separate* D input-noise
draws cached in actual call order. The bank and fit do not advance training
RNG, optimizer moments, data clocks or learning rates. This is an unqualified
research binding; no residual_student gate result is implied.
"""

import ast
from contextlib import ExitStack, contextmanager
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import torch

from benchmarks.locked_shared.hosts import residual_student
from particlegan.gan_loss import GANLoss
from reports.toy100 import extra_adam_scratch as source_adapter
from reports.toy100 import pr84_critic_refinement as warm
from reports.toy100 import pr84_critic_refinement_cold as cold
from reports.toy100 import pr84_critic_relaxation as fit


METHOD = "pr84_alternating_bounded_empirical_critic_refinement_residual_student"
HOST_FUNCTION_SHA256 = "436cecfac68a59864c1da383100e870016150b68e1f9759c9030eb6938afc136"
NATIVE_ROWS = 12


def _alias(local):
    if "generator" in local and local["generator"] is not local.get("head"):
        raise ValueError("residual host generator alias changed")
    if "head" not in local or "critic" not in local:
        raise ValueError("residual host locals changed")
    return {**local, "generator": local["head"]}


@torch.no_grad()
def fixed_bank(local):
    """Cache the fixed conditional training law from cloned noise streams."""
    local = _alias(local)
    policy = local.get("noise_policy")
    if policy is None or policy.output_scale is not None:
        raise ValueError("residual refinement requires fixed output-noise scale")
    gan, regularizer = local["gan"], local["regularizer"]
    if (gan.mode != "rp" or gan.loss_type != "logistic" or regularizer.arm != "b_cap"
            or regularizer.lazy_k != 1 or regularizer.method != "autograd"
            or regularizer.target_anneal != "none"):
        raise ValueError("residual refinement requires the declared sharp Rp b_cap D objective")
    head = getattr(local["head"], "model", local["head"])
    critic = getattr(local["critic"], "model", local["critic"])
    slow, real = local["slow"].detach(), local["paired"].detach()
    if (not isinstance(head, residual_student.ResidualHead)
            or not isinstance(critic, residual_student._Critic)
            or slow.shape[0] != NATIVE_ROWS or real.shape[0] != NATIVE_ROWS
            or next(head.parameters()).device.type != "cpu"):
        raise ValueError("residual_student architecture or native 12-row batch changed")
    input_stream = cold._clone_stream(policy.input_stream)
    output_stream = cold._clone_stream(policy.output_stream) if policy.output_stream is not None else None
    rows = []
    with torch.random.fork_rng(devices=[]):
        for _ in range(warm.BANK_BATCHES):
            clean = head(slow, local["prior"].z)
            if policy.output_sigma:
                noise = (torch.randn_like(clean) if output_stream is None else
                         torch.randn(clean.shape, generator=output_stream,
                                     dtype=clean.dtype, device=clean.device))
                fake = clean + policy.output_sigma * noise
            else:
                fake = clean
            perturbations = ([torch.randn(real.shape, generator=input_stream,
                                          dtype=real.dtype, device=real.device)
                              for _ in range(4)] if policy.input_sigma else None)
            rows.append(dict(real=real, fake=fake.detach(), slow=slow,
                             input_noise=perturbations, task="residual_student",
                             input_sigma=policy.input_sigma, penalty_step=local["step"]))
    bank = dict(real=torch.cat([row["real"] for row in rows]),
                fake=torch.cat([row["fake"] for row in rows]),
                slow=torch.cat([row["slow"] for row in rows]),
                input_noise=None if rows[0]["input_noise"] is None else [
                    torch.cat([row["input_noise"][call] for row in rows])
                    for call in range(4)],
                task="residual_student", input_sigma=policy.input_sigma,
                penalty_step=local["step"])
    return rows[0], bank


class ResidualRefinementRecorder(cold.ColdCriticRefinementRecorder):
    def __init__(self, *, start_step=0, refinement=True):
        super().__init__(task="residual_student", start_step=start_step, refinement=refinement)

    def phases(self, step, opt_d, opt_g, local):
        for phase in super().phases(step, opt_d, opt_g, _alias(local)):
            yield phase
        if self._active:
            self.refinement_records[-1]["conditioning"] = "unchanged native slow coordinates"

    def receipt(self):
        value = super().receipt()
        value.update(method=METHOD, scratch_optimizer_policy=METHOD,
                     refinement_scope="CPU residual_student, fixed output scale, native sharp conditional D objective",
                     bank_batch_size=NATIVE_ROWS,
                     gradient_query_size_note="three native 12-row host batches/player, one 12-row D parity check, eight-by-12-pair fit closures",
                     adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     cold_adapter_sha256=hashlib.sha256(Path(cold.__file__).read_bytes()).hexdigest(),
                     native_g_objective="unaltered adversarial + cover + particle + spread + supervised residual")
        return value


@contextmanager
def pr84_residual_refinement(*, start_step=0, refinement=True):
    """Yield ``(recorder, generated_source)`` for one unchanged host episode."""
    with patch.dict(source_adapter.HOSTS, {"residual_student": "train"}):
        tree, _, original_sha = source_adapter.transformed_function(residual_student, "residual_student")
    if original_sha != HOST_FUNCTION_SHA256:
        raise RuntimeError("residual_student training function source changed")
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "phases"]
    if len(calls) != 1:
        raise RuntimeError("expected exactly one transformed game-phase iterator")
    calls[0].args.append(ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]))
    ast.fix_missing_locations(tree)
    generated = ast.unparse(tree) + "\n"
    recorder = ResidualRefinementRecorder(start_step=start_step, refinement=refinement)
    recorder.host_source = dict(task="residual_student", original_function_sha256=original_sha,
                                generated_function_sha256=source_adapter.sha(generated.encode()))
    ordinary_step = torch.optim.Adam.step
    original_d_loss = GANLoss.d_loss

    def observed_d_loss(gan, real_logits, fake_logits):
        value = original_d_loss(gan, real_logits, fake_logits)
        if recorder.phase == 0 and recorder.advantage is None:
            recorder.advantage = math.log(2) - float(value.detach())
        return value

    with ExitStack() as stack:
        stack.enter_context(patch.dict(residual_student.__dict__, {"_extra_state": recorder}))
        namespace = {}
        exec(compile(tree, "<pr84-residual-student>", "exec"), residual_student.__dict__, namespace)
        stack.enter_context(patch.object(residual_student, "train", namespace["train"]))
        stack.enter_context(patch.object(torch.optim.Adam, "step",
            lambda optimizer, closure=None: recorder.step(optimizer, ordinary_step, closure)))
        stack.enter_context(patch.object(GANLoss, "d_loss", observed_d_loss))
        stack.enter_context(patch.object(warm, "fixed_bank", fixed_bank))
        stack.enter_context(patch.object(fit, "d_loss", cold.cached_d_loss))
        yield recorder, generated
