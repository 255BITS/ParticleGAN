"""ROOT-paid, CPU-only copied existing MoG AE initializer metadata check.

Import is inert. The actual initializer wrapper is inspected but never called;
no model/Recipe/policy/owner, tensor, draw or scientific context is constructed.
"""
from __future__ import annotations
from contextlib import ExitStack, contextmanager
import inspect
import json
import os
from pathlib import Path
import sys

SCHEMA = "forge_atlas_existing_mog_ae_initializer_preflight_v1"
ADAPTER_SHA256 = '179ef8632da9d6f03d9561f276c86ad2ce87848c90db89251b4ccb7a88fee42b'
TASK_ID = "ae_gan_hold"


@contextmanager
def _fences(torch, numpy):
    """Allow static library/type imports, forbid all physical scientific calls."""
    from unittest.mock import patch
    counts = {name: 0 for name in ("model_constructors", "recipe_constructors", "policy_constructors",
        "owner_constructors", "initializer_calls", "forwards", "updates", "evaluation_draws",
        "array_constructors", "context_entries")}
    def refuse(kind):
        def denied(*args, **kwargs):
            counts[kind] += 1
            raise ValueError("initializer metadata proof forbids " + kind)
        return denied
    with ExitStack() as stack:
        def guard(owner, name, kind):
            if hasattr(owner, name):
                stack.enter_context(patch.object(owner, name, refuse(kind)))
        guard(torch.nn.Module, "__init__", "model_constructors")
        guard(torch.nn.Module, "_call_impl", "forwards")
        guard(torch.optim.Optimizer, "__init__", "updates")
        guard(torch.optim.Optimizer, "step", "updates")
        for name in ("rand", "randn", "randint", "rand_like", "randn_like", "randperm", "multinomial", "normal", "bernoulli", "manual_seed", "seed"):
            guard(torch, name, "evaluation_draws")
        for name in ("tensor", "as_tensor", "zeros", "ones", "empty", "full", "arange", "linspace", "eye"):
            guard(torch, name, "array_constructors")
        for name in ("normal_", "uniform_", "random_", "bernoulli_", "exponential_", "cauchy_", "log_normal_", "geometric_"):
            guard(torch.Tensor, name, "evaluation_draws")
        for name in ("random", "random_sample", "rand", "randn", "randint", "choice", "normal", "uniform", "shuffle", "permutation", "seed", "default_rng"):
            guard(numpy.random, name, "evaluation_draws")
        for name in ("no_grad", "enable_grad", "inference_mode", "set_grad_enabled"):
            context = getattr(torch, name, None)
            if context is not None:
                guard(context, "__enter__", "context_entries")
        yield counts, guard



def copied_initializer_preflight(request, task, root):
    """Actual metadata only. ROOT persists and joins this before reservation."""
    from . import atlas_existing_mog_ae as adapter
    from .contracts import stable_hash
    root = Path(root).resolve()
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or Path(__file__).resolve() != root / "experiments/forge/atlas_existing_mog_ae_probe.py"
            or task["id"] != TASK_ID or task["resources"]["cpu_threads"] != 1
            or task["resources"]["gpus"] != 0):
        raise ValueError("owned copied existing MoG AE initializer proof requires CPU1/CUDA hidden")
    tasks = request["tasks"]
    if (not isinstance(tasks, dict) or TASK_ID not in tasks
            or adapter.canonical(tasks[TASK_ID]) != adapter.canonical(task)):
        raise ValueError("initializer proof task must be the exact ordinary frozen request task")
    adapter.source_guard(request, root)
    binding = adapter.resolve_binding(root, request["candidate"], task, request["protocol"])
    if binding["adapter_sha256"] != ADAPTER_SHA256:
        raise ValueError("exact reviewed existing MoG AE initializer guard required")
    import torch
    import numpy as np
    if torch.get_num_threads() != 1:
        raise ValueError("initializer metadata imports require CPU1 environment")
    with _fences(torch, np) as (counts, guard):
        from particlegan.init import deterministic_orthogonal_
        from particlegan.recipes import Recipe
        from particlegan.policy import UpdatePolicy
        guard(Recipe, "__init__", "recipe_constructors")
        guard(UpdatePolicy, "__init__", "policy_constructors")
        guard(adapter.AEOwner, "__init__", "owner_constructors")
        value = deterministic_orthogonal_
        unwrapped = inspect.unwrap(value)
        protected = {value.__code__, unwrapped.__code__, adapter.construct_owner.__code__,
            adapter.run_case.__code__, adapter.run_behavior.__code__,
            adapter.owner_initial_receipt.__code__, adapter.derive_host_train.__code__}
        previous_profile = sys.getprofile()
        def profile(frame, event, arg):
            if event == "call" and frame.f_code in protected:
                kind = ("initializer_calls" if frame.f_code in {value.__code__, unwrapped.__code__}
                        else "owner_constructors")
                counts[kind] += 1
                raise ValueError("initializer metadata proof entered forbidden " + kind)
            if previous_profile is not None:
                previous_profile(frame, event, arg)
        sys.setprofile(profile)
        try:
            adapter.source_guard(request, root)
            adapter._loaded_source(root, value, "particlegan/init.py", no_grad=torch.no_grad)
            adapter.source_guard(request, root)
        finally:
            sys.setprofile(previous_profile)
        if any(counts.values()):
            raise ValueError("initializer metadata proof attempted a physical scientific call")
        receipt = {"schema": SCHEMA, "status": "PASS_METADATA_ONLY", "task_id": TASK_ID,
            "source_origin_commit": request["source"]["origin_commit"],
            "source_digest": request["source"]["digest"],
            "frozen_request_digest": stable_hash(request), "task_digest": stable_hash(task),
            "candidate_digest": stable_hash(request["candidate"]),
            "binding_digest": stable_hash(binding), "candidate_id": request["candidate"].get("id"),
            "adapter_sha256": ADAPTER_SHA256,
            "source_contract_sha256": binding["source_contract_sha256"],
            "initializer_pin": binding["source_contract"]["files"]["particlegan/init.py"],
            "source_export_identity": {"module": value.__module__, "name": value.__name__,
                "qualname": value.__qualname__, "single_public_no_grad_wrapper": value.__wrapped__ is unwrapped},
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "operation_counts": dict(counts), "actual_owner_initialized": False,
            "science_admission_granted": False, "accepted_numeric_credit": False}
    return receipt


def main():
    if len(sys.argv) != 3 or sys.argv[1] != "--request":
        raise SystemExit("Use ROOT's paid isolated CPU proof: --request frozen-request.json")
    from .contracts import read_json
    request = read_json(sys.argv[2])
    tasks = request["tasks"]
    if not isinstance(tasks, dict) or TASK_ID not in tasks:
        raise ValueError("one exact ID-keyed frozen existing MoG AE task required")
    print(json.dumps(copied_initializer_preflight(request, tasks[TASK_ID],
                    Path(__file__).resolve().parents[2]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
