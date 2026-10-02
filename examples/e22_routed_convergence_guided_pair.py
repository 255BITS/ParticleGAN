"""One guidance-pair transfer of the held, exactly reachable routed toy.

Every native update/checkpoint/restore/score helper remains the held helper.
Only the host execution is changed to two shared-parameter conditional halves
and their final CFG=3 combination. Derived targets and whitening follow it.
This is a standalone particle-cloud diagnostic, outside Forge qualification.
"""

from copy import deepcopy
import hashlib
from inspect import unwrap
import math
import marshal
from pathlib import Path
import sys
from types import FunctionType

import torch

if __package__:
    from . import e22_routed_convergence as baseline
    from . import e22_routed_convergence_neutral as neutral
    from . import e22_routed_convergence_rotated_teacher as parent
else:
    import e22_routed_convergence as baseline
    import e22_routed_convergence_neutral as neutral
    import e22_routed_convergence_rotated_teacher as parent


TASK = "routed_convergence_guided_pair_v1"
ARMS, SITES, RHO = parent.ARMS, parent.SITES, parent.RHO
AUDIT_SHA256 = parent.AUDIT_SHA256
GUIDANCE = 3.
CFG_BUFFERS = ("cfg_source_projection", "cfg_unconditional_source")


class GuidedHost(baseline.Host):
    """The held two sites, called on [conditional B, unconditional B]."""

    def __init__(self, particle=False, *, source_projection, unconditional_source):
        super().__init__(particle)
        self.register_buffer("cfg_source_projection", source_projection.detach().clone())
        self.register_buffer("cfg_unconditional_source", unconditional_source.detach().clone())

    def half_inputs(self, context):
        conditional = context[..., :baseline.WIDTH]
        source = context[:, 0, baseline.WIDTH:baseline.WIDTH + 768]
        # Reuse the exact conditional input panel. Replacing only its fixed
        # source contribution gives the other half the same latent/time input;
        # no independent latent, time, source or Gaussian draw occurs here.
        replacement = (self.cfg_unconditional_source[None] - source) @ self.cfg_source_projection.T
        unconditional = conditional + replacement[:, None]
        return torch.cat((conditional, unconditional), dim=0)

    @staticmethod
    def combine(halves):
        conditional, unconditional = halves.chunk(2, dim=0)
        # Match Supra's actual FP32 arithmetic order; mathematically 3c - 2u.
        return unconditional + GUIDANCE * (conditional - unconditional)

    def forward(self, context):
        value = self.half_inputs(context)
        hidden = self.first(value).to(torch.bfloat16).tanh().float()
        return self.combine(self.second(hidden))

    def forward_routed(self, context, router, candidate, routing):
        value = self.half_inputs(context)
        first_logits = router.first_query(value) @ candidate.table.T / math.sqrt(baseline.Z_DIM)
        conditional, unconditional = first_logits.chunk(2, dim=0)
        first_codes = routing.mix("first", torch.stack((conditional, unconditional), dim=1))
        first_codes = torch.cat((first_codes[:, 0], first_codes[:, 1]), dim=0)
        hidden = self.first(value, first_codes).to(torch.bfloat16).tanh().float()
        second_logits = router.second_query(hidden) @ candidate.table.T / math.sqrt(baseline.Z_DIM)
        conditional, unconditional = second_logits.chunk(2, dim=0)
        second_codes = routing.mix("second", torch.stack((conditional, unconditional), dim=1))
        second_codes = torch.cat((second_codes[:, 0], second_codes[:, 1]), dim=0)
        return self.combine(self.second(hidden, second_codes))


def guided_model_forward(models, context, candidate, routing):
    return models["generator"].forward_routed(context, models["router"], candidate, routing)


def factory_namespace(data):
    """Bind new callbacks around the exact held factory code, without mutation.

    FunctionType receives a fresh globals dictionary. Neither a held module's
    globals nor the native optimizer implementation is changed or copied.
    The authoritative checkpoint law records this adapter and held file SHA.
    """
    def host_factory(particle=False):
        return GuidedHost(particle, source_projection=data["source_projection"],
                          unconditional_source=data["sources"].mean(0))

    namespace = dict(vars(baseline))
    namespace.update(Host=host_factory, model_forward=guided_model_forward)
    factory = FunctionType(baseline.make_loop.__code__, namespace, baseline.make_loop.__name__,
                           baseline.make_loop.__defaults__, baseline.make_loop.__closure__)
    factory.__kwdefaults__ = deepcopy(baseline.make_loop.__kwdefaults__)
    return factory


def factory_binding_manifest():
    return {
        "id": "isolated_held_code_namespace_v1",
        "held_function": "examples.e22_routed_convergence.make_loop",
        "held_file_sha256": hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
        "held_code_object_sha256": hashlib.sha256(marshal.dumps(baseline.make_loop.__code__)).hexdigest(),
        "code_object_runtime": {"python": sys.version, "co_filename": baseline.make_loop.__code__.co_filename},
        "bindings": {"Host": "GuidedHost bound to frozen projection/mean-source data",
                     "model_forward": "guided_model_forward"},
        "held_globals_mutated": False,
        "update_checkpoint_restore_score": "held examples.e22_routed_convergence functions, unchanged",
    }


@torch.no_grad()
def make_guided_data(original=None):
    original = parent.make_rotated_data() if original is None else original
    if original["digest"] != baseline.digest({key: value for key, value in original.items() if key != "digest"}):
        raise ValueError("parent data changed after its digest was bound")
    rotation = original.get("teacher_span_rotation", {})
    if (rotation.get("id") != parent.TASK or rotation.get("rho") != RHO
            or rotation.get("audit_sha256") != AUDIT_SHA256 or "guided_pair" in original):
        raise ValueError("expected the held rotated reachable teacher")
    data = deepcopy(original)
    data.pop("digest")
    mean_source = data["sources"].mean(0)
    for role in ("initial_ordinary", "initial_particle", "teacher"):
        data[role].update(cfg_source_projection=data["source_projection"].clone(),
                          cfg_unconditional_source=mean_source.clone())
    with torch.random.fork_rng(devices=[]):
        teacher = GuidedHost(source_projection=data["source_projection"], unconditional_source=mean_source)
        ordinary = GuidedHost(source_projection=data["source_projection"], unconditional_source=mean_source)
    teacher.load_state_dict(data["teacher"], strict=True)
    ordinary.load_state_dict(data["initial_ordinary"], strict=True)
    teacher.eval().requires_grad_(False)
    ordinary.eval().requires_grad_(False)
    for pool in baseline.SPLITS:
        data[pool]["targets"] = baseline.batched_host(teacher, data[pool]["context"])
        data[pool]["base"] = baseline.batched_host(ordinary, data[pool]["context"])
    data["raw_scale"] = (data["fit"]["targets"] - data["fit"]["base"]).std(dim=(0, 1))
    data["scale"] = data["raw_scale"].clamp_min(.04)
    data["guided_pair"] = {
        "id": TASK, "guidance": GUIDANCE, "parent_data_digest": original["digest"],
        "unconditional_source": "fixed FP32 mean of the unchanged six parent source vectors",
        "half_order": "physical conditional B then unconditional B; routed mix stacks [B,2,T,N]",
        "half_inputs": "conditional input stays exact; unconditional adds (mean-source)@source_projection.T",
        "latent_time": "same held latent/time input, no new draws; FP32 source-replacement arithmetic",
        "combination": "unconditional + 3*(conditional-unconditional), FP32; mathematically 3conditional-2unconditional",
        "derived_changes": "guided teacher outputs, guided frozen base, fit residual coordinate scale",
        "buffers": CFG_BUFFERS,
        "factory_adapter": factory_binding_manifest(),
    }
    data["digest"] = baseline.digest(data)
    return data


def make_guided_loop(arm, data, *, bindings=None):
    if arm not in ARMS:
        raise ValueError("unknown guidance-pair arm")
    if data.get("guided_pair", {}).get("id") != TASK or data["guided_pair"].get("guidance") != GUIDANCE:
        raise ValueError("expected the declared fixed guidance-pair law")
    rotation = data.get("teacher_span_rotation", {})
    if (rotation.get("id") != parent.TASK or rotation.get("rho") != RHO
            or rotation.get("audit_sha256") != AUDIT_SHA256):
        raise ValueError("expected the held rotated reachable teacher")
    for role in ("initial_ordinary", "initial_particle", "teacher"):
        if not torch.equal(data[role]["cfg_unconditional_source"], data["sources"].mean(0)):
            raise ValueError("unconditional source must be the exact fixed parent-source mean")
        if not torch.equal(data[role]["cfg_source_projection"], data["source_projection"]):
            raise ValueError("CFG source projection must match the unchanged fixed source projection")
    actual = neutral.make_neutral_data(data) if arm == ARMS[2] else data
    native_arm = ARMS[1] if arm == ARMS[2] else arm
    loop = factory_namespace(actual)(native_arm, actual, bindings=bindings)
    loop.law.update(task=TASK, arm=arm, guided_pair=deepcopy(data["guided_pair"]),
                    teacher_span_rotation=deepcopy(data["teacher_span_rotation"]),
                    common_task_data_digest=data["digest"],
                    factory_adapter=factory_binding_manifest())
    if arm == ARMS[2]:
        loop.law["intervention"] = deepcopy(actual["intervention"])
    return loop


@torch.no_grad()
def reachability_witness(data):
    namespace = dict(vars(baseline))
    namespace["make_loop"] = make_guided_loop
    held = unwrap(baseline.reachability_witness)
    witness = FunctionType(held.__code__, namespace, held.__name__, held.__defaults__, held.__closure__)
    return witness(data)


@torch.no_grad()
def initial_difference_witness(original, candidate_data=None):
    """Check the sole neutral initialization change in actual FAST/EMA owners."""
    candidate_data = neutral.make_neutral_data(original) if candidate_data is None else candidate_data
    expected_data = neutral.make_neutral_data(original)
    if baseline.digest(candidate_data) != baseline.digest(expected_data):
        raise ValueError("neutral data differs from the sole declared H/b initialization change")
    reference = make_guided_loop(ARMS[1], original)
    candidate = make_guided_loop(ARMS[2], original)
    expected, actual = deepcopy(reference.policy.state_dict()), candidate.policy.state_dict()
    changed = []
    for family, role in (("models", "generator"), ("averages", "average_generator")):
        for name, value in expected[family]["generator"].items():
            if name.endswith("bridge.weight"):
                value[:, :baseline.RANK].zero_()
                changed.append(role + "." + name + "[H]")
            elif name.endswith("bridge.bias"):
                value.zero_()
                changed.append(role + "." + name)
    if baseline.digest(expected) != baseline.digest(actual):
        raise AssertionError("neutral guidance arm changed more than initial H/b")
    for site in SITES:
        if getattr(candidate.G, site).bridge.weight[:, baseline.RANK:].count_nonzero() == 0:
            raise AssertionError("neutral initialization removed sampled particle C")
    return {"changed_initial_coordinates": changed, "other_initial_owners_equal": True,
            "teacher_data_scales_equal": True, "native_recipe_equal": True,
            "all_non_generator_native_initial_state_equal": True, "sampled_C_nonzero": True,
            "original_data_digest": original["digest"], "neutral_data_digest": candidate_data["digest"],
            "cfg_fast_ema_frozen_buffers_exact": True}


def initialization_witness(data):
    return initial_difference_witness(data)


@torch.no_grad()
def feasibility_witness(original=None):
    original = parent.make_rotated_data() if original is None else original
    before, rng = baseline.digest(original), torch.get_rng_state().clone()
    data = make_guided_data(original)
    for role in ("initial_ordinary", "initial_particle", "teacher"):
        for name, value in original[role].items():
            if not torch.equal(value, data[role][name]):
                raise AssertionError("guidance changed an existing student/teacher tensor")
    for key in ("sources", "source_projection", "initialization", "teacher_span_rotation"):
        if baseline.digest(data[key]) != baseline.digest(original[key]):
            raise AssertionError("guidance changed held source/basis/initialization law")
    for pool in baseline.SPLITS:
        for key in ("context", "subjects", "times", "latent_ids"):
            if not torch.equal(data[pool][key], original[pool][key]):
                raise AssertionError("guidance changed held input/time/latent panels")
    initialization_witness(data)
    reachable = reachability_witness(data)
    if baseline.digest(original) != before or not torch.equal(torch.get_rng_state(), rng):
        raise AssertionError("guidance preparation changed parent data or global RNG")
    return {
        "optimizer_updates": 0, "guidance": GUIDANCE, "rho": RHO,
        "parent_data_digest": original["digest"], "data_digest": data["digest"],
        "existing_student_and_teacher_tensors_unchanged": True,
        "conditional_sources_and_input_panels_unchanged": True,
        "unconditional_source_is_exact_parent_mean": True,
        "neutral_fast_ema_initial_change_only_H_b": True,
        "global_rng_unchanged": True, "reachability": reachable,
        "raw_scale_min": float(data["raw_scale"].min()),
        "raw_scale_max": float(data["raw_scale"].max()),
        "scale_floor_coordinates": int((data["raw_scale"] < .04).count_nonzero()),
        "target_and_base_dtypes": {pool: {key: str(data[pool][key].dtype) for key in ("targets", "base")}
                                   for pool in baseline.SPLITS},
    }
