"""Reproduce the zero-update basis and game-tangent diagnostic on private clones.

PYTHONPATH=. python examples/diagnose_e22_routed_convergence_basis.py \
    --parent runs/routed-convergence-v1 --out runs/routed-basis-audit-reproduced

This inspects the teacher-aligned diagnostic family; it gives no Forge or Supra
quality credit. H/b are zeroed only on a private initial FAST probe. C, particles,
native controls, and all parent artifacts remain intact. There are no optimizer,
critic, controller, or training updates, and no output-error objective.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import particlegan

from e22_routed_convergence import (
    RANK, WIDTH, checkpoint, digest, forward, make_data, make_loop, restore,
)


ROOT = Path(__file__).resolve().parents[1]
SITES = ("first", "second")
NATIVE_ARMS = ("ordinary_native_game", "particle_native_game")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(value):
    return float(value.detach().double().square().mean().sqrt())


def cosine(a, b):
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


def basis_error(features, target):
    features = features.detach().double().flatten(0, -2)
    target = target.detach().double().flatten(0, -2)
    solution = torch.linalg.lstsq(features, target, driver="gelsd").solution
    return rms(features @ solution - target) / rms(target)


def capture(loop, context):
    records, handles = {}, []
    for name in SITES:
        def hook(owner, args, name=name):
            hidden = owner.down(args[0].float())
            if hasattr(owner, "bridge"):
                extra = F.linear(hidden, owner.bridge.weight[:, :RANK], owner.bridge.bias).tanh()
                gate = F.linear(args[1], owner.bridge.weight[:, RANK:]).tanh()
            else:
                extra, gate = torch.zeros_like(hidden), torch.zeros_like(hidden)
            records[name] = (hidden, extra, gate, hidden + extra + hidden * gate)
        handles.append(getattr(loop.G, name).register_forward_pre_hook(hook))
    try:
        return forward(loop, context), records
    finally:
        for handle in handles:
            handle.remove()


class Wrapper(nn.Module):
    def __init__(self, loop):
        super().__init__()
        self.G, self.loop = loop.G, loop

    def forward(self, context):
        return forward(self.loop, context)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_table(path, basis, tangent):
    lines = ["Zero-update private-clone diagnostic; teacher-aligned family only.", "",
             "Initial outputs remain exact; these are basis and autograd comparisons,",
             "not a convergence result or a finite derivative of BF16 rounding.", "",
             "| Fit basis | Site | H modulation / h | h recovery residual | Gate RMS |",
             "|---|---|---:|---:|---:|"]
    for label, pools in basis.items():
        for site, values in pools["fit"].items():
            lines.append(f"| {label} | {site} | {values['hidden_modulation_over_hidden']:.6f} | "
                         f"{values['recover_hidden_from_phi_relative_rms']:.6f} | {values['gate_rms']:.6f} |")
    lines.extend(["", "| Fixed learned judge | Direction | Particle cosine to ordinary | H+b neutral cosine |",
                  "|---|---|---:|---:|"])
    for judge, directions in tangent.items():
        for direction, values in directions.items():
            lines.append(f"| {judge} | {direction} | {values['particle']['cosine_to_ordinary_output_tangent']:.6f} | "
                         f"{values['H_bias_neutral']['cosine_to_ordinary_output_tangent']:.6f} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; preserve it and choose a fresh directory")
    if Path(particlegan.__file__).resolve().parent != ROOT / "particlegan":
        parser.error("set PYTHONPATH to this checkout; imported a different ParticleGAN")
    torch.set_num_threads(1)
    parent = args.parent.resolve()
    receipt = json.loads((parent / "receipt.json").read_text())
    assert receipt["status"] == "complete", "parent diagnostic is incomplete"
    bindings = receipt["bindings"]
    native_hash = digest({str(path.relative_to(ROOT)): sha(path)
                          for path in sorted((ROOT / "particlegan").rglob("*.py"))})
    assert native_hash == bindings["native_source_hash"], "native source differs from parent"

    def check_sources():
        for relative, expected in bindings["source_hashes"].items():
            assert sha(ROOT / relative) == expected, f"held parent source differs: {relative}"

    check_sources()
    data = make_data()
    assert data["digest"] == receipt["data_digest"]
    inputs = {"receipt.json": sha(parent / "receipt.json")}

    def load(arm, step):
        relative = f"{arm}/step-{step:04d}.pt"
        path = parent / relative
        inputs[relative] = sha(path)
        recorded = {item["step"]: item["sha256"] for item in receipt["arms"][arm]["checkpoints"]}
        if step in recorded:
            assert inputs[relative] == recorded[step], "parent checkpoint hash differs"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["law"]["bindings"] == bindings and saved["step"] == step
        with torch.random.fork_rng(devices=[]):
            loop = make_loop(arm, data, bindings=bindings)
            restore(loop, saved)
            assert digest(checkpoint(loop)) == digest(saved), "strict native restore differs"
        if step == 0:
            assert digest(loop.G.state_dict()) == receipt["arms"][arm]["initial_model_digest"]
        return loop

    loops = {"ordinary": load(NATIVE_ARMS[0], 0), "particle": load(NATIVE_ARMS[1], 0),
             "H_bias_neutral": load(NATIVE_ARMS[1], 0)}
    with torch.no_grad():
        for name in SITES:
            branch = getattr(loops["H_bias_neutral"].G, name)
            branch.bridge.weight[:, :RANK].zero_()
            branch.bridge.bias.zero_()
    assert all(torch.equal(getattr(loops["particle"].G, name).bridge.weight[:, RANK:],
                           getattr(loops["H_bias_neutral"].G, name).bridge.weight[:, RANK:]) for name in SITES)
    initial_states = {name: digest(checkpoint(loop)) for name, loop in loops.items()}
    result = {"schema": "routed_convergence_basis_zero_update_audit_v1", "training_updates": 0,
              "baseline_bindings": bindings, "data_digest": data["digest"],
              "scope": "Private CPU clones; clean native routing; no optimizer/controller updates. Initial tangent probes use frozen common learned critics and one fixed native generator Gaussian, without a critic update.",
              "basis": {}, "initial_gradient": {}}
    basis_loops = [("initial_particle", loops["particle"]), ("initial_H_bias_neutral", loops["H_bias_neutral"]),
                   ("final_particle", load(NATIVE_ARMS[1], 6400))]
    for label, loop in basis_loops:
        before = digest(checkpoint(loop))
        result["basis"][label] = {}
        for pool in ("fit", "guard", "test"):
            with torch.no_grad():
                prediction, records = capture(loop, data[pool]["context"])
            sites = {}
            for name, (hidden, extra, gate, phi) in records.items():
                branch = getattr(loop.G, name)
                H_norm = float(torch.linalg.matrix_norm(branch.bridge.weight[:, :RANK].detach(), ord=2))
                sites[name] = {"hidden_rms": rms(hidden), "hidden_modulation_rms": rms(extra),
                               "hidden_modulation_over_hidden": rms(extra) / rms(hidden), "gate_rms": rms(gate),
                               "code_modulation_over_hidden": rms(hidden * gate) / rms(hidden),
                               "phi_over_hidden_rms": rms(phi) / rms(hidden), "phi_hidden_cosine": cosine(phi, hidden),
                               "recover_hidden_from_phi_relative_rms": basis_error(phi, hidden),
                               "H_operator_norm": H_norm, "tanh_bias_rms": rms(branch.bridge.bias.tanh()),
                               "H_norm_upper_bound_nonsingular_zero_code_hidden_jacobian": H_norm < 1}
            result["basis"][label][pool] = sites
            if label.startswith("initial"):
                with torch.no_grad():
                    assert torch.equal(prediction, forward(loops["ordinary"], data[pool]["context"]))
        assert before == digest(checkpoint(loop)), "clean basis audit changed native state"

    trace_path = parent / NATIVE_ARMS[0] / "trace.jsonl"
    inputs[str(trace_path.relative_to(parent))] = sha(trace_path)
    with trace_path.open() as stream:
        first = json.loads(stream.readline())
    indices = torch.randint(len(data["fit"]["context"]), (4,), generator=torch.Generator().manual_seed(7)).tolist()
    assert first["step"] == 1 and indices == first["batch_indices"]
    context, targets = (data["fit"][key][indices] for key in ("context", "targets"))
    generator = torch.Generator().manual_seed(43)
    critic_base = torch.randn(targets.shape, generator=generator)
    generator_base = torch.randn(targets.shape, generator=generator)
    assert digest((critic_base, generator_base)) == first["paired_base_digest"]
    gaussian = .125 * generator_base
    result["initial_gradient"].update(batch_indices=indices, paired_generator_gaussian_digest=digest(gaussian))
    tangent = {"schema": "routed_convergence_initial_output_tangent_v1", "training_updates": 0,
               "scope": "Clean native routing at zero up weights; frozen common critics; same fixed native generator Gaussian. Autograd tangent uses BF16 cast straight-through gradients and is not a finite-difference derivative of the quantized forward.",
               "baseline_bindings": bindings, "data_digest": data["digest"], "judges": {}}
    for judge_arm in NATIVE_ARMS:
        for step in (800, 6400):
            judge = deepcopy(load(judge_arm, step).policy.D).eval().requires_grad_(False)
            name, values, jvps = f"{judge_arm}@{step}", {}, {}
            assert digest(judge.state_dict()) == receipt["judges"][name]
            for label, loop in loops.items():
                before = digest(checkpoint(loop))
                prediction, _ = capture(loop, context)
                condition = context[:, 0, WIDTH:]
                loss = loop.policy.recipe.make_loss().g_loss(
                    judge(gaussian + (prediction - targets) / data["scale"], condition), judge(gaussian, condition).detach())
                weights = tuple(getattr(loop.G, site).up.weight for site in SITES)
                grads = torch.autograd.grad(loss, (prediction,) + weights)
                values[label] = (float(loss.detach()), grads[0], dict(zip(SITES, grads[1:])))
                group = loop.policy.opt_g.param_groups[0]
                assert not loop.policy.opt_g.state and group["betas"][0] == 0 and group["weight_decay"] == 0
                wrapper = Wrapper(loop)
                def function(a, b):
                    return torch.func.functional_call(wrapper, {"G.first.up.weight": a, "G.second.up.weight": b}, (context,))
                directions = {"raw_gradient": tuple(-g for g in grads[1:]),
                              "initial_native_Adam": tuple(-group["lr"] * g / (g.abs() + group["eps"]) for g in grads[1:])}
                for direction, deltas in directions.items():
                    base, output_tangent = torch.autograd.functional.jvp(function, weights, deltas, strict=True)
                    assert torch.equal(base, prediction)
                    jvps[label, direction] = output_tangent.detach(), float((grads[0] * output_tangent).sum())
                assert before == digest(checkpoint(loop)), "gradient or JVP changed native state"
            assert len({value[0] for value in values.values()}) == 1
            assert all(torch.equal(value[1], values["ordinary"][1]) for value in values.values())
            result["initial_gradient"][name] = {"game_equal": True, "output_gradient_exact": True, "sites": {}}
            for site in SITES:
                reference = values["ordinary"][2][site]
                result["initial_gradient"][name]["sites"][site] = {
                    label: {"gradient_norm": float(value[2][site].norm()), "cosine_to_ordinary": cosine(value[2][site], reference),
                            "relative_difference_from_ordinary": float((value[2][site] - reference).norm() / reference.norm())}
                    for label, value in values.items()}
            tangent["judges"][name] = {}
            for direction in directions:
                reference = jvps["ordinary", direction][0]
                tangent["judges"][name][direction] = {
                    label: {"cosine_to_ordinary_output_tangent": cosine(jvps[label, direction][0], reference),
                            "relative_output_tangent_difference": float((jvps[label, direction][0] - reference).norm() / reference.norm()),
                            "game_gradient_dot_output_tangent": jvps[label, direction][1],
                            "output_tangent_rms": float(jvps[label, direction][0].square().mean().sqrt())} for label in loops}
    assert all(initial_states[name] == digest(checkpoint(loop)) for name, loop in loops.items())
    check_sources()
    result["checks"] = {"initial_clean_output_exact_all_three_models_all_pools": True, "C_columns_bit_exact": True,
                        "all_private_native_state_unchanged": True, "held_baseline_sources_unchanged": True,
                        "data_digest_exact": True, "common_initial_output_gradient_bit_exact_all_four_judges": True}
    tangent["checks"] = {"same_base_outputs_all_function_tangents": True,
                         "private_full_native_states_unchanged": True, "C_unchanged": True}
    result.update(input_sha256=inputs, audit_source_sha256=sha(__file__))
    tangent.update(input_sha256=inputs, audit_source_sha256=sha(__file__))
    args.out.mkdir(parents=True)
    write_json(args.out / "report.json", result)
    tangent["parent_audit_sha256"] = sha(args.out / "report.json")
    write_json(args.out / "function-tangent.json", tangent)
    write_table(args.out / "tables.md", result["basis"], tangent["judges"])
    print(json.dumps({"out": str(args.out), "training_updates": 0, "checks": result["checks"]}))


if __name__ == "__main__":
    main()
