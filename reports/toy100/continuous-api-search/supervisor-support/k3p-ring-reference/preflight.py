"""Standard-library source preflight only: no Torch import, model or GPU work."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True


def sha(data):
    return hashlib.sha256(data).hexdigest()


def named(tree, name):
    return next(n for n in tree.body if getattr(n, "name", None) == name or
                isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def verify(bundle=None):
    bundle = Path(bundle or __file__).resolve()
    if bundle.is_file():
        bundle = bundle.parent
    assert "torch" not in sys.modules, "source preflight must run before importing Torch"
    manifest = json.loads((bundle / "bundle-sha256.json").read_text())
    for name, wanted in manifest["files"].items():
        assert sha((bundle / name).read_bytes()) == wanted, name
    expected_files = set(manifest["files"]) | {"bundle-sha256.json", "preflight-result.json"}
    actual_files = {str(p.relative_to(bundle)) for p in bundle.rglob("*") if p.is_file()}
    assert actual_files <= expected_files, "unsealed files in bundle: " + repr(sorted(actual_files-expected_files))
    declaration = json.loads((bundle / "declaration.json").read_text())
    receipt = json.loads((bundle / "original-source-receipt.json").read_text())
    assert declaration["release"]["commit"] == receipt["commit"] == "0ff9a7afe5dcb828239369446cfe71971bce687b"
    assert declaration["release"]["patches"] == []
    with zipfile.ZipFile(bundle / "public-k3p-v0.8.0.zip") as archive:
        assert set(archive.namelist()) == set(receipt["sha256"])
        for name, wanted in receipt["sha256"].items():
            assert sha(archive.read(name)) == wanted == sha((bundle / "source" / name).read_bytes())
    assert len(receipt["sha256"]) == 12
    with zipfile.ZipFile(bundle / "rp5-recovery-host-reference.zip") as archive:
        for name in archive.namelist():
            assert archive.read(name) == (bundle / "original-host" / name).read_bytes()
    recipe_source = ast.parse((bundle / "source/particlegan/recipes.py").read_text())
    recipe_class = named(recipe_source, "Recipe")
    defaults = {n.target.id:ast.literal_eval(n.value) for n in recipe_class.body
                if isinstance(n,ast.AnnAssign) and n.value is not None}
    assert defaults["total_steps"] == 7000 and defaults["name"] == "k3p"
    defaults.update(declaration["recipe_overrides"])
    assert json.loads(json.dumps(defaults)) == declaration["recipe"]
    assert declaration["recipe_overrides"] == {"total_steps":4600}
    recipe = declaration["recipe"]
    assert (recipe["num_particles"],recipe["z_dim"],recipe["batch_size"]) == (20000,2,2048)
    assert recipe["total_steps"] * recipe["input_noise_anneal_end"] == 460
    assert recipe["total_steps"] * recipe["output_noise_warmup"] == 920
    assert recipe["total_steps"] * recipe["lr_anneal_start"] == 2760
    assert recipe["network_lr_horizon_cap"] * recipe["lr_anneal_start"] == 960
    assert recipe["network_lr_floor"] == .01 and recipe["lr_floor"] == .05
    assert declaration["schedules"] == dict(input_zero_at_completed_step=460,output_full_at_completed_step=920,
        network_decay_start_completed_step=960,network_floor_at_completed_step=1600,
        prior_decay_start_completed_step=2760,prior_floor_at_completed_step=4600,final_update_schedule_index=4599)
    retained = json.loads((bundle / "original-host/receipts/rp5-single-initial.json").read_text())
    assert declaration["expected_initial_fixture"] == {k:v for k,v in retained.items() if k not in ("trainer_sha256","optimizers_sha256")}
    protocol = json.loads((bundle / "original-host/reports/reversible-precision/evaluation-protocols.json").read_text())
    assert declaration["evaluation"]["total_updates"] == protocol["protocols"]["single_shift"]["total_updates"] == 4600
    assert declaration["evaluation"]["target_changes"] == protocol["protocols"]["single_shift"]["target_changes"]
    frozen = ast.parse((bundle / "source/frozen_host.py").read_text())
    selections = json.loads((bundle / "extraction.json").read_text())
    for selection in selections:
        original = (bundle / "original-host" / selection["original"]).read_text()
        span = "".join(original.splitlines(keepends=True)[selection["first_line"]-1:selection["last_line"]])
        assert sha(span.encode()) == selection["source_span_sha256"]
        assert ast.dump(named(ast.parse(original),selection["name"]),include_attributes=False) == ast.dump(named(frozen,selection["name"]),include_attributes=False)
    host_imports = [n for n in ast.walk(frozen) if isinstance(n,(ast.Import,ast.ImportFrom))]
    for node in host_imports:
        names = [node.module] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names]
        assert all(n in ("torch","math","json","hashlib") for n in names)
    worker = ast.parse((bundle / "worker.py").read_text())
    calls = [n for n in ast.walk(worker) if isinstance(n,ast.Call)]
    recipes = [n for n in calls if isinstance(n.func,ast.Name) and n.func.id == "get_recipe"]
    assert len(recipes) == 1 and {k.arg:ast.literal_eval(k.value) for k in recipes[0].keywords} == declaration["recipe_overrides"]
    steps = [n for n in calls if isinstance(n.func,ast.Attribute) and ast.unparse(n.func) == "trainer.step"]
    contexts = [n for n in ast.walk(worker) if isinstance(n,ast.With) and any(
        isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=="serial_step" for i in n.items)]
    assert len(steps) == len(contexts) == 1 and steps[0] in list(ast.walk(contexts[0]))
    assert {k.arg:ast.unparse(k.value) for k in steps[0].keywords}["generator_real"] == "real"
    forbidden = ("backward","set_default_device")
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in calls)
    assert not any(isinstance(n,ast.Attribute) and ast.unparse(n)=="torch.optim" for n in ast.walk(worker))
    loads = [n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="load_state_dict"]
    assert len(loads) == 1 and ast.unparse(loads[0].func)=="frozen.load_state_dict"
    assert not any(isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=="torch.load" for n in calls)
    trainer_calls = [n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="GANTrainer"]
    assert len(trainer_calls)==1 and all(k.arg not in ("serial_backward","prior") for k in trainer_calls[0].keywords)
    for n in ast.walk(worker):
        if isinstance(n,ast.ImportFrom):
            assert not (n.module or "").startswith(("benchmarks","public_worker","source_helpers"))
    execution = ast.parse((bundle / "execution_contract.py").read_text())
    assert not any(isinstance(n,ast.Import) and any(a.name=="torch" for a in n.names) or
                   isinstance(n,ast.ImportFrom) and n.module=="torch" for n in ast.walk(execution))
    assert "torch" not in sys.modules
    return dict(status="PASS",scope="stdlib-only source preflight; no Torch import/model/GPU/training/tests",
        package_files=12,exact_host_definitions=len(selections),
        checks=["sealed file hashes and exact release ZIP", "complete original host provenance",
                "exact extracted source spans and AST", "release defaults plus only total_steps4600",
                "public-ring resources and expected initial RNG/model receipts", "scheduled noise460/920 and LR rules",
                "whole public step inside serial context", "no optimizer monkeypatch/backward or Torch load",
                "only separate frozen trainer restores; main uninterrupted", "no candidate benchmark import chain"],
        runtime_checks="NOT_RUN; sealed worker assertions validate imports, Torch/Adam source, initial models/RNG and native counters during authorized execution",
        execution_blockers=["Independent second source review required before external owner execution",
                            "Pinned CUDA runtime and selected lane GPU must satisfy worker assertions",
                            "No model/initialization/first-step native Adam verification has been executed during preparation"],
        bundle_manifest_sha256=sha((bundle / "bundle-sha256.json").read_bytes()))


if __name__ == "__main__":
    print(json.dumps(verify(),indent=2))
