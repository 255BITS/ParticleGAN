"""Standard-library-only source/fixture/contract checks. Never imports Torch."""
from pathlib import Path
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import sys
import zipfile

sys.dont_write_bytecode = True
from execution_contract import EXECUTION, host_cuda, serial_step, validate_checkpoint


def sha(data):
    return hashlib.sha256(data).hexdigest()


def node_named(tree, name):
    matches = [n for n in tree.body if getattr(n, "name", None) == name or
               isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(matches) == 1, name
    return matches[0]


def check_context_contracts():
    class Generator:
        def __new__(cls, device="cpu"):
            value = object.__new__(cls)
            value.device = device
            return value

        def __init__(self, device="cpu"):
            pass

    class FakeTorch:
        def __init__(self):
            self.current_device = "cpu"
            self.multithreading = True
            self.autograd = self

        def get_default_device(self):
            return self.current_device

        @contextmanager
        def device(self, name):
            before = self.current_device
            self.current_device = name
            try:
                yield
            finally:
                self.current_device = before

        def is_multithreading_enabled(self):
            return self.multithreading

        @contextmanager
        def set_multithreading_enabled(self, value):
            before = self.multithreading
            self.multithreading = value
            try:
                yield
            finally:
                self.multithreading = before

    FakeTorch.Generator = Generator
    fake = FakeTorch()
    for raise_error in (False, True):
        try:
            with host_cuda(fake):
                assert fake.get_default_device() == "cuda:0"
                assert fake.Generator().device == "cuda:0"
                assert fake.Generator(device="cpu").device == "cpu"
                if raise_error:
                    raise LookupError("deliberate restoration check")
        except LookupError:
            assert raise_error
        assert fake.get_default_device() == "cpu" and fake.Generator is Generator
    for previous in (True, False):
        for raise_error in (False, True):
            fake.multithreading = previous
            try:
                with serial_step(fake):
                    assert not fake.multithreading
                    with serial_step(fake):
                        assert not fake.multithreading
                    if raise_error:
                        raise LookupError("deliberate restoration check")
            except LookupError:
                assert raise_error
            assert fake.multithreading is previous
    envelope = dict(schema=1, execution=EXECUTION, identity={"pinned": "source"}, trainer={"schema": 3}, data_rng=None)
    validate_checkpoint(envelope, envelope["identity"])
    for key, value in (("execution", {"serial_backward": False}), ("identity", {}),
                       ("schema", 2), ("trainer", {"schema": 4})):
        bad = deepcopy(envelope)
        bad[key] = value
        try:
            validate_checkpoint(bad, envelope["identity"])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid checkpoint contract accepted")


def verify(bundle=None):
    bundle = Path(__file__).resolve().parent if bundle is None else Path(bundle)
    manifest = json.loads((bundle / "bundle-sha256.json").read_text())
    assert manifest["schema"] == 1
    for name, want in manifest["files"].items():
        path = bundle / name
        assert path.resolve().is_relative_to(bundle.resolve())
        assert sha(path.read_bytes()) == want, name
        if path.suffix == ".py":
            compile(ast.parse(path.read_text()), str(path), "exec")
    d = json.loads((bundle / "declaration.json").read_text())
    assert d["status"] == "PREPARED_NOT_RUN" and d["schema"] == 1
    assert d["release"]["tag"] == "v0.8.0" and d["release"]["commit"] == "0ff9a7afe5dcb828239369446cfe71971bce687b"
    assert d["release"]["patches"] == []
    receipt = json.loads((bundle / "release-receipt.json").read_text())
    assert receipt["sha256"] == d["release"]["source_sha256"]
    with zipfile.ZipFile(bundle / "public-k3p-v0.8.0.zip") as archive:
        assert set(archive.namelist()) == set(receipt["sha256"]) and len(archive.namelist()) == 12
        for name, want in receipt["sha256"].items():
            assert sha(archive.read(name)) == sha((bundle / "source" / name).read_bytes()) == want
        tree = ast.parse(archive.read("particlegan/recipes.py"))
        recipe = node_named(tree, "Recipe")
        released_defaults = {n.target.id: ast.literal_eval(n.value) for n in recipe.body if isinstance(n, ast.AnnAssign)}
        resolved = {**released_defaults, "total_steps": 1200, "num_particles": 256, "z_dim": 4, "batch_size": 128}
        assert json.loads(json.dumps(resolved)) == d["recipe"]
    assert sha((bundle / "public-k3p-v0.8.0.zip").read_bytes()) == d["release"]["archive_sha256"]
    assert set((bundle / "source/particlegan").glob("*.py")) == {bundle / "source" / name for name in receipt["sha256"]}
    for name, want in d["source_preparation"]["original_source_sha256"].items():
        assert sha((bundle / "original-host" / name).read_bytes()) == want
    original = json.loads((bundle / "original-host/benchmarks/transfer_suite/plans/default_comparison.json").read_text())
    spec = next(row["spec"] for row in original if row["spec"]["name"] == "vector_unequal_mass")
    card = json.loads((bundle / "original-host/reports/transfer_suite/unadjusted/leading_profile.json").read_text())["discriminators"][spec["name"]]
    expected_spec = {**spec, "d_hidden": card["width"], "d_layers": card["layers"], "fourier": card["fourier"], "research_discriminator": card}
    assert spec == d["original_spec"] and expected_spec == d["spec"] and card == d["card"]
    assert card["implementation"] == "shared_batch_feature_v1" and card["name"] == "batchfeat_center6_distance_head"
    assert (d["spec"]["particles"], d["spec"]["z_dim"], d["spec"]["batch"], d["spec"]["steps"]) == (256, 4, 128, 1200)
    assert (d["spec"]["d_every"], d["spec"]["g_every"]) == (1, 1)
    assert d["evaluation"]["observations"] == list(range(50, 1201, 50))
    assert d["evaluation"]["output_noise_seed"] == 2303 and d["evaluation"]["final_passing_suffix"] == 5
    assert d["recipe"]["input_noise_anneal_end"] * 1200 == 120
    assert d["recipe"]["output_noise_warmup"] * 1200 == 240
    assert min(1200, d["recipe"]["network_lr_horizon_cap"]) * d["recipe"]["lr_anneal_start"] == 720
    assert sha((bundle / "initial-values.pt").read_bytes()) == d["fixture"]["sha256"]
    with zipfile.ZipFile(bundle / "initial-values.pt") as archive:
        storages = sorted((n for n in archive.namelist() if "/data/" in n), key=lambda n: int(n.rsplit("/", 1)[1]))
        params = sum(d["fixture"]["initial_parameter_groups"], [])
        assert len(storages) == len(params) == 15
        for name, parameter in zip(storages, params):
            raw = archive.read(name)
            assert sha(raw) == parameter["sha256"] and len(raw) == math.prod(parameter["shape"]) * 4
    frozen = ast.parse((bundle / "source/frozen_host.py").read_text())
    extractions = json.loads((bundle / "extraction.json").read_text())
    for selected in extractions:
        original = (bundle / "original-host" / selected["original"]).read_text()
        span = "".join(original.splitlines(keepends=True)[selected["first_line"] - 1:selected["last_line"]])
        assert sha(span.encode()) == selected["source_span_sha256"]
        before = node_named(ast.parse(original), selected["name"])
        after = node_named(frozen, selected["name"])
        assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    worker = ast.parse((bundle / "worker.py").read_text())
    calls = [n for n in ast.walk(worker) if isinstance(n, ast.Call)]
    recipes = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "get_recipe"]
    assert len(recipes) == 1 and {k.arg: ast.literal_eval(k.value) for k in recipes[0].keywords} == d["recipe_overrides"]
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in ("backward", "set_default_device") for n in calls)
    steps = [n for n in calls if isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == "trainer.step"]
    serial_withs = [n for n in ast.walk(worker) if isinstance(n, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Name) and i.context_expr.func.id == "serial_step" for i in n.items)]
    assert len(steps) == len(serial_withs) == 1 and steps[0] in list(ast.walk(serial_withs[0]))
    assert not any(isinstance(n, ast.Attribute) and ast.unparse(n) == "torch.optim" for n in ast.walk(worker))
    check_context_contracts()
    assert "torch" not in sys.modules, "source preflight unexpectedly imported Torch"
    return dict(status="PASS", scope="stdlib-only preparation; no Torch/GPU/training", package_files=12,
                exact_host_definitions=len(extractions), fixture_tensors=15,
                checks=["bundle/source hashes", "exact release bytes", "recipe defaults plus four declared overrides",
                        "frozen task/card", "15 raw fixture storage hashes", "exact host AST/source spans",
                        "public step enclosed by serial context", "host-only CUDA restoration on exception",
                        "nested serial restoration", "checkpoint identity/mode rejection"],
                runtime_checks="NOT_RUN; model/import/hash and Adam device assertions execute only in authorized worker",
                bundle_manifest_sha256=sha((bundle / "bundle-sha256.json").read_bytes()))


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2))
