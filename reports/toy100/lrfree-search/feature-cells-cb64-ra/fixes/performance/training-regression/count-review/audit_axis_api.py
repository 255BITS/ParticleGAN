"""Independent CPU review of cached integer axes and frozen native row-ID calls."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
import ast
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace, MethodType
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEW = ROOT / "performance/sampler-regression/cpu-plan-review/pkg-AXIS-ID"
OLD = ROOT / "pkg-CB64-RA3"
HARNESS = Path("/ml2/hypergan/lrfree-20260926/harness/screen.py")
INPUT = ROOT / "integration/review/training-regression/snapshot-1000.pt"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths = [Path(__file__), HARNESS, INPUT] + list((NEW / "particlegan").glob("*.py")) + list((OLD / "particlegan").glob("*.py"))
before = {str(p): sha(p) for p in paths}
assert sha(HARNESS) == "ee8193adbdf09e93511befae7b6491143c26de88612eddf065cbb92eb2153c3c"


def package(name, root):
    holder = ModuleType(name)
    holder.__path__ = [str(root / "particlegan")]
    sys.modules[name] = holder
    return importlib.import_module(name + ".feature_cells"), importlib.import_module(name + ".training")


old_fc, old_training = package("independent_axis_old", OLD)
new_fc, new_training = package("independent_axis_new", NEW)
payload = torch.load(INPUT, map_location="cpu", weights_only=False)
checks = []


def checked(name, **kw):
    checks.append(dict(name=name, status="PASS", **kw))


def stream():
    return torch.Generator(device="cpu").set_state(payload["planning_rng"])


old_files = {p.name: sha(p) for p in (OLD / "particlegan").glob("*.py")}
new_files = {p.name: sha(p) for p in (NEW / "particlegan").glob("*.py")}
assert old_files.keys() == new_files.keys()
assert {name for name in old_files if old_files[name] != new_files[name]} == {"feature_cells.py", "training.py"}
checked("package_changes_limited_to_cache_axis_conversion_and_generate_signature",
        changed_files=["feature_cells.py", "training.py"], unchanged_python_files=len(old_files)-2)


# Compile only the exact option resolver and draw argument construction from
# the frozen harness. No harness imports, seeds, scoring or GPU policy run.
parsed = ast.parse(HARNESS.read_text())
default = next(n for n in parsed.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "DEFAULT_OPTIONS" for t in n.targets))
resolve = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_options")
namespace = dict(inspect=inspect)
exec(compile(ast.Module(body=[default, resolve], type_ignores=[]), str(HARNESS), "exec"), namespace)
draw = next(n for n in ast.walk(parsed) if isinstance(n, ast.FunctionDef) and n.name == "draw"
            and [a.arg for a in n.args.args] == ["n", "ema", "latent_seed", "noise_seed"])
body = next(n.body for n in ast.walk(draw) if isinstance(n, ast.With)
            and any(isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Attribute)
                    and item.context_expr.func.attr == "fork_rng" for item in n.items))
start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "arguments" for t in n.targets))
fragment = deepcopy(body[start:start+3])
assert ast.unparse(fragment[1].body[0]) == "arguments.append(indices)"
wrapper = ast.parse("def native(trainer,model,latent,latent_stream,indices,options):\n    pass\n").body[0]
wrapper.body = fragment + [ast.Return(value=ast.Name(id="clean", ctx=ast.Load()))]
ast.fix_missing_locations(wrapper)
exec(compile(ast.Module(body=[wrapper], type_ignores=[]), str(HARNESS), "exec"), namespace)
old_options, _ = namespace["resolve_options"](SimpleNamespace(GANTrainer=old_training.GANTrainer), {}, {})
new_options, _ = namespace["resolve_options"](SimpleNamespace(GANTrainer=new_training.GANTrainer), {}, {})
assert old_options["evaluation_generate"] == "plain" and new_options["evaluation_generate"] == "indexed"
assert inspect.signature(new_training.GANTrainer._generate).parameters["indices"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


z = torch.nn.Parameter(torch.tensor([[-10., 0.], [0., 0.], [.01, 0.], [10., 1.],
                                     [20., -1.], [30., .5]], dtype=torch.float64))
prior = SimpleNamespace(z=z)
graph = new_fc.LatentLineage(6, 1, "cpu")
graph.register_copies(torch.tensor([1]), torch.tensor([2]))
geometry = new_fc.BoundedLatentGeometry(rank=1, neighbors=2, chunk=2, lineage=graph)
backend = new_fc.FeatureCellBirthDeath.__new__(new_fc.FeatureCellBirthDeath)
backend.latent_geometry = geometry
trainer = SimpleNamespace(prior=prior, ema_prior=SimpleNamespace(z=z.detach().clone()),
                          G=torch.nn.Identity(), ema_G=torch.nn.Identity(), birth_death=backend,
                          controller=SimpleNamespace(latent_bandwidth=torch.ones(2, dtype=torch.float64)),
                          noise_generator=None)
trainer._generate = MethodType(new_training.GANTrainer._generate, trainer)
forwarded = []
actual = backend.perturb_latent


def observed(latent, *args, rows=None, **kw):
    forwarded.append(rows)
    return actual(latent, *args, rows=rows, **kw)


backend.perturb_latent = observed
rows = torch.tensor([1], dtype=torch.long)
latent = z[rows]
native = namespace["native"](trainer, trainer.G, latent, stream(), rows, new_options)
assert forwarded[-1] is rows
keyword = trainer._generate(trainer.G, latent, 0., stream(), rows=rows)
assert torch.equal(native, keyword)
plain = trainer._generate(trainer.G, latent, 0., stream())
assert not torch.equal(native, plain), "known-copy augmentation was not observable in native indexed call"
with_ids = geometry.radius(latent, prior, rows=rows)
without_ids = geometry.radius(latent, prior)
assert float(with_ids[0]) == .005 and float(without_ids[0]) == 5.
checked("exact_frozen_harness_detection_and_fifth_positional_call_reach_known_copy",
        harness_sha256=sha(HARNESS), old_auto_mode="plain", new_auto_mode="indexed",
        indexed_radius=float(with_ids[0]), plain_radius=float(without_ids[0]), sampled_rows_forwarded=True)


# All legacy four-argument and rows-keyword calls retain output and stream
# bits; the new positional/indices-keyword forms are the same law.
rows = torch.tensor([1, 2, 1, 0], dtype=torch.long)
comparisons = 0
for ema in (False, True):
    model, table = (trainer.ema_G, trainer.ema_prior) if ema else (trainer.G, trainer.prior)
    latent = table.z[rows]
    for sigma in (0., .029):
        a, b = stream(), stream()
        old = old_training.GANTrainer._generate(trainer, model, latent, sigma, a)
        new = trainer._generate(model, latent, sigma, b)
        assert torch.equal(old, new) and torch.equal(a.get_state(), b.get_state())
        a, b, c, d = stream(), stream(), stream(), stream()
        old = old_training.GANTrainer._generate(trainer, model, latent, sigma, a, rows=rows)
        positional = trainer._generate(model, latent, sigma, b, rows)
        indices_keyword = trainer._generate(model, latent, sigma, c, indices=rows)
        rows_keyword = trainer._generate(model, latent, sigma, d, rows=rows)
        assert torch.equal(old, positional) and torch.equal(positional, indices_keyword) and torch.equal(indices_keyword, rows_keyword)
        assert torch.equal(a.get_state(), b.get_state()) and torch.equal(b.get_state(), c.get_state()) and torch.equal(c.get_state(), d.get_state())
        comparisons += 2
rng = stream()
saved_rng, saved_graph, saved_work = rng.get_state().clone(), graph.neighbors.clone(), deepcopy(geometry.work)
try:
    trainer._generate(trainer.G, z[rows], .029, rng, rows, rows=rows)
except ValueError:
    pass
else:
    raise AssertionError("ambiguous row aliases were accepted")
assert torch.equal(saved_rng, rng.get_state()) and torch.equal(saved_graph, graph.neighbors)
assert saved_work == geometry.work
trainer._generate(trainer.G, z[rows], 0., stream(), rows).sum().backward()
assert torch.equal(z.grad, torch.bincount(rows, minlength=len(z)).double()[:, None].expand_as(z))
checked("legacy_alias_live_ema_output_rng_parity_atomic_ambiguity_and_identity_gradient",
        live_ema_sigma_comparisons=comparisons, ambiguous_alias_rejected_before_rng_and_work=True,
        indexed_identity_derivative=True)


# Independently compare cache output on cold/warm calls and a version change.
cache_rows = torch.tensor([0, 1, 2, 1, 5], dtype=torch.long)
noise = torch.tensor([[.3, -.2], [-.4, .5], [.2, .7], [-.1, -.6], [.8, .1]], dtype=torch.float64)
old_geometry = old_fc.BoundedLatentGeometry(rank=1, neighbors=2, chunk=2, lineage=graph)
new_geometry = new_fc.BoundedLatentGeometry(rank=1, neighbors=2, chunk=2, lineage=graph)
records = []
for stage in ("cold", "warm", "version_changed", "all_tied"):
    with torch.no_grad():
        if stage == "version_changed":
            z[5, 1] += .01
        elif stage == "all_tied":
            z.fill_(1.)
    for geo in (old_geometry, new_geometry):
        geo._local_geometry(z[cache_rows], prior, rows=cache_rows)
    a = old_geometry.displacement(z[cache_rows], prior, torch.ones(2), noise, rows=cache_rows)
    b = new_geometry.displacement(z[cache_rows], prior, torch.ones(2), noise, rows=cache_rows)
    assert torch.equal(a, b) and old_geometry.work == new_geometry.work
    assert all(type(axis) is int for axis, _, _ in new_geometry._orders(z))
    records.append(dict(stage=stage, bit_identical=True, cached_axes_python_int=True,
                        builds=new_geometry.work["builds"]))
assert new_geometry.work["builds"] == 3
checked("cache_axis_representation_cold_warm_versions_and_ties_preserve_bits", records=records)

assert before == {str(p): sha(p) for p in paths}, "reviewed sources changed during audit"
assert not torch.cuda.is_initialized()
receipt = dict(status="PASS", checks=checks, source_sha256=before,
               cuda_initialized=False, new_seeds=0, optimizer_updates=0,
               scope="Exact source composition, frozen harness row API and deterministic cache parity; no performance/quality claim")
(HERE / "axis-api-review.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2), flush=True)
