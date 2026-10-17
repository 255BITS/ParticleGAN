"""Independent CPU review of frozen RA4 composition, API and diagnostic bindings."""
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
OUTPUT = HERE / "ra4-source-review.json"
assert not OUTPUT.exists()
READY = ROOT / "integration/iteration-4/READY.json"
COMPOSITION = ROOT / "integration/iteration-4/COMPOSITION.json"
FREEZE = ROOT / "validation-ra4/source-freeze.json"
NEW = ROOT / "pkg-CB64-RA4/particlegan"
AXIS = ROOT / "performance/sampler-regression/cpu-plan-review/pkg-AXIS-ID/particlegan"
PLAN = ROOT / "performance/sampler-regression/cpu-plan-review/plan-batching/pkg-PLAN-FINAL/particlegan"
INPUT = ROOT / "integration/review/training-regression/snapshot-1000.pt"
HARNESS = Path("/ml2/hypergan/lrfree-20260926/harness/screen.py")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
dump = lambda node: ast.dump(node, include_attributes=False)
ready, composition, freeze = read(READY), read(COMPOSITION), read(FREEZE)
paths = {Path(__file__), READY, COMPOSITION, FREEZE, INPUT, HARNESS}
paths.update(map(Path, ready["numerical_source_sha256"]))
paths.update(AXIS.glob("*.py"))
paths.update(PLAN.glob("*.py"))
paths.update(Path(p) for p in freeze["external_sources"])
paths.update(ROOT / "validation-ra4" / p for p in freeze["local_sources"])
before = {str(p): sha(p) for p in sorted(paths)}
for p, expected in ready["numerical_source_sha256"].items():
    assert before[p] == expected, p
for p, expected in freeze["external_sources"].items():
    assert before[p] == expected, p
for p, expected in freeze["local_sources"].items():
    assert before[str(ROOT / "validation-ra4" / p)] == expected, p
checks = []
def checked(name, **details):
    checks.append(dict(name=name, status="PASS", **details))
    print(json.dumps(checks[-1]), flush=True)

digest = hashlib.sha256()
for p in sorted(NEW.rglob("*.py")):
    name = str(p.relative_to(NEW))
    assert sha(p) == ready["package_source_sha256"][name]
    assert sha(p) == composition["package_source_sha256"]["particlegan/" + name]
    digest.update(name.encode() + b"\0" + p.read_bytes() + b"\0")
assert digest.hexdigest() == ready["package_sha256"] == "e34bcb21aaa64caa0601cea5dc1f9b8eaebee9578686ebff39b459676063deb2"
assert ready["config_sha256"] == "d2b1018854671ded2ffe92b2f30c97a3871cf15911c6c241ceaadd281d94e7e7"
assert sha(NEW / "training.py") == sha(AXIS / "training.py")
assert {p.name for p in NEW.glob("*.py")} == {p.name for p in AXIS.glob("*.py")} == {p.name for p in PLAN.glob("*.py")}
for p in NEW.glob("*.py"):
    if p.name not in ("feature_cells.py", "training.py"):
        assert sha(p) == sha(AXIS / p.name) == sha(PLAN / p.name)
checked("complete_package_config_and_all_frozen_numerical_sources", package_sha256=digest.hexdigest(),
        numerical_source_files=len(ready["numerical_source_sha256"]), unchanged_python_files=25,
        training_api_bytes_exact_axis=True)

trees = [ast.parse((p / "feature_cells.py").read_text()) for p in (NEW, AXIS, PLAN)]
nodes = [{n.name:n for n in t.body if hasattr(n, "name")} for t in trees]
new, axis, plan = nodes
assert dump(new["FeatureCellSnapshot"]) == dump(plan["FeatureCellSnapshot"])
assert dump(new["_group_integer_allocate"]) == dump(plan["_group_integer_allocate"])
for name in ("LatentLineage", "BoundedLatentGeometry"):
    assert dump(new[name]) == dump(axis[name]), name
backend = new["FeatureCellBirthDeath"]
axis_backend = axis["FeatureCellBirthDeath"]
methods = lambda node: {n.name:n for n in node.body if isinstance(n, ast.FunctionDef)}
new_methods, axis_methods = methods(backend), methods(axis_backend)
setting_node = lambda method: next(n for n in ast.walk(method) if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Attribute) and t.attr == "settings" for t in n.targets))
last_node = lambda method: next(n for n in ast.walk(method) if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "last" for t in n.targets)
    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == "dict")
settings = setting_node(new_methods["__init__"])
setting_values = {kw.arg:kw.value for kw in settings.value.keywords}
expected_strings = dict(mass_policy="joint_mass_local_global_common_3K_plus_2_unique_parents_v1",
    count_partition="even_fit_score_order_statistic_2K",
    count_family="original_K_plus_support_2K_plus_global_2_common_Q_over_3K_plus_2")
for name, value in expected_strings.items():
    assert ast.literal_eval(setting_values[name]) == value
count_fields = {"count_partition", "count_boundary", "count_categories", "count_multiplicity", "count_cutoff",
                "ordinary_mass_moves", "ordinary_support_moves", "ordinary_global_moves", "ordinary_death_policy"}
new_last = last_node(new_methods["maybe_apply"])
plan_last = last_node(methods(plan["FeatureCellBirthDeath"])["maybe_apply"])
bindings = {kw.arg:dump(kw.value) for kw in new_last.value.keywords if kw.arg in count_fields}
assert bindings == {kw.arg:dump(kw.value) for kw in plan_last.value.keywords if kw.arg in count_fields}
assert set(bindings) == set(composition["count_diagnostic_fields"]) == count_fields
# Reconstruct the entire reviewed AXIS module by undoing only the count class,
# helper, three setting strings and nine diagnostic fields. This covers all
# lifecycle, copy, generated-pool identity propagation and checkpoint code.
reconstructed = deepcopy(trees[0])
reconstructed.body = [n for n in reconstructed.body if not (hasattr(n,"name") and n.name == "_group_integer_allocate")]
for i, n in enumerate(reconstructed.body):
    if isinstance(n, ast.ClassDef) and n.name == "FeatureCellSnapshot":
        reconstructed.body[i] = deepcopy(axis["FeatureCellSnapshot"])
    elif isinstance(n, ast.ClassDef) and n.name == "FeatureCellBirthDeath":
        m = methods(n)
        setting_node(m["__init__"]).value = deepcopy(setting_node(axis_methods["__init__"]).value)
        last_node(m["maybe_apply"]).value.keywords = [kw for kw in last_node(m["maybe_apply"]).value.keywords
                                                    if kw.arg not in count_fields]
assert dump(reconstructed) == dump(trees[1])
checked("whole_module_exact_approved_composition", count_snapshot_and_planners_exact_plan=True,
        lineage_and_geometry_exact_axis=True, all_other_module_ast_exact=True,
        count_settings=expected_strings, diagnostic_fields=sorted(count_fields))

holder = ModuleType("independent_ra4_source")
holder.__path__ = [str(NEW)]
sys.modules[holder.__name__] = holder
fc = importlib.import_module(holder.__name__ + ".feature_cells")
training = importlib.import_module(holder.__name__ + ".training")
payload = torch.load(INPUT, map_location="cpu", weights_only=False)
recipe = SimpleNamespace(birth_death_space="critic", birth_death_isolation=True,
    birth_death_feature_scale="std", birth_death_cells=64, birth_death_metric_rank=8,
    birth_death_chunk=256, birth_death_parent_policy="real_anchor")
q = payload["q"].clone()
trainer = SimpleNamespace(prior=SimpleNamespace(z=torch.nn.Parameter(q.clone())),
    G=torch.nn.Identity(), D=torch.nn.Linear(q.shape[1], 1, device="meta"),
    recipe=recipe, device=torch.device("cpu"), dtype=q.dtype, completed_steps=1000,
    controller=SimpleNamespace(latent_bandwidth=torch.ones(q.shape[1], dtype=q.dtype)))
bd = fc.FeatureCellBirthDeath(trainer, payload["seed"])
for name, value in expected_strings.items():
    assert bd.settings[name] == value
assert bd.BACKEND_SCHEMA == 4 and bd.settings["lineage_degree"] == 8
assert bd.settings["latent_candidate_bound"] == 72
assert bd.lineage.neighbors.shape == (1024,8)
assert bd.settings["latent_kernel"] == "bounded_local_dv12_lineage"
initial = bd.state_dict()
bad = deepcopy(initial)
bad["settings"]["mass_policy"] = "reference_topology_vacancies_unique_parents_v4"
bad["settings"].pop("count_family")
bad["settings"].pop("count_partition")
before_s, before_graph, before_rng = bd.S.clone(), bd.lineage.neighbors.clone(), bd.stream.get_state().clone()
try:
    bd.load_state_dict(bad)
except ValueError:
    pass
else:
    raise AssertionError("old RA3 law accepted")
assert torch.equal(before_s,bd.S) and torch.equal(before_graph,bd.lineage.neighbors)
assert torch.equal(before_rng,bd.stream.get_state())
assert "snapshot" not in initial and "latent_geometry" not in initial
checked("actual_backend_settings_and_distinct_law_atomic_checkpoint_rejection",
        backend_schema=4, lineage_degree=8, latent_candidate_bound=72,
        old_ra3_settings_rejected_before_mutation=True, transient_caches_not_checkpointed=True)

# Run the production maybe_apply method on one retained mechanical fixture.
# Captured features and move effects are replaced by CPU hooks; the actual
# fit, tests, three planners, isolation, metadata and counters run unchanged.
# This is a diagnostic wiring test, not a trajectory or learned-quality test.
bd.stream.set_state(payload["planning_rng"])
bd.reservoir = payload["real_features"].clone()
bd.sample_shape = (q.shape[1],)
bd.fill = bd.rows_since_eval = bd.N
captures, moves = [], []
bd._features = lambda owner, raw, **kw: raw.clone()
def captured(owner, latent, *, sigma=0., jitter=False, rows=None):
    captures.append(dict(jitter=jitter, rows=None if rows is None else rows.clone(), count=len(latent)))
    return payload["fake_features"].clone() if jitter else latent.clone()
bd._capture_generated = captured
def moved(owner, child, parent):
    moves.append((child.clone(),parent.clone()))
    bd.lineage.register_copies(child,parent)
bd._move = moved
ordinary_detail = []
original = fc.FeatureCellSnapshot.ordinary_transport
def observed(snapshot, *args, **kw):
    answer = original(snapshot,*args,**kw)
    ordinary_detail.append(answer[2])
    return answer
fc.FeatureCellSnapshot.ordinary_transport = observed
try:
    last = bd.maybe_apply(trainer,.029)
finally:
    fc.FeatureCellSnapshot.ordinary_transport = original
assert last is not None and len(ordinary_detail) == 1
detail = ordinary_detail[0]
k = bd.snapshot.cells
assert last["count_partition"] == dict(bd.snapshot.count_partition)
assert last["count_boundary"] == float(bd.snapshot.count_boundary)
assert last["count_categories"] == 2*k and last["count_multiplicity"] == 3*k+2
assert last["count_cutoff"] == fc.Q/(3*k+2)
for phase in ("mass","support","global"):
    assert last["ordinary_"+phase+"_moves"] == detail[phase+"_moves"]
assert sum(last["ordinary_"+p+"_moves"] for p in ("mass","support","global")) == last["ordinary_moves"]
assert last["ordinary_death_policy"] == detail["death_policy"]
assert last["moves"] == last["ordinary_moves"]+last["iso_moves"] == sum(len(c) for c,p in moves)
assert bd.counters["ordinary_moves"] == last["ordinary_moves"]
assert bd.counters["iso_moves"] == last["iso_moves"]
assert captures[0]["rows"] is None and captures[1]["jitter"]
assert captures[1]["rows"].shape == (bd.N,) and captures[1]["rows"].dtype == torch.long
assert trainer.G.training and trainer.D.training
checked("production_maybe_apply_count_phase_diagnostics_and_sampled_row_ids",
        fixture="retained snapshot-1000 features; CPU capture/move hooks; actual fit/tests/planners/diagnostics",
        cells=k, categories=last["count_categories"], multiplicity=last["count_multiplicity"],
        cutoff=last["count_cutoff"], mass_moves=last["ordinary_mass_moves"],
        local_moves=last["ordinary_support_moves"], global_moves=last["ordinary_global_moves"],
        ordinary_moves=last["ordinary_moves"], isolation_moves=last["iso_moves"],
        total_moves=last["moves"], generated_fake_row_ids_present=True)

# Exact frozen harness option resolver, with no harness imports or RNG setup.
parsed = ast.parse(HARNESS.read_text())
default = next(n for n in parsed.body if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=="DEFAULT_OPTIONS" for t in n.targets))
resolve = next(n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name=="resolve_options")
namespace = dict(inspect=inspect)
exec(compile(ast.Module(body=[default,resolve],type_ignores=[]),str(HARNESS),"exec"),namespace)
options, _ = namespace["resolve_options"](SimpleNamespace(GANTrainer=training.GANTrainer),{}, {})
assert options["evaluation_generate"] == "indexed"
assert inspect.signature(training.GANTrainer._generate).parameters["indices"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
seen = []
spy_backend = SimpleNamespace(perturb_latent=lambda latent,stream,controller,**kw: seen.append(kw["rows"]) or latent)
spy = SimpleNamespace(birth_death=spy_backend,prior=trainer.prior,ema_prior=None,
    G=trainer.G,ema_G=None,controller=trainer.controller,noise_generator=None)
rows = torch.tensor([1,2,1])
stream = torch.Generator(device="cpu").set_state(payload["planning_rng"])
out = training.GANTrainer._generate(spy,spy.G,q[rows],0.,stream,rows)
assert seen[-1] is rows and torch.equal(out,q[rows])
checked("frozen_native_harness_indexed_mode_and_positional_api", harness_sha256=sha(HARNESS),
        mode="indexed", fifth_positional_indices_forwarded=True,
        complete_training_api_exact_independently_reviewed_axis=True)

assert before == {str(p):sha(p) for p in sorted(paths)}, "frozen source changed during review"
assert not torch.cuda.is_initialized()
receipt = dict(status="PASS", checks=checks, package_sha256=ready["package_sha256"],
    ready_sha256=sha(READY), source_sha256=before, cpu_only=True,cuda_initialized=False,
    optimizer_updates=0,new_seeds=0, high_dimensional_support_law_qualified=False,
    learned_quality_qualified=False, scope=__doc__)
OUTPUT.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(status="PASS",receipt=str(OUTPUT),checks=len(checks),cuda_initialized=False)),flush=True)
