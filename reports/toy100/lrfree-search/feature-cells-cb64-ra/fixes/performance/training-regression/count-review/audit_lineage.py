"""Independent CPU checks of bounded lineage state and latent row semantics."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from copy import deepcopy

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def package(name, path):
    holder = ModuleType(name)
    holder.__path__ = [str(path / "particlegan")]
    sys.modules[name] = holder
    return importlib.import_module(name + ".feature_cells")


base_path = ROOT / "pkg-CB64-RA2/particlegan/feature_cells.py"
proposed_path = ROOT / "geometry/training-regression/pkg-LINEAGE/particlegan/feature_cells.py"
source_paths = [Path(__file__), base_path, proposed_path,
                proposed_path.with_name("training.py")]
start_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths}
base = package("review_lineage_base", ROOT / "pkg-CB64-RA2")
proposed = package("review_lineage_proposed", proposed_path.parents[1])
checks = []


def checked(name, **values):
    checks.append(dict(name=name, status="PASS", **values))


def rows(values):
    return torch.tensor(values, dtype=torch.long)


def edges(graph):
    graph.validate(graph.neighbors)
    return {(row, int(partner)) for row in range(graph.n)
            for partner in graph.neighbors[row] if int(partner) >= 0}


def expect_rejected(graph, value):
    try:
        graph.validate(value)
    except ValueError:
        return
    raise AssertionError("malformed lineage state was accepted")


# No random seeds are introduced. These are small, specified counterexamples.
graph = proposed.LatentLineage(12, 2, torch.device("cpu"))
graph.register_copies(rows([]), rows([]))
assert not edges(graph) and graph.work["updates"] == 0
graph.register_copies(rows([1, 2, 3]), rows([0, 0, 0]))
assert edges(graph) == {(0, 2), (2, 0), (0, 3), (3, 0)}
assert graph.neighbors[0].tolist() == [3, 2]
graph.register_copies(rows([2]), rows([4]))
assert edges(graph) == {(0, 3), (3, 0), (2, 4), (4, 2)}
graph.register_copies(rows([5]), rows([0]))
graph.register_copies(rows([6]), rows([0]))
assert edges(graph) == {(0, 5), (5, 0), (0, 6), (6, 0), (2, 4), (4, 2)}
graph.register_copies(rows([4, 7]), rows([8, 4]))
assert edges(graph) == {(0, 5), (5, 0), (0, 6), (6, 0), (4, 8), (8, 4)}
assert int((graph.neighbors >= 0).sum(1).max()) <= graph.degree
assert graph.work["maximum_reciprocal_slots"] <= 3 * graph.degree ** 2
checked("overwrite_reciprocal_eviction_simultaneous_parent_and_empty_batch",
        edges=len(edges(graph)) // 2, degree=graph.degree,
        maximum_reciprocal_slots=graph.work["maximum_reciprocal_slots"])

empty = proposed.LatentLineage(12, 2, torch.device("cpu"))
for invalid in (-2, 12):
    broken = empty.neighbors.clone()
    broken[1, 0] = invalid
    expect_rejected(empty, broken)
broken = empty.neighbors.clone()
broken[1, 0] = 1
expect_rejected(empty, broken)
broken = empty.neighbors.clone()
broken[1] = rows([2, 2])
broken[2, 0] = 1
expect_rejected(empty, broken)
broken = empty.neighbors.clone()
broken[1, 0] = 2
expect_rejected(empty, broken)
try:
    proposed.LatentLineage(100, 65, torch.device("cpu"))
except ValueError:
    pass
else:
    raise AssertionError("unbounded lineage degree was accepted")
checked("malformed_graph_rejected_and_degree_capped")

# The graph's padding must not change the historical kernel or its gradients.
z = torch.tensor([[-10., 0.], [0., 0.], [.01, 0.], [10., 1.],
                  [20., -1.], [30., .5]], dtype=torch.float64,
                 requires_grad=True)
prior = SimpleNamespace(z=z)
sample_rows = rows([1, 2, 1, 0])
latent = z[sample_rows]
noise = torch.tensor([[.3, -.2], [-.4, .5], [.2, .7], [-.1, -.6]],
                     dtype=torch.float64)
bandwidth = torch.tensor([1., 1.], dtype=torch.float64)
old_geometry = base.BoundedLatentGeometry(rank=1, neighbors=2, chunk=2)
new_graph = proposed.LatentLineage(len(z), 1, torch.device("cpu"))
new_geometry = proposed.BoundedLatentGeometry(rank=1, neighbors=2, chunk=2,
                                             lineage=new_graph)
old_delta = old_geometry.displacement(latent, prior, bandwidth, noise)
new_delta = new_geometry.displacement(latent, prior, bandwidth, noise, rows=sample_rows)
assert torch.equal(old_delta, new_delta)
(latent + new_delta).sum().backward()
expected = torch.bincount(sample_rows, minlength=len(z)).double()[:, None].expand_as(z)
assert torch.equal(z.grad, expected)
checked("empty_graph_bitwise_kernel_and_indexed_identity_derivative")

# A nearest copy just above the query is absent from this two-slot sort window.
before = new_geometry.radius(z[rows([1])], prior, rows=rows([1]))
new_graph.register_copies(rows([1]), rows([2]))
after = new_geometry.radius(z[rows([1])], prior, rows=rows([1]))
assert torch.equal(after, torch.tensor([.005], dtype=torch.float64))
assert bool((after < before).all())
assert new_geometry.work["max_candidates"] <= new_geometry.neighbors + new_graph.degree
saved_graph = new_graph.neighbors.clone()
restored_graph = proposed.LatentLineage(len(z), 1, torch.device("cpu"))
restored_graph.validate(saved_graph)
restored_graph.neighbors = saved_graph.clone()
restored_geometry = proposed.BoundedLatentGeometry(rank=1, neighbors=2, chunk=2,
                                                  lineage=restored_graph)
assert torch.equal(new_geometry.displacement(latent, prior, bandwidth, noise,
                                            rows=sample_rows),
                   restored_geometry.displacement(latent, prior, bandwidth, noise,
                                                  rows=sample_rows))
checked("known_copy_candidate_and_graph_only_resume",
        radius_before=float(before.item()), radius_after=float(after.item()),
        max_candidates=new_geometry.work["max_candidates"])

# Backend state uses the existing saved seed; no training or model initialization.
payload_path = ROOT / "integration/review/training-regression/snapshot-1000.pt"
payload = torch.load(payload_path, map_location="cpu", weights_only=False)
recipe = SimpleNamespace(birth_death_space="critic", birth_death_isolation=True,
                         birth_death_feature_scale="reference", birth_death_cells=64,
                         birth_death_metric_rank=8, birth_death_chunk=256,
                         birth_death_parent_policy="nearest")
trainer = SimpleNamespace(prior=SimpleNamespace(z=torch.nn.Parameter(
                              torch.arange(2048, dtype=torch.float64).reshape(1024, 2))),
                          D=torch.nn.Linear(2, 1, device="meta"),
                          recipe=recipe, device=torch.device("cpu"), dtype=torch.float64,
                          controller=SimpleNamespace(latent_bandwidth=torch.ones(2)))
backend = proposed.FeatureCellBirthDeath(trainer, payload["seed"])
backend.lineage.register_copies(rows([1, 2]), rows([0, 0]))
saved_backend = deepcopy(backend.state_dict())
restored_backend = proposed.FeatureCellBirthDeath(trainer, payload["seed"])
restored_backend.load_state_dict(saved_backend)
assert torch.equal(restored_backend.lineage.neighbors, backend.lineage.neighbors)
assert restored_backend.snapshot is None and not restored_backend.latent_geometry._entries
before = restored_backend.lineage.neighbors.clone()
old_backend = base.FeatureCellBirthDeath(trainer, payload["seed"])
try:
    restored_backend.load_state_dict(old_backend.state_dict())
except ValueError:
    pass
else:
    raise AssertionError("old kernel checkpoint was accepted")
assert torch.equal(before, restored_backend.lineage.neighbors)
bad = deepcopy(saved_backend)
bad["lineage_neighbors"][1, 0] = 1024
try:
    restored_backend.load_state_dict(bad)
except ValueError:
    pass
else:
    raise AssertionError("invalid graph checkpoint was accepted")
assert torch.equal(before, restored_backend.lineage.neighbors)
checked("backend_semantic_checkpoint_cache_discard_and_rejection_before_mutation",
        backend_schema=backend.BACKEND_SCHEMA,
        latent_kernel=backend.settings["latent_kernel"],
        graph_shape=list(backend.lineage.neighbors.shape),
        reused_input_sha256=hashlib.sha256(payload_path.read_bytes()).hexdigest())

stop_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in source_paths}
assert start_hashes == stop_hashes, "a reviewed source changed during the checks"
assert not torch.cuda.is_initialized()
receipt = dict(status="PASS", checks=checks, source_sha256=start_hashes,
               cuda_initialized=False, optimizer_updates=0, new_seeds=0,
               scope="Specified CPU copy/latent contracts; no learned quality qualification")
(HERE / "lineage-review.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
