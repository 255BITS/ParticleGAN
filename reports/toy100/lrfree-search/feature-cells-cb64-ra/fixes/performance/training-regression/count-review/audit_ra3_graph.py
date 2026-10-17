"""CPU supplement: compare actual saved lineage/backend metadata with RA3 readiness."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
import hashlib
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VALIDATION = ROOT / "validation-ra3"
LEARNED = VALIDATION / "learned"
AUDIT = ROOT / "integration/review/ra3-artifact-audit-state-review"
READY = ROOT / "integration/iteration-3/READY.json"
CHECKER = ROOT / "integration/review/audit_learned.py"
CHECKPOINTS = (0, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000)
VARIANT = "CB64-RA3"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())

ready = read(READY)
inputs = read(LEARNED / "INPUTS.json")
summary = read(AUDIT / "summary.json")
assert summary["complete"] and summary["source_integrity"]["status"] == "VALID"
assert summary["cpu_only"] and not summary["cuda_initialized"]
assert read(AUDIT / "AUDITOR-IDENTITY.json")["checker_sha256"] == sha(CHECKER)
for name, row in summary["records"].items():
    assert row["evidence_status"] == "VALID", name
    assert row["primary_status"] == ("PASS" if name.startswith("replay-") else "COMPLETE"), name
package = Path(ready["package_root"])
selected = inputs["variants"][VARIANT]
assert selected["package_root"] == str(package)
assert selected["package_sha256"] == ready["package_sha256"]
assert selected["config_sha256"] == ready["config_sha256"]
assert selected["source_sha256"] == ready["package_source_sha256"]
assert sha(selected["config_path"]) == ready["config_sha256"]
sources = [Path(__file__), READY, CHECKER, AUDIT / "summary.json", AUDIT / "AUDITOR-IDENTITY.json",
           LEARNED / "INPUTS.json", LEARNED / "SOURCE-FREEZE.json", VALIDATION / "source-freeze.json",
           Path(selected["config_path"])] + sorted((package / "particlegan").rglob("*.py"))
digest = hashlib.sha256()
for p in sorted((package / "particlegan").rglob("*.py")):
    name = str(p.relative_to(package / "particlegan"))
    assert sha(p) == ready["package_source_sha256"][name]
    digest.update(name.encode() + b"\0" + p.read_bytes() + b"\0")
assert digest.hexdigest() == ready["package_sha256"]
before = {str(p): sha(p) for p in sources}
artifact_hashes = {}


def load(path):
    artifact_hashes[str(path)] = sha(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def audit_graph(path, state, recipe):
    bd = state["birth_death"]
    settings = bd["settings"]
    n = len(state["models"]["prior"]["z"])
    degree = min(recipe["birth_death_metric_rank"], 64, n-1)
    expected = dict(cells=recipe["birth_death_cells"], rank=recipe["birth_death_metric_rank"],
                    chunk=recipe["birth_death_chunk"], parent_policy=recipe["birth_death_parent_policy"],
                    lloyd=4, power=4, parent_reservoir=64, real_anchors=1, parent_rank=None,
                    latent_kernel=ready["latent_kernel"], latent_neighbors=64,
                    latent_rank=recipe["birth_death_metric_rank"], lineage_degree=degree,
                    lineage_policy="symmetric_copy_links_newest_first_invalidate_overwrites",
                    latent_candidate_bound=64+degree, mass_policy=ready["mass_policy"],
                    isolation_parent_pool="supported_same_cell_without_replacement")
    assert settings == expected, str(path)
    assert bd["backend"] == "feature_cells" and bd["backend_schema"] == ready["backend_schema"] == 4
    assert state["schema"] == 4
    assert "snapshot" not in bd and "latent_geometry" not in bd
    policy = bd["population_policy"]
    assert policy["population"] == n and policy["calibration_rows"] == n//2
    assert policy["actual_backend"] == policy["matching_sampler"] == "feature_cells"
    assert policy["finite_resolution_feasible"]
    assert policy["minimum_bh_flags"] == math.ceil(n / (.05 * (n//2 + 1)))
    assert policy["maximum_guard_flags"] == math.floor(.05*n)
    graph = bd["lineage_neighbors"]
    assert graph.dtype == torch.long and graph.shape == (n, degree)
    # Independent sparse validation requires O(N*degree) edge storage.
    directed = set()
    maximum_degree = 0
    rows_with_padding_gaps = 0
    for row, neighbors in enumerate(graph.tolist()):
        valid = [v for v in neighbors if v != -1]
        # Reciprocal invalidation can leave -1 slots between retained links;
        # the semantic graph accepts padding in any slot and masks it on use.
        rows_with_padding_gaps += neighbors != valid + [-1] * (degree-len(valid))
        assert len(valid) == len(set(valid)) and row not in valid
        assert all(0 <= v < n for v in valid)
        maximum_degree = max(maximum_degree, len(valid))
        directed.update((row, v) for v in valid)
    assert all((b, a) in directed for a, b in directed)
    assert maximum_degree <= degree
    return dict(path=str(path.relative_to(ROOT)), completed_steps=state["completed_steps"],
                backend_schema=bd["backend_schema"], settings=settings,
                lineage_shape=list(graph.shape), lineage_edges=len(directed)//2,
                maximum_degree=maximum_degree, rows_with_padding_gaps=rows_with_padding_gaps,
                lineage_sha256=hashlib.sha256(graph.contiguous().numpy().tobytes()).hexdigest(),
                ordinary_moves=bd["counters"]["ordinary_moves"], isolation_moves=bd["counters"]["iso_moves"],
                transient_snapshot_and_coordinate_cache_absent=True)


training = {}
for problem in ("toy", "mnist"):
    folder = LEARNED / "training" / problem / VARIANT
    receipt = read(folder / "config.json")
    recipe = receipt["recipe"]
    assert receipt["package"] == selected
    rows = []
    for step in CHECKPOINTS:
        path = folder / f"checkpoint-{step:04d}.pt"
        saved = load(path)
        state = saved["trainer"]
        assert json.loads(json.dumps(state["recipe"])) == recipe
        assert state["completed_steps"] == step and saved["data_position"] == 2*step*128
        row = audit_graph(path, state, recipe)
        assert row["lineage_edges"] == saved["record"]["diagnostics"]["birth_death"]["lineage_edges"]
        if step == 0:
            assert row["lineage_edges"] == row["ordinary_moves"] == row["isolation_moves"] == 0
        rows.append(row)
    assert rows[-1]["lineage_edges"] > 0 and rows[-1]["ordinary_moves"] > 0
    training[problem] = rows

replay = {}
aggregate_path = LEARNED / f"replay-{VARIANT}.json"
sources.append(aggregate_path)
before[str(aggregate_path)] = sha(aggregate_path)
for problem, result in read(aggregate_path).items():
    recipe = read(LEARNED / "training" / problem / VARIANT / "config.json")["recipe"]
    rows, graphs = [], []
    for branch in result["branches"]:
        path = Path(branch["endpoint"])
        saved = load(path)
        state = saved["trainer"]
        assert state["completed_steps"] == 1010 and saved["data_position"] == 2*1010*128
        rows.append(audit_graph(path, state, recipe))
        graphs.append(state["birth_death"]["lineage_neighbors"])
    assert torch.equal(graphs[0], graphs[1])
    replay[problem] = dict(branches=rows, lineage_graph_bit_identical=True,
                          original_full_semantic_and_loss_fingerprints_valid=True)

assert before == {str(p): sha(p) for p in sources}
assert artifact_hashes == {p: sha(p) for p in artifact_hashes}
assert not torch.cuda.is_initialized()
receipt = dict(status="PASS", ready_sha256=sha(READY), package_sha256=ready["package_sha256"],
               original_completed_artifact_audit_valid=True, training=training, replay=replay,
               source_sha256=before, checkpoint_endpoint_sha256=artifact_hashes,
               cuda_initialized=False, optimizer_updates=0, new_seeds=0,
               scope="Actual saved sparse graph/backend/settings and exact completed artifact receipts; no CUDA execution")
(HERE / "ra3-graph-review.json").write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps(dict(status="PASS", training={p:rows[-1] for p, rows in training.items()},
                     replay={p:dict(lineage_graph_bit_identical=v["lineage_graph_bit_identical"]) for p,v in replay.items()},
                     cuda_initialized=False), indent=2), flush=True)
