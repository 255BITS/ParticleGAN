"""Independent narrow review of immutable paired-copy sources and receipts."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path("/ml2/hypergan/gan-attempts/feature-cells-fixes-20260929")
OWNER = ROOT / "performance/sampler-regression/cpu-plan-review/post-ra4-quality"
OUT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ready = json.loads((OWNER / "READY.json").read_text())
assert sha(OWNER / "READY.json") == "6ecd770ffaf2646cb6329647b78adc1ac62b866d4417a9960de8e62ff4bd1e23"
checked = {str(OWNER / "READY.json"): sha(OWNER / "READY.json")}
for key in ("helper_source_sha256", "numerical_source_sha256"):
    for path, expected in ready[key].items():
        assert sha(path) == expected, (key, path)
        checked[path] = expected
base_root = Path(ready["base_package_root"]) / "particlegan"
new_root = Path(ready["package_root"]) / "particlegan"
changed = []
for name, expected in ready["package_source_sha256"].items():
    assert sha(new_root / name) == expected
    checked[str(new_root / name)] = expected
    checked[str(base_root / name)] = sha(base_root / name)
    if sha(new_root / name) != sha(base_root / name):
        changed.append(name)
assert changed == ["feature_cells.py"]
old_tree, new_tree = (ast.parse((root / "feature_cells.py").read_text()) for root in (base_root, new_root))
old_cls, new_cls = (next(n for n in tree.body if isinstance(n, ast.ClassDef)
    and n.name == "FeatureCellBirthDeath") for tree in (old_tree, new_tree))
method = lambda cls, name: next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
old_move, new_move = method(old_cls, "_move"), method(new_cls, "_move")
calls = [n for n in ast.walk(new_move) if isinstance(n, ast.Call)]
assert sum(ast.unparse(n.func) == "torch.randn" for n in calls) == 1
paired = [n for n in calls if ast.unparse(n.func) == "self.latent_geometry.displacement"]
assert len(paired) == 1
assert [ast.unparse(a) for a in paired[0].args] == ["ema_zp", "ema", "trainer.controller.latent_bandwidth", "noise"]
assert [(k.arg, ast.unparse(k.value)) for k in paired[0].keywords] == [("rows", "parent")]
try_node = next(n for n in new_move.body if isinstance(n, ast.Try))
ema_compute = next(i for i,n in enumerate(try_node.body) if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "ema_delta" for t in n.targets))
live_write = next(i for i,n in enumerate(try_node.body) if isinstance(n, ast.Assign)
    and any(ast.unparse(t) == "prior.z[child]" for t in n.targets))
ema_write = next(n for n in try_node.body if isinstance(n, ast.Assign)
    and any(ast.unparse(t) == "ema.z[child]" for t in n.targets))
assert ema_compute < live_write and ast.unparse(ema_write.value) == "ema_zp + ema_delta"
new_cls.body[new_cls.body.index(new_move)] = deepcopy(old_move)
schema = next(n for n in new_cls.body if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "BACKEND_SCHEMA" for t in n.targets))
assert schema.value.value == 5
schema.value.value = 4
init = method(new_cls, "__init__")
extra = [n for n in init.body if isinstance(n, ast.Assign)
    and any(ast.unparse(t) == "self.settings['copy_noise_policy']" for t in n.targets)]
assert len(extra) == 1 and extra[0].value.value == "shared_noise_separate_live_ema_current_geometry_v1"
init.body.remove(extra[0])
assert ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)

copy = json.loads((OWNER / "cpu-copy-contract-01.json").read_text())
control = json.loads((OWNER / "cpu-equal-control-02.json").read_text())
assert copy["status"] == control["status"] == "PASS" and len(copy["cases"]) == 9
for row in copy["cases"]:
    for key in ("fast_z_moments_history_graph_rng_bit_identical", "single_noise_draw_accounting",
                "ema_exact_own_geometry_placement", "both_versioned_caches_rebuilt"):
        assert row[key]
    assert row["new_ema_radius_violation_rows"] == 0
assert copy["resume"]["old_schema_and_forged_schema_missing_policy_rejected_before_mutation"]
assert copy["resume"]["post_copy_save_load_and_two_copy_continuation_bit_identical"]
assert control["equal_priors_preserve_old_ema_bits"] and copy["cuda_initialized"] is False
assert copy["optimizer_updates"] == control["numerical_updates"] == 0
for obj in (copy, control):
    for path, expected in obj["source_sha256"].items():
        assert sha(path) == expected
assert all(sha(p) == expected for p, expected in checked.items())
receipt = dict(status="PASS", scope="independent source/AST/freeze and saved CPU receipt review; no duplicate numerical suite",
    checks=dict(all_owner_hashes=True, full_module_restoration=True, only_changed_module=changed,
        one_shared_noise_draw=True, both_displacements_before_writes=True,
        separate_live_ema_geometry=True, owner_copy_contracts=9, owner_equal_controls=1,
        checkpoint_atomic_distinct_law=True, exact_continuation=True),
    reviewed_hashes=checked, optimizer_updates=0, cuda_contexts=0, new_seeds=0,
    limitations=["Historical RA4 copy actions were not reconstructed.",
        "Native saved 2D proof is RA2, not RA4.", "Strict learned/native quality effect remains unqualified."])
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
frozen = dict(status="PASS", files={str(OUT / n):sha(OUT / n) for n in ("audit.py", "receipt.json")},
    reviewed_hashes=checked)
(OUT / "FROZEN.json").write_text(json.dumps(frozen, indent=2) + "\n")
print(json.dumps(dict(status="PASS", receipt=str(OUT / "receipt.json"),
    receipt_sha256=sha(OUT / "receipt.json"), freeze_sha256=sha(OUT / "FROZEN.json")), indent=2))
