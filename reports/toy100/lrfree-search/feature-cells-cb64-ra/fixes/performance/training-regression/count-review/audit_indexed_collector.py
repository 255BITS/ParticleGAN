"""CPU pilot of one declared-API expectation change; strict records stay intact."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode = True
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LANE = ROOT / "validation-ra4/screens"
OUTPUT = HERE / "indexed-collector-pilot"
assert not OUTPUT.exists()
OUTPUT.mkdir()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
SOURCE = LANE / "collect.py"
READY = ROOT / "integration/iteration-4/READY.json"
API_RECEIPT = HERE / "ra4-source-review.json"
STRICT = ROOT / "integration/review/ra4-validation-monitor/canonical-receipts/screens/runs"
TASKS = ("mode_hold", "img_blobs4", "vector_unequal_mass")
ready = read(READY)
api = read(API_RECEIPT)
assert api["status"] == "PASS" and api["package_sha256"] == ready["package_sha256"] == "e34bcb21aaa64caa0601cea5dc1f9b8eaebee9578686ebff39b459676063deb2"
paths = [Path(__file__), SOURCE, LANE/"lane.py", READY, API_RECEIPT, HERE/"RA4-SOURCE-FROZEN.json",
         LANE/"source-freeze.json", LANE/"READY.json", LANE/"candidate-options.json"]
for task in TASKS:
    paths += [LANE/"runs"/task/name for name in ("result.json", "job-header.json", "execution-receipt.json", "metrics.jsonl", "final-state.pt")]
    paths.append(STRICT/task/"acceptance-receipt.json")
before = {str(p):sha(p) for p in paths}
spec = importlib.util.spec_from_file_location("lane", LANE/"lane.py")
lane = importlib.util.module_from_spec(spec)
sys.modules["lane"] = lane
spec.loader.exec_module(lane)
assert lane.verify_frozen()["status"] == "VALID"
assert lane.PACKAGE_SHA == ready["package_sha256"]
tree = ast.parse(SOURCE.read_text())
collect = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="collect")
assignment = next(n for n in ast.walk(collect) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=="expected_options" for t in n.targets))
keyword = next(k for k in assignment.value.keywords if k.arg=="evaluation_generate")
assert ast.literal_eval(keyword.value) == "plain"
expected_plain = {kw.arg:ast.literal_eval(kw.value) for kw in assignment.value.keywords}
keyword.value = ast.copy_location(ast.Constant(value="indexed"),keyword.value)
adapted_ast = deepcopy(tree)
keyword.value = ast.copy_location(ast.Constant(value="plain"),keyword.value)
assert ast.dump(tree,include_attributes=False) == ast.dump(ast.parse(SOURCE.read_text()),include_attributes=False)
adapter = ModuleType("review_declared_indexed_collector")
adapter.__file__ = str(SOURCE)
exec(compile(adapted_ast,str(SOURCE),"exec"),adapter.__dict__)
# CPU mechanism reads retain all checks of the actual physical-GPU receipts.
adapter.ENV = dict(adapter.ENV, CUDA_VISIBLE_DEVICES="")
def redirected_write(path,value):
    p = Path(path).resolve()
    assert p.is_relative_to(LANE)
    dest = OUTPUT / "adapted-receipts" / p.relative_to(LANE)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(value,indent=2)+"\n")
adapter.write = redirected_write
records = []
for task in TASKS:
    original = read(LANE/"runs"/task/"result.json")
    header = read(LANE/"runs"/task/"job-header.json")
    strict = read(STRICT/task/"acceptance-receipt.json")
    assert original["header"] == header
    options = header["options"]
    assert options == dict(expected_plain,evaluation_generate="indexed")
    assert header["auto_detected"]["evaluation_generate"] == "indexed"
    assert not original["warnings"] and original["stream_deviations"] == 0
    assert len(strict["validity_reasons"]) == 1 and strict["validity_reasons"][0].startswith("original resolved options differ: ")
    assert strict["primary_status"] == "PASS" and strict["acceptance_status"] == "ERROR" and strict["canonical_fixture_validity"] == "INVALID"
    adapted = adapter.collect(task,require_attempt=True)
    assert adapted["primary_status"] == strict["primary_status"]
    assert adapted["acceptance_status"] == "PASS"
    assert adapted["canonical_fixture_validity"] == "VALID" and not adapted["validity_reasons"]
    modified = {"collected_at", "acceptance_status", "canonical_fixture_validity", "validity_reasons", "canonical_gpu_acceptance"}
    assert {k:v for k,v in adapted.items() if k not in modified} == {k:v for k,v in strict.items() if k not in modified}
    records.append(dict(task=task,primary_status=strict["primary_status"],strict_acceptance="ERROR",
        adapted_acceptance=adapted["acceptance_status"],adapted_validity=adapted["canonical_fixture_validity"],
        only_declared_option_changed=True, all_other_canonical_record_fields_exact=True,
        warnings=[],stream_deviations=0, fixture_drift_observed=False))

# A second option mismatch and a package mismatch remain INVALID. Inputs are
# copied only in memory; no saved headers/results or strict receipts change.
real_read = adapter.read
result_path = LANE/"runs"/TASKS[0]/"result.json"
guards = []
for name, mutate in (("other_option",lambda r:r["header"]["options"].update(strict_streams=False)),
                     ("wrong_candidate",lambda r:r["header"].update(package_sha256="0"*64))):
    def changed(path, mutate=mutate):
        value = real_read(path)
        if Path(path) == result_path:
            value = deepcopy(value)
            mutate(value)
        return value
    adapter.read = changed
    adapter.write = lambda *args:None
    rejected = adapter.collect(TASKS[0],require_attempt=True)
    assert rejected["acceptance_status"] == "ERROR" and rejected["canonical_fixture_validity"] == "INVALID"
    guards.append(dict(case=name,reasons=rejected["validity_reasons"],status="PASS"))
adapter.read = real_read
assert before == {str(p):sha(p) for p in paths}
assert lane.verify_frozen()["status"] == "VALID" and not torch.cuda.is_initialized()
receipt = dict(status="PASS",records=records,guards=guards,source_sha256=before,
    expected_options_historical_plain=expected_plain,
    expected_options_declared_ra4=dict(expected_plain,evaluation_generate="indexed"),
    adapter_change="Only the in-memory collect.expected_options evaluation_generate literal: plain -> indexed",
    strict_receipts_unchanged=True,validation_inputs_unchanged=True,
    original_primary_verdicts_and_quality_metrics_unchanged=True,
    data_init_streams_schedules_scorers_thresholds_and_budgets_unchanged=True,
    historical_plain_evaluation_semantically_identical_claim=False,
    scope="Saved artifacts accepted under the predeclared RA4 indexed sampler; historical plain-option mismatch remains recorded",
    cpu_only=True,cuda_initialized=False,optimizer_updates=0,new_seeds=0)
(OUTPUT/"review.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(status="PASS",records=records,guards=guards,receipt=str(OUTPUT/"review.json"))),flush=True)
