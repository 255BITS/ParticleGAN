"""Independent read-only CPU review of the declared indexed collector adapter."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
sys.dont_write_bytecode=True
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OWNER=ROOT/"performance/sampler-regression/cpu-plan-review/indexed-metadata"
COLLECTOR=ROOT/"validation-ra4/screens/collect.py"
MONITOR=ROOT/"integration/review/ra4-indexed-api-monitor"
READY=OWNER/"READY.json"
PILOT=OWNER/"PILOT.json"
OUTPUT=HERE/"indexed-owner-review.json"
assert not OUTPUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
declaration,pilot=read(READY),read(PILOT)
assert sha(READY)=="e76a712d99f3a224a99f1e26a4dcfe813eb61617f217577dd60d69f12af5dee4"
assert sha(PILOT)=="cb9620d116112d710abd4d31f17f101f1d21d58c31b24bcc2bf8c2b738fd6b03"
assert pilot["status"]=="VALID" and pilot["ready_sha256"]==sha(READY)
paths={Path(__file__),READY,PILOT,HERE/"ra4-source-review.json",HERE/"RA4-SOURCE-FROZEN.json",
       HERE/"INDEXED-COLLECTOR-PILOT-FROZEN.json",HERE/"indexed-collector-e22-reference.json",
       HERE/"ra4-settings-ast-supplement.json"}
paths.update(map(Path,declaration["exact_source_sha256"]))
paths.update(map(Path,pilot["source_and_input_sha256"]))
before={str(p):sha(p) for p in paths}
for p,expected in declaration["exact_source_sha256"].items():assert before[p]==expected,p
for p,expected in pilot["source_and_input_sha256"].items():assert before[p]==expected,p
spec=importlib.util.spec_from_file_location("independent_indexed_owner",OWNER/"indexed_adapter.py")
adapter=importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
assert adapter.guard()["status"]=="VALID"
original,adapted,proof=adapter.collector_trees()
assert proof==declaration["collector_ast_proof"]==pilot["collector_ast_proof"]
assert ast.dump(original,include_attributes=False)==ast.dump(ast.parse(COLLECTOR.read_text()),include_attributes=False)
# Independently undo the single change, without using the owner's undo helper.
reverted=deepcopy(adapted)
function=next(n for n in reverted.body if isinstance(n,ast.FunctionDef) and n.name=="collect")
assignment=next(n for n in ast.walk(function) if isinstance(n,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=="expected_options" for t in n.targets))
keyword=next(kw for kw in assignment.value.keywords if kw.arg=="evaluation_generate")
assert keyword.value.value=="indexed"
keyword.value.value="plain"
assert ast.dump(reverted,include_attributes=False)==ast.dump(original,include_attributes=False)
api=adapter.resolve_declared_api()
assert api==pilot["api"]
assert set(api["resolved_options"])==set(read(HERE/"indexed-collector-pilot/review.json")["expected_options_declared_ra4"])
assert api["resolved_options"]==read(HERE/"indexed-collector-pilot/review.json")["expected_options_declared_ra4"]
assert "image_steps" not in api["resolved_options"] and "native_steps" not in api["resolved_options"]

summary_path=MONITOR/"summary.json"
summary=read(summary_path)
snapshot=HERE/"indexed-owner-monitor-snapshot-attempt2.json"
assert not snapshot.exists()
snapshot.write_bytes(summary_path.read_bytes())
observed=[row for row in summary["records"] if row["acceptance_status"]!="PENDING"]
assert {row["task"] for row in observed}=={"mode_hold","img_blobs4","vector_unequal_mass","ring_shift"}
records=[]
for row in observed:
    task=row["task"]
    base=MONITOR/"canonical-receipts/screens/runs"/task
    strict_path=base/"strict-plain-acceptance-receipt.json"
    adapted_path=base/"acceptance-receipt.json"
    strict,current=read(strict_path),read(adapted_path)
    # The monitor writer puts a compact declaration annotation in each
    # canonical receipt; the returned summary row retains the detailed strict
    # interpretation. All original collector fields must still agree.
    assert {k:v for k,v in row.items() if k!="indexed_api_expectation_adapter"}=={k:v for k,v in current.items() if k!="indexed_api_expectation_adapter"}
    assert strict["primary_status"]==current["primary_status"]=="PASS"
    assert strict["acceptance_status"]=="ERROR" and strict["canonical_fixture_validity"]=="INVALID"
    assert len(strict["validity_reasons"])==1 and strict["validity_reasons"][0].startswith("original resolved options differ: ")
    assert current["acceptance_status"]=="PASS" and current["canonical_fixture_validity"]=="VALID"
    assert not current["validity_reasons"]
    excluded=adapter.STATUS_FIELDS|{"indexed_api_expectation_adapter"}
    assert {k:v for k,v in strict.items() if k not in excluded}=={k:v for k,v in current.items() if k not in excluded}
    compact=current["indexed_api_expectation_adapter"]
    assert compact["declaration_ready_sha256"]==sha(READY)
    assert compact["all_other_canonical_checks_unchanged"] and compact["quality_gates_unchanged"]
    assert compact["exact_ast_delta"]==proof["exact_ast_delta"]
    annotation=row["indexed_api_expectation_adapter"]
    assert annotation["original_strict_reasons"]==strict["validity_reasons"]
    assert annotation["original_primary_status"]==current["primary_status"]
    assert annotation["quality_verdict_unchanged"] and annotation["all_other_output_fields_exact"]
    assert annotation["collector_ast_proof"]==proof
    result_path=ROOT/"validation-ra4/screens/runs"/task/"result.json"
    result=read(result_path)
    assert result["header"]["options"]==api["resolved_options"]
    assert result["header"]["auto_detected"]==api["auto_detected"]
    assert sha(result_path)==current["result_sha256"]
    old_path=Path(annotation["original_error_receipt_path"])
    assert sha(old_path)==annotation["original_error_receipt_sha256"]
    paths.update((strict_path,adapted_path,result_path,old_path))
    for p in (strict_path,adapted_path,result_path,old_path):before[str(p)]=sha(p)
    records.append(dict(task=task,primary="PASS",strict="ERROR/INVALID",declared_indexed="PASS/VALID",
        all_other_collector_fields_exact=True,original_error_receipt_unchanged=True))
identity=read(MONITOR/"CHECKER-IDENTITY.json")
assert identity["original_canonical_checks_unchanged"] is False and identity["write_redirection_only"] is False
assert identity["indexed_api_expectation_adapter"]["all_other_canonical_checks_unchanged"]
assert identity["indexed_api_expectation_adapter"]["quality_gates_unchanged"]
controls={row["name"]:row for row in pilot["negative_controls"]}
assert set(controls)=={"wrong_other_option","wrong_package","wrong_stream","wrong_api_mode","quality_fail"}
for name,row in controls.items():
    assert row["memory_overlay_only"]
    assert (row["status"],row["validity"])==(("FAIL","VALID") if name=="quality_fail" else ("ERROR","INVALID"))
assert before=={str(p):sha(p) for p in paths}
assert not torch.cuda.is_initialized()
receipt=dict(status="PASS",source_sha256=before,ready_sha256=sha(READY),pilot_sha256=sha(PILOT),
    records=records,collector_ast_proof=proof,api=api,negative_controls_verified=controls,
    monitor_snapshot_path=str(snapshot),monitor_snapshot_sha256=sha(snapshot),
    live_summary_sha256_observed=sha(summary_path),live_monitor_output_expected_to_evolve=True,
    exact_host_unset_budget_key_removal=True,old_strict_error_receipts_preserved=True,
    all_original_numerical_data_init_stream_schedule_scorer_gate_checks_retained=True,
    inherited_settings_ast_supplement_sha256=sha(HERE/"ra4-settings-ast-supplement.json"),
    historical_e22_reference_sha256=sha(HERE/"indexed-collector-e22-reference.json"),
    cpu_only=True,cuda_initialized=False,optimizer_updates=0,new_seeds=0,numerical_reruns=0,
    scope="Declared indexed RA4 metadata acceptance; strict historical plain interpretation is retained and quality verdicts are unchanged")
OUTPUT.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(status="PASS",receipt=str(OUTPUT),receipt_sha256=sha(OUTPUT),records=records,
                     ready_sha256=sha(READY),pilot_sha256=sha(PILOT),cuda_initialized=False)),flush=True)
