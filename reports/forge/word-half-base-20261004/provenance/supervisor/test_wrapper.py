"""Synthetic model-free supervisor controls; no preparation or scientific calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location("word_half_supervisor_controls",Path(__file__).with_name("run_supervised.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def wire_request():
    ids=[s.TASK if p==s.PARENT else p for p in s.PARENTS]
    task=dict(policy_family=s.FAMILY,task_cohort=s.COHORT,policy_parent={"id":s.PARENT},preflight_blockers=[],
        execution=dict(steps=20001,original_schedule_horizon=20000,execution_path="public_components",device="cuda",
            resources=dict(num_particles=11,z_dim=2,batch_size=256),prior=dict(kind="particle_cloud",sigma=0.,standardize=False)),
        evaluation=dict(thresholds=deepcopy(s.THRESHOLDS),observations=24,minimum_stable_checks=5,eval_samples=1024,scoring_weights="state_selected"),
        resources=dict(timeout_seconds=900))
    return dict(candidate=dict(id=s.CANDIDATE,word_rate_profile=s.PROFILE,task_cohort=s.COHORT,trainer_family=s.FAMILY,
            recipe_preset="atlas",recipe_overrides=deepcopy(s.OVERRIDES),execution_path="public_trainer"),
        view=dict(assignments=[dict(task=name,qualification_tier=1 if i<5 else 2 if i<24 else 3,importance="required") for i,name in enumerate(ids)],
            revision=4,policy_family=s.FAMILY,task_cohort=s.COHORT),
        tasks={name:deepcopy(task) if name==s.TASK else {"id":name} for name in ids},protocol={"seed":0},
        jobs=[dict(task_id=s.TASK,task_ids=[s.TASK],budget_seconds=900)],source={"digest":"a"*64})


def test_json_roundtrip_preserves_full26_and_real_order_without_dict_order_dependence():
    value=json.loads(json.dumps(wire_request(),sort_keys=True))
    assert s.validate_request(value)["task_ids"]==[s.TASK]
    assert list(value["tasks"])!=[a["task"] for a in value["view"]["assignments"]]


@pytest.mark.parametrize("key,value",[("id","word-min11-quarter_base-rates-v1"),("word_rate_profile","quarter_base"),
    ("trainer_family","atlas_word_joint_min11"),("task_cohort","word_joint_policy_min11_v1"),
    ("execution_path","public_components"),("recipe_preset","ka2")])
def test_candidate_profile_family_reference_context_fail_closed(key,value):
    data=wire_request(); data["candidate"][key]=value
    with pytest.raises(ValueError,match="candidate/profile/tuple"): s.validate_request(data)


@pytest.mark.parametrize("key,value",[("lr",.0053125),("prior_lr_mult",.15),("d_lr_mult",True),("new_unknown_field",1)])
def test_only_fixed_global_half_base_tuple_can_run(key,value):
    data=wire_request(); data["candidate"]["recipe_overrides"][key]=value
    with pytest.raises(ValueError): s.validate_request(data)


@pytest.mark.parametrize("mutation",["missing","extra","order","tier","optional","seed","boolean_seed","old_family","preflight","steps","horizon","N5","prior","sigma","standardize","gate","cadence","hold","samples","weights","cap","multi_job"])
def test_domain_gates_prior_and_full_denominator_mutations_refused(mutation):
    data=wire_request(); task=data["tasks"][s.TASK]
    if mutation=="missing": del data["tasks"]["trajectory"]
    elif mutation=="extra": data["tasks"]["foreign"]={}
    elif mutation=="order": data["view"]["assignments"].reverse()
    elif mutation=="tier": data["view"]["assignments"][0]["qualification_tier"]=2
    elif mutation=="optional": data["view"]["assignments"][0]["importance"]="optional"
    elif mutation=="seed": data["protocol"]["seed"]=1
    elif mutation=="boolean_seed": data["protocol"]["seed"]=False
    elif mutation=="old_family": task["policy_family"]="atlas_word_joint_min11"
    elif mutation=="preflight": task["preflight_blockers"]=["missing owner"]
    elif mutation=="steps": task["execution"]["steps"]=20000
    elif mutation=="horizon": task["execution"]["original_schedule_horizon"]=20001
    elif mutation=="N5": task["execution"]["resources"]["num_particles"]=5
    elif mutation=="prior": task["execution"]["prior"]["kind"]="mog"
    elif mutation=="sigma": task["execution"]["prior"]["sigma"]=.025
    elif mutation=="standardize": task["execution"]["prior"]["standardize"]=True
    elif mutation=="gate": task["evaluation"]["thresholds"][1][2]=.9
    elif mutation=="cadence": task["evaluation"]["observations"]=23
    elif mutation=="hold": task["evaluation"]["minimum_stable_checks"]=1
    elif mutation=="samples": task["evaluation"]["eval_samples"]=128
    elif mutation=="weights": task["evaluation"]["scoring_weights"]="forced_ema"
    elif mutation=="cap": task["resources"]["timeout_seconds"]=901
    else: data["jobs"][0]["task_ids"].append("trajectory")
    with pytest.raises(ValueError): s.validate_request(data)


def test_original24_reads_and_nine_frames_have_no_fabricated_zero_or_new_hold():
    assert s.STEPS==[834,1667,2501,3334,4167,5001,5834,6667,7501,8334,9168,10001,10834,11668,12501,13334,14168,15001,15835,16668,17501,18335,19168,20001]
    assert s.STEPS[-5:]==[16668,17501,18335,19168,20001]
    assert s.MEDIA_STEPS==[834,3334,5834,8334,10834,12501,15001,17501,20001]
    assert s.protocol()["attempts"]==1 and s.protocol()["export_grace_seconds"]==0


@pytest.mark.parametrize("field,value",[("attempts",2),("allowance_seconds",901),("seed",1),("physical_gpu","0"),("export_grace_seconds",60),("required_slots",1)])
def test_protocol_is_one_attempt_exact900_no_retries_no_reduced_view(field,value):
    p=s.protocol(); p[field]=value
    with pytest.raises(ValueError): s.validate_protocol(p)


def test_original_history_once_and_remaining_full_allowance():
    prior=dict(prior_charged_seconds=s.PRIOR_TOTAL,prior_lane_charged_seconds=s.PRIOR_LANES)
    assert s.allowance_fits(prior)
    assert s.allowance_fits(prior,1424.5869376168)
    assert not s.allowance_fits(prior,1424.586938)
    assert not s.allowance_fits(prior,1300,200)
    prior["prior_lane_charged_seconds"]={"0":0,"1":0}
    with pytest.raises(ValueError): s.allowance_fits(prior)


def test_durable_measurement_vs_unknown_interruption_reserve():
    assert s.charge(20,None)==dict(paid_wall_seconds=20,unmeasured_interrupt_reserved_seconds=880,charged_seconds=900)
    assert s.charge(900.4,{"attempt_status":"timeout"})==dict(paid_wall_seconds=900.4,unmeasured_interrupt_reserved_seconds=0.,charged_seconds=900.4)
    assert s.charge(10,{"attempt_status":"completed"})["charged_seconds"]==10
    with pytest.raises(ValueError): s.charge(True,None)


@pytest.mark.parametrize("status",["error","cancelled","timeout"])
def test_noncompleted_terminal_keeps_measured_paid_and_full_allowance_reserve(status):
    assert s.charge(10.,{"attempt_status":status})==dict(paid_wall_seconds=10.,unmeasured_interrupt_reserved_seconds=890.,charged_seconds=900.)
    assert s.charge(901.,{"attempt_status":status})==dict(paid_wall_seconds=901.,unmeasured_interrupt_reserved_seconds=0.,charged_seconds=901.)


@pytest.mark.parametrize("child_exit",[0,1])
def test_completed_numeric_fail_or_nonzero_infrastructure_exit_closes_measured_cost(child_exit):
    assert s.charge(10.,dict(attempt_status="completed",child_returncode=child_exit))["charged_seconds"]==10.


def test_full_json_wire_identity_accepts_tuple_serialization_without_dropping_fields():
    value=wire_request(); value["candidate"]["resolved_recipe"]={"betas":(0.,.999),"alpha_bar":(1.,2.),"direct_particle_betas":(0.,.9)}
    wire=json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
    assert value!=wire
    s.require_wire_identity(value,wire)
    for area in ("source","gates","resources","seed","ownership","annotation"):
        changed=deepcopy(wire)
        if area=="source": changed["source"]["digest"]="b"*64
        elif area=="gates": changed["tasks"][s.TASK]["evaluation"]["thresholds"][1][2]=.9
        elif area=="resources": changed["tasks"][s.TASK]["execution"]["resources"]["num_particles"]=5
        elif area=="seed": changed["protocol"]["seed"]=False
        elif area=="ownership": changed["candidate"]["resolved_recipe"]["direct_particle_betas"][0]=.1
        else: changed["tasks"][s.TASK]["field_ownership"]={"foreign":True}
        with pytest.raises(ValueError,match="planner/JSON/request"): s.require_wire_identity(value,changed)


def test_actual_planner_roundtrip_and_actual_maintained_recovery_match_without_models(tmp_path):
    """Real metadata APIs, synthetic terminal files and no source preparation.

    Only hardware profile/source-origin inspection are stubbed: the latter
    reads the complete current file-byte manifest with the root-supplied f9
    identity, avoiding Git invocation. All declaration/context planning and
    maintained _released_attempt accounting remain actual source functions.
    """
    script=r'''
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import importlib.util,json,sys
helper,root,output=map(Path,sys.argv[1:])
spec=importlib.util.spec_from_file_location("half_v2_real_metadata",helper)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
s.guards().select_snapshot_path(root)
import torch
from experiments.forge import planning,runtime,evaluate,views,policy_execution
torch.set_num_threads(1)
assert s.sha(root/s.CONTRACT)==s.CONTRACT_SHA
assert not torch.cuda.is_initialized()
before=torch.get_rng_state().clone()
def forbidden(*a,**k): raise AssertionError("structural control cannot construct/forward/draw/score/execute")
files=s.source_paths(root)
manifest=dict(schema_version=1,origin_commit=s.ORIGIN,files=files,digest=s.digest(files))
with ExitStack() as stack:
    for owner,name in ((torch.nn.Module,"__init__"),(torch.nn.Module,"__call__"),(runtime,"execute"),(evaluate,"evaluate"),(views,"grade_result")):
        stack.enter_context(patch.object(owner,name,forbidden))
    for name in ("rand","randn","randint","randperm","multinomial","normal","bernoulli"):
        stack.enter_context(patch.object(torch,name,forbidden))
    stack.enter_context(patch.object(planning,"compute_profile",lambda backend,*a,**k:dict(backend=backend,model="synthetic_metadata_A6000",availability="available",threads=1)))
    stack.enter_context(patch.object(planning,"inspect_source",lambda *a,**k:manifest))
    value=s.build_request(root)
    wire=json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
    assert value!=wire
    differing=[k for k,v in value["candidate"]["resolved_recipe"].items() if v!=wire["candidate"]["resolved_recipe"][k]]
    assert set(differing)=={"alpha_bar","betas","direct_particle_betas"}
    s.require_wire_identity(value,wire)
    changed=json.loads(json.dumps(wire));changed["tasks"][s.TASK]["evaluation"]["thresholds"][1][2]=.9
    try:s.require_wire_identity(value,changed)
    except ValueError:pass
    else:raise AssertionError("full canonical task mutation accepted")
    assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
records=[]
for i,(status,paid,exitcode) in enumerate((("completed",10.,0),("completed",10.,1),("error",10.,None),("cancelled",10.,-15),("timeout",10.,-9),("timeout",901.,-9),(None,0.,None))):
    directory=output/f"terminal-{i}";directory.mkdir(parents=True)
    terminal=None if status is None else dict(token="software_owned",attempt_status=status,paid_wall_seconds=paid,child_returncode=exitcode)
    if terminal is not None:s.write(directory/"supervisor-terminal.json",terminal)
    entry=dict(lease_path=str(directory/"execution.lock"),token="software_owned",allowance_seconds=s.LIMIT)
    policy_execution._released_attempt(entry)
    new=s.charge(paid,terminal)
    assert new["charged_seconds"]==entry["charged_seconds"]
    assert new["paid_wall_seconds"]==paid
    records.append(dict(status=status,child_exit=exitcode,new=new,maintained_charged_seconds=entry["charged_seconds"]))
assert torch.equal(before,torch.get_rng_state()) and not torch.cuda.is_initialized()
s.write(output/"metadata-control.json",dict(status="PASS_METADATA_ONLY",tuple_fields=differing,tasks=len(wire["tasks"]),word_preflight=wire["tasks"][s.TASK]["preflight_blockers"],full_identity=s.digest(wire),recovery=records,model_constructions=0,forwards=0,draws=0,updates=0,scorer_calls=0,cuda_initialized=False,global_rng_unchanged=True,source_snapshot_prepared=False,queue_actions=0))
'''
    env={**s.os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1"}
    process=subprocess.run([sys.executable,"-c",script,str(Path(__file__).with_name("run_supervised.py")),
        "/ml2/hypergan/ParticleGAN-atlas-forge-unblock-20261003",str(tmp_path)],env=env,capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stdout+process.stderr
    value=s.read(tmp_path/"metadata-control.json")
    assert value["status"]=="PASS_METADATA_ONLY" and value["tasks"]==26 and value["word_preflight"]==[]
    assert value["cuda_initialized"] is False and value["global_rng_unchanged"] is True
    assert len(value["recovery"])==7


@pytest.mark.parametrize("row,okay",[("12288,82,NVIDIA RTX A6000",True),("12287,70,NVIDIA RTX A6000",False),
    ("13000,83,NVIDIA RTX A6000",False),("13000,60,Other",False)])
def test_telemetry_exact_gpu1_thresholds(row,okay):
    def query(command,**kwargs):
        assert command[1]=="--id=1"; return row
    if okay: assert s.readiness(query)["physical_gpu"]=="1"
    else:
        with pytest.raises(ValueError): s.readiness(query)


def test_strict_namespace_and_torch_pseudo_source_controls(tmp_path,monkeypatch):
    root=tmp_path/"snapshot"; root.mkdir(); (root/"experiments").mkdir(); module=root/"experiments"/"member.py"; module.write_text("value=1\n")
    source=dict(snapshot_path=str(root),files={"experiments/member.py":s.sha(module)})
    guards=s.guards(); monkeypatch.setattr(sys,"path",[str(root),str(root/"."),str(tmp_path)])
    guards.select_snapshot_path(root)
    assert sum(Path(p).resolve()==root for p in sys.path)==1
    good={"experiments":SimpleNamespace(__file__=None,__path__=[str(root/"experiments")]),
        "experiments.member":SimpleNamespace(__file__=str(module)),"torch.ops":SimpleNamespace(__file__="_ops.py")}
    assert set(guards.guard_imports(source,good))=={"experiments.member"}
    good["experiments"].__path__.append(str(root/"experiments"))
    with pytest.raises(ValueError,match="namespace"): guards.guard_imports(source,good)
    for path in ("_ops.py",str(tmp_path/"foreign.py")):
        modules={"experiments.member":SimpleNamespace(__file__=path)}
        with pytest.raises(ValueError): guards.guard_imports(source,modules)
    with pytest.raises(ValueError,match="unpinned"):
        guards.guard_imports({**source,"files":{}},{"foreign_alias":SimpleNamespace(__file__=str(module))})


def synthetic_history(tmp_path,monkeypatch):
    raw=tmp_path/"original-terminal.json"; s.write(raw,{"status":"error","paid_wall_seconds":10})
    current=649.2088527791202
    costs=dict(current_reserved_seconds=0,original_cap_seconds=10500,inclusive_charged_seconds=s.PRIOR_TOTAL,current_paid_seconds=current,
        lanes={g:dict(inclusive_charged_seconds=s.PRIOR_LANES[g],cap_seconds=7500 if g=="0" else 3000) for g in s.PRIOR_LANES})
    published=tmp_path/"results.json"; index=tmp_path/"input-index.json"; card=tmp_path/"card.json"
    s.write(published,dict(cost=costs,source={"digest":"a"*64},qualification_input=False))
    s.write(index,dict(file_count=1,files=[s.pin(raw)])); s.write(card,dict(schema="synthetic_known_history",qualification_input=False,prepared=s.pin(raw),studies={}))
    monkeypatch.setattr(s,"PRIOR_PINS",{k:(p,s.sha(p)) for k,p in {"results":published,"index":index,"card":card}.items()})
    calls=[]
    class Inputs:
        def json(self,p): return s.read(s.checked(p))
        def recheck(self): calls.append("recheck")
    def step(name,value):
        def f(*args): calls.append(name); return value
        return f
    old=SimpleNamespace(INPUT_SCHEMA="synthetic_known_history",Inputs=Inputs,FAMILIES={"fake0":{},"fake1":{}},
        DEBITS={"0":12.873334385920316,"1":12.449620655039325},
        PREVIOUS_DEBITS={"0":221.9527463898994,"1":13.754588949028403},
        packet_identity=step("source",None),engineering=step("v1",{"paid_seconds":25.32295504095964}),
        continuation=step("v3",{"paid_seconds":235.7073353389278}),validate_lanes=step("lanes",None),
        project_family=lambda inputs,packet,f,item,studies: dict(physical_gpu="0" if f=="fake0" else "1",charged_seconds=0 if f=="fake0" else current))
    monkeypatch.setattr(s,"delegate",lambda *a:old)
    return published,index,raw,calls


def test_history_uses_all_old_source_terminal_projection_steps_cost_only(tmp_path,monkeypatch):
    _,_,_,calls=synthetic_history(tmp_path,monkeypatch); value=s.history()
    assert calls==["source","v1","v3","lanes","recheck"]
    assert value["prior_charged_seconds"]==s.PRIOR_TOTAL and value["outcomes_reused"] is False


@pytest.mark.parametrize("mutation",["reset","lane","duplicate","old_terminal_bytes","index_bytes"])
def test_coherent_history_and_original_bytes_tamper_rejected(tmp_path,monkeypatch,mutation):
    published,index,raw,_=synthetic_history(tmp_path,monkeypatch)
    if mutation in {"reset","lane"}:
        data=s.read(published)
        if mutation=="reset": data["cost"]["inclusive_charged_seconds"]=0
        else: data["cost"]["lanes"]["1"]["inclusive_charged_seconds"]=0
        s.write(published,data); s.PRIOR_PINS["results"]=(published,s.sha(published))
    elif mutation=="duplicate":
        data=s.read(index); data["files"].append(data["files"][0]); data["file_count"]=2
        s.write(index,data); s.PRIOR_PINS["index"]=(index,s.sha(index))
    elif mutation=="old_terminal_bytes": raw.write_text("foreign")
    else: index.write_text("foreign")
    with pytest.raises(ValueError): s.history()


def result_fixture(tmp_path,monkeypatch,paid=10,returncode=0,status="completed"):
    attempt=tmp_path/"queue"/"attempt"; attempt.mkdir(parents=True)
    terminal=attempt/"supervisor-terminal.json"; s.write(terminal,dict(token="owned",paid_wall_seconds=paid,attempt_status=status,child_returncode=returncode))
    admission=dict(lease_path=str(attempt/"execution.lock"),token="owned")
    monkeypatch.setattr(s,"outcome",lambda *args: dict(original_gate="FAIL",qualification_input=False))
    return admission,terminal


def test_numeric_fail_is_a_complete_result_only_after_full_attestation(tmp_path,monkeypatch):
    admission,_=result_fixture(tmp_path,monkeypatch)
    result=s.result_for(tmp_path,{"source":{}},admission,{})
    assert result["status"]=="COMPLETE" and result["original_gate"]=="FAIL"
    assert result["charged_seconds"]==10 and result["unmeasured_interrupt_reserved_seconds"]==0


def test_missing_final_attestation_never_becomes_numeric_fail(tmp_path,monkeypatch):
    admission,_=result_fixture(tmp_path,monkeypatch)
    def absent(*a): raise ValueError("missing final attestation")
    monkeypatch.setattr(s,"outcome",absent)
    value=s.result_for(tmp_path,{"source":{}},admission,{})
    assert value["status"]=="INVALID" and value["original_gate"]=="UNAVAILABLE"


def test_completed_raw_fails_to_qualify_after_deadline(tmp_path,monkeypatch):
    admission,_=result_fixture(tmp_path,monkeypatch,paid=900.1)
    value=s.result_for(tmp_path,{"source":{}},admission,{})
    assert value["status"]=="BUDGET_EXCEEDED" and value["original_gate"]=="UNAVAILABLE"
    assert "outcome" not in value and value["retained_unaccepted_outcome"]["original_gate"]=="FAIL"


def test_source_or_model_exit_cannot_count_as_numeric_fail(tmp_path,monkeypatch):
    admission,_=result_fixture(tmp_path,monkeypatch,returncode=1)
    assert s.result_for(tmp_path,{"source":{}},admission,{})["status"]=="INVALID"


def test_dropped_supervisor_reserves_full_allowance_without_retry(tmp_path):
    path=tmp_path/"queue"; path.mkdir(); output=tmp_path/"attempt"; output.mkdir()
    error=RuntimeError("supervisor lost")
    result=s.result_for(output,{"source":{"digest":"a"*64}},dict(lease_path=str(path/"execution.lock"),token="owned"),{},error)
    assert result["status"]=="INCOMPLETE" and result["charged_seconds"]==900
    assert result["paid_wall_seconds"]==0 and result["unmeasured_interrupt_reserved_seconds"]==900
    assert s.read(s.checked(result["launch_error"]))["error"]=="RuntimeError: supervisor lost"


def test_protocol_file_matches_single_profile_declaration():
    assert s.validate_protocol(s.read(Path(__file__).with_name("protocol.json")))==s.protocol()


def stage_fixture(tmp_path):
    paths={s.CONTRACT,"experiments/forge/word_joint_policy_adapters.py","experiments/forge/runtime.py",
        "experiments/forge/evaluate.py","experiments/forge/api.py","experiments/forge/views.py",
        "experiments/forge/sampling.py","particlegan/policy.py","particlegan/recipes.py","benchmarks/toy_audit/api_run.py",s.SELF}
    source=dict(digest="a"*64,files={p:"b"*64 for p in paths})
    imports={p:dict(path=p,sha256="b"*64) for p in paths}
    for name in ("execution","evaluation"):
        s.write(tmp_path/(name+"-control.json"),dict(source_digest=source["digest"],imports=imports,code=0,**s.FLAGS))
    return dict(source=source)


def test_actual_execution_and_evaluator_stage_lists_are_both_required(tmp_path):
    packet=stage_fixture(tmp_path)
    assert set(s.stage_proof(tmp_path,packet))=={"execution","evaluation"}
    (tmp_path/"evaluation-control.json").unlink()
    with pytest.raises(FileNotFoundError): s.stage_proof(tmp_path,packet)


@pytest.mark.parametrize("mutation",["source","foreign_import","missing_real_producer","failed_code","boolean_code","credit"])
def test_coherent_stage_json_cannot_claim_foreign_or_unexecuted_code(tmp_path,mutation):
    packet=stage_fixture(tmp_path)
    for name in ("execution","evaluation"):
        path=tmp_path/(name+"-control.json"); value=s.read(path)
        if mutation=="source": value["source_digest"]="c"*64
        elif mutation=="foreign_import": value["imports"][s.CONTRACT]["sha256"]="c"*64
        elif mutation=="missing_real_producer": del value["imports"][s.CONTRACT]
        elif mutation=="failed_code": value["code"]=1
        elif mutation=="boolean_code": value["code"]=False
        else: value["qualification_input"]=True
        s.write(path,value)
    with pytest.raises(ValueError): s.stage_proof(tmp_path,packet)


def final_fixture(tmp_path):
    """Explicit software bytes, not training arrays/checkpoints or qualification."""
    from PIL import Image
    packet=stage_fixture(tmp_path); metadata=dict(word_rate_binding=dict(schema_version=1,owner="particlegan.Recipe",
        profile=s.PROFILE,tuple_id=s.CANDIDATE,overrides=s.OVERRIDES,resolved_recipe={"software_fixture":True}))
    metadata["word_rate_binding"]["resolved_recipe_sha256"]=s.digest(metadata["word_rate_binding"]["resolved_recipe"])
    root=tmp_path/"word-joint-policy"; (root/"observations").mkdir(parents=True)
    (root/"state.pt").write_bytes(b"synthetic metadata checkpoint bytes")
    for step in s.STEPS: (root/"observations"/f"step_{step:06d}.npz").write_bytes(f"synthetic observation bytes {step}".encode())
    files={p.relative_to(root).as_posix():dict(size=p.stat().st_size,sha256=s.sha(p)) for p in root.rglob("*") if p.is_file()}
    manifest=dict(schema_version=1,files=files,sha256=s.digest(files),file_count=len(files),total_bytes=sum(p["size"] for p in files.values()))
    observations=[dict(step=step,sample_count=1024,quality_fraction=1.,modes=5,mass_tv=0.,reconstruction_exact=1,
        minimum_reconstruction_token_probability=1.) for step in s.STEPS]
    audits=[dict(completed_steps=step,pure=True,before_sha256="d"*64,after_sha256="d"*64,
        global_rng_before_sha256="e"*64,global_rng_after_sha256="e"*64) for step in s.STEPS]
    raw=dict(task_id=s.TASK,device="cuda:0",execution_path="public_components",cost={"completed_steps":20001},
        applied=dict(recipe=metadata["word_rate_binding"]["resolved_recipe"],family=s.FAMILY,task_cohort=s.COHORT,
            execution_path="public_components",actual_resources=dict(num_particles=11,z_dim=2,batch_size=256),
            policy_lifecycle=dict(owner="particlegan.UpdatePolicy",completed_steps=20001,external_max_steps=20001,
                controls={"word_rate_binding":metadata["word_rate_binding"]})),
        evidence=dict(observations=observations,live=observations[-1],scoring_weights="state_selected",
            policy_controls={"word_rate_binding":metadata["word_rate_binding"]},
            guards=dict(all_finite=True,optimizer_updates={r:20001 for r in ("generator","encoder","prior","discriminator")}),
            policy_observations=[dict(completed_steps=step,family=s.FAMILY) for step in s.STEPS],policy_purity=audits,
            artifact_root=str(root),artifact_manifest=manifest,
            checkpoint=dict(path="state.pt",sha256=files["state.pt"]["sha256"],state_sha256="f"*64,digest_kind="typed_policy_state_v1")))
    s.write(tmp_path/"raw-result.json",raw)
    s.write(tmp_path/"graded-result.json",dict(raw_hash=s.digest(raw),source_digest=packet["source"]["digest"],grades={s.TASK:{"gate_status":"PASS"}}))
    frames=[Image.new("RGB",(2,2),(i*20,0,0)) for i in range(9)]
    frames[0].save(tmp_path/"goal.gif",save_all=True,append_images=frames[1:],duration=100,loop=0)
    media=dict(schema=s.SCHEMA+"_media",source_digest=packet["source"]["digest"],candidate_id=s.CANDIDATE,profile=s.PROFILE,
        original_gate="PASS",actual_steps=s.MEDIA_STEPS,selected_indices=s.SELECTED,
        inputs=[s.pin(root/"observations"/f"step_{step:06d}.npz") for step in s.MEDIA_STEPS],gif=s.pin(tmp_path/"goal.gif"),
        frames=9,renderer_sha256="b"*64,wrapper_sha256="b"*64,numerical_observations_changed=False,draws=0,updates=0,**s.FLAGS)
    s.write(tmp_path/"media.json",media)
    def control():
        s.write(tmp_path/"word-control.json",dict(schema=s.SCHEMA+"_control",source=packet["source"],candidate_id=s.CANDIDATE,
            profile=s.PROFILE,word_rate_binding=metadata["word_rate_binding"],raw=s.pin(tmp_path/"raw-result.json"),
            grading=s.pin(tmp_path/"graded-result.json"),media=s.pin(tmp_path/"media.json"),gif=s.pin(tmp_path/"goal.gif"),
            stages=s.stage_proof(tmp_path,packet),imports={s.CONTRACT:dict(path=s.CONTRACT,sha256="b"*64)},**s.FLAGS))
    control()
    return packet,metadata,control


def test_final_metadata_joins_real24_clocks_checkpoint_and_nine_decoded_frames(tmp_path):
    packet,metadata,_=final_fixture(tmp_path)
    value=s.outcome(tmp_path,packet,metadata)
    assert value["original_gate"]=="PASS"
    assert not any(k.startswith("experiments.") for k in sys.modules)


@pytest.mark.parametrize("mutation",["N5","recipe","profile","missing_binding","owner","steps","clock","weights","optimizer",
    "pure","global_rng","metrics_nonfinite","checkpoint","missing_array"])
def test_coherently_rehashed_raw_metadata_cannot_evade_final_contract(tmp_path,mutation):
    packet,metadata,rebind=final_fixture(tmp_path); path=tmp_path/"raw-result.json"; raw=s.read(path); evidence=raw["evidence"]
    if mutation=="N5": raw["applied"]["actual_resources"]["num_particles"]=5
    elif mutation=="recipe": raw["applied"]["recipe"]["lr"]=.0053125
    elif mutation=="profile": evidence["policy_controls"]["word_rate_binding"]["profile"]="quarter_base"
    elif mutation=="missing_binding": del raw["applied"]["policy_lifecycle"]["controls"]["word_rate_binding"]
    elif mutation=="owner": raw["applied"]["policy_lifecycle"]["owner"]="disabled"
    elif mutation=="steps": raw["cost"]["completed_steps"]=20000
    elif mutation=="clock": evidence["policy_observations"][0]["completed_steps"]=0
    elif mutation=="weights": evidence["scoring_weights"]="forced_ema"
    elif mutation=="optimizer": evidence["guards"]["optimizer_updates"]["encoder"]=0
    elif mutation=="pure": evidence["policy_purity"][0]["pure"]=False
    elif mutation=="global_rng": evidence["policy_purity"][0]["global_rng_after_sha256"]="c"*64
    elif mutation=="metrics_nonfinite": evidence["observations"][0]["modes"]=None
    elif mutation=="checkpoint": evidence["checkpoint"]["sha256"]="c"*64
    else:
        del evidence["artifact_manifest"]["files"][f"observations/step_{s.STEPS[0]:06d}.npz"]
        manifest=evidence["artifact_manifest"]; manifest.update(sha256=s.digest(manifest["files"]),file_count=len(manifest["files"]),
            total_bytes=sum(p["size"] for p in manifest["files"].values()))
    s.write(path,raw); grading=s.read(tmp_path/"graded-result.json"); grading["raw_hash"]=s.digest(raw)
    s.write(tmp_path/"graded-result.json",grading); rebind()
    with pytest.raises(ValueError): s.outcome(tmp_path,packet,metadata)


@pytest.mark.parametrize("mutation",["steps","indices","inputs","renderer","source","gate","draws","credit","frames","actual_frames"])
def test_coherently_rehashed_media_stays_retained_and_exact_source_bound(tmp_path,mutation):
    packet,metadata,rebind=final_fixture(tmp_path); path=tmp_path/"media.json"; value=s.read(path)
    if mutation=="steps": value["actual_steps"][0]=0
    elif mutation=="indices": value["selected_indices"][0]=1
    elif mutation=="inputs": value["inputs"][0]=value["inputs"][1]
    elif mutation=="renderer": value["renderer_sha256"]="c"*64
    elif mutation=="source": value["source_digest"]="c"*64
    elif mutation=="gate": value["original_gate"]="FAIL"
    elif mutation=="draws": value["draws"]=1
    elif mutation=="credit": value["qualification_input"]=0
    elif mutation=="frames": value["frames"]=8
    else:
        from PIL import Image
        Image.new("RGB",(2,2)).save(tmp_path/"goal.gif"); value["gif"]=s.pin(tmp_path/"goal.gif")
    s.write(path,value); rebind()
    with pytest.raises(ValueError): s.outcome(tmp_path,packet,metadata)


def test_nonfinite_later_metric_cannot_hide_behind_early_failed_bound():
    row=dict(sample_count=1,quality_fraction=0.,modes=None,mass_tv=1.,reconstruction_exact=0,minimum_reconstruction_token_probability=0.)
    with pytest.raises(ValueError): s.point_pass(row)


def retained_fixture(tmp_path,monkeypatch):
    key="a"*64; queue=tmp_path/"queue"; directory=tmp_path/"scientific"; directory.mkdir()
    parent=queue/"policy/attempts"/key; parent.mkdir(parents=True); monkeypatch.setattr(s,"QUEUE",queue)
    source=dict(snapshot_path=str(tmp_path/"snapshot"),digest="c"*64)
    terminal=dict(token="owned",paid_wall_seconds=10.,attempt_status="completed",child_returncode=1)
    s.write(parent/"supervisor-terminal.json",terminal)
    command=[sys.executable,"-u",str(Path(source["snapshot_path"])/s.SELF),"--child",str(directory/"resolved.json"),"--lease-fd","8"]
    supervisor=dict(source=source,token="owned",command=command,lease_fds=[7,8],started_monotonic=100.,deadline_monotonic=1000.)
    s.write(parent/"supervisor-request.json",supervisor)
    result=dict(status="INVALID",original_gate="UNAVAILABLE",attempt_key=key,
        terminal=s.pin(parent/"supervisor-terminal.json"),token_sha256=s.hashlib.sha256(b"owned").hexdigest(),
        overrun_seconds=0.,**s.charge(10.,terminal),**s.FLAGS)
    return directory,dict(source=source),result,parent


def test_retained_invalid_is_cost_only_and_never_automatically_retried_or_regraded(tmp_path,monkeypatch):
    directory,packet,value,_=retained_fixture(tmp_path,monkeypatch)
    def forbidden(*args): raise AssertionError("old INVALID cannot be recertified")
    monkeypatch.setattr(s,"outcome",forbidden)
    assert s.verify_retained_result(directory,packet,value,{})==value


@pytest.mark.parametrize("mutation",["cost","reserve","overrun","budget","credit","key","source","command","fd","deadline"])
def test_retained_cost_source_fencing_and_no_cap_reset(tmp_path,monkeypatch,mutation):
    directory,packet,value,parent=retained_fixture(tmp_path,monkeypatch)
    if mutation=="cost": value["paid_wall_seconds"]=0.
    elif mutation=="reserve": value["unmeasured_interrupt_reserved_seconds"]=890.
    elif mutation=="overrun": value["overrun_seconds"]=1.
    elif mutation=="budget": value["status"]="BUDGET_EXCEEDED"
    elif mutation=="credit": value["qualification_input"]=False; value["original_gate"]="PASS"
    elif mutation=="key": value["attempt_key"]="b"*64
    else:
        path=parent/"supervisor-request.json"; data=s.read(path)
        if mutation=="source": data["source"]["digest"]="d"*64
        elif mutation=="command": data["command"][0]="foreign-python"
        elif mutation=="fd": data["lease_fds"]=[8,8]
        else: data["deadline_monotonic"]=1001.
        s.write(path,data)
    with pytest.raises(ValueError): s.verify_retained_result(directory,packet,value,{})


def test_saved_full26_slots_and_inclusive_cost_cannot_pool_old_passes(tmp_path):
    packet={k:{} for k in ("spec","spec_sha256","source","execution_source","request","history","case_definitions","inputs",
        "runtime_contract","lane_runtime","family_paid_budget_seconds","capacity_preflight")}
    packet["request"]=wire_request()
    saved=deepcopy(packet); saved.update(executed_family=s.FAMILY,result=dict(status="INVALID",original_gate="UNAVAILABLE",charged_seconds=10.),
        slots={name:{"status":"NOT_RUN"} for name in packet["request"]["tasks"]},status="INVALID",spent_seconds=10.,
        inclusive_lane_charged_seconds=s.PRIOR_LANES["1"]+10.,inclusive_total_charged_seconds=s.PRIOR_TOTAL+10.)
    saved["slots"][s.TASK]={"status":"INVALID"}; s.verify_saved_study(saved,packet)
    for mutation in ("denominator","old_pass","cost_reset","source"):
        data=deepcopy(saved)
        if mutation=="denominator": del data["slots"]["trajectory"]
        elif mutation=="old_pass": data["slots"]["trajectory"]["status"]="PASS"
        elif mutation=="cost_reset": data["inclusive_total_charged_seconds"]=10.
        else: data["source"]={"foreign":True}
        with pytest.raises(ValueError): s.verify_saved_study(data,packet)


def test_no_protected_ml_import_during_stdlib_control_module_load():
    assert "torch" not in sys.modules
    assert not any(k.startswith("experiments.") for k in sys.modules)
