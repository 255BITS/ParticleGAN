"""Passive stdlib/Pillow publication of one trusted half-base terminal cut.

This module imports no Forge, Torch, NumPy, producer, evaluator or scorer. It
checks retained bytes and recorded decisions; it performs no numerical replay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys

from PIL import Image
import PIL

SCHEMA = "pg_word_half_base_publication_v1"
CARD_SCHEMA = "pg_word_half_base_publication_inputs_v1"
RUN_SCHEMA = "pg_word_half_base_supervision_v1"
ORIGIN = "f9f7ed9d7a06c48d4ec56999107658983d7e8efc"
SOURCE_DIGEST = "5995590d3c303207c664fe3b0ba7dc1e09dd3da7a90abf87086634c789175026"
PREPARED_SHA = "70aef35d0d237ae20d273cdb3221f7756af5c87e12d6fedd70e3f3cadc1a6433"
PREFLIGHT_SHA = "e2fad29d9f049d8427b02e319eba4b650624fbec7d6432aabdd3df0cc452266e"
ROOT_REVIEW_SHA = "b9b1efdb3c49ca09c5cfaacc303f426f09db0b089edb3f0417303fc0d9b423c1"
INDEPENDENT_SHA = "fab44ce9db3faa50a6908e08d10b02e73b540a178f89879c8fdd455f7b2f67de"
RECOVERY_SHA = "84c447356f26f2370a1cf1183f28736dc4726a639dbddcd8f094af5194e5252b"
RECIPE_SHA = "ea470cca5c55a31e6f726945402f0dfad945fea2ec95b654afba922e93c8ac69"
DIRECTORY = "observer-controls/word-half-base-v1"
SELF = DIRECTORY + "/run_supervised.py"
IMPLEMENTATIONS = {
    SELF:"dae842935a1ea3c4c7aae1694cb91e2f6c8b55815095f7230e64c7dc0ec368a1",
    DIRECTORY+"/protocol.json":"2d24165fcecdbb26444a3fae4ffec253431974b598a601e5f7a2995782c50f6b",
    DIRECTORY+"/source_guards.py":"1152522e2cc3215e0e707cf265697f596f3aa19244d905ffe0eea411b5daee97",
    DIRECTORY+"/history_projection.py":"99970901e2793961c433164f5f8c73094316c19d2f92c77c5c8d3e5e6617187a",
    "experiments/forge/word_joint_rate_policy_contracts.py":"0c725c5c4b8ea7e09554bf7022b62950bb42a3d41ea4fb49dca932428a61ca11"}
COHORT = "word_joint_policy_min11_rates_v1"
FAMILY = "atlas_word_joint_min11_rates"
PROFILE = "half_base"
CANDIDATE = "word-min11-half_base-rates-v1"
PARENT = "five_word_joint_acquisition"
TASK = PARENT+"_"+COHORT
OVERRIDES = dict(lr=.00265625,prior_lr_mult=1.5,d_lr_mult=1.)
LIMIT = 900
PRIOR_TOTAL = 910.2391431590077
PRIOR_LANES = {"0":234.82608077581972,"1":675.4130623831879}
LANE_CAPS = {"0":7500,"1":3000}
PARENTS = ["two_pole","unused_token_hold","ae_gan_hold","ring16_acquisition",PARENT,
    "trajectory","residual_student","unipolar","cover_leftover","mid_scale_identity","mode_hold",
    "vector_two_broad","vector_unequal_mass","vector_unequal_width","vector_anisotropic","vector_overlap",
    "vector_spiral","img_stripes2","img_bars4","img_blobs4","img_intensity2","grid100","rotated100",
    "staggered100","ring_hold","ring_extension"]
STEPS = [math.ceil(i*20001/24) for i in range(1,25)]
INDICES = [0,3,6,9,12,14,17,20,23]
MEDIA_STEPS = [STEPS[i] for i in INDICES]
THRESHOLDS = [["sample_count",">=",1024],["quality_fraction",">=",.95],["modes","==",5],
    ["mass_tv","<=",.1],["reconstruction_exact","==",1],["minimum_reconstruction_token_probability",">=",.9]]
FLAGS = dict(qualification_input=False,ordinary_tier_credit=False,historical_credit=False,
    default_adoption=False,speed_ranking=False,cross_tuple_pooling=False)
OWNERS = {"continuous_controller","stationarity_lr","row_evidence","birth_death","learned_output_noise",
    "selected_averaging","optimizer_surprise","reopen_guard"}
HOOKS = ["begin_step","after_critic_step","after_generator_backward","after_generator_step","finish_step"]
PRIVATE_KEYS = {"token","lease_fd","lease_fds","lease_path","password","secret","credentials","credential",
    "api_key","authorization","access_token","refresh_token"}
HISTORY_PINS = {"results":"0a175519b59b5180078d9669519060b79dc30d596daca3c8be1aca2dd9ce1d5f",
    "index":"8c57ff6e02a69eda41b39cb10c0bdd8c3f41040504b3eda1f11534b2a6d82ce4",
    "card":"f287d51757c4e9397afb5435e7b3848877915b6326ba59d0659b32fab20c6be2"}
OPTIONAL = ("control","raw","grading","media")
SOURCE_FILES = 1444
HISTORY_FILES = 3039
QUEUE = Path("/ml2/hypergan/ParticleGAN-single-recipe/runs/forge")
CARD_INPUTS = {"prepared","source_manifest","metadata_preflight","root_preflight_review",
    "independent_review","recovery_proof","study","cost","resolved","supervisor_request",
    "supervisor_terminal",*OPTIONAL}


def require(condition,message):
    if not condition: raise ValueError(message)


def read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"duplicate JSON field")
            result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite JSON")))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def pin(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)


def relative(value):
    require(isinstance(value,str),"relative path must be text")
    p=PurePosixPath(value)
    require(bool(value) and value!="." and not p.is_absolute() and ".." not in p.parts
        and "\\" not in value and p.as_posix()==value,"unsafe relative file")
    return value


def count(value):
    require(type(value) is int and value>=0,"integer evidence count required")
    return value


def number(value):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,"finite nonnegative cost required")
    return value


def close(a,b):
    require(math.isclose(number(a),number(b),rel_tol=0,abs_tol=1e-8),"cost arithmetic differs")


def no_credit(value):
    require(all(value.get(k) is False for k in FLAGS),"no qualification/default/speed/pooling credit")


def public(value,secrets=()):
    if isinstance(value,dict):
        require(not (set(map(str.lower,value))&PRIVATE_KEYS),"private field in public output")
        for item in value.values():public(item,secrets)
    elif isinstance(value,(list,tuple)):
        for item in value:public(item,secrets)
    elif isinstance(value,str):
        require(not any(secret and secret in value for secret in secrets),"private nonce in public output")
    return value


class Inputs:
    def __init__(self):self.files={};self.labels={};self.secrets=set()
    def check(self,label,item,expected=None,expected_sha=None):
        require(isinstance(item,dict) and set(item)=={"path","sha256","bytes"},"complete input pin required: "+label)
        p=Path(item["path"])
        require(p.is_absolute() and not p.is_symlink() and p.is_file() and not any(x.is_symlink() for x in p.parents),"unsafe/missing input: "+label)
        require(type(item["bytes"]) is int and item["bytes"]>=0 and re.fullmatch("[a-f0-9]{64}",str(item["sha256"]))
            and pin(p)==item,"changed input bytes: "+label)
        require(expected is None or p==Path(expected).resolve(),"borrowed input: "+label)
        require(expected_sha is None or item["sha256"]==expected_sha,"wrong frozen input: "+label)
        require(str(p) not in self.files or self.files[str(p)]==item,"input changed between reads")
        require(label not in self.labels or self.labels[label]==item,"conflicting input label")
        self.files[str(p)]=deepcopy(item);self.labels[label]=deepcopy(item);return p
    def file(self,label,path,expected_sha=None):return self.check(label,pin(path),path,expected_sha)
    def json(self,label,item,expected=None,expected_sha=None):return read(self.check(label,item,expected,expected_sha))
    def recheck(self):
        for p,item in self.files.items():require(pin(p)==item,"input changed during projection")
    def index(self):
        return dict(schema=SCHEMA+"_input_index",unique_file_count=len(self.files),label_count=len(self.labels),
            files=[dict(label=k,sha256=v["sha256"],bytes=v["bytes"],availability="LOCAL_ONLY") for k,v in sorted(self.labels.items())])


def imports(value,source,required=()):
    require(isinstance(value,dict) and value,"actual imported-source manifest required")
    paths=set()
    for item in value.values():
        require(isinstance(item,dict) and set(item)=={"path","sha256"},"typed imported-source identity required")
        p=relative(item["path"])
        require(source["files"].get(p)==item["sha256"],"foreign/unbound actual import")
        paths.add(p)
    require(set(required).issubset(paths),"actual producer/evaluator import absent")
    return paths


def source_packet(packet,card,inputs):
    require(packet.get("schema")==RUN_SCHEMA+"_packet","wrong prepared protocol")
    no_credit(packet)
    source=packet["source"];require(digest(source)==digest(packet["execution_source"]),"executed source differs")
    require(type(source.get("schema_version")) is int and source["schema_version"]==1
        and source.get("origin_commit")==ORIGIN and source.get("digest")==SOURCE_DIGEST
        and digest(source["files"])==SOURCE_DIGEST and len(source["files"])==SOURCE_FILES,"wrong full source origin/digest")
    root=Path(source["snapshot_path"]);require(root.is_absolute() and not root.is_symlink(),"exact snapshot root required")
    header=inputs.json("source:header",card["source_manifest"],root/"forge-source.json")
    require(digest(header)==digest({k:v for k,v in source.items() if k!="snapshot_path"}),"source header origin collision")
    for name,h in source["files"].items():inputs.file("source:"+relative(name),root/name,h)
    for name,h in IMPLEMENTATIONS.items():require(source["files"].get(name)==h,"changed implementation closure")
    parent=packet["parent_source"]
    additions={SELF,DIRECTORY+"/protocol.json",DIRECTORY+"/source_guards.py",DIRECTORY+"/history_projection.py"}
    require(parent.get("origin_commit")==ORIGIN and digest(parent["files"])==parent.get("digest")
        and set(source["files"])==set(parent["files"])|additions
        and all(source["files"].get(k)==v for k,v in parent["files"].items()),"parent/derived source closure differs")
    require(set(packet["inputs"])=={"wrapper","protocol","source_guards","history_projection"},"external supervisor closure missing")
    for name,path in {"wrapper":SELF,"protocol":DIRECTORY+"/protocol.json","source_guards":DIRECTORY+"/source_guards.py","history_projection":DIRECTORY+"/history_projection.py"}.items():
        inputs.check("supervisor-input:"+name,packet["inputs"][name],expected_sha=source["files"][path])
    protocol=read(root/(DIRECTORY+"/protocol.json"))
    require(protocol.get("schema")==RUN_SCHEMA and protocol.get("id")==CANDIDATE and protocol.get("canonical_origin_commit")==ORIGIN
        and protocol.get("profile")==PROFILE and protocol.get("recipe_overrides")==OVERRIDES
        and protocol.get("allowance_seconds")==LIMIT and protocol.get("export_grace_seconds")==0
        and protocol.get("attempts")==1 and protocol.get("retries")==0 and protocol.get("physical_gpu")=="1"
        and protocol.get("metric_steps")==STEPS and protocol.get("terminal_steps")==STEPS[-5:]
        and protocol.get("media_steps")==MEDIA_STEPS and protocol.get("media_indices")==INDICES
        and protocol.get("thresholds")==THRESHOLDS and protocol.get("required_slots")==26,"full frozen decision protocol differs")
    require(packet["spec"].get("id")==CANDIDATE and packet["spec"].get("profile")==PROFILE
        and packet["spec"].get("recipe_overrides")==OVERRIDES and packet["spec"].get("paid_cap_seconds")==LIMIT
        and packet["spec"].get("export_grace_seconds")==0 and packet["spec"].get("frames")==9
        and digest(packet["spec"])==packet["spec_sha256"] and packet["family_paid_budget_seconds"]=={FAMILY:LIMIT},"fixed one-shot spec differs")
    require(packet["spec"]["history_sha256"]==digest(packet["history"])
        and packet["spec"]["representation_card"]==packet["inputs"]["protocol"],"cost history/protocol identity differs")
    lane=packet["lane_runtime"]
    require(lane.get("physical_gpu")=="1" and lane.get("device")=="cuda:0" and lane.get("torch_threads")==1
        and lane.get("compute",{}).get("model")=="NVIDIA RTX A6000","physical/logical runtime differs")
    return source,root,protocol


def task_request(packet,root):
    request=packet["request"];candidate=request["candidate"];view=request["view"]
    expected=dict(id=CANDIDATE,word_rate_profile=PROFILE,trainer_family=FAMILY,task_cohort=COHORT,
        recipe_preset="atlas",recipe_overrides=OVERRIDES,execution_path="public_trainer")
    require(digest({k:candidate.get(k) for k in expected})==digest(expected),"fixed global half-base tuple differs")
    ids=[TASK if p==PARENT else p for p in PARENTS]
    require(set(request["tasks"])==set(ids) and [x["task"] for x in view["assignments"]]==ids
        and len(view["assignments"])==26 and [sum(x["qualification_tier"]==t for x in view["assignments"]) for t in (1,2,3)]==[5,19,2]
        and all(x["importance"]=="required" for x in view["assignments"])
        and view.get("revision")==4 and view.get("task_cohort")==COHORT and view.get("policy_family")==FAMILY
        and type(request["protocol"].get("seed")) is int and request["protocol"]["seed"]==0,"full26 view/seed/order differs")
    canonical={}
    for parent in PARENTS:
        task=request["tasks"][TASK if parent==PARENT else parent]
        require(isinstance(task.get("preflight_blockers"),list) and all(type(v) is str for v in task["preflight_blockers"])
            and ("field_ownership" not in task or isinstance(task["field_ownership"],dict)),"malformed compiler annotations")
        raw={k:v for k,v in task.items() if k not in {"preflight_blockers","field_ownership"}}
        source_path=(root/f"configs/forge/task-variants/{COHORT}/{TASK}.json" if parent==PARENT else root/f"configs/forge/tasks/{parent}.json")
        require(digest(raw)==digest(read(source_path)),"source-bound whole task/parent differs")
        canonical[task["id"]]=raw
    require(view["parent_view_fingerprint"]==digest(read(root/"configs/forge/views/discriminator_stability.json"))
        and view["cohort_fingerprint"]==digest(canonical),"full original/actual view fingerprints differ")
    task=request["tasks"][TASK];execution,evaluation=task["execution"],task["evaluation"]
    require(task["preflight_blockers"]==[] and task.get("policy_family")==FAMILY and task.get("task_cohort")==COHORT
        and task["policy_parent"].get("id")==PARENT and task["policy_parent"]["task_sha256"]==sha(root/f"configs/forge/tasks/{PARENT}.json")
        and execution.get("steps")==20001 and execution.get("original_schedule_horizon")==20000
        and execution.get("execution_path")=="public_components" and execution.get("device")=="cuda"
        and execution.get("resources")==dict(num_particles=11,z_dim=2,batch_size=256)
        and execution["prior"]["kind"]=="particle_cloud" and execution["prior"]["sigma"]==0 and execution["prior"]["standardize"] is False
        and evaluation.get("thresholds")==THRESHOLDS and evaluation.get("observations")==24
        and evaluation.get("minimum_stable_checks")==5 and evaluation.get("eval_samples")==1024
        and evaluation.get("scoring_weights")=="state_selected" and evaluation.get("eval_output_noise")=="clean"
        and task["resources"]["timeout_seconds"]==LIMIT,"physical word resources/law/gates/cadence differ")
    jobs=[j for j in request["jobs"] if TASK in j["task_ids"]]
    require(len(jobs)==1 and jobs[0]["task_ids"]==[TASK] and jobs[0]["budget_seconds"]==LIMIT,"one word job required")
    require(digest(packet["source"])==digest(request["source"]),"request/source join differs")
    return task,jobs[0]


def metadata(packet,card,inputs,source):
    review=inputs.json("provenance:root-preflight",card["root_preflight_review"],expected_sha=ROOT_REVIEW_SHA)
    independent=inputs.json("provenance:independent-review",card["independent_review"],expected_sha=INDEPENDENT_SHA)
    recovery=inputs.json("provenance:maintained-recovery",card["recovery_proof"],expected_sha=RECOVERY_SHA)
    require(independent.get("status")=="CLEARED_SOURCE_AND_SOFTWARE_CONTROLS" and independent.get("canonical_scientific_origin")==ORIGIN,"independent source review absent")
    pre=inputs.json("preflight:receipt",review["preflight"],expected_sha=PREFLIGHT_SHA)
    require(card["metadata_preflight"]==review["preflight"],"card/preflight byte identity differs")
    require(review.get("status")=="PASS_METADATA_ONLY" and review.get("global_rng_pure") is True
        and review.get("canonical_study_absent_before_admission") is True and review.get("queue_admission_before_review") is False
        and review.get("full_required_slots")==26 and review.get("original_tiers")=={"1":5,"2":19,"3":2}
        and digest(review["source"])==digest(source) and review["packet"]==card["prepared"],"root pre-admission source proof differs")
    require(pre.get("schema")==RUN_SCHEMA+"_metadata" and pre.get("status")=="PASS_METADATA_ONLY"
        and digest(pre["source"])==digest(source) and pre.get("wrapper_sha256")==IMPLEMENTATIONS[SELF]
        and pre.get("request_sha256")==digest(packet["request"]) and pre.get("full26_task_sha256")==digest(packet["request"]["tasks"])
        and pre.get("canonical_snapshot_entries")==1 and pre.get("cuda_initialized") is False
        and pre.get("global_rng_before_sha256")==pre.get("global_rng_after_sha256")
        and re.fullmatch("[a-f0-9]{64}",str(pre.get("global_rng_before_sha256"))),"real copied-source metadata proof differs")
    for obj in (pre,review):
        require(all(type(obj.get(k)) is int and obj[k]==0 for k in ("model_constructions","forwards","draws","updates","scorer_calls"))
            and obj.get("cuda_initialized") is False,"preflight must be model/draw/scorer/CUDA free")
    imports(pre["imported_sources"],source)
    binding=pre["word_rate_binding"]
    require(binding.get("profile")==PROFILE and binding.get("tuple_id")==CANDIDATE and binding.get("owner")=="particlegan.Recipe"
        and binding.get("overrides")==OVERRIDES and digest(binding["resolved_recipe"])==RECIPE_SHA
        and binding.get("resolved_recipe_sha256")==RECIPE_SHA,"complete effective Recipe differs")
    require(digest(review["word_rate_binding"])==digest(binding),"root/full Recipe binding differs")
    return pre


def prior(packet,inputs):
    history=packet["history"];require(history.get("terminal_cost_joins_verified") is True and history.get("outcomes_reused") is False,"old history must be cost-only")
    values={k:inputs.json("prior:"+k,item,expected_sha=HISTORY_PINS[k]) for k,item in history["pins"].items()}
    require(set(values)==set(HISTORY_PINS),"missing history cut")
    index=values["index"];require(index.get("file_count")==HISTORY_FILES and len(index["files"])==HISTORY_FILES,"full old consumed-pin history required")
    require(len({x["path"] for x in index["files"]})==HISTORY_FILES,"duplicate old history input")
    for i,item in enumerate(index["files"]):inputs.check(f"prior-consumed:{i:04d}",item)
    costs=values["results"]["cost"]
    close(history["prior_charged_seconds"],PRIOR_TOTAL);close(costs["inclusive_charged_seconds"],PRIOR_TOTAL)
    require(costs["current_reserved_seconds"]==0 and costs["original_cap_seconds"]==10500,"old scope/cost reset")
    for gpu in PRIOR_LANES:
        close(history["prior_lane_charged_seconds"][gpu],PRIOR_LANES[gpu]);close(costs["lanes"][gpu]["inclusive_charged_seconds"],PRIOR_LANES[gpu])
        require(costs["lanes"][gpu]["cap_seconds"]==LANE_CAPS[gpu],"old lane cap reset")
    word=values["results"]["families"]["atlas_word_joint_min11"]
    require(word.get("status")=="INVALID" and word["attempts"][0]["numerical_gate"]=="UNAVAILABLE","old invalid word cannot be recertified")
    return dict(source=values["results"]["source"],status="INVALID",accepted_numeric="UNAVAILABLE",paid_seconds=word["paid_seconds"],
        already_in_prior_total=True,recertified=False,qualification_input=False)


def durable(card,packet,study,inputs,directory,job):
    result=study["result"];terminal=inputs.json("durable:terminal",card["supervisor_terminal"])
    terminal_path=Path(card["supervisor_terminal"]["path"]);request=inputs.json("durable:request",card["supervisor_request"],terminal_path.with_name("supervisor-request.json"))
    inputs.secrets.add(terminal["token"])
    require(request.get("token")==terminal.get("token") and hashlib.sha256(terminal["token"].encode()).hexdigest()==result.get("token_sha256"),"durable token fence differs")
    require(result.get("terminal")==card["supervisor_terminal"] and result.get("attempt_key")==terminal_path.parent.name
        and terminal_path==QUEUE/"policy/attempts"/result["attempt_key"]/"supervisor-terminal.json"
        and re.fullmatch("[a-f0-9]{64}",str(result["attempt_key"]))
        and terminal_path.name=="supervisor-terminal.json","wrong durable attempt")
    resolved=inputs.json("run:resolved",card["resolved"],directory/"resolved.json")
    require(set(resolved["packet"])==set(packet)|{"coordinator","spent_seconds","executed_family"}
        and all(digest(resolved["packet"].get(k))==digest(packet.get(k)) for k in packet)
        and resolved["packet"]["coordinator"]==study["coordinator"]
        and resolved["packet"]["spent_seconds"]==0 and resolved["packet"]["executed_family"]==FAMILY,"resolved packet drift")
    require(set(study["coordinator"])=={"canonical_output","queue_root","study_key"}
        and study["coordinator"]["canonical_output"]==str(directory.parent)
        and study["coordinator"]["queue_root"]==str(QUEUE)
        and re.fullmatch("[a-f0-9]{64}",str(study["coordinator"]["study_key"])),"foreign shared-lane coordinator")
    require(digest(resolved["request"])==digest(packet["request"]) and digest(resolved["job"])==digest(job)
        and resolved.get("resolved_path")==str(directory/"resolved.json") and resolved["worker"]["device"]=="cuda:0"
        and resolved["worker"]["token"]==terminal["token"] and resolved["worker"]["attempt"]==result["attempt_key"]
        and resolved["study_output"]==str(directory.parent)
        and resolved["metadata_preflight"]==card["metadata_preflight"],"actual wire request/worker differs")
    command=request.get("command",[])
    require(len(command)==7 and command[1:5]==["-u",str(Path(packet["source"]["snapshot_path"])/SELF),"--child",str(directory/"resolved.json")]
        and command[5]=="--lease-fd" and command[6].isdigit() and len(request["lease_fds"])==2
        and len(set(request["lease_fds"]))==2 and int(command[6]) in request["lease_fds"]
        and all(type(fd) is int and fd>=0 for fd in request["lease_fds"])
        and digest(request["source"])==digest(packet["source"]),"actual source/command/inherited leases differ")
    close(request["deadline_monotonic"]-request["started_monotonic"],LIMIT)
    require(terminal.get("attempt_status") in {"completed","timeout","error","cancelled"},"unknown durable supervisor status")
    paid=number(terminal["paid_wall_seconds"]);completed=terminal.get("attempt_status")=="completed"
    reserve=0. if completed else max(0.,LIMIT-paid);charged=paid+reserve
    for key,value in (("paid_wall_seconds",paid),("unmeasured_interrupt_reserved_seconds",reserve),("charged_seconds",charged),
            ("overrun_seconds",max(0.,charged-LIMIT)),("spent_seconds",charged)):
        close(result[key] if key!="spent_seconds" else study[key],value)
    close(study["inclusive_lane_charged_seconds"],PRIOR_LANES["1"]+charged);close(study["inclusive_total_charged_seconds"],PRIOR_TOTAL+charged)
    require((charged>LIMIT)==(result["status"]=="BUDGET_EXCEEDED"),"budget overrun status differs")
    require(result["status"] in {"COMPLETE","INVALID","INCOMPLETE","BUDGET_EXCEEDED"},"unknown terminal status")
    if result["status"]=="COMPLETE":require(completed and type(terminal.get("child_returncode")) is int and terminal["child_returncode"]==0 and charged<=LIMIT,"complete outcome requires timely successful child")
    else:require(result.get("original_gate")=="UNAVAILABLE","fault/partial cannot receive accepted numerical credit")
    if terminal["attempt_status"]=="timeout" and charged<=LIMIT:
        require(result["status"]=="INCOMPLETE","timed out partial must remain INCOMPLETE")
    return dict(paid_seconds=paid,reserved_seconds=reserve,charged_seconds=charged,overrun_seconds=max(0.,charged-LIMIT),
        allowance_seconds=LIMIT,export_grace_seconds=0,prior_charged_seconds=PRIOR_TOTAL,
        inclusive_charged_seconds=PRIOR_TOTAL+charged,original_cap_seconds=10500,
        lanes={"0":dict(prior_charged_seconds=PRIOR_LANES["0"],current_charged_seconds=0.,inclusive_charged_seconds=PRIOR_LANES["0"],cap_seconds=7500),
            "1":dict(prior_charged_seconds=PRIOR_LANES["1"],current_charged_seconds=charged,inclusive_charged_seconds=PRIOR_LANES["1"]+charged,cap_seconds=3000)},
        speed_comparison_available=False,paid_vs_reserve_separate=True)


def artifacts(raw,inputs,directory):
    evidence=raw["evidence"];root=Path(evidence["artifact_root"])
    require(root.is_absolute() and root.resolve().is_relative_to(directory.resolve()),"foreign artifact root")
    manifest=evidence["artifact_manifest"];files=manifest["files"]
    expected={"state.pt",*(f"observations/step_{step:06d}.npz" for step in STEPS)}
    require(set(files)==expected and manifest.get("file_count")==25 and manifest.get("sha256")==digest(files)
        and manifest.get("total_bytes")==sum(count(v["size"]) for v in files.values()),"complete 25-artifact manifest differs")
    for name,item in files.items():inputs.check("artifact:"+relative(name),dict(path=str(root/name),sha256=item["sha256"],bytes=item["size"]),root/name)
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual==expected,"added/missing artifact files")
    checkpoint=evidence["checkpoint"]
    require(checkpoint.get("path")=="state.pt" and checkpoint.get("sha256")==files["state.pt"]["sha256"]
        and checkpoint.get("digest_kind")=="typed_policy_state_v1" and re.fullmatch("[a-f0-9]{64}",str(checkpoint.get("state_sha256"))),"typed checkpoint join differs")
    return root


def accepted(card,packet,study,pre,inputs,directory,task):
    raw=inputs.json("run:raw",card["raw"],directory/"raw-result.json")
    grade=inputs.json("run:grading",card["grading"],directory/"graded-result.json")
    media=inputs.json("run:media",card["media"],directory/"media.json")
    control=inputs.json("run:control",card["control"],directory/"word-control.json")
    outcome=study["result"]["outcome"]
    for key in ("raw","grading","media","control"):
        require(outcome[key]==card[key],"accepted outcome/card bytes differ")
    no_credit(control);no_credit(outcome);no_credit(media)
    require(control.get("schema")==RUN_SCHEMA+"_control" and control.get("candidate_id")==CANDIDATE and control.get("profile")==PROFILE
        and digest(control["source"])==digest(packet["source"]) and digest(control["word_rate_binding"])==digest(pre["word_rate_binding"]),"final actual source/rate attestation differs")
    source=packet["source"];paths=set()
    for stage in ("execution","evaluation"):
        value=inputs.json("stage:"+stage,control["stages"][stage],directory/(stage+"-control.json"))
        no_credit(value);require(value.get("source_digest")==source["digest"] and type(value.get("code")) is int and value["code"]==0,"stage source/exit differs")
        paths.update(imports(value["imports"],source))
    required={"experiments/forge/word_joint_rate_policy_contracts.py","experiments/forge/word_joint_policy_adapters.py",
        "experiments/forge/runtime.py","experiments/forge/evaluate.py","experiments/forge/api.py","experiments/forge/views.py",
        "experiments/forge/sampling.py","particlegan/policy.py","particlegan/recipes.py","benchmarks/toy_audit/api_run.py"}
    require(required.issubset(paths),"both actual execution/evaluator source stages required")
    imports(control["imports"],source)
    for key in ("raw","grading","media"):
        require(control[key]==card[key],"final source control pin differs")
    require(raw.get("task_id")==TASK and raw.get("device")=="cuda:0" and raw.get("execution_path")=="public_components"
        and grade.get("raw_hash")==digest(raw) and grade.get("source_digest")==source["digest"] and set(grade["grades"])=={TASK},"raw/independent grade/source join differs")
    verdict=grade["grades"][TASK]
    require(verdict.get("status")==verdict.get("gate_status") and verdict["gate_status"] in {"PASS","FAIL"}
        and study["result"]["original_gate"]==verdict["gate_status"] and outcome["original_gate"]==verdict["gate_status"],"accepted original numerical grade differs")
    evaluated=verdict["evaluator_result"]
    require(evaluated.get("status")==verdict["gate_status"]
        and evaluated.get("passed") is (verdict["gate_status"]=="PASS")
        and evaluated.get("attempted") is True,"independent recorded decision is incoherent")
    applied=raw["applied"];evidence=raw["evidence"];binding=pre["word_rate_binding"]
    require(applied.get("family")==FAMILY and applied.get("task_cohort")==COHORT and applied.get("execution_path")=="public_components"
        and digest(applied.get("recipe"))==RECIPE_SHA and applied.get("actual_resources")==dict(num_particles=11,z_dim=2,batch_size=256)
        and raw["cost"].get("completed_steps")==20001 and evidence.get("scoring_weights")=="state_selected","full Recipe/resources/law/clock differ")
    lifecycle=applied["policy_lifecycle"];controls=evidence["policy_controls"]
    require(lifecycle.get("owner")=="particlegan.UpdatePolicy" and lifecycle.get("completed_steps")==20001
        and lifecycle.get("external_max_steps")==20001 and digest(lifecycle["controls"])==digest(controls)
        and digest(controls.get("word_rate_binding"))==digest(binding),"actual owner/profile receipt differs")
    require(controls.get("completed_steps")==20001 and controls.get("implementation_observed") is True
        and controls.get("requested_owners_bound") is True and set(controls["requested"])==OWNERS
        and set(controls["enabled"])==OWNERS and all(controls["requested"][k] is True and controls["enabled"][k] is True for k in OWNERS)
        and controls.get("row_semantics")=="independent" and controls.get("actual_prior_rows")==11
        and controls.get("joint_atom_code")=="same_effective_code" and controls.get("output_noise_coordinates")=="words168_only"
        and controls["execution"].get("model_devices")==["cuda:0"]
        and controls["execution"].get("floating_dtypes")==["torch.float32"]
        and controls["execution"].get("autocast_enabled") is False
        and controls.get("actual_birth_death")==dict(rows=11,neighbours=5,reference_half=6,isolation=True),"full actual public owners/device/law missing")
    audit=controls["lifecycle"]
    require(audit.get("complete") is True and audit.get("owner")=="particlegan.UpdatePolicy" and audit.get("start_completed_steps")==0
        and audit.get("end_completed_steps")==20001 and audit.get("observed_updates")==20001
        and audit.get("calls")=={h:20001 for h in HOOKS} and audit.get("last_order")==HOOKS
        and audit.get("pending")==[] and audit.get("order_errors")==0,"full ordered lifecycle evidence missing")
    guards=evidence["guards"]
    require(guards.get("all_finite") is True and guards.get("hooks_exercised") is True and guards.get("unintended_rng_deviations")==0
        and guards.get("optimizer_updates")=={k:20001 for k in ("generator","encoder","prior","discriminator")},"full optimizer/state/RNG evidence missing")
    points=evidence["observations"]
    require([p["step"] for p in points]==STEPS and all(type(p["step"]) is int for p in points)
        and digest(evidence["live"])==digest(points[-1]) and digest(verdict["metrics"])==digest(points[-1]),"original 24 reads/final endpoint differ")
    for point in points:
        require(all(type(point.get(key)) in (int,float) and math.isfinite(point[key]) for key,_,_ in THRESHOLDS),"finite original metrics required")
    require([(m["metric"],m["op"],m["threshold"],m["value"]) for m in evaluated["metrics"]]
        ==[(key,op,bound,points[-1][key]) for key,op,bound in THRESHOLDS],"published endpoint/threshold summaries differ")
    convergence=evaluated["convergence"]
    require(convergence.get("complete") is True and convergence.get("observations")==24
        and convergence.get("minimum_stable_checks")==5
        and count(convergence.get("passing_observations"))<=24
        and count(convergence.get("passing_suffix"))<=convergence["passing_observations"],"recorded full-cadence decision summary differs")
    observed=evidence["policy_observations"];purity=evidence["policy_purity"]
    require([p["completed_steps"] for p in observed]==STEPS and [p["completed_steps"] for p in purity]==STEPS,"every read needs actual policy/RNG proof")
    for observation,proof in zip(observed,purity):
        require(observation.get("family")==FAMILY and observation.get("policy_owner")=="particlegan.UpdatePolicy"
            and observation.get("observed") is True and observation.get("weight_selector")=="state_selected"
            and observation.get("output_noise") is False and observation.get("latent_policy")=="actual_selected_public_policy"
            and observation.get("sampler")=="ServedModel.generate" and observation.get("row_selection")=="uniform_eleven_actual_prior_rows"
            and observation.get("controller")=="dv12" and observation.get("diagnostic_credit") is False
            and observation.get("selected_source") in {"fast","averaged"}
            and proof.get("pure") is True and proof.get("before_sha256")==proof.get("after_sha256")
            and proof.get("global_rng_before_sha256")==proof.get("global_rng_after_sha256")
            and all(re.fullmatch("[a-f0-9]{64}",str(proof.get(k))) for k in ("before_sha256","global_rng_before_sha256")),"selected observer/train/global RNG purity differs")
    root=artifacts(raw,inputs,directory)
    expected_media=dict(schema=RUN_SCHEMA+"_media",source_digest=source["digest"],candidate_id=CANDIDATE,profile=PROFILE,
        original_gate=verdict["gate_status"],actual_steps=MEDIA_STEPS,selected_indices=INDICES,
        inputs=[pin(root/"observations"/f"step_{step:06d}.npz") for step in MEDIA_STEPS],gif=pin(directory/"goal.gif"),frames=9,
        renderer_sha256=source["files"]["benchmarks/toy_audit/api_run.py"],wrapper_sha256=IMPLEMENTATIONS[SELF],
        numerical_observations_changed=False,draws=0,updates=0,**FLAGS)
    require(digest(media)==digest(expected_media) and control["gif"]==media["gif"] and outcome["gif"]==media["gif"],"exact nine retained media/source join differs")
    gif=inputs.check("media:goal",media["gif"],directory/"goal.gif")
    with Image.open(gif) as image:require(image.n_frames==9,"decoded actual GIF must have nine frames")
    return dict(original_gate=verdict["gate_status"],final_metrics=deepcopy(points[-1]),
        original_grader_summary=deepcopy(verdict.get("evaluator_result",{}).get("convergence",{})),
        metric_steps=STEPS,terminal_steps=STEPS[-5:],completed_steps=20001,optimizer_updates=guards["optimizer_updates"],
        policy_owner="particlegan.UpdatePolicy",requested_owners=sorted(OWNERS),enabled_owners=sorted(OWNERS),
        lifecycle_calls=deepcopy(audit["calls"]),actual_birth_death=deepcopy(controls["actual_birth_death"]),
        independent_atlas_qualification=False,synthetic_mechanism_probes_are_learning_credit=False,
        selected_source=controls.get("served_source"),actual_goal_gif=dict(path="media/goal.gif",sha256=media["gif"]["sha256"],bytes=media["gif"]["bytes"],frames=9,actual_steps=MEDIA_STEPS),
        artifact_manifest_sha256=evidence["artifact_manifest"]["sha256"],checkpoint_sha256=evidence["checkpoint"]["sha256"],
        checkpoint_typed_state_sha256=evidence["checkpoint"]["state_sha256"],raw_local_only=True,**FLAGS),gif


def project(card_path,trusted_sha):
    require(re.fullmatch("[a-f0-9]{64}",str(trusted_sha)) and sha(card_path)==trusted_sha,"root trusted immutable-card SHA required")
    inputs=Inputs();card=inputs.json("root:terminal-card",pin(card_path),expected_sha=trusted_sha)
    require(card.get("schema")==CARD_SCHEMA and card.get("terminal_immutable") is True and card.get("qualification_input") is False,"immutable root terminal cut required")
    require(set(card)=={"schema","terminal_immutable","qualification_input","study_dir","inputs"}
        and isinstance(card.get("inputs"),dict) and set(card["inputs"])==CARD_INPUTS,"complete root terminal input roster required")
    declared=card["inputs"]
    packet=inputs.json("prepared:packet",declared["prepared"],expected_sha=PREPARED_SHA)
    source,root,protocol=source_packet(packet,declared,inputs);task,job=task_request(packet,root)
    pre=metadata(packet,declared,inputs,source);old_word=prior(packet,inputs)
    study_path=inputs.check("study:terminal",declared["study"]);study=read(study_path);directory=study_path.parent/"attempt"
    require(str(study_path.parent)==card["study_dir"],"borrowed canonical study directory")
    for key in packet:require(digest(study.get(key))==digest(packet[key]) or key=="status","terminal study changed prepared identities")
    require(study.get("executed_family")==FAMILY and study.get("status")==study["result"]["status"],"terminal family/status differs")
    no_credit(study);no_credit(study["result"])
    cost_result=inputs.json("study:cost",declared["cost"],study_path.with_name("cost.json"))
    require(digest(cost_result)==digest(study["result"]),"study/compact cost receipt differs")
    costs=durable(declared,packet,study,inputs,directory,job)
    status=study["result"]["status"];expected_slots={name:{"status":"NOT_RUN"} for name in packet["request"]["tasks"]}
    expected_slots[TASK]={"status":study["result"]["original_gate"] if status=="COMPLETE" else status}
    require(digest(study["slots"])==digest(expected_slots),"one current word/full26/not-run slots cannot pool historical grades")
    if status=="COMPLETE":
        require(all(declared.get(k) is not None for k in OPTIONAL),"accepted result needs complete final attestation/export")
        result,gif=accepted(declared,packet,study,pre,inputs,directory,task)
    else:
        for key in OPTIONAL:
            if declared.get(key) is not None:inputs.check("available-unaccepted:"+key,declared[key],directory/{"control":"word-control.json","raw":"raw-result.json","grading":"graded-result.json","media":"media.json"}[key])
        result=dict(original_gate="UNAVAILABLE",completed_steps=None,accepted_original_gif=None,
            reason="Original full protocol/export attestation was not accepted; retained failure detail remains local.",**FLAGS);gif=None
    output=dict(schema=SCHEMA,candidate_id=CANDIDATE,profile=PROFILE,family=FAMILY,task_cohort=COHORT,status=status,
        accepted_numeric=result["original_gate"],required_slots=26,scheduled_slots=1,not_run_slots=25,tiers={"1":5,"2":19,"3":2},slots=expected_slots,
        counts=dict(Counter(x["status"] for x in expected_slots.values())),task_id=TASK,parent_task_id=PARENT,
        question="Acquire all five words and their paired free-E reconstructions under the complete same-code joint cloud law.",
        recipe_overrides=OVERRIDES,resolved_recipe=pre["word_rate_binding"]["resolved_recipe"],resolved_recipe_sha256=RECIPE_SHA,
        source=dict(origin_commit=ORIGIN,digest=SOURCE_DIGEST,files=len(source["files"]),supervisor_sha256=IMPLEMENTATIONS[SELF]),
        protocol=dict(seed=0,updates=20001,schedule_horizon=20000,metric_steps=STEPS,eval_samples=1024,terminal_steps=STEPS[-5:],thresholds=THRESHOLDS,
            serving="actual policy-selected G/E/raw ParticlePrior; DV12 retained, output noise off",resources=dict(num_particles=11,z_dim=2,batch_size=256),
            objective="original joint RpGAN plus original regularizers; free continuous E; no reconstruction training loss"),
        runtime=packet["lane_runtime"],result=result,cost=costs,historical_word=old_word,
        provenance=dict(trusted_card_sha256=trusted_sha,prepared_sha256=PREPARED_SHA,preflight_sha256=PREFLIGHT_SHA,
            root_preflight_review_sha256=ROOT_REVIEW_SHA,independent_review_sha256=INDEPENDENT_SHA,recovery_proof_sha256=RECOVERY_SHA),
        raw_availability="LOCAL_ONLY; bulk states, arrays, traces and private supervision fields are not copied",**FLAGS)
    public(output,inputs.secrets);inputs.recheck();return output,inputs,gif


def markdown(value):
    row=value["result"];cost=value["cost"];word=value["historical_word"]
    text=f"""# One half-base joint-word result

**{value['status']} — accepted original word gate {value['accepted_numeric']}.**
This one tuple retains all 26 required slots (tiers 5/19/2): one word case and
25 NOT_RUN. Earlier passes stay in their own source cohorts and do not fill
this row. No default, ordinary-tier, speed or cross-variant qualification.

| Tuple | Word decision | Remaining slots |
|---|---|---|
| `{CANDIDATE}` | {value['accepted_numeric']} ({value['status']}) | 25 NOT_RUN |

The global base LR is 0.00265625, prior multiplier 1.5 and critic multiplier 1.
All nominal role rates inherit that base; this is not a generator-only change.
Endogenous rates/displacements remain policy behavior. Actual resources are
11 learned raw 2D ParticlePrior rows, batch 256, original G/E/joint-D and free
continuous E. A generated atom is `(G(z_effective), z_effective)` with actual
DV12; training output noise affects only 168 word coordinates. Evaluation uses
actual policy-selected G/E/prior without output noise. There is no new
reconstruction loss, row-to-word teacher or unseen-word claim.

The target words are apple, grape, lemon, melon and berry. All 20,001 updates,
24 reads of 1,024 generated rows and five paired inputs, complete public-owner
clocks, and final five reads are required. Original bounds: sample count >=
1024; quality >= .95; all five modes; mass TV <= .1; exact paired reconstruction;
minimum paired token probability >= .9. Final reads are updates 16,668 / 17,501 /
18,335 / 19,168 / 20,001. No gates are weakened or new acquisition gate added.

"""
    if value["status"]=="COMPLETE":
        metrics=row["final_metrics"];summary=row["original_grader_summary"]
        text+=f"The retained independent grader records {summary['passing_observations']} / 24\npassing reads and a passing suffix of {summary['passing_suffix']}. Terminal\nmetrics: {metrics['modes']} / 5 modes, quality {metrics['quality_fraction']:.9g},\nmass TV {metrics['mass_tv']:.9g}, paired exact reconstruction\n{metrics['reconstruction_exact']:.9g}, and minimum paired token probability\n{metrics['minimum_reconstruction_token_probability']:.9g}.\n\n"
        text+="The byte-original nine-frame GIF shows actual generated rows, all five paired\nfree-E reconstructions, confidence and padding. Argmax words are display only;\nquality and mode masses use the original probability/confidence test over all\n1,024 samples. A displayed spelling such as apple can therefore have zero\nqualified apple mass. The footer's Default test names the original full\nprotocol gate; it does not claim a shipped-family default.\n\n![Actual word training; accepted original "+value["accepted_numeric"]+"](media/goal.gif)\n\n"
    else:text+="No accepted numerical result or qualifying GIF is published. Retained raw\nstamps/export fragments cannot replace the missing complete attestation or\nextend the deadline.\n\n"
    text+=f"""The new attempt paid {cost['paid_seconds']:.9f} seconds, reserved
{cost['reserved_seconds']:.9f}, charged {cost['charged_seconds']:.9f}, and overran
{cost['overrun_seconds']:.9f}. The 900-second allowance includes construction,
training, all reads, checkpoint, grading, original goal GIF and final attestation;
there is zero grace and no retry. Prior named-lane cost {PRIOR_TOTAL:.9f} is
included once, giving {cost['inclusive_charged_seconds']:.9f} / 10,500 seconds.
GPU1 prior {PRIOR_LANES['1']:.9f} plus this attempt gives
{cost['lanes']['1']['inclusive_charged_seconds']:.9f} / 3,000; GPU0 remains
{PRIOR_LANES['0']:.9f} / 7,500. Paid and interruption reserve stay separate.
Separate CUDA engineering controls and the Gaussian study are separate budgets.

The old f380 word attempt remains INVALID / numerical UNAVAILABLE, paid
{word['paid_seconds']:.9f} seconds, already included in that prior debit. It
is not recertified under revised health guards. Cold representation evidence
and structural controls confer no learned credit.

Canonical origin `{ORIGIN}`; executed source digest `{SOURCE_DIGEST}`.
The exact full Recipe, source/cadence/provenance and cost joins are in
[results.json](results.json); consumed bytes are in [input-index.json](input-index.json),
and the passive checks/copies in [verification.json](verification.json).
Raw sources/states/arrays/traces are LOCAL_ONLY. This publisher hashes retained
inputs and copies accepted original media; it draws, restores, trains, renders
and rescores nothing.
"""
    return text


def publish(card_path,trusted_sha,output):
    output=Path(output).resolve();require(not output.exists(),"fresh publication output required")
    value,inputs,gif=project(card_path,trusted_sha)
    value["publisher_source"]=dict(sha256=sha(__file__),bytes=Path(__file__).stat().st_size,
        runtime=dict(python=sys.version.split()[0],pillow=PIL.__version__))
    output.mkdir(parents=True)
    if gif is not None:
        (output/"media").mkdir();shutil.copyfile(gif,output/"media/goal.gif")
        require(sha(output/"media/goal.gif")==value["result"]["actual_goal_gif"]["sha256"],"copied GIF bytes differ")
    inputs.recheck();public(value,inputs.secrets)
    def save(name,obj):(output/name).write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n")
    index=inputs.index();public(index,inputs.secrets)
    save("results.json",value);save("input-index.json",index);(output/"README.md").write_text(markdown(value))
    context=dict(schema=SCHEMA+"_media_context",status=value["status"],accepted_numeric=value["accepted_numeric"],
        task_id=TASK,source=value["source"],required_slots=26,not_run_slots=25,
        argmax_words="Display only; qualified quality and mode masses use the original probability/confidence test over all 1,024 samples.",
        footer="Default test refers to the original full protocol gate, not a shipped-family default.",
        original_gif=None if gif is None else value["result"]["actual_goal_gif"],**FLAGS)
    public(context,inputs.secrets);save("media-context.json",context)
    verification=dict(schema=SCHEMA+"_verification",status="PASS_PASSIVE_PROJECTION",trusted_card_sha256=trusted_sha,
        consumed_files=len(inputs.files),copied_original_gifs=int(gif is not None),draws=0,restores=0,updates=0,scorer_calls=0,
        numerical_decision_recomputed=False,private_fields_copied=False,accepted_numeric=value["accepted_numeric"],
        outputs={p.relative_to(output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.rglob("*") if p.is_file()},**FLAGS)
    public(verification,inputs.secrets);save("verification.json",verification);return verification


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--card",required=True,type=Path)
    parser.add_argument("--trusted-sha256",required=True);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv);print(json.dumps(publish(args.card,args.trusted_sha256,args.output),sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
