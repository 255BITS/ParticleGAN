"""Self-contained synthetic byte archives; no numerical experiment is executed."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

SPEC=importlib.util.spec_from_file_location("passive_publisher_under_test",Path(__file__).with_name("publish_results.py"))
p=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(p)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")
    return p.pin(path)


class Archive:
    def __init__(self,base,monkeypatch,status="FAIL"):
        self.base=base;self.snapshot=base/"snapshot";self.snapshot.mkdir()
        self.study_dir=base/"study";self.directory=self.study_dir/"attempt";self.directory.mkdir(parents=True)
        self.queue=base/"queue";monkeypatch.setattr(p,"QUEUE",self.queue)
        self.durable_dir=self.queue/"policy/attempts"/("1"*64);self.durable_dir.mkdir(parents=True)
        self.files={}
        def source(name,value):
            file=self.snapshot/name;file.parent.mkdir(parents=True,exist_ok=True)
            if isinstance(value,(dict,list)):save(file,value)
            else:file.write_text(value)
            self.files[name]=p.sha(file)
        self.parent_tasks={name:dict(id=name,schema_version=1,execution=dict(steps=80),evaluation=dict(thresholds=[])) for name in p.PARENTS}
        for name,task in self.parent_tasks.items():source(f"configs/forge/tasks/{name}.json",task)
        tiers=[1]*5+[2]*19+[3]*2
        self.original_view=dict(id="discriminator_stability",revision=3,
            assignments=[dict(task=name,qualification_tier=tier,importance="required") for name,tier in zip(p.PARENTS,tiers)])
        source("configs/forge/views/discriminator_stability.json",self.original_view)
        self.task=dict(id=p.TASK,policy_family=p.FAMILY,task_cohort=p.COHORT,
            policy_parent=dict(id=p.PARENT,task_sha256=self.files[f"configs/forge/tasks/{p.PARENT}.json"]),
            execution=dict(steps=20001,original_schedule_horizon=20000,execution_path="public_components",device="cuda",
                resources=dict(num_particles=11,z_dim=2,batch_size=256),prior=dict(kind="particle_cloud",sigma=0.,standardize=False)),
            evaluation=dict(thresholds=p.THRESHOLDS,observations=24,minimum_stable_checks=5,eval_samples=1024,
                scoring_weights="state_selected",eval_output_noise="clean"),resources=dict(timeout_seconds=900))
        source(f"configs/forge/task-variants/{p.COHORT}/{p.TASK}.json",self.task)
        for name in p.IMPLEMENTATIONS:
            if name!=p.DIRECTORY+"/protocol.json":source(name,"# synthetic immutable source: "+name+"\n")
        for name in ["experiments/forge/word_joint_policy_adapters.py","experiments/forge/runtime.py",
            "experiments/forge/evaluate.py","experiments/forge/api.py","experiments/forge/views.py",
            "experiments/forge/sampling.py","particlegan/policy.py","particlegan/recipes.py","benchmarks/toy_audit/api_run.py"]:
            source(name,"# synthetic immutable source: "+name+"\n")
        self.protocol=dict(schema=p.RUN_SCHEMA,id=p.CANDIDATE,profile=p.PROFILE,canonical_origin_commit=p.ORIGIN,
            recipe_overrides=p.OVERRIDES,allowance_seconds=900,export_grace_seconds=0,attempts=1,retries=0,
            physical_gpu="1",metric_steps=p.STEPS,terminal_steps=p.STEPS[-5:],media_steps=p.MEDIA_STEPS,
            media_indices=p.INDICES,thresholds=p.THRESHOLDS,required_slots=26)
        source(p.DIRECTORY+"/protocol.json",self.protocol)
        monkeypatch.setattr(p,"IMPLEMENTATIONS",{name:self.files[name] for name in p.IMPLEMENTATIONS})
        monkeypatch.setattr(p,"SOURCE_FILES",len(self.files))
        monkeypatch.setattr(p,"SOURCE_DIGEST",p.digest(self.files))
        self.source=dict(schema_version=1,origin_commit=p.ORIGIN,digest=p.SOURCE_DIGEST,files=self.files,snapshot_path=str(self.snapshot))
        additions={p.SELF,p.DIRECTORY+"/protocol.json",p.DIRECTORY+"/source_guards.py",p.DIRECTORY+"/history_projection.py"}
        parent_files={k:v for k,v in self.files.items() if k not in additions}
        parent_source=dict(schema_version=1,origin_commit=p.ORIGIN,digest=p.digest(parent_files),files=parent_files)
        self.header_pin=save(self.snapshot/"forge-source.json",{k:v for k,v in self.source.items() if k!="snapshot_path"})
        self.imports={"synthetic_"+str(i):dict(path=k,sha256=v) for i,(k,v) in enumerate(self.files.items()) if k.endswith(".py")}
        old_file=base/"prior-original.bin";old_file.write_bytes(b"synthetic old immutable raw bytes")
        self.old_results=dict(source=dict(origin_commit="f"*40,digest="f"*64),
            cost=dict(inclusive_charged_seconds=p.PRIOR_TOTAL,current_reserved_seconds=0,original_cap_seconds=10500,
                lanes={g:dict(inclusive_charged_seconds=p.PRIOR_LANES[g],cap_seconds=p.LANE_CAPS[g]) for g in p.PRIOR_LANES}),
            families=dict(atlas_word_joint_min11=dict(status="INVALID",paid_seconds=558.5739127129782,
                attempts=[dict(numerical_gate="UNAVAILABLE")])))
        self.old_index=dict(file_count=1,files=[p.pin(old_file)])
        self.history_pins=dict(results=save(base/"old-results.json",self.old_results),
            index=save(base/"old-index.json",self.old_index),card=save(base/"old-card.json",{"synthetic":True}))
        monkeypatch.setattr(p,"HISTORY_FILES",1)
        monkeypatch.setattr(p,"HISTORY_PINS",{k:v["sha256"] for k,v in self.history_pins.items()})
        self.history=dict(terminal_cost_joins_verified=True,outcomes_reused=False,pins=self.history_pins,
            prior_charged_seconds=p.PRIOR_TOTAL,prior_lane_charged_seconds=p.PRIOR_LANES)
        self.recipe=dict(name="atlas",lr=.00265625,prior_lr_mult=1.5,d_lr_mult=1.,num_particles=11,z_dim=2,
            batch_size=256,betas=[0.,.999],alpha_bar=[1.,.9,.5,.05,.0001],direct_particle_betas=[0.,.9],
            encoder_mode="none",prior_kind="particles",sigma_rel=0.,standardize=False,row_policy="independent",
            output_noise_mode="learnable",continuous_policy="dv12",birth_death_isolation=True)
        monkeypatch.setattr(p,"RECIPE_SHA",p.digest(self.recipe))
        self.binding=dict(profile=p.PROFILE,tuple_id=p.CANDIDATE,owner="particlegan.Recipe",overrides=p.OVERRIDES,
            resolved_recipe=self.recipe,resolved_recipe_sha256=p.RECIPE_SHA)
        tasks={name:deepcopy(task) for name,task in self.parent_tasks.items() if name!=p.PARENT}
        tasks[p.TASK]=deepcopy(self.task)
        canonical=deepcopy(tasks)
        for task in tasks.values():task.update(preflight_blockers=[],field_ownership={"synthetic_structural_only":True})
        assignments=[dict(x,task=p.TASK if x["task"]==p.PARENT else x["task"]) for x in self.original_view["assignments"]]
        self.view=dict(id="discriminator_stability",revision=4,task_cohort=p.COHORT,policy_family=p.FAMILY,assignments=assignments,
            parent_view_fingerprint=p.digest(self.original_view),cohort_fingerprint=p.digest(canonical))
        self.job=dict(task_ids=[p.TASK],budget_seconds=900,id="synthetic-word-job")
        self.request=dict(candidate=dict(id=p.CANDIDATE,word_rate_profile=p.PROFILE,trainer_family=p.FAMILY,
            task_cohort=p.COHORT,recipe_preset="atlas",recipe_overrides=p.OVERRIDES,execution_path="public_trainer"),
            view=self.view,tasks=tasks,protocol=dict(seed=0),jobs=[self.job],source=self.source)
        external={k:p.pin(self.snapshot/name) for k,name in dict(wrapper=p.SELF,protocol=p.DIRECTORY+"/protocol.json",
            source_guards=p.DIRECTORY+"/source_guards.py",history_projection=p.DIRECTORY+"/history_projection.py").items()}
        spec=dict(id=p.CANDIDATE,profile=p.PROFILE,recipe_overrides=p.OVERRIDES,paid_cap_seconds=900,export_grace_seconds=0,
            frames=9,history_sha256=p.digest(self.history),representation_card=external["protocol"])
        self.packet=dict(schema=p.RUN_SCHEMA+"_packet",status="PREPARED",source=self.source,execution_source=self.source,
            parent_source=parent_source,inputs=external,spec=spec,spec_sha256=p.digest(spec),history=self.history,
            family_paid_budget_seconds={p.FAMILY:900},request=self.request,
            lane_runtime=dict(physical_gpu="1",device="cuda:0",torch_threads=1,compute=dict(model="NVIDIA RTX A6000")),**p.FLAGS)
        self.packet_pin=save(base/"prepared.json",self.packet)
        monkeypatch.setattr(p,"PREPARED_SHA",self.packet_pin["sha256"])
        self.pre=dict(schema=p.RUN_SCHEMA+"_metadata",status="PASS_METADATA_ONLY",source=self.source,
            wrapper_sha256=p.IMPLEMENTATIONS[p.SELF],request_sha256=p.digest(self.request),full26_task_sha256=p.digest(tasks),
            canonical_snapshot_entries=1,cuda_initialized=False,global_rng_before_sha256="a"*64,global_rng_after_sha256="a"*64,
            imported_sources=self.imports,word_rate_binding=self.binding,
            **dict.fromkeys(("model_constructions","forwards","draws","updates","scorer_calls"),0))
        self.pre_pin=save(base/"preflight.json",self.pre)
        monkeypatch.setattr(p,"PREFLIGHT_SHA",self.pre_pin["sha256"])
        review=dict(status="PASS_METADATA_ONLY",global_rng_pure=True,canonical_study_absent_before_admission=True,
            queue_admission_before_review=False,full_required_slots=26,original_tiers={"1":5,"2":19,"3":2},
            source=self.source,packet=self.packet_pin,preflight=self.pre_pin,word_rate_binding=self.binding,cuda_initialized=False,
            **dict.fromkeys(("model_constructions","forwards","draws","updates","scorer_calls"),0))
        self.review_pin=save(base/"root-review.json",review);monkeypatch.setattr(p,"ROOT_REVIEW_SHA",self.review_pin["sha256"])
        self.independent_pin=save(base/"independent.json",dict(status="CLEARED_SOURCE_AND_SOFTWARE_CONTROLS",canonical_scientific_origin=p.ORIGIN))
        self.recovery_pin=save(base/"recovery.json",dict(synthetic_control=True))
        monkeypatch.setattr(p,"INDEPENDENT_SHA",self.independent_pin["sha256"]);monkeypatch.setattr(p,"RECOVERY_SHA",self.recovery_pin["sha256"])
        self.token="PRIVATE-SYNTHETIC-NONCE-DO-NOT-PUBLISH"
        self.supervisor=dict(token=self.token,source=self.source,started_monotonic=100.,deadline_monotonic=1000.,lease_fds=[10,11],
            command=["/synthetic/python","-u",str(self.snapshot/p.SELF),"--child",str(self.directory/"resolved.json"),"--lease-fd","11"])
        self.terminal=dict(token=self.token,attempt_status="completed",child_returncode=0,paid_wall_seconds=20.)
        self.coordinator=dict(canonical_output=str(self.study_dir),queue_root=str(self.queue),study_key="2"*64)
        self.resolved=dict(packet=dict(deepcopy(self.packet),coordinator=self.coordinator,spent_seconds=0.,executed_family=p.FAMILY),
            request=self.request,job=self.job,resolved_path=str(self.directory/"resolved.json"),study_output=str(self.study_dir),
            metadata_preflight=self.pre_pin,worker=dict(device="cuda:0",token=self.token,attempt="1"*64))
        self.artifact_root=self.directory/"word-joint-policy";self.artifact_root.mkdir()
        self.artifact_files={}
        for name in ["state.pt",*[f"observations/step_{step:06d}.npz" for step in p.STEPS]]:
            f=self.artifact_root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(("synthetic no-model artifact "+name).encode())
            self.artifact_files[name]=dict(sha256=p.sha(f),size=f.stat().st_size)
        self.points=[dict(step=step,sample_count=1024,quality_fraction=.98 if status=="PASS" else .5,modes=5 if status=="PASS" else 2,
            mass_tv=.02 if status=="PASS" else .6,reconstruction_exact=1 if status=="PASS" else 0,
            minimum_reconstruction_token_probability=.95 if status=="PASS" else .001) for step in p.STEPS]
        self.controls=dict(completed_steps=20001,implementation_observed=True,requested_owners_bound=True,
            requested=dict.fromkeys(p.OWNERS,True),enabled=dict.fromkeys(p.OWNERS,True),row_semantics="independent",actual_prior_rows=11,
            joint_atom_code="same_effective_code",output_noise_coordinates="words168_only",word_rate_binding=self.binding,
            execution=dict(model_devices=["cuda:0"],floating_dtypes=["torch.float32"],autocast_enabled=False),
            actual_birth_death=dict(rows=11,neighbours=5,reference_half=6,isolation=True),served_source="fast",
            lifecycle=dict(complete=True,owner="particlegan.UpdatePolicy",start_completed_steps=0,end_completed_steps=20001,
                observed_updates=20001,calls=dict.fromkeys(p.HOOKS,20001),last_order=p.HOOKS,pending=[],order_errors=0))
        observations=[dict(completed_steps=step,family=p.FAMILY,policy_owner="particlegan.UpdatePolicy",observed=True,
            weight_selector="state_selected",output_noise=False,latent_policy="actual_selected_public_policy",sampler="ServedModel.generate",
            row_selection="uniform_eleven_actual_prior_rows",controller="dv12",diagnostic_credit=False,selected_source="fast") for step in p.STEPS]
        purity=[dict(completed_steps=step,pure=True,before_sha256="b"*64,after_sha256="b"*64,
            global_rng_before_sha256="c"*64,global_rng_after_sha256="c"*64) for step in p.STEPS]
        self.raw=dict(task_id=p.TASK,device="cuda:0",execution_path="public_components",cost=dict(completed_steps=20001),
            applied=dict(family=p.FAMILY,task_cohort=p.COHORT,execution_path="public_components",recipe=self.recipe,
                actual_resources=dict(num_particles=11,z_dim=2,batch_size=256),
                policy_lifecycle=dict(owner="particlegan.UpdatePolicy",completed_steps=20001,external_max_steps=20001,controls=deepcopy(self.controls))),
            evidence=dict(policy_controls=self.controls,observations=self.points,live=deepcopy(self.points[-1]),scoring_weights="state_selected",
                guards=dict(all_finite=True,hooks_exercised=True,unintended_rng_deviations=0,
                    optimizer_updates=dict.fromkeys(("generator","encoder","prior","discriminator"),20001)),
                policy_observations=observations,policy_purity=purity,artifact_root=str(self.artifact_root),
                artifact_manifest=dict(files=self.artifact_files,file_count=25,sha256=p.digest(self.artifact_files),
                    total_bytes=sum(x["size"] for x in self.artifact_files.values())),
                checkpoint=dict(path="state.pt",sha256=self.artifact_files["state.pt"]["sha256"],digest_kind="typed_policy_state_v1",state_sha256="d"*64)))
        self.grade=dict(source_digest=p.SOURCE_DIGEST,raw_hash=p.digest(self.raw),grades={p.TASK:dict(status=status,gate_status=status,
            metrics=deepcopy(self.points[-1]),evaluator_result=dict(status=status,passed=status=="PASS",attempted=True,
                metrics=[dict(metric=k,op=op,threshold=bound,value=self.points[-1][k]) for k,op,bound in p.THRESHOLDS],
                convergence=dict(complete=True,observations=24,minimum_stable_checks=5,
                    passing_observations=24 if status=="PASS" else 0,passing_suffix=24 if status=="PASS" else 0)))})
        frames=[Image.new("RGB",(12,12),(i*20,0,100)) for i in range(9)]
        frames[0].save(self.directory/"goal.gif",save_all=True,append_images=frames[1:],duration=200,loop=0)
        self.media=dict(schema=p.RUN_SCHEMA+"_media",source_digest=p.SOURCE_DIGEST,candidate_id=p.CANDIDATE,profile=p.PROFILE,
            original_gate=status,actual_steps=p.MEDIA_STEPS,selected_indices=p.INDICES,
            inputs=[p.pin(self.artifact_root/"observations"/f"step_{step:06d}.npz") for step in p.MEDIA_STEPS],gif=p.pin(self.directory/"goal.gif"),
            frames=9,renderer_sha256=self.files["benchmarks/toy_audit/api_run.py"],wrapper_sha256=p.IMPLEMENTATIONS[p.SELF],
            numerical_observations_changed=False,draws=0,updates=0,**p.FLAGS)
        self.stages={name:dict(source_digest=p.SOURCE_DIGEST,code=0,imports=self.imports,**p.FLAGS) for name in ("execution","evaluation")}
        self.control=dict(schema=p.RUN_SCHEMA+"_control",candidate_id=p.CANDIDATE,profile=p.PROFILE,source=self.source,
            word_rate_binding=self.binding,imports=self.imports,**p.FLAGS)
        self.outcome=dict(original_gate=status,**p.FLAGS)
        self.result=dict(status="COMPLETE",original_gate=status,paid_wall_seconds=20.,charged_seconds=20.,overrun_seconds=0.,
            unmeasured_interrupt_reserved_seconds=0.,attempt_key="1"*64,token_sha256=p.hashlib.sha256(self.token.encode()).hexdigest(),**p.FLAGS)
        self.study=dict(deepcopy(self.packet),executed_family=p.FAMILY,status="COMPLETE",result=self.result,coordinator=self.coordinator,
            spent_seconds=20.,inclusive_lane_charged_seconds=p.PRIOR_LANES["1"]+20.,inclusive_total_charged_seconds=p.PRIOR_TOTAL+20.,
            slots={k:dict(status=status if k==p.TASK else "NOT_RUN") for k in self.request["tasks"]})
        self.card=dict(schema=p.CARD_SCHEMA,terminal_immutable=True,qualification_input=False,study_dir=str(self.study_dir),inputs={})
        self.flush()

    def flush(self):
        # Rebind dependent JSON pins, so semantic negative controls reach validators.
        self.grade["raw_hash"]=p.digest(self.raw)
        raw=save(self.directory/"raw-result.json",self.raw);grade=save(self.directory/"graded-result.json",self.grade)
        media=save(self.directory/"media.json",self.media)
        stages={k:save(self.directory/(k+"-control.json"),v) for k,v in self.stages.items()}
        self.control.update(raw=raw,grading=grade,media=media,gif=self.media["gif"],stages=stages)
        control=save(self.directory/"word-control.json",self.control)
        self.outcome.update(raw=raw,grading=grade,media=media,control=control,gif=self.media["gif"])
        self.result["outcome"]=self.outcome
        self.result["terminal"]=save(self.durable_dir/"supervisor-terminal.json",self.terminal)
        self.card["inputs"].update(prepared=self.packet_pin,source_manifest=self.header_pin,metadata_preflight=self.pre_pin,
            root_preflight_review=self.review_pin,independent_review=self.independent_pin,recovery_proof=self.recovery_pin,
            resolved=save(self.directory/"resolved.json",self.resolved),supervisor_request=save(self.durable_dir/"supervisor-request.json",self.supervisor),
            supervisor_terminal=self.result["terminal"],raw=raw,grading=grade,media=media,control=control,
            study=save(self.study_dir/"study.json",self.study),cost=save(self.study_dir/"cost.json",self.result))
        self.card_path=self.base/"root-card.json";save(self.card_path,self.card)

    def project(self):return p.project(self.card_path,p.sha(self.card_path))

    def terminal_status(self,status,paid,attempt_status="completed",child_returncode=0):
        self.terminal.update(paid_wall_seconds=paid,attempt_status=attempt_status,child_returncode=child_returncode)
        reserve=0. if attempt_status=="completed" else max(0.,900-paid);charge=paid+reserve
        self.result.update(status=status,original_gate="UNAVAILABLE" if status!="COMPLETE" else self.outcome["original_gate"],
            paid_wall_seconds=paid,charged_seconds=charge,unmeasured_interrupt_reserved_seconds=reserve,overrun_seconds=max(0.,charge-900))
        self.study.update(status=status,spent_seconds=charge,inclusive_lane_charged_seconds=p.PRIOR_LANES["1"]+charge,
            inclusive_total_charged_seconds=p.PRIOR_TOTAL+charge)
        self.study["slots"][p.TASK]={"status":self.result["original_gate"] if status=="COMPLETE" else status};self.flush()


@pytest.fixture
def archive(tmp_path,monkeypatch):return Archive(tmp_path,monkeypatch)


@pytest.mark.parametrize("gate",["PASS","FAIL"])
def test_complete_original_decision_not_recomputed(tmp_path,monkeypatch,gate):
    a=Archive(tmp_path,monkeypatch,gate);value,inputs,gif=a.project()
    assert value["status"]=="COMPLETE" and value["accepted_numeric"]==gate
    assert value["counts"]=={gate:1,"NOT_RUN":25} and value["required_slots"]==26
    assert value["result"]["actual_goal_gif"]["frames"]==9 and gif==a.directory/"goal.gif"
    assert value["cost"]["inclusive_charged_seconds"]==p.PRIOR_TOTAL+20.
    assert value["historical_word"]["status"]=="INVALID" and not value["historical_word"]["recertified"]
    assert all(value[k] is False for k in p.FLAGS)


def test_passive_export_copies_only_original_media_and_pins(archive,tmp_path):
    before={str(f):p.pin(f) for f in archive.base.rglob("*") if f.is_file()}
    verification=p.publish(archive.card_path,p.sha(archive.card_path),tmp_path/"publication")
    output=tmp_path/"publication"
    assert p.sha(output/"media/goal.gif")==p.sha(archive.directory/"goal.gif")
    assert verification["copied_original_gifs"]==1 and not verification["numerical_decision_recomputed"]
    assert not any(x in str(list(output.rglob("*"))) for x in ("state.pt",".npz","supervisor-request"))
    for file in output.rglob("*"):
        if file.is_file() and file.suffix!=".gif":
            assert archive.token not in file.read_text() and str(archive.directory) not in file.read_text()
    assert before=={f:p.pin(f) for f in before}
    assert {"README.md","results.json","input-index.json","media-context.json","verification.json","media/goal.gif"}=={f.relative_to(output).as_posix() for f in output.rglob("*") if f.is_file()}
    assert p.read(output/"input-index.json")["label_count"]>=p.read(output/"input-index.json")["unique_file_count"]


@pytest.mark.parametrize("case",["missing-control","owner-disabled","clock","full-cadence","purity","global-rng","dtype","row-count",
    "joint-code","word-noise","birth-death","hook-order","pending-hook","optimizer-clock","endpoint","grade-endpoint",
    "grade-threshold","convergence","forced-ema","sampling","latent","diagnostic-credit","source-stage","rate-profile",
    "recipe","family","resources","media-steps","media-source","media-claimed-draws","media-short-gif","artifact-missing",
    "artifact-extra","typed-checkpoint","original-grade","missing-stage","token-fence","source-command","single-lease",
    "deadline","resolved-foreign-key","foreign-coordinator","not-run-credit","denominator"])
def test_coherently_repinned_semantic_tampering_fails(archive,case):
    a=archive;c=a.raw["evidence"]["policy_controls"];e=a.raw["evidence"];v=a.grade["grades"][p.TASK]
    if case=="missing-control":a.card["inputs"]["control"]=None;save(a.card_path,a.card)
    elif case=="owner-disabled":c["enabled"]["birth_death"]=False
    elif case=="clock":a.raw["cost"]["completed_steps"]=20000
    elif case=="full-cadence":e["observations"].pop(0)
    elif case=="purity":e["policy_purity"][0]["pure"]=False
    elif case=="global-rng":e["policy_purity"][0]["global_rng_after_sha256"]="e"*64
    elif case=="dtype":c["execution"]["floating_dtypes"]=["torch.float64"]
    elif case=="row-count":c["actual_prior_rows"]=5
    elif case=="joint-code":c["joint_atom_code"]="split_code"
    elif case=="word-noise":c["output_noise_coordinates"]="all170"
    elif case=="birth-death":c["actual_birth_death"]["isolation"]=False
    elif case=="hook-order":c["lifecycle"]["last_order"]=list(reversed(p.HOOKS))
    elif case=="pending-hook":c["lifecycle"]["pending"]=["begin_step"]
    elif case=="optimizer-clock":e["guards"]["optimizer_updates"]["encoder"]=20000
    elif case=="endpoint":e["live"]["modes"]=5
    elif case=="grade-endpoint":v["metrics"]["modes"]=5
    elif case=="grade-threshold":v["evaluator_result"]["metrics"][1]["threshold"]=.5
    elif case=="convergence":v["evaluator_result"]["convergence"]["observations"]=23
    elif case=="forced-ema":e["policy_observations"][0]["weight_selector"]="forced_ema"
    elif case=="sampling":e["policy_observations"][0]["sampler"]="raw_live"
    elif case=="latent":e["policy_observations"][0]["latent_policy"]="disabled"
    elif case=="diagnostic-credit":e["policy_observations"][0]["diagnostic_credit"]=True
    elif case=="source-stage":a.stages["evaluation"]["source_digest"]="f"*64
    elif case=="rate-profile":a.control["word_rate_binding"]=dict(a.binding,profile="quarter_base")
    elif case=="recipe":a.raw["applied"]["recipe"]=dict(a.recipe,total_steps=20000)
    elif case=="family":a.raw["applied"]["family"]="atlas"
    elif case=="resources":a.raw["applied"]["actual_resources"]["num_particles"]=5
    elif case=="media-steps":a.media["actual_steps"]=p.MEDIA_STEPS[:-1]+[20000]
    elif case=="media-source":a.media["renderer_sha256"]="e"*64
    elif case=="media-claimed-draws":a.media["draws"]=1
    elif case=="media-short-gif":
        Image.new("RGB",(12,12)).save(a.directory/"goal.gif");a.media["gif"]=p.pin(a.directory/"goal.gif")
    elif case=="artifact-missing":(a.artifact_root/"state.pt").unlink()
    elif case=="artifact-extra":(a.artifact_root/"unclaimed-state.pt").write_bytes(b"unbound")
    elif case=="typed-checkpoint":e["checkpoint"]["digest_kind"]="untyped"
    elif case=="original-grade":v["gate_status"]="PASS"
    elif case=="missing-stage":a.control["stages"]={"execution":p.pin(a.directory/"execution-control.json")}
    elif case=="token-fence":a.terminal["token"]="foreign"
    elif case=="source-command":a.supervisor["command"][2]="/foreign/caller.py"
    elif case=="single-lease":a.supervisor["lease_fds"]=[11]
    elif case=="deadline":a.supervisor["deadline_monotonic"]=1001.
    elif case=="resolved-foreign-key":a.resolved["packet"]["ignore_guard"]=True
    elif case=="foreign-coordinator":a.coordinator["queue_root"]="/foreign/queue"
    elif case=="not-run-credit":a.study["slots"]["trajectory"]={"status":"PASS"}
    elif case=="denominator":a.study["slots"].pop("trajectory")
    if case not in {"missing-control","missing-stage"}:a.flush()
    if case=="missing-stage":
        # Keep the altered final control and rebind its outcome/card identities only.
        a.card["inputs"]["control"]=save(a.directory/"word-control.json",a.control)
        a.outcome["control"]=a.card["inputs"]["control"]
        a.card["inputs"]["study"]=save(a.study_dir/"study.json",a.study)
        a.card["inputs"]["cost"]=save(a.study_dir/"cost.json",a.result);save(a.card_path,a.card)
    with pytest.raises((ValueError,KeyError)):a.project()


@pytest.mark.parametrize("status,paid,terminal,code,reserve",[
    ("INVALID",10.,"completed",1,0.),("INCOMPLETE",10.,"timeout",None,890.),
    ("INVALID",10.,"error",None,890.),("INVALID",10.,"cancelled",None,890.),
    ("BUDGET_EXCEEDED",901.,"completed",0,0.),("BUDGET_EXCEEDED",901.,"timeout",None,0.)])
def test_incomplete_invalid_and_overrun_never_gain_numeric_or_media(archive,status,paid,terminal,code,reserve):
    archive.terminal_status(status,paid,terminal,code)
    value,_,gif=archive.project()
    assert value["accepted_numeric"]=="UNAVAILABLE" and gif is None
    assert value["cost"]["paid_seconds"]==paid and value["cost"]["reserved_seconds"]==reserve
    assert value["cost"]["charged_seconds"]==paid+reserve
    assert value["cost"]["inclusive_charged_seconds"]==p.PRIOR_TOTAL+paid+reserve


@pytest.mark.parametrize("case",["reset-prior","reset-lane","omit-reserve","double-prior","overrun-complete"])
def test_cost_reset_or_undercharge_is_rejected(archive,case):
    if case=="reset-prior":archive.study["inclusive_total_charged_seconds"]=20.
    elif case=="reset-lane":archive.study["inclusive_lane_charged_seconds"]=20.
    elif case=="double-prior":archive.study["inclusive_total_charged_seconds"]+=p.PRIOR_TOTAL
    elif case=="omit-reserve":
        archive.terminal_status("INVALID",10.,"error",None);archive.result["charged_seconds"]=10.;archive.result["unmeasured_interrupt_reserved_seconds"]=0.
    elif case=="overrun-complete":archive.terminal_status("COMPLETE",901.,"completed",0)
    archive.flush()
    with pytest.raises(ValueError):archive.project()


@pytest.mark.parametrize("field,value",[("steps",20000),("device","cpu"),("resources",dict(num_particles=5,z_dim=2,batch_size=256)),
    ("prior",dict(kind="mog",sigma=.025,standardize=False))])
def test_current_task_transform_is_byte_bound(archive,field,value):
    packet=deepcopy(archive.packet);packet["request"]["tasks"][p.TASK]["execution"][field]=value
    with pytest.raises(ValueError):p.task_request(packet,archive.snapshot)


@pytest.mark.parametrize("case",["source-byte","source-digest","parent-closure","protocol-gates","foreign-import","missing-import","duplicate-json",
    "nonfinite-json","private-field","private-value","unknown-card-input","missing-terminal","missing-preflight","untrusted-card","symlink-input"])
def test_source_trust_privacy_and_missing_evidence_fail_closed(archive,case,tmp_path):
    if case=="source-byte":(archive.snapshot/"particlegan/policy.py").write_text("changed protected source")
    elif case=="source-digest":archive.card["inputs"]["source_manifest"]=save(archive.snapshot/"forge-source.json",{"digest":"f"*64})
    elif case=="parent-closure":
        packet=deepcopy(archive.packet);packet["parent_source"]["files"].pop("particlegan/policy.py")
        with pytest.raises(ValueError):p.source_packet(packet,archive.card["inputs"],p.Inputs())
        return
    elif case=="protocol-gates":
        packet=deepcopy(archive.packet);packet["spec"]["paid_cap_seconds"]=901;packet["spec_sha256"]=p.digest(packet["spec"])
        with pytest.raises(ValueError):p.source_packet(packet,archive.card["inputs"],p.Inputs())
        return
    elif case=="foreign-import":
        archive.stages["evaluation"]["imports"]={"fake":dict(path="outside/producer.py",sha256="f"*64)};archive.flush()
    elif case=="missing-import":
        archive.stages["evaluation"]["imports"]={k:v for k,v in archive.imports.items() if v["path"]!="experiments/forge/evaluate.py"}
        archive.stages["execution"]["imports"]=deepcopy(archive.stages["evaluation"]["imports"]);archive.flush()
    elif case=="duplicate-json":
        path=tmp_path/"duplicate.json";path.write_text('{"status":"PASS","status":"FAIL"}')
        with pytest.raises(ValueError):p.read(path)
        return
    elif case=="nonfinite-json":
        path=tmp_path/"nan.json";path.write_text('{"metric":NaN}')
        with pytest.raises(ValueError):p.read(path)
        return
    elif case=="private-field":
        with pytest.raises(ValueError):p.public({"token":"private"})
        return
    elif case=="private-value":
        with pytest.raises(ValueError):p.public({"reason":archive.token},{archive.token})
        return
    elif case=="unknown-card-input":archive.card["inputs"]["unbound_guard_waiver"]={}
    elif case=="missing-terminal":archive.card["inputs"]["supervisor_terminal"]=None
    elif case=="missing-preflight":archive.card["inputs"]["metadata_preflight"]=None
    elif case=="untrusted-card":
        with pytest.raises(ValueError):p.project(archive.card_path,"f"*64)
        return
    elif case=="symlink-input":
        link=tmp_path/"link.json";link.symlink_to(archive.directory/"raw-result.json")
        archive.card["inputs"]["raw"]={"path":str(link),"sha256":p.sha(link),"bytes":link.stat().st_size}
    save(archive.card_path,archive.card)
    with pytest.raises((ValueError,KeyError,TypeError)):archive.project()


def test_fault_output_contains_no_raw_failure_text_or_qualifying_media(archive,tmp_path):
    archive.terminal_status("INVALID",20.,"completed",1)
    archive.result["reason"]=archive.token+" "+str(archive.directory);archive.flush()
    output=tmp_path/"fault-publication";v=p.publish(archive.card_path,p.sha(archive.card_path),output)
    assert v["copied_original_gifs"]==0 and not (output/"media").exists()
    for file in output.iterdir():assert archive.token not in file.read_text()


def test_helper_has_no_scientific_or_dynamic_import_dependency():
    import ast
    tree=ast.parse(Path(p.__file__).read_text())
    names={name.name.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.Import) for name in node.names}
    names|={node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module}
    assert not names.intersection({"torch","numpy","scipy","experiments","benchmarks","particlegan","importlib","subprocess"})
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {"eval","exec","__import__"} for node in ast.walk(tree))
