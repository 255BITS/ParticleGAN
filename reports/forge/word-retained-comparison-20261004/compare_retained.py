"""Compare fixed saved word arrays. No models, sampling or official grading."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("run with CUDA explicitly hidden")
import numpy as np

CHARS="abcdefghijklmnopqrstuvwxyz_ "
WORDS=["apple_","grape_","lemon_","melon_","berry_"]
STEPS=[math.ceil(i*20001/24) for i in range(1,25)]
DETAIL_STEPS={834,10001,15835,20001}
OLD_ORIGIN="fb7acc775b3a1a6184d36b55e035b9da04531492"
OLD_DIGEST="f380eed990931bacb205e6537beaf387fdbb676ffc97b6f32f19ff903ae1cfed"
NEW_ORIGIN="f9f7ed9d7a06c48d4ec56999107658983d7e8efc"
NEW_DIGEST="5995590d3c303207c664fe3b0ba7dc1e09dd3da7a90abf87086634c789175026"
OLD_TASK="five_word_joint_acquisition_word_joint_policy_min11_v1"
NEW_TASK="five_word_joint_acquisition_word_joint_policy_min11_rates_v1"
KEYS={"target","generated","reconstruction","prior","generated_raw_code","generated_effective_code",
    "encoded_code","reconstruction_effective_code"}
SOURCES=["experiments/forge/word_joint_policy_adapters.py","particlegan/policy.py","particlegan/continuous.py",
    "particlegan/recipes.py","particlegan/gan_loss.py","particlegan/particle_prior.py",
    "benchmarks/toy_audit/api_images.py","benchmarks/toy_audit/definition_quality.py"]


def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"duplicate JSON field");result[key]=value
        return result
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda x:(_ for _ in ()).throw(ValueError("nonfinite JSON")))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def pin(path):
    p=Path(path).resolve();return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


class Inputs:
    def __init__(self):self.files={}
    def checked(self,item,label):
        p=Path(item["path"])
        require(p.is_absolute() and p.is_file() and not p.is_symlink(),"missing/unsafe retained input: "+label)
        value=pin(p);require(value==item,"changed retained input: "+label)
        require(str(p) not in self.files or self.files[str(p)]==value,"changed during reading")
        self.files[str(p)]=value;return p
    def file(self,path,label,expected=None):
        value=pin(path)
        if expected is not None:require(value["sha256"]==expected,"wrong frozen input: "+label)
        return self.checked(value,label)
    def json(self,item,label):return read(self.checked(item,label))
    def recheck(self):require(all(pin(p)==v for p,v in self.files.items()),"retained bytes changed during analysis")


def decoded(array):
    return ["".join(CHARS[int(i)] for i in row) for row in array.argmax(1)]


def pair_stats(points):
    d=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    upper=d[np.triu_indices(len(points),1)]
    return dict(minimum=float(upper.min()),median=float(np.median(upper)),maximum=float(upper.max()))


def rms_rows(value):return float(np.sqrt(np.mean(np.sum(value.astype(np.float64)**2,axis=1))))


def safe_ratio(numerator,denominator):return float(numerator/denominator) if denominator>0 else None


def descriptor(arrays,recorded,step):
    require(set(arrays)==KEYS,"saved array key set differs")
    expected={"target":(5,28,6),"generated":(1024,28,6),"reconstruction":(5,28,6),"prior":(11,2),
        "generated_raw_code":(1024,2),"generated_effective_code":(1024,2),"encoded_code":(5,2),
        "reconstruction_effective_code":(5,2)}
    for key,value in arrays.items():
        require(value.shape==expected[key] and np.issubdtype(value.dtype,np.floating)
            and np.isfinite(value).all(),"saved array shape/type/health differs")
    target,generation,reconstruction=[arrays[k] for k in ("target","generated","reconstruction")]
    require(decoded(target)==WORDS and np.array_equal(target.sum(1),np.ones((5,6))),"canonical five-word targets differ")
    for value in (generation,reconstruction):
        require((value>=0).all() and np.allclose(value.sum(1),1.,rtol=0.,atol=1e-6),"saved words must be normalized probabilities")
    words=decoded(reconstruction);individual=[a==b for a,b in zip(words,WORDS)]
    # These are descriptors of retained values, never calls to the official scorer.
    true_reconstruction=(target*reconstruction).sum(1)
    all_true=np.einsum("nct,wct->nwt",generation,target).min(2)
    argmax=generation.argmax(1);tokens=target.argmax(1)
    spelling_counts=[int(np.all(argmax==word,axis=1).sum()) for word in tokens]
    encoded,prior=arrays["encoded_code"],arrays["prior"]
    e_pair=pair_stats(encoded);p_pair=pair_stats(prior)
    generated_delta=arrays["generated_effective_code"]-arrays["generated_raw_code"]
    inverse_delta=arrays["reconstruction_effective_code"]-encoded
    reconstruction_rms=rms_rows(inverse_delta);generated_rms=rms_rows(generated_delta)
    e_to_prior=np.linalg.norm(encoded[:,None,:]-prior[None,:,:],axis=2).min(1)
    generated_to_encoder=np.linalg.norm(arrays["generated_effective_code"][:,None,:]-encoded[None,:,:],axis=2).min(1)
    query_nearest=np.linalg.norm(arrays["reconstruction_effective_code"][:,None,:]-encoded[None,:,:],axis=2).argmin(1)
    equal_to_real=(arrays["generated_effective_code"][:,None,:]==encoded[None,:,:]).all(2).any(1)
    generated_matches_prior=(arrays["generated_raw_code"][:,None,:]==prior[None,:,:]).all(2).any(1)
    require(generated_matches_prior.all(),"saved raw generation codes are not actual selected rows")
    output=dict(step=step,recorded_metrics=deepcopy(recorded),recorded_grade_recomputed=False,
        reconstruction=dict(decoded_words=words,individually_exact_argmax_count=sum(individual),
            individual_exact_argmax=individual,true_token_minimum=true_reconstruction.min(1).astype(float).tolist(),
            true_token_mean=true_reconstruction.mean(1).astype(float).tolist(),
            true_token_probabilities=true_reconstruction.astype(float).tolist() if step in DETAIL_STEPS else None,
            argmax_is_display_diagnostic_only=True),
        generation=dict(decoded_canonical_counts=dict(zip(WORDS,spelling_counts)),
            decoded_other_count=1024-sum(spelling_counts),
            best_minimum_true_token_probability=dict(zip(WORDS,all_true.max(0).astype(float).tolist())),
            median_minimum_true_token_probability=dict(zip(WORDS,np.median(all_true,axis=0).astype(float).tolist())),
            counts_do_not_replace_confidence_filtered_original_masses=True),
        codes=dict(prior_unique_rows=len(np.unique(prior,axis=0)),encoded_unique_rows=len(np.unique(encoded,axis=0)),
            prior_axis_std=prior.std(0).astype(float).tolist(),encoder_axis_std=encoded.std(0).astype(float).tolist(),
            prior_pair_distances=p_pair,encoder_pair_distances=e_pair,encoded_to_nearest_prior_distance=e_to_prior.astype(float).tolist(),
            generated_effective_to_nearest_encoder_median=float(np.median(generated_to_encoder)),
            generated_effective_equal_to_any_encoded_count=int(equal_to_real.sum()),
            generated_effective_changed_count=int(np.any(generated_delta!=0,axis=1).sum()),
            inverse_effective_changed_count=int(np.any(inverse_delta!=0,axis=1).sum()),
            inverse_effective_nearest_original_encoded_label=[WORDS[int(i)] for i in query_nearest],
            generated_dv12_rms=generated_rms,inverse_dv12_rms=reconstruction_rms,
            inverse_dv12_rms_over_minimum_encoder_separation=safe_ratio(reconstruction_rms,e_pair["minimum"]),
            generated_dv12_rms_over_prior_rms_axis_std=safe_ratio(generated_rms,float(np.linalg.norm(prior.std(0)))),
            terminal_encoded_codes=encoded.astype(float).tolist() if step==20001 else None,
            terminal_inverse_effective_codes=arrays["reconstruction_effective_code"].astype(float).tolist() if step==20001 else None))
    return output


def ast_functions(path):
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(ast.parse(Path(path).read_text()))
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}


def compare(spec_path):
    inputs=Inputs();spec=read(spec_path);inputs.file(spec_path,"reproduction specification")
    old_analysis=inputs.json(spec["old_goal_analysis"],"old goal analysis")
    old_law=inputs.json(spec["old_law_review"],"old law review")
    for path,item in old_law["inputs"].items():
        inputs.checked(dict(path=path,**item),"old 46-pin proof")
    old_raw_path=next(Path(k) for k in old_analysis["inputs"] if k.endswith("/raw-result.json"))
    old_raw=read(old_raw_path)
    old_source=Path(old_law["source"]["snapshot_path"])
    old_resolved_path=next(Path(k) for k in old_analysis["inputs"] if k.endswith("/resolved.json"))
    old_resolved=read(old_resolved_path)
    old_header=inputs.json(old_law["source"]["manifest"],"old source manifest")
    require(old_header["origin_commit"]==OLD_ORIGIN and old_header["digest"]==OLD_DIGEST,"old source identity differs")
    card=inputs.json(spec["new_terminal_card"],"new trusted terminal card")
    require(card["terminal_immutable"] is True,"current terminal cut required")
    current={name:inputs.json(card["inputs"][name],"new "+name) for name in
        ("prepared","source_manifest","study","cost","resolved","raw","grading","control","media","metadata_preflight")}
    publication=inputs.json(spec["new_publication_results"],"new passive publication")
    index=inputs.json(spec["new_publication_index"],"new publication input index")
    require(publication["status"]=="COMPLETE" and publication["accepted_numeric"]=="FAIL"
        and publication["counts"]=={"FAIL":1,"NOT_RUN":25} and publication["provenance"]["trusted_card_sha256"]==spec["new_terminal_card"]["sha256"],"new accepted result identity differs")
    indexed={x["label"]:x for x in index["files"]}
    for name,item in card["inputs"].items():
        if item is not None:inputs.checked(item,"new root card "+name)
    new_raw=current["raw"];new_source=Path(card["inputs"]["source_manifest"]["path"]).parent
    require(current["source_manifest"]["origin_commit"]==NEW_ORIGIN and current["source_manifest"]["digest"]==NEW_DIGEST
        and publication["source"]["origin_commit"]==NEW_ORIGIN and publication["source"]["digest"]==NEW_DIGEST,"new source identity differs")
    old_recipe=old_raw["applied"]["recipe"];new_recipe=new_raw["applied"]["recipe"]
    differences={key:dict(old=old_recipe.get(key),new=new_recipe.get(key)) for key in set(old_recipe)|set(new_recipe)
        if old_recipe.get(key)!=new_recipe.get(key)}
    require(differences=={"lr":dict(old=.0053125,new=.00265625)},"comparison has additional Recipe changes")
    require(old_raw["task_id"]==OLD_TASK and new_raw["task_id"]==NEW_TASK
        and old_raw["applied"]["initialization"]==new_raw["applied"]["initialization"],"task/named initialization parity differs")
    task_defs=[old_resolved["request"]["tasks"][OLD_TASK],current["resolved"]["request"]["tasks"][NEW_TASK]]
    require(task_defs[0]["evaluation"]["thresholds"]==task_defs[1]["evaluation"]["thresholds"]
        and task_defs[0]["execution"]["steps"]==task_defs[1]["execution"]["steps"]==20001
        and old_resolved["request"]["protocol"]["seed"]==current["resolved"]["request"]["protocol"]["seed"]==0,"gates/clock/seed differ")
    source_checks={}
    for name in SOURCES:
        old=inputs.file(old_source/name,"old "+name,old_header["files"][name])
        new=inputs.file(new_source/name,"new "+name,current["source_manifest"]["files"][name])
        source_checks[name]=dict(old_sha256=sha(old),new_sha256=sha(new),byte_equal=sha(old)==sha(new))
    af,bf=[ast_functions(root/SOURCES[0]) for root in (old_source,new_source)]
    core={name:af[name]==bf[name] for name in ("join_words","joint_generation","words_only_noise","fake_joint","generator_objective","step")}
    require(all(core.values()) and all(v["byte_equal"] for k,v in source_checks.items() if k!=SOURCES[0]),"physical model/loss/policy/scorer source drift")
    cohorts=[]
    for label,raw,root in (("original_n11",old_raw,old_raw_path.parent),("half_base",new_raw,Path(card["inputs"]["raw"]["path"]).parent)):
        evidence=raw["evidence"];controls=evidence["policy_controls"]
        require([x["step"] for x in evidence["observations"]]==STEPS and [x["completed_steps"] for x in evidence["policy_observations"]]==STEPS,
            "24 matched clocks required")
        require(all(x["selected_source"]=="fast" and x["controller"]=="dv12" and x["output_noise"] is False for x in evidence["policy_observations"])
            and all(x["pure"] is True for x in evidence["policy_purity"]),"saved observation law/purity differs")
        summaries=[]
        for recorded in evidence["observations"]:
            step=recorded["step"];name=f"observations/step_{step:06d}.npz";entry=evidence["artifact_manifest"]["files"][name]
            path=Path(evidence["artifact_root"])/name
            inputs.checked(dict(path=str(path),sha256=entry["sha256"],bytes=entry["size"]),label+" saved arrays")
            if label=="half_base":
                require(indexed["artifact:"+name]["sha256"]==entry["sha256"],"accepted publication array pin differs")
            with np.load(path,allow_pickle=False) as archive:
                summaries.append(descriptor({k:archive[k] for k in archive.files},recorded,step))
        sigma=float(controls["output_sigma"]);controller=controls["diagnostics"]["controller"]
        require(math.isfinite(sigma) and sigma>=0,"recorded terminal sigma is finite")
        source_status=(dict(status="INVALID",accepted_numeric="UNAVAILABLE",paid_seconds=558.5739127129782)
            if label=="original_n11" else dict(status="COMPLETE",accepted_numeric="FAIL",paid_seconds=publication["cost"]["paid_seconds"]))
        terminal_noise=dict(recorded_output_sigma=sigma,per_coordinate_units="post-softmax word-coordinate training noise",
            expected_word_noise_l2=sigma*math.sqrt(168),expected_noise_l2_over_canonical_word_l2=sigma*math.sqrt(168)/math.sqrt(6),
            expected_noise_is_analytic_scale_not_a_draw=True,
            recorded_training_latent_bandwidth=controller["latent_bandwidth"],
            last_two_recorded_training_latent_applications=controller["latent_applications"],
            per_checkpoint_training_output_sigma="NOT_RETAINED; endpoint amplitude only",
            eval_output_noise=False,eval_dv12_remains=True)
        cohorts.append(dict(id=label,preserved_status=source_status,recipe=deepcopy(raw["applied"]["recipe"]),
            recorded_owner_updates=deepcopy(evidence["guards"]["optimizer_updates"]),selected_sources=["fast"],
            all_five_recorded_reconstruction_flags=[x["reconstruction_exact"] for x in evidence["observations"]],
            five_mode_recorded_steps=[x["step"] for x in evidence["observations"] if x["modes"]==5],
            maximum_individual_exact_argmax=max(x["reconstruction"]["individually_exact_argmax_count"] for x in summaries),
            terminal_noise=terminal_noise,terminal_birth_counters=deepcopy(controls["diagnostics"]["birth_death"]["counters"]),
            terminal_surprise=deepcopy(controls["diagnostics"]["surprise"]),saved_checkpoints=summaries))
    pairs=[dict(step=step,original_n11=cohorts[0]["saved_checkpoints"][i],half_base=cohorts[1]["saved_checkpoints"][i]) for i,step in enumerate(STEPS)]
    for cohort in cohorts:cohort.pop("saved_checkpoints")
    require(not any(name=="torch" or name.startswith(("experiments.","particlegan.","benchmarks.")) for name in sys.modules),"scientific module imported")
    inputs.recheck()
    return dict(schema="pg_word_saved_checkpoint_comparison_v1",status="RETAINED_DIAGNOSTIC_NO_REGRADING",
        old_source=dict(origin_commit=OLD_ORIGIN,digest=OLD_DIGEST),new_source=dict(origin_commit=NEW_ORIGIN,digest=NEW_DIGEST),
        source_checks=source_checks,training_function_ast_equal=core,recipe_differences=differences,
        named_initialization_equal=True,matched_steps=STEPS,same_seed=0,same_targets=WORDS,same_full_horizon=20001,
        original_thresholds=task_defs[0]["evaluation"]["thresholds"],cohorts=cohorts,matched_checkpoints=pairs,
        input_pins=list(inputs.files.values()),input_count=len(inputs.files),
        one_causal_repair_proposal=dict(status="PROPOSAL_ONLY_NOT_IMPLEMENTED_OR_EXECUTED",
            kind="explicit_new_objective_and_family_variant",
            change="Add paired six-token NLL for all five original known words through the existing stochastic public G(E(word)) path, at fixed reconstruction_weight=1.",
            path="Use the actual owned differentiable fast G/E and public UpdatePolicy.generate inside the generator update, never an inference snapshot. DV12 remains in the effective query; sigma=0 matches the inverse measurement. The auxiliary NLL bypasses D and supplies a direct word -> E -> public generator -> correct-token gradient.",
            implementation_requirements="Use numerically stable log probabilities from the same generator logits and prove forward/gradient parity; taking log of underflowed saved softmax values or a hard-clamped probability is insufficient. Declare a separate auxiliary training-noise stream; no silent reuse/shift of original fake or evaluation streams.",
            existing_loss_retained="Current joint RpGAN and original regularizers remain; add exactly this explicit auxiliary term. Recipe.reconstruction_weight=1 is currently unused by the word caller.",
            contrast_conditions="Preserve current half-base nominal rates, all public owners, same-code generation callback, N11/raw prior, architecture, initialization, seed, DV12/noise laws, full horizon and original gates.",
            source_reason="Identical old/new generator_objective has no direct inverse loss. Paired inverse remains incomplete at all 24 clocks despite occasional individual recovery.",
            prediction="The original five paired minimum-token probabilities and all-five exact inverse criterion should improve persistently; all generation/quality/mass gates still must independently pass.",
            falsifier="No persistent improvement of those original finite inverse measurements, or continued original full-gate failure, rejects the proposal as a complete repair; coverage and inverse must not be pooled across checkpoints.",
            limits="No proof of exact joint-law equality, stochastic capacity, DV12 causality, or default/speed benefit. Compressed code neighborhoods, the finite discriminator, moving support and asymmetric noisy joint distributions remain competing explanations.",
            learning_rate_conclusion="This one failed global LR reduction changes trajectories but does not establish that LR is irrelevant.",
            old_or_current_variant_credit=False,new_candidate_declared=False),
        scope=dict(model_constructions=0,checkpoint_deserializations=0,model_restores=0,forwards=0,samples_drawn=0,
            optimizer_updates=0,official_scorer_calls=0,cuda_initialized=False,numerical_grades_changed=False,
            old_invalid_recertified=False,default_credit=False,speed_credit=False,qualification_input=False),
        missing=["Same-state G(E(word)) with DV12 disabled", "Per-update loss/parameter-delta history",
            "Earlier complete model checkpoints", "Per-checkpoint training output-noise amplitude",
            "Capacity witness under this exact stochastic joint/serving law", "An intervention identifying the actual cause"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--inputs",required=True,type=Path);parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv);require(not args.output.exists(),"fresh comparison output required")
    result=compare(args.inputs);result["reproducer"]=pin(__file__)
    args.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"],inputs=result["input_count"],matched_clocks=len(result["matched_steps"]),
        output_sha256=sha(args.output),**result["scope"]),sort_keys=True))


if __name__=="__main__":raise SystemExit(main())
