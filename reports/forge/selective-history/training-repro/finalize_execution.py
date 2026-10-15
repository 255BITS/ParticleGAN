import hashlib,json,time
from pathlib import Path
root=Path("/ml2/hypergan/particlegan-selective-history-20261015")
out=Path("/ml2/hypergan/combination-training-20261015")
queue=out/"queue"
campaign="bcap-selective-history-pole-ring-v2"
planned=json.loads((out/"planned-roster.json").read_text())
expected={j["key"] for a in planned["arms"] for j in a["jobs"]}
state=json.loads((queue/"queue/state.json").read_text())
assert set(state["jobs"])==expected and len(expected)==8
assert all(j["status"]=="terminal" for j in state["jobs"].values()),"not all eight jobs terminal"
assert set(state["campaigns"])=={campaign}
rows=[]
for key,job in state["jobs"].items():
 result=job["result"]
 aid=result["attempt_id"];directory=queue/campaign/aid
 wrapper=json.loads((directory/"request.json").read_text());request=wrapper["request"]
 assert request["source"]["digest"]==planned["source_digest"]
 assert request["protocol"]["seed"]==0
 assert request["campaign_id"]==campaign
 assert "retry_of" not in result
 assert len(result["task_results"])==1
 row=result["task_results"][0];assert row["raw_status"]=="completed"
 durable=root/"reports/forge/attempts"/aid
 assert all((durable/n).is_file() for n in ["request.json","result.json","evidence.json"])
 hashes={n:hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in ["request.json","raw-result.json","graded-result.json","terminal.json","result.json"]}
 rows.append({"candidate":request["candidate"]["id"],"study":request["study"]["id"],"request_id":wrapper["request_id"],"attempt_id":aid,"task":row["task_id"],"job_key":key,"gate_status":row["gate_status"],"raw_status":row["raw_status"],"reason":row["reason"],"wall_seconds":row["cost"]["wall_seconds"],"device":row["cost"]["device"],"metrics":{k:v for k,v in row.get("metrics",{}).items() if isinstance(v,(int,float))},"external_artifact_directory":str(directory),"durable_receipt_directory":str(durable),"file_sha256":hashes,"saved_frame_files":[p.name for p in directory.iterdir() if p.name in ["observed-samples.pt","constraint_geometry-scored-outputs.pt"]],"existing_visual_files":[p.name for p in directory.iterdir() if p.suffix in [".gif",".png",".mp4"]]})
charges=state["charges"]
assert len(charges)==8 and len({c["attempt_id"] for c in charges})==8
actual=sum(c["seconds"] for c in charges)
assert actual==state["campaigns"][campaign]["spent_seconds"]
assert actual<=2400 and state["campaigns"][campaign]["reserved_seconds"]==0
by_candidate={}
for row in rows:by_candidate[row["candidate"]]=by_candidate.get(row["candidate"],0)+row["wall_seconds"]
assert all(v<=600 for v in by_candidate.values())
amendment=json.loads((out/"authorization.json").read_text())
for n,sha in amendment["historical_frozen_file_sha256"].items():assert hashlib.sha256((root/"reports/forge/selective-history"/n).read_bytes()).hexdigest()==sha
transition=json.loads((out/"coordinator-transition.json").read_text())
for aid,value in transition["completed_receipts_before"].items():assert hashlib.sha256((queue/campaign/aid/"terminal.json").read_bytes()).hexdigest()==value["sha256"]
summary={"schema":"selective_history_training_completion_v2","status":"COMPLETE","completed_at_unix":time.time(),"campaign":campaign,"source_digest":planned["source_digest"],"source_origin_commit":planned["source_origin_commit"],"scientific_attempts":8,"completed_attempts":8,"scientific_retries":0,"spent_seconds":actual,"campaign_cap_seconds":2400,"remaining_seconds":2400-actual,"reserved_seconds":0,"candidate_spend_seconds":by_candidate,"qualification_input":False,"scientific_scope":"research_diagnostic","coordinator_repair":"Interrupted full-memory compilation after two completed baseline jobs, then resumed same public Queue/drain with deferred compilation; no training was interrupted or repeated.","jobs":sorted(rows,key=lambda r:(r["candidate"],r["task"]))}
(out/"completion.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="jobs"},indent=2))
