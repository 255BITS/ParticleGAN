import json,os,subprocess,time
from pathlib import Path
root=Path("/ml2/hypergan/particlegan-selective-history-20261015")
out=Path("/ml2/hypergan/combination-training-20261015")
python="/ml2/hypergan/.venvs/particlegan-selective-history-20261015/bin/python"
base=[python,"-m","experiments.forge","--root",str(root),"--queue-root",str(out/"queue")]
commands=[]
manifest={"campaign":"bcap-selective-history-pole-ring-v2","source_digest":"39fad3b03e5178bcdc68cc17ba012ac4083d5887147bef91f0bbeee1205baea6","launch_time_unix":time.time(),"queue_root":str(out/"queue"),"submissions":[],"commands":commands}
def save():
 (out/"launch-manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
for arm,candidate in [("b","recipe-prior-refactor-bcap-pure-v1"),("t","bcap-selective-history-transport-v1"),("w","bcap-selective-history-whole-v1"),("s","bcap-selective-history-adversarial-v1")]:
 command=base+["enqueue",candidate,"--study",f"bcap-selective-history-{arm}-pole-ring-study-v2"]
 commands.append(command);save()
 result=subprocess.run(command,cwd=root,capture_output=True,text=True)
 (out/"logs"/f"enqueue-{arm}.stdout.json").write_text(result.stdout)
 (out/"logs"/f"enqueue-{arm}.stderr.log").write_text(result.stderr)
 if result.returncode: raise RuntimeError(f"enqueue {arm} failed: {result.stderr}")
 receipt=json.loads(result.stdout)
 manifest["submissions"].append({"arm":arm.upper(),"receipt":receipt});save()
 print(f"ENQUEUED {arm.upper()} "+json.dumps(receipt),flush=True)
command=base+["drain","--gpus","0,1","--workers-per-gpu","1","--campaign",manifest["campaign"]]
commands.append(command)
manifest["driver_pid"]=os.getpid();manifest["drain_started_unix"]=time.time();save()
print("DRAIN START "+json.dumps(command),flush=True)
with (out/"logs"/"drain.stdout.json").open("w") as stdout,(out/"logs"/"drain.stderr.log").open("w") as stderr:
 process=subprocess.Popen(command,cwd=root,stdout=stdout,stderr=stderr)
 manifest["drain_pid"]=process.pid;save()
 code=process.wait()
manifest["drain_exit_code"]=code;manifest["drain_finished_unix"]=time.time();save()
print("DRAIN EXIT "+str(code),flush=True)
raise SystemExit(code)
