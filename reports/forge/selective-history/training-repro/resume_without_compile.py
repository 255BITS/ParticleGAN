import json,os,time
from pathlib import Path
from experiments.forge.queue import Queue,drain
root=Path("/ml2/hypergan/particlegan-selective-history-20261015")
out=Path("/ml2/hypergan/combination-training-20261015")
queue=Queue(out/"queue",report_root=root/"reports/forge",on_completion=None)
record=json.loads((out/"coordinator-transition.json").read_text())
record.update(new_coordinator_pid=os.getpid(),resume_started_unix=time.time(),new_driver=str(Path(__file__).resolve()),workers_per_gpu=1,devices=["0","1"])
(out/"coordinator-transition.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("RESUME",json.dumps(record),flush=True)
drain(queue,["0","1"],workers_per_gpu=1,allow_sharing=False,campaign="bcap-selective-history-pole-ring-v2")
record["resume_finished_unix"]=time.time()
(out/"coordinator-transition.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print("DRAIN COMPLETE",flush=True)
