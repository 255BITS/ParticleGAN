"""Execution-only two-GPU coordinator; retains frozen requests and live workers."""
from run import ROOT,QUEUE,PROGRESS,CAMPAIGN,save
import argparse
from experiments.forge.contracts import read_json,utc_now
from experiments.forge.queue import Queue,drain
import json
parser=argparse.ArgumentParser();parser.add_argument('--gpus',default='0');args=parser.parse_args();devices=args.gpus.split(',')
p=read_json(PROGRESS);p.update(phase='running',coordinator_note='Execution-only coordinator resource adjustment; running workers and frozen requests retained.');save(p)
print(json.dumps(dict(time=utc_now(),event='coordinator-resume',devices=devices,requests=p['requests'])),flush=True)
q=Queue(QUEUE,report_root=ROOT/'reports/forge',on_completion=None)
drain(q,devices,workers_per_gpu=1,allow_sharing=True,watch=False,campaign=CAMPAIGN)
p=read_json(PROGRESS);p['phase']='trained';save(p);print(json.dumps(dict(time=utc_now(),event='drain-complete')),flush=True)
