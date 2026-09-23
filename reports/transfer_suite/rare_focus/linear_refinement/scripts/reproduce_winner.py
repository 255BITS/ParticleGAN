"""Replay the rare-mode winner from a restored exact source checkout.

python reproduce_winner.py --source-root /path/to/restored/checkout --output /tmp/new-replay
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.source_root.resolve()))
import torch
from benchmarks.transfer_suite import suite,vector_tasks
from benchmarks.transfer_suite.protocol import test_verdict
from benchmarks.transfer_suite.linear_skip_refinement_research import ARCHITECTURES,constructor

torch.set_num_threads(1)
args.output.mkdir(parents=True,exist_ok=False)
architecture=deepcopy(next(c for c in ARCHITECTURES if c['name']=='linear_skip_d96_beta5'))
original=deepcopy(next(s for s in suite.manifest()['tasks'] if s['name']=='vector_unequal_mass'))
spec=original|dict(d_hidden=96,d_layers=2,research_discriminator=architecture)
protocol=suite.snapshot(args.output)
policy=vector_tasks.fixed_policy('cosine')
saved=vector_tasks.SimpleMLPDiscriminator
vector_tasks.SimpleMLPDiscriminator=constructor(architecture)
print('START linear_skip_d96_beta5 vector_unequal_mass',flush=True)
try:result=vector_tasks.run_episode(spec,policy,fixed=True)
finally:vector_tasks.SimpleMLPDiscriminator=saved
suite.verify_source(protocol)
verdict=test_verdict(spec,result)
record=dict(original_spec=original,spec=spec,architecture=architecture,policy=policy,result=result,verdict=verdict,protocol=protocol)
(args.output/'result.json').write_text(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps(dict(status=verdict['status'],live=result.get('live'),convergence=verdict.get('convergence')),indent=2))
