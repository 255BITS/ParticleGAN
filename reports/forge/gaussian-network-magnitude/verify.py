"""Exact CUDA restores via the existing saved-state verifier; no training."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from benchmarks.toy_audit import gaussian_network_magnitude as study
path=ROOT/'reports/forge/bcap-past-extrapolation/verify.py'
spec=importlib.util.spec_from_file_location('network_saved_verifier',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.build=study.build;module.declaration=study.declaration

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();rows=module.verify(args.raw,device=args.device)
    print(json.dumps(dict(exact_restores=len(rows),training_updates=0,model_sampling_draws=0)))
