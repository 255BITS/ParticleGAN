"""CPU proof of the optional bounded MST optimization, no GPU context."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import torch
import plan_pair_common as common
import mst_pair_checks as mst


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package-root',type=Path,default=common.BASE)
    parser.add_argument('--package-root',type=Path,default=common.HERE/'pkg-PLAN')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Evidence already exists')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    maps=[common.sources(p) for p in (args.reference_package_root,args.package_root)]
    input_sha=common.sha(common.INPUT)
    data=torch.load(common.INPUT,map_location='cpu',weights_only=False)
    old=common.module('mst_cpu_old',args.reference_package_root)
    new=common.module('mst_cpu_new',args.package_root)
    rows=mst.checks(old,new,data,'cpu');profiles=[]
    for name in ('saved_toy_1000','nominal'):
        row=dict(name=name)
        for label,module in (('reference',old),('optimized',new)):
            snap=common.prepare(module,data['cases'][name],'cpu')[0]
            _,row[label]=common.profile(snap._mass_topology,'cpu')
        assert row['optimized']['aten::_local_scalar_dense']<row['reference']['aten::_local_scalar_dense']
        profiles.append(row);print(json.dumps(dict(profile=row)),flush=True)
    assert maps==[common.sources(p) for p in (args.reference_package_root,args.package_root)]
    assert input_sha==common.sha(common.INPUT) and not torch.cuda.is_initialized()
    result=dict(status='PASS',scope='CPU optional MST selection/cut proof; GPU inclusion still conditional',
        cpu_threads=1,cuda_initialized=False,new_seeds=0,cases=rows,profiles=profiles,
        reference_source_sha256=maps[0],proposal_source_sha256=maps[1],input_sha256=input_sha,
        script_sha256=common.sha(__file__),common_sha256=common.sha(common.HERE/'plan_pair_common.py'),
        mst_checks_sha256=common.sha(common.HERE/'mst_pair_checks.py'))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(rows),output=str(args.output))),flush=True)


if __name__=='__main__':main()
