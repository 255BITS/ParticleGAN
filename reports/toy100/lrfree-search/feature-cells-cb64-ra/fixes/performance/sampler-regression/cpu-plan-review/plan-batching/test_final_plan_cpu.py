"""CPU integer/action/RNG proof against the same final statistical count law."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import argparse
import itertools
import json
from pathlib import Path
import time
import torch
import plan_pair_common as common
import plan_pair_final as final


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package-root',type=Path,default=common.BASE)
    parser.add_argument('--package-root',type=Path,required=True)
    parser.add_argument('--input',type=Path,default=common.INPUT)
    parser.add_argument('--contract-root',type=Path,default=common.COUNT)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Output evidence already exists')
    final.configure(args.reference_package_root,args.input,args.contract_root)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    old_map,new_map=common.sources(common.BASE),common.sources(args.package_root)
    input_sha=common.sha(common.INPUT);contract_sha=common.sha(common.COUNT/'contract_cases.py')
    law=final.law_checks(common.BASE,args.package_root)
    old=common.module('final_plan_cpu_old',common.BASE)
    new=common.module('final_plan_cpu_new',args.package_root)
    start=time.perf_counter();checks=0
    for capacity in itertools.product(range(4),repeat=4):
        cap=torch.tensor(capacity)
        for groups in ((0,0,1,1),(0,1,0,1),(0,0,0,0)):
            group=torch.tensor(groups)
            for totals in ((0,0),(1,1),(2,3),(7,7),(-1,3),(2,0),(1,4),(4,2)):
                total=torch.tensor(totals)
                expected=sum((old._integer_allocate(cap*(group==g),int(total[g])) for g in range(2)),torch.zeros_like(cap))
                actual=new._group_integer_allocate(cap,group,total)
                assert torch.equal(expected,actual),(capacity,groups,totals)
                assert bool(((actual>=0)&(actual<=cap)).all());checks+=1
    data=torch.load(common.INPUT,map_location='cpu',weights_only=False)
    cases=common.paired(old,new,data,'cpu')
    quotas=final.grouped_quota_checks(old,new,data,'cpu')
    profiles=[]
    for name in ('saved_toy_1000','nominal','supported_mass_imbalance','supported_mass_small_holes'):
        value=data['cases'][name];row=dict(name=name,temperature='warm')
        outputs={}
        for label,mod in (('reference',old),('optimized',new)):
            prepared=common.prepare(mod,value,'cpu',warm=True)
            outputs[label],row[label]=common.profile(lambda:final.actions(prepared),'cpu')
        assert common.same(outputs['reference'],outputs['optimized']),name+' profiled actions changed'
        assert row['optimized']['aten::_local_scalar_dense']<row['reference']['aten::_local_scalar_dense'],name
        profiles.append(row);print(json.dumps(dict(profile=row)),flush=True)
    assert old_map==common.sources(common.BASE) and new_map==common.sources(args.package_root)
    assert input_sha==common.sha(common.INPUT) and contract_sha==common.sha(common.COUNT/'contract_cases.py')
    assert not torch.cuda.is_initialized()
    result=dict(status='PASS',scope='same count law CPU exact integer/action/RNG/accounting parity; no quality verdict',
        cpu_threads=1,cuda_initialized=False,new_seeds=0,seconds=time.perf_counter()-start,
        exhaustive_integer_cases=checks,cases=cases,quota_checks=quotas,profiles=profiles,law_checks=law,
        package_root=str(args.package_root.resolve()),source_sha256=new_map,base_source_sha256=old_map,
        reference_package_root=str(common.BASE.resolve()),input_path=str(common.INPUT.resolve()),
        input_sha256=input_sha,contract_sha256=contract_sha,script_sha256=common.sha(__file__),
        common_sha256=common.sha(common.HERE/'plan_pair_common.py'),final_common_sha256=common.sha(common.HERE/'plan_pair_final.py'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(cases),integer_cases=checks,output=str(args.output))),flush=True)


if __name__=='__main__':main()
