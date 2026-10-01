"""Root-only fixed planner parity/operator profile under the same count law."""
import os
import sys
os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
import plan_pair_common as common
import plan_pair_final as final
import mst_pair_checks as mst

UUID='GPU-72c1b506-891d-b8bc-b353-e020585e1c47'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-package-root',type=Path,default=common.BASE)
    parser.add_argument('--package-root',type=Path,required=True)
    parser.add_argument('--input',type=Path,default=common.INPUT)
    parser.add_argument('--contract-root',type=Path,default=common.COUNT)
    parser.add_argument('--include-mst',action='store_true',help='Optional MST edge/cut parity; excludes it by default')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Output evidence already exists')
    final.configure(args.reference_package_root,args.input,args.contract_root)
    old_map,new_map=common.sources(common.BASE),common.sources(args.package_root)
    input_sha=common.sha(common.INPUT);contract_sha=common.sha(common.COUNT/'contract_cases.py')
    law=final.law_checks(common.BASE,args.package_root,args.include_mst)
    inventory=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],
                             check=True,capture_output=True,text=True).stdout
    physical=dict(line.strip().split(', ',1) for line in inventory.strip().splitlines())
    assert physical.get('0')==UUID
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.cuda.set_device(0)
    props=torch.cuda.get_device_properties(0)
    assert 'GPU-'+str(props.uuid).removeprefix('GPU-').lower()==UUID
    torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=common.module('plan_gpu_reference',common.BASE)
    new=common.module('plan_gpu_proposal',args.package_root)
    data=torch.load(common.INPUT,map_location='cpu',weights_only=False)
    exact=common.paired(old,new,data,'cuda:0')
    quotas=final.grouped_quota_checks(old,new,data,'cuda:0')
    mst_checks=mst.checks(old,new,data,'cuda:0') if args.include_mst else []
    profiles=[];timings=[]
    for name in ('saved_toy_1000','nominal','supported_mass_imbalance','supported_mass_small_holes'):
        value=data['cases'][name]
        for temperature in ('warm','cold'):
            row=dict(name=name,temperature=temperature,scope='ordinary_transport plus select_parents only')
            timing=dict(name=name,temperature=temperature,scope='fixed planner call; preparation and parity checks excluded',milliseconds={})
            outputs={}
            for label,module in (('reference',old),('optimized',new)):
                warm=temperature=='warm'
                # Prime kernels using an independent copy of the same fixture/stream.
                final.actions(common.prepare(module,value,'cuda:0',warm=warm))
                samples=[]
                for _ in range(3):
                    prepared=common.prepare(module,value,'cuda:0',warm=warm)
                    torch.cuda.synchronize(0);start=time.perf_counter()
                    final.actions(prepared)
                    torch.cuda.synchronize(0);samples.append((time.perf_counter()-start)*1000)
                timing['milliseconds'][label]=dict(samples=samples,median=sorted(samples)[1])
                prepared=common.prepare(module,value,'cuda:0',warm=warm)
                outputs[label],row[label]=common.profile(lambda:final.actions(prepared),'cuda:0')
            assert common.same(outputs['reference'],outputs['optimized']),name+' profiled actions changed'
            assert row['optimized']['aten::_local_scalar_dense']<row['reference']['aten::_local_scalar_dense'],name
            row['actions_exact']=True
            profiles.append(row);timings.append(timing)
            print(json.dumps(dict(operator_profile=row)),flush=True)
            print(json.dumps(dict(timing=timing)),flush=True)
    mst_profiles=[]
    if args.include_mst:
        for name in ('saved_toy_1000','nominal'):
            row=dict(name=name,temperature='cold',scope='bounded K-centre MST only')
            topologies={}
            for label,module in (('reference',old),('optimized',new)):
                snap=common.prepare(module,data['cases'][name],'cuda:0')[0]
                _,row[label]=common.profile(snap._mass_topology,'cuda:0')
                topologies[label]=(snap.mass_group_ids,snap.mass_topology)
            assert common.same(topologies['reference'],topologies['optimized']),name+' MST profile cut changed'
            assert row['optimized']['aten::_local_scalar_dense']<row['reference']['aten::_local_scalar_dense']
            mst_profiles.append(row);print(json.dumps(dict(mst_profile=row)),flush=True)
    assert old_map==common.sources(common.BASE) and new_map==common.sources(args.package_root)
    assert input_sha==common.sha(common.INPUT) and contract_sha==common.sha(common.COUNT/'contract_cases.py')
    result=dict(status='PASS',scope='same count law fixed GPU actions/RNG/accounting parity and separate operator/timing observations',
        gpu_uuid=UUID,cpu_threads=1,deterministic=True,tf32=False,new_seeds=0,quality_updates=0,
        reference_package_root=str(common.BASE.resolve()),package_root=str(args.package_root.resolve()),
        reference_source_sha256=old_map,proposal_source_sha256=new_map,input_path=str(common.INPUT.resolve()),
        input_sha256=input_sha,contract_sha256=contract_sha,script_sha256=common.sha(__file__),
        common_sha256=common.sha(common.HERE/'plan_pair_common.py'),final_common_sha256=common.sha(common.HERE/'plan_pair_final.py'),
        mst_checks_sha256=common.sha(common.HERE/'mst_pair_checks.py'),law_checks=law,
        exact_contracts=exact,quota_checks=quotas,operator_profiles=profiles,timing_observations=timings,
        optional_mst_included=args.include_mst,mst_exact_contracts=mst_checks,mst_operator_profiles=mst_profiles,
        peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',output=str(args.output))),flush=True)


if __name__=='__main__':main()
