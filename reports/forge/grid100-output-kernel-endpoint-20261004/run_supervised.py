"""One corrected endpoint attempt with the original 180s ceiling, prior debit retained."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PRODUCER = Path('/ml2/hypergan/pg-grid100-output-kernel-diagnostic-v2-20261003/run_grid100_output_kernel.py')
PROTOCOL = PRODUCER.with_name('protocol.json')
PRODUCER_SHA = '596a30fa791297c000037c9b8e8bd3971d45878fe16775755119057f7e6ec5fd'
PROTOCOL_SHA = '9626b5c39966d425dd49422069c5f4d80040f0a7afd5a9f24fd1926bbdf03257'
COMMIT = '9563dea57bb150f2a0275bbe8d785bf76210fca3'
DIGEST = 'db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037'
MAINTAINED = Path('/ml2/hypergan/ParticleGAN-atlas-forge-unblock-20261003')
QUEUE = Path('/ml2/hypergan/ParticleGAN-single-recipe/runs/forge')
FAMILY = 'atlas_grid100_output_kernel_endpoint_corrected'
CASE = 'grid100_zero_update_output_kernel_endpoint_v2'
ORIGINAL_CAP = 180.
PRIOR_PAID = 5.461433995049447
REMAINING_CAP = ORIGINAL_CAP - PRIOR_PAID
PRIOR_REFERENCE = {'source': {'commit': '9563dea57bb150f2a0275bbe8d785bf76210fca3',
            'digest': 'db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037'},
 'status': 'INVALID',
 'paid_seconds': 5.461433995049447,
 'reserved_seconds': 0.0,
 'qualification_input': False,
 'outcomes_reused': False,
 'artifacts': {'study': {'path': '/ml2/hypergan/pg-grid100-output-kernel-endpoint-supervised-20261003/study.json',
                         'sha256': 'b5adfb765a00f8e716410f630ed6fece7b61a8b2a6e023c1ed10a20b9ef644ab',
                         'bytes': 346557},
               'cost': {'path': '/ml2/hypergan/pg-grid100-output-kernel-endpoint-supervised-20261003/cost.json',
                        'sha256': '6426f36ef4cea394f943c5689fd3acc7d48b404fa75cb6741d93aa6fe07b7fa0',
                        'bytes': 819},
               'resolved': {'path': '/ml2/hypergan/pg-grid100-output-kernel-endpoint-supervised-20261003/resolved.json',
                            'sha256': '7c4fa12beba3e6439a97f5086d98ddbb0c22805ce1d39c21915c09e8a6f3123a',
                            'bytes': 351756},
               'prepared': {'path': '/ml2/hypergan/.pg-grid100-output-kernel-endpoint-supervised-20261003.endpoint-supervision.json',
                            'sha256': '8a706932b9f9a0f957364f7dd27bc2a5f760bc6c6b4c9c83c23052b571dc7004',
                            'bytes': 345314},
               'terminal': {'path': '/ml2/hypergan/ParticleGAN-single-recipe/runs/forge/policy/attempts/9ed410c1c0777d98597248c576477442a7084f85c7bacfa4ad62ab503e1235fd/supervisor-terminal.json',
                            'sha256': '9c9f58d4e72d8bbb6985ce33b2e758d1a5fbacc437b78f572767b94b228b531c',
                            'bytes': 150},
               'supervisor_request': {'path': '/ml2/hypergan/ParticleGAN-single-recipe/runs/forge/policy/attempts/9ed410c1c0777d98597248c576477442a7084f85c7bacfa4ad62ab503e1235fd/supervisor-request.json',
                                      'sha256': 'f09effe6e32f09aa5031103c7e37a25985c9917d2b9ee443c5fdd50e753eb5f8',
                                      'bytes': 171287},
               'invalid_receipt': {'path': '/ml2/hypergan/pg-grid100-output-kernel-endpoint-supervised-20261003/endpoint/invalid.json',
                                   'sha256': 'c14e6bd37c177d141c8484777620cf2b1b5edebb8938eb828956070ff528bf9e',
                                   'bytes': 276},
               'source_manifest': {'path': '/ml2/hypergan/diagnostic-source/snapshots/db5492df4aa5ef60ce9e7d5869b2b6be8d492f19191a2889967274555f8d0037/forge-source.json',
                                   'sha256': '8b7bf13530a37b62d944d86e6393165d2675346a4c963fd9f162cddeefd1cc98',
                                   'bytes': 167821},
               'original_maintained_coordinator': {'path': '/ml2/hypergan/ParticleGAN-atlas-forge-unblock-20261003/experiments/forge/policy_execution.py',
                                                   'sha256': 'fa99cb761903211d17fa1eaded93ff3f5d7a92c9d3f15c327a42d92e9c9e197a',
                                                   'bytes': 22816},
               'original_protocol': {'path': '/ml2/hypergan/pg-grid100-output-kernel-diagnostic-20261003/protocol.json',
                                     'sha256': '13d57b7d937099d0e2892b21381f128e23f5b7d509f0b0407205e2e852bb4bab',
                                     'bytes': 181598},
               'original_script': {'path': '/ml2/hypergan/pg-grid100-output-kernel-diagnostic-20261003/run_grid100_output_kernel.py',
                                   'sha256': 'e1adcb72a60b9b24e6e046bbb306910b7987fb5cfc986cfa916d1803219ed713',
                                   'bytes': 19396},
               'original_wrapper': {'path': '/ml2/hypergan/pg-grid100-output-kernel-supervisor-20261003/run_supervised.py',
                                    'sha256': '10ea7b965ea89df21fd4d7c814d008a7071714d51af2a921880e946c0afd7817',
                                    'bytes': 13984}}}
OLD_DRIVER = 'reports/forge/atlas-current-gpu-diagnostics-v1/run_diagnostics.py'
COORDINATOR = 'experiments/forge/policy_execution.py'
ENV = dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def checked(item):
    path=Path(item['path'])
    if path.is_symlink() or not path.is_file() or pin(path)!=item: raise ValueError('changed pinned diagnostic/source input')
    return path


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temporary.replace(path)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def validate_prior_documents(documents):
    """Validate the recorded engineering charge; no old model or grade credit."""
    study=documents['study'];cost=documents['cost'];prepared=documents['prepared']
    resolved=documents['resolved'];terminal=documents['terminal'];supervisor=documents['supervisor_request']
    invalid=documents['invalid_receipt'];source=study['source']
    identity=('schema','spec','spec_sha256','source','execution_source','case_definitions',
              'runtime_contract','lane_runtime','family_paid_budget_seconds')
    if (study.get('schema')!='pg_grid100_endpoint_supervision_v1' or study.get('status')!='INVALID'
            or any(study.get(key)!=prepared.get(key) or study.get(key)!=resolved['packet'].get(key) for key in identity)
            or study.get('execution_source')!=source or study.get('result')!=cost
            or cost.get('status')!='INVALID' or cost.get('paid_wall_seconds')!=PRIOR_PAID
            or cost.get('charged_seconds')!=PRIOR_PAID or cost.get('unmeasured_interrupt_reserved_seconds')!=0.
            or cost.get('paid_cap_seconds')!=ORIGINAL_CAP or study.get('spent_seconds')!=PRIOR_PAID
            or any(cost.get(key) is not False for key in ('qualification_input','convergence_credit','default_adoption','speed_ranking','retry_authorized'))
            or source.get('origin_commit')!=COMMIT or source.get('digest')!=DIGEST or digest(source.get('files'))!=DIGEST
            or documents['source_manifest']!={key:value for key,value in source.items() if key!='snapshot_path'}
            or terminal.get('attempt_status')!='completed' or terminal.get('child_returncode')!=1
            or terminal.get('paid_wall_seconds')!=PRIOR_PAID
            or terminal.get('token')!=supervisor.get('token') or terminal.get('token')!=resolved['worker'].get('token')
            or hashlib.sha256(str(terminal.get('token')).encode()).hexdigest()!=cost.get('token_sha256')
            or supervisor.get('source')!=source
            or not math.isclose(supervisor['deadline_monotonic']-supervisor['started_monotonic'],ORIGINAL_CAP,rel_tol=0,abs_tol=1e-6)
            or invalid.get('status')!='INVALID' or invalid.get('diagnostic_optimizer_updates')!=0
            or invalid.get('qualification') is not False or invalid.get('retry_authorized') is not False):
        raise ValueError('original INVALID source/terminal/once-only engineering debit changed')
    old_case=study['case_definitions'].get('grid100_zero_update_output_kernel_endpoint_v1',{})
    expected={key:PRIOR_REFERENCE['artifacts']['original_'+key] for key in ('script','protocol','wrapper','maintained_coordinator')}
    if (old_case.get('external_files')!=expected or study['spec'].get('paid_cap_seconds')!=ORIGINAL_CAP
            or cost.get('terminal')!=PRIOR_REFERENCE['artifacts']['terminal']):
        raise ValueError('original producer/wrapper or full allowance changed')
    return source


def verify_carryover():
    documents={name:read(checked(item)) for name,item in PRIOR_REFERENCE['artifacts'].items()
               if not name.startswith('original_')}
    for name,item in PRIOR_REFERENCE['artifacts'].items():
        if name.startswith('original_'):checked(item)
    return validate_prior_documents(documents)


def declared_spec():
    return dict(id=CASE,paid_cap_seconds=REMAINING_CAP,original_campaign_cap_seconds=ORIGINAL_CAP,prior_engineering_paid_seconds=PRIOR_PAID,engineering_carryover=deepcopy(PRIOR_REFERENCE),export_grace_seconds=0,frames=0,representation_card=pin(PROTOCOL),
                resources=dict(host_memory_mb=2048,cpu_threads=1,memory_fraction=.2,minimum_free_gpu_memory_mib=12288,maximum_gpu_temperature_c=82),
                qualification_input=False,convergence_credit=False,default_adoption=False,speed_ranking=False,physical_attempt_limit=1,retries=0)


def verify(packet):
    case=packet['case_definitions'][CASE]
    for item in case['external_files'].values():checked(item)
    expected_files=dict(script=pin(PRODUCER),protocol=pin(PROTOCOL),wrapper=pin(__file__),maintained_coordinator=pin(MAINTAINED/COORDINATOR))
    protocol=read(PROTOCOL)
    if (packet.get('schema')!='pg_grid100_endpoint_supervision_v2'
            or set(packet['case_definitions'])!={CASE} or case['external_files']!=expected_files
            or sha(PRODUCER)!=PRODUCER_SHA or sha(PROTOCOL)!=PROTOCOL_SHA
            or packet['source']!=packet['execution_source'] or packet['source']!=protocol['source']
            or protocol['source']['origin_commit']!=COMMIT or protocol['source']['digest']!=DIGEST
            or digest(protocol['source']['files'])!=DIGEST or packet['lane_runtime']!={**protocol['runtime'],'physical_gpu':'1'}
            or protocol['claim']['training_updates']!=0 or protocol['claim']['ordinary_qualification'] is not False
            or protocol['claim']['convergence_credit'] is not False or protocol['claim']['default_adoption'] is not False
            or protocol['claim']['speed_ranking'] is not False
            or protocol.get('schema')!='pg_grid100_public_output_kernel_endpoint_v2'
            or any(protocol.get('budget',{}).get(key)!=value for key,value in
                   (('cumulative_paid_cap_seconds',ORIGINAL_CAP),('prior_engineering_paid_seconds',PRIOR_PAID),
                    ('proposed_supervisor_wall_cap_seconds',REMAINING_CAP),('export_grace_seconds',0)))
            or packet['spec']!=declared_spec() or packet['spec_sha256']!=digest(packet['spec'])
            or packet['family_paid_budget_seconds']!={FAMILY:REMAINING_CAP}
            or any(packet.get(key) is not False for key in ('qualification_input','convergence_credit','default_adoption','speed_ranking'))
            or case.get('engineering_carryover')!=PRIOR_REFERENCE or case.get('max_seconds')!=REMAINING_CAP
            or verify_carryover()!=packet['source'] or PRIOR_PAID+REMAINING_CAP!=ORIGINAL_CAP):
        raise ValueError('fixed corrected endpoint/source/inclusive180s/nonqualification contract changed')
    if case.get('protocol_sha256')!=PROTOCOL_SHA or case.get('optimizer_updates')!=0 or case.get('qualification_input') is not False:
        raise ValueError('endpoint declaration changed')
    # Metadata-only producer verification: hashes/JSON/source, no ML imports.
    producer=load(PRODUCER,'_grid100_endpoint_metadata_producer')
    producer.verify_inputs(protocol)
    return protocol


def build_packet():
    """Hash-bound metadata only; no output write, owner, reservation or model."""
    protocol=read(PROTOCOL)
    case=dict(id=CASE,claim='Paired output-kernel contribution at one retained endpoint; no convergence or training claim.',
              protocol_sha256=PROTOCOL_SHA,optimizer_updates=0,qualification_input=False,
              max_seconds=REMAINING_CAP,engineering_carryover=deepcopy(PRIOR_REFERENCE),
              external_files=dict(script=pin(PRODUCER),protocol=pin(PROTOCOL),wrapper=pin(__file__),
                                  maintained_coordinator=pin(MAINTAINED/COORDINATOR)))
    spec=declared_spec()
    packet=dict(schema='pg_grid100_endpoint_supervision_v2',status='PREPARED',spec=spec,spec_sha256=digest(spec),
                source=deepcopy(protocol['source']),execution_source=deepcopy(protocol['source']),case_definitions={CASE:case},
                capacity_preflight=dict(kind='hash_bound_endpoint_protocol',model_capacity_proved=False,learned_quality_proved=False,qualification_input=False),
                runtime_contract=protocol['runtime'],lane_runtime={**protocol['runtime'],'physical_gpu':'1'},
                family_paid_budget_seconds={FAMILY:REMAINING_CAP},qualification_input=False,convergence_credit=False,default_adoption=False,speed_ranking=False)
    verify(packet);return packet


def prepare(output):
    output=Path(output).resolve();sidecar=output.parent/('.'+output.name+'.endpoint-supervision.json')
    if sidecar.exists() or output.exists():raise ValueError('fresh once-only diagnostic output required')
    write(sidecar,build_packet());return sidecar


def readiness(query=None):
    lines=(query or (lambda:subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True)))()
    rows={p[0].strip():(int(p[1]),int(p[2])) for p in (line.split(',') for line in lines.splitlines()) if len(p)==3}
    if '1' not in rows or rows['1'][0]<12288 or rows['1'][1]>82:raise RuntimeError('physical GPU1 unsafe/unavailable')
    return dict(physical_gpu='1',free_memory_mib=rows['1'][0],temperature_c=rows['1'][1])


def charge(paid,terminal):
    if type(paid) not in (int,float) or not math.isfinite(paid) or paid<0:raise ValueError('invalid durable paid cost')
    reserve=0. if terminal is not None and terminal.get('attempt_status')=='completed' else max(0.,REMAINING_CAP-paid)
    return dict(paid_wall_seconds=paid,unmeasured_interrupt_reserved_seconds=reserve,charged_seconds=paid+reserve)


def inclusive_cost(paid,terminal):
    current=charge(paid,terminal)
    return {**current,'paid_cap_seconds':REMAINING_CAP,'original_campaign_cap_seconds':ORIGINAL_CAP,
            'prior_engineering_paid_seconds':PRIOR_PAID,'inclusive_paid_seconds':PRIOR_PAID+current['paid_wall_seconds'],
            'inclusive_charged_seconds':PRIOR_PAID+current['charged_seconds'],
            'overrun_seconds':max(0.,PRIOR_PAID+current['charged_seconds']-ORIGINAL_CAP)}


def verify_retained_result(result,packet):
    if (result.get('status') not in {'COMPLETE_DIAGNOSTIC','INVALID','INCOMPLETE'}
            or any(result.get(key) is not False for key in ('qualification_input','convergence_credit','default_adoption','speed_ranking','retry_authorized'))):
        raise ValueError('retained terminal status/credit cannot change; never retry')
    if 'terminal' in result:
        terminal=read(checked(result['terminal']))
        supervisor=read(checked(result['supervisor_request']));resolved=read(checked(result['resolved']))
        fields=('schema','spec','spec_sha256','source','execution_source','case_definitions','runtime_contract','lane_runtime','family_paid_budget_seconds')
        command=supervisor.get('command',[])
        if (hashlib.sha256(str(terminal.get('token')).encode()).hexdigest()!=result.get('token_sha256')
                or supervisor.get('token')!=terminal.get('token') or resolved['worker'].get('token')!=terminal.get('token')
                or supervisor.get('source')!=packet['execution_source']
                or any(resolved['packet'].get(key)!=packet.get(key) for key in fields)
                or resolved.get('output')!=packet.get('output')
                or Path(result['terminal']['path']).parent.name!=result.get('attempt_key')
                or len(command)!=7 or command[1:5]!=['-u',str(Path(__file__).resolve()),'--child',result['resolved']['path']]
                or command[5]!='--lease-fd' or not str(command[6]).isdigit()
                or not math.isclose(supervisor['deadline_monotonic']-supervisor['started_monotonic'],REMAINING_CAP,rel_tol=0,abs_tol=1e-6)):
            raise ValueError('retained terminal attribution differs; never retry')
        expected=inclusive_cost(terminal['paid_wall_seconds'],terminal)
    else:
        error=read(checked(result['launch_error'])) if 'launch_error' in result else None
        if error is not None and (error.get('token_sha256')!=result.get('token_sha256') or error.get('source_digest')!=DIGEST):
            raise ValueError('retained interruption attribution differs; never retry')
        expected=inclusive_cost(error.get('paid_wall_seconds',0.) if error else 0.,None)
    if any(result.get(key)!=value for key,value in expected.items()):
        raise ValueError('retained cumulative paid/reserve/180s debit differs; never retry')
    if result.get('endpoint_receipt') is not None:
        checked(result['endpoint_receipt'])
        if result['status']!='COMPLETE_DIAGNOSTIC' or result['endpoint_receipt']!=outcome(Path(packet['output']),packet):
            raise ValueError('retained endpoint receipt differs; never retry')
    return result


def child(path,fd):
    resolved=read(path);packet=resolved['packet'];protocol=verify(packet)
    if any(os.environ.get(k)!=v for k,v in ENV.items()):raise ValueError('original physical GPU1/thread/determinism environment required')
    snapshot=Path(packet['execution_source']['snapshot_path'])
    sys.path.insert(0,str(snapshot))
    utility=load(snapshot/OLD_DRIVER,'_grid100_original_lease_guard')
    utility.verify_lease(fd,resolved);utility.guard_imports(packet['source'],execution_root=snapshot)
    import torch
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.2,0)
    # Producer owns all actual restore/sampling/scoring and its exact runtime.
    import runpy
    previous_argv=sys.argv
    sys.argv=[str(PRODUCER),'run','--output',str(Path(resolved['output'])/'endpoint')]
    try:runpy.run_path(str(PRODUCER),run_name='__main__')
    finally:sys.argv=previous_argv
    utility.guard_imports(packet['source'],execution_root=snapshot)
    return 0


def outcome(output,packet):
    path=Path(output)/'endpoint'/'receipt.json';value=read(path)
    if (value.get('status')!='COMPLETE_DIAGNOSTIC' or value.get('protocol_sha256')!=PROTOCOL_SHA
            or value.get('reproducer_sha256')!=PRODUCER_SHA or value.get('source_commit')!=COMMIT or value.get('source_digest')!=DIGEST
            or value.get('diagnostic_optimizer_updates')!=0 or value.get('training_or_default_qualification') is not False
            or value.get('convergence_or_speed_credit') is not False or value.get('physical_gpu')!='1'
            or value.get('runtime')!=packet['runtime_contract']):raise ValueError('no complete exact zero-update diagnostic receipt')
    for row in value['observations']:checked(row['arrays'])
    return pin(path)


def run(output):
    output=Path(output).resolve();sidecar=output.parent/('.'+output.name+'.endpoint-supervision.json')
    packet=read(sidecar);verify(packet);os.environ.update(ENV)
    sys.path.insert(0,str(MAINTAINED))
    from experiments.forge.policy_execution import PolicyCoordinator
    coordinator=PolicyCoordinator(QUEUE,report_root=MAINTAINED/'reports/forge')
    if (output/'study.json').exists():
        saved=read(output/'study.json')
        if saved.get('result') is not None:
            if any(saved.get(k)!=packet[k] for k in ('spec','spec_sha256','source','execution_source','case_definitions','runtime_contract','family_paid_budget_seconds')):
                raise ValueError('retained source/protocol/quota changed; never retry')
            return verify_retained_result(saved['result'],{**packet,'output':str(output)}) # Never retry a retained outcome.
    readiness();key,actual=coordinator.register(packet,output,FAMILY,packet['lane_runtime'])
    if actual!=output:raise ValueError('compatible attempt already registered elsewhere; no fresh retry')
    with coordinator.study_lease(key) as study_lease:
        if study_lease is None:raise RuntimeError('compatible study has a live owner')
        row=dict(id=CASE,timeout_seconds=REMAINING_CAP)
        attempt=coordinator.attempt_key(packet,dict(family=FAMILY,recipe_overrides={}),row)
        with coordinator.admit(attempt,packet,row,'cuda:0') as (admission,lease):
            if admission['status']=='busy':raise RuntimeError('shared GPU1/host admission busy; no launch')
            token=admission['token'];resolved_path=output/'resolved.json';error=None
            if admission['status']=='running' and lease is not None:
                try:
                    readiness();verify(packet)
                    resolved=dict(packet=packet,output=str(output),worker=dict(token=token,lease_path=admission['lease_path']))
                    write(resolved_path,resolved)
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--child',str(resolved_path),'--lease-fd',str(lease.fileno())]
                    coordinator.launch(command,packet,output/'run.log',(study_lease,lease),REMAINING_CAP)
                except BaseException as exc:error=exc
            terminal_path=Path(admission['lease_path']).parent/'supervisor-terminal.json'
            terminal=read(terminal_path) if terminal_path.exists() else None
            if terminal is not None and terminal.get('token')!=token:raise ValueError('foreign terminal')
            paid=terminal.get('paid_wall_seconds',0.) if terminal else getattr(error,'paid_wall_seconds',0.)
            result=dict(status='INCOMPLETE',attempt_key=attempt,token_sha256=hashlib.sha256(token.encode()).hexdigest(),
                        **inclusive_cost(paid,terminal),qualification_input=False,convergence_credit=False,default_adoption=False,speed_ranking=False,retry_authorized=False)
            if terminal is not None:
                result.update(terminal=pin(terminal_path),supervisor_request=pin(terminal_path.with_name('supervisor-request.json')),
                              resolved=pin(resolved_path))
            elif error is not None:
                error_path=output/'launch-error.json'
                write(error_path,dict(token_sha256=result['token_sha256'],source_digest=DIGEST,
                                     type=type(error).__name__,paid_wall_seconds=paid))
                result['launch_error']=pin(error_path)
            if terminal and terminal.get('attempt_status')=='completed':
                result['status']='INVALID'
                if terminal.get('child_returncode')==0:
                    try:result.update(status='COMPLETE_DIAGNOSTIC',endpoint_receipt=outcome(output,packet))
                    except (OSError,ValueError,KeyError):pass
            if admission['status'] in {'running','awaiting_certification'}:coordinator.complete(attempt,result)
            elif admission.get('charged_seconds')!=result['charged_seconds']:raise ValueError('local/central charge differs')
            verify_retained_result(result,{**packet,'output':str(output)})
            packet.update(status=result['status'],result=result,spent_seconds=result['charged_seconds'],
                          prior_engineering_paid_seconds=PRIOR_PAID,inclusive_spent_seconds=result['inclusive_charged_seconds'])
            write(output/'study.json',packet);write(output/'cost.json',result)
            return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--child',type=Path);p.add_argument('--lease-fd',type=int);a=p.parse_args(argv)
    if a.child:
        if a.lease_fd is None:raise ValueError('child requires admitted descriptor')
        return child(a.child,a.lease_fd)
    if a.output is None:raise ValueError('explicit fresh output required')
    if a.prepare_only:
        path=prepare(a.output);print(json.dumps(dict(status='PREPARED',packet_sha256=sha(path),qualification_input=False)));return 0
    result=run(a.output)
    print(json.dumps({k:result[k] for k in ('status','paid_wall_seconds','unmeasured_interrupt_reserved_seconds','charged_seconds',
                                          'prior_engineering_paid_seconds','inclusive_charged_seconds','overrun_seconds','qualification_input')}))
    return 0 if result['status']=='COMPLETE_DIAGNOSTIC' else 2


if __name__=='__main__':raise SystemExit(main())
