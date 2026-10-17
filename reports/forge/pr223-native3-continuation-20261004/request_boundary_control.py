"""Private maintained-executor request simulation, stopped before learning.

The copied preflight calls this only with CUDA hidden. The real coordinator
register/admit/launch methods use a private in-memory Queue seam. A fake ledger
MODULE loader prevents any access to the canonical ledger. The actual durable
request, inherited descriptors and copied helper validators run in a fresh
metadata-only subprocess; observer/model/scorer imports are forbidden.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

DIRECTORY='reports/forge/pr223-native3-continuation-20261004'
SELF=DIRECTORY+'/request_boundary_control.py'
SCHEMA='pg_pr223_native3_request_boundary_control_v1'
OUTCOMES={'maintained_current_request':'PASS_STOPPED_BEFORE_OBSERVER',
          'declared_attempt':'REFUSED_DECLARED_ATTEMPT',
          'historical_credit':'REFUSED_PRIOR_CREDIT',
          'wrong_token':'REFUSED_CURRENT_TOKEN','case_drift':'REFUSED_CASE_DRIFT',
          'foreign_lease':'REFUSED_FOREIGN_LEASE','expired_lease':'REFUSED_EXPIRED_LEASE'}


def guard_private_path(filename,source_root):
    """Reject foreign workspace/link targets before their metadata is probed."""
    def allowed(path):
        return not (path.is_relative_to('/home/martyn/dev/ParticleGAN/artifacts') or
            (path.is_relative_to('/ml2/hypergan') and not path.is_relative_to(source_root)))
    path=Path(os.path.abspath(os.fsdecode(filename)))
    for _ in range(40):
        if not allowed(path):raise AssertionError('actual workspace/ledger/queue/raw access forbidden')
        current=Path(path.anchor);parts=path.parts[1:]
        for index,part in enumerate(parts):
            current=current/part
            try:mode=os.lstat(current).st_mode
            except (FileNotFoundError,NotADirectoryError):return
            if stat.S_ISLNK(mode):
                target=Path(os.readlink(current))
                if not target.is_absolute():target=current.parent/target
                path=Path(os.path.abspath(os.fspath(target.joinpath(*parts[index+1:]))))
                if not allowed(path):raise AssertionError('foreign source alias forbidden before metadata')
                break
        else:return
    raise AssertionError('cyclic source aliases refused')


def binding(packet,helper):
    source=packet['execution_source'];root=Path(source['snapshot_path'])
    names=(helper.SELF,DIRECTORY+'/native3_contract.py',SELF)
    for name in names:
        if helper.sha(root/name)!=source['files'].get(name):
            raise ValueError('copied request-boundary source changed')
    return dict(source_digest=source['digest'],helper_sha256=source['files'][helper.SELF],
                contract_sha256=source['files'][names[1]],control_sha256=source['files'][SELF])


def validate_proof(proof,packet,helper):
    expected=dict(schema=SCHEMA,status='PASS_SYNTHETIC_MAINTAINED_REQUEST_BOUNDARY',
        binding=binding(packet,helper),outcomes=OUTCOMES,synthetic_only=True,
        actual_queue_calls=0,actual_admissions=0,models=0,sampler_calls=0,scorer_calls=0,
        simulated_coordinator_requests=1,simulated_registrations=1,simulated_admissions=1,
        observer_calls=0,numerical_credit=False,old_arrays_read=False)
    if proof!=expected:raise ValueError('copied current-request boundary proof missing, foreign or incomplete')
    return proof


def run(packet,helper):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('request-boundary control must hide CUDA')
    pins=binding(packet,helper)
    policy=sys.modules[helper.PolicyCoordinator.__module__]
    real_popen=subprocess.Popen
    real_ledger=helper.ledger_module()
    captured={}
    with tempfile.TemporaryDirectory(prefix='pr223-private-request-boundary-') as directory:
        private=Path(directory);output=private/'out';queue=private/'metadata-ownership'
        value=deepcopy(packet)
        value.update(prepared_output=str(output),queue_root=str(queue),
                     copied_preflight_receipt_path=str(helper.native3_module().copied_preflight_path(output)))
        class PrivateQueue:
            def __init__(self,root,**kwargs):
                if Path(root).resolve()!=queue:raise AssertionError('only private ownership state may be constructed')
                self.root=queue;self.data={'jobs':{},'policy_attempts':{},'policy_studies':{},'policy_outputs':{}}
            @contextmanager
            def state(self):yield self.data
            def collect(self,**kwargs):return None
        class PrivateLedger:
            path=packet['metadata_ledger_path']
            def snapshot(self):return deepcopy(packet['metadata_history']['predecessor_closed_state'])
            @contextmanager
            def phase(self,name):yield self
            @contextmanager
            def pause(self):yield self
        ledger=PrivateLedger()
        fake_module=SimpleNamespace(**vars(real_ledger))
        fake_module.SharedMetadataLedger=lambda path:ledger
        def runtime():
            return dict(python='synthetic-metadata',torch='not-imported',numpy='not-imported',
                device='cuda:0',physical_gpu=1,torch_threads=1,memory_fraction=.2,
                deterministic=True,tf32=False,cuda_device_model='synthetic-no-hardware-query')
        def no_science(*args,**kwargs):raise AssertionError('observer/science/attestation execution is forbidden')
        class StopBeforeObserver(RuntimeError):pass
        class PrivateSupervisor:
            def __init__(self,argv,**kwargs):
                durable=Path(argv[-1]);request=helper.read_json(durable)
                child_request=helper.read_json(output/'native/grid100/request.json')
                captured['request']=child_request
                # A fresh child imports only the actual copied metadata helper.
                # It consumes the actual coordinator-written durable record.
                code="""import importlib.util,json,os,stat,sys
from pathlib import Path
root=Path(sys.argv[1]);req=Path(sys.argv[2]);proof=Path(sys.argv[3])
_PRIVATE_PATH_GUARD_
def deny(event,values):
    if event=='import' and values[0].split('.')[0] in {'torch','numpy','particlegan','benchmarks','lib'}:
        raise AssertionError('model/scorer import forbidden')
    if event=='open' and isinstance(values[0],(str,bytes)):
        guard_private_path(values[0],root)
sys.addaudithook(deny)
sys.path.insert(0,str(root))
s=importlib.util.spec_from_file_location('_private_current_request',root/'reports/forge/pr223-original-full-retest-20261004/run_retest.py')
h=importlib.util.module_from_spec(s);sys.modules[s.name]=h;s.loader.exec_module(h)
r=h.read_json(req);p,row,target=h.validate_request(r)
h.verify_lease(r,r['worker']['lease_fd']);h.guard_imports(p['execution_source'])
# The actual child entry also validates its declared placement before observer
# construction. Only that metadata lookup is simulated; real CUDA stays hidden.
from types import SimpleNamespace
env=SimpleNamespace(get=lambda name,default=None:h.protocol.ENVIRONMENT.get(name,os.environ.get(name,default)))
h.os=SimpleNamespace(**{**vars(os),'environ':env})
class StopBeforeObserver(RuntimeError):pass
def observer_boundary():
    h.verify_lease(r,r['worker']['lease_fd']);h.guard_imports(p['execution_source'])
    raise StopBeforeObserver('all actual nonlearning child checks passed')
h.observer=observer_boundary
try:h.child(req,r['worker']['lease_fd'])
except StopBeforeObserver:pass
else:raise AssertionError('actual child must stop before observer import')
try:h.verify_lease(r,r['worker']['lease_fd'],now=r['worker']['deadline_monotonic'])
except TimeoutError:pass
else:raise AssertionError('exact deadline must refuse while descriptors are valid')
from copy import deepcopy
foreign=deepcopy(r);foreign['worker']['lease_path']=str(proof.parent/'foreign.lock')
try:h.verify_lease(foreign,foreign['worker']['lease_fd'])
except ValueError:pass
else:raise AssertionError('foreign lease must refuse while descriptors are valid')
assert not any(n.split('.')[0] in {'torch','numpy','particlegan','benchmarks','lib'} for n in sys.modules)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
proof.write_text(json.dumps({'status':'PASS_STOPPED_BEFORE_OBSERVER','source_digest':p['execution_source']['digest'],'case_id':row['id'],'scientific_calls':0,'expired_lease':'REFUSED_EXPIRED_LEASE','foreign_lease':'REFUSED_FOREIGN_LEASE'}))
"""
                # root may be /ml2/hypergan in an actual copied preflight; permit
                # ONLY immutable copied source and this private tree in child.
                source_root=Path(value['execution_source']['snapshot_path'])
                code=code.replace('_PRIVATE_PATH_GUARD_',inspect.getsource(guard_private_path))
                proof=private/'child-boundary.json'
                process=real_popen([sys.executable,'-B','-c',code,str(source_root),
                    str(output/'native/grid100/request.json'),str(proof)],
                    cwd=private,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(source_root),
                        'PYTHONDONTWRITEBYTECODE':'1'},pass_fds=tuple(request['lease_fds']),
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                stdout,stderr=process.communicate(timeout=15)
                if process.returncode!=0:raise AssertionError('private actual child validation failed: '+stderr.decode())
                captured['child']=helper.read_json(proof)
                helper.atomic_json(durable.parent/'supervisor-terminal.json',dict(token=request['token'],
                    attempt_status='completed',child_returncode=2,paid_wall_seconds=1.,synthetic_only=True))
                self.pid=process.pid;self.first=True
            def wait(self,timeout=None):
                if self.first:self.first=False;raise StopBeforeObserver('private metadata child stopped before observation/science')
                return 0
            def kill(self):pass
        with ExitStack() as stack:
            stack.enter_context(patch.object(helper,'ledger_module',lambda:fake_module))
            stack.enter_context(patch.object(helper.native3_module(),'require_existing_ledger',lambda:Path(ledger.path)))
            stack.enter_context(patch.object(helper,'prepare',lambda *args,**kwargs:deepcopy(value)))
            # The exact copied preflight is proved by its enclosing caller; this
            # seam isolates subsequent maintained registration/request transport.
            stack.enter_context(patch.object(helper,'require_copied_preflight',lambda p:None))
            stack.enter_context(patch.object(helper,'runtime_metadata',runtime))
            stack.enter_context(patch.object(helper.legacy,'gpu_readiness',lambda:{'ready':True}))
            stack.enter_context(patch.object(policy,'Queue',PrivateQueue))
            stack.enter_context(patch.object(policy,'host_capacity',lambda:{'cpu_threads':4,'memory_mb':8192,'available_memory_mb':8192}))
            stack.enter_context(patch.object(policy,'physical_device',lambda device:'1'))
            stack.enter_context(patch.object(policy.subprocess,'Popen',PrivateSupervisor))
            for name in ('science','observer','verify_attestation'):
                stack.enter_context(patch.object(helper,name,no_science))
            # Placement is simulated only for the parent metadata guard. Child
            # and real environment keep CUDA hidden; no hardware query occurs.
            real_environ=os.environ
            environ=SimpleNamespace(get=lambda name,default=None:'1' if name=='CUDA_VISIBLE_DEVICES' else real_environ.get(name,default),
                copy=lambda:real_environ.copy())
            private_os=SimpleNamespace(**{**vars(helper.os),'environ':environ})
            stack.enter_context(patch.object(helper,'os',private_os))
            result=helper.run(output,root=helper.ROOT,queue_root=queue,max_new_attempts=1,
                              native3_anchor=value['metadata_history']['closed_anchor'])
        if result['rows'][0]['status']!='INVALID' or result['completed']!=0 or result['required']!=3:
            raise AssertionError('private metadata stop must confer no numerical completion')
        if captured.get('child',{}).get('status')!=OUTCOMES['maintained_current_request']:
            raise AssertionError('missing actual maintained request/descriptor boundary')
        request=captured['request'];scope=helper.native3_module()
        bad=deepcopy(request);bad['packet']['status']='DECLARED'
        try:helper.validate_request(bad)
        except ValueError:pass
        else:raise AssertionError('declared admission was not refused')
        for scenario in ('historical_credit','wrong_token','case_drift'):
            bad=deepcopy(request)
            if scenario=='historical_credit':
                bad['row']['original_gate']='PASS';bad['packet']['rows'][0]=deepcopy(bad['row'])
            elif scenario=='wrong_token':bad['worker']['token']='0'*32
            else:bad['packet']['case_definitions'][scope.IDS[0]]['original_host']['steps']=600
            try:helper.validate_request(bad)
            except ValueError:pass
            else:raise AssertionError('current request negative accepted: '+scenario)
        for scenario in ('foreign_lease','expired_lease'):
            if captured['child'].get(scenario)!=OUTCOMES[scenario]:
                raise AssertionError('missing open-descriptor lease negative: '+scenario)
    proof=dict(schema=SCHEMA,status='PASS_SYNTHETIC_MAINTAINED_REQUEST_BOUNDARY',binding=pins,
        outcomes=OUTCOMES,synthetic_only=True,actual_queue_calls=0,actual_admissions=0,
        simulated_coordinator_requests=1,simulated_registrations=1,simulated_admissions=1,
        models=0,sampler_calls=0,scorer_calls=0,observer_calls=0,numerical_credit=False,old_arrays_read=False)
    return validate_proof(proof,packet,helper)
