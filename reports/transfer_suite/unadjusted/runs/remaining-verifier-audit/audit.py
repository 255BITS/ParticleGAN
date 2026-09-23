"""Mutation tests against the real archived replay verifier; no GAN training."""
from contextlib import redirect_stdout
from copy import deepcopy
import gzip,hashlib,io,json,runpy,shutil,sys,tempfile
from pathlib import Path
REPO=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path('/ml2/hypergan/ParticleGAN-pr36-verification')
ROOT=REPO/'reports/transfer_suite/unadjusted/runs/remaining-replays'
sys.path.insert(0,str(REPO))
verify=(ROOT/'verify.py').read_bytes()
original_protocol=json.loads((ROOT/'control/protocol.json').read_text())
reference=(REPO/original_protocol['reference']).read_bytes()
original_checks=json.loads((ROOT/'control/checks.json').read_text())
original_actual=json.loads(gzip.decompress((ROOT/'control/episode.json.gz').read_bytes()))
results=[]
for name,reject in [('valid',False),('empty_checks',True),('missing_result_check',True),
                    ('truthy_nonboolean',True),('changed_metric',True),('timing_only',False),
                    ('empty_source_manifest',True),('bad_reference_hash',True),('no_replays',True)]:
 with tempfile.TemporaryDirectory(prefix='pr38-replay-mutant-') as temp:
  repo=Path(temp)/'repo';root=repo/'reports/transfer_suite/unadjusted/runs/remaining-replays'
  root.mkdir(parents=True);(root/'verify.py').write_bytes(verify)
  if name!='no_replays':
   shutil.copytree(ROOT/'control',root/'control')
   ref=repo/original_protocol['reference'];ref.parent.mkdir(parents=True,exist_ok=True);ref.write_bytes(reference)
   protocol=deepcopy(original_protocol);checks=deepcopy(original_checks);actual=deepcopy(original_actual)
   if name=='empty_checks':checks['checks']={}
   elif name=='missing_result_check':del checks['checks']['result']
   elif name=='truthy_nonboolean':checks['checks']['result']=1
   elif name=='changed_metric':actual['result']['observations'][-1]['sw1_normalized']+=.01
   elif name=='timing_only':
    actual['result']['seconds']+=10
    actual['result']['observations'][-1]['seconds']+=10
    actual['result']['convergence']['confirmed_seconds']+=10
   elif name=='empty_source_manifest':
    protocol['source_sha256']={};actual['source_sha256']={}
   elif name=='bad_reference_hash':protocol['reference_sha256']='0'*64
   raw=(json.dumps(actual,sort_keys=True,allow_nan=False)+'\n').encode()
   checks['uncompressed_sha256']=hashlib.sha256(raw).hexdigest()
   (root/'control/episode.json.gz').write_bytes(gzip.compress(raw,mtime=0))
   (root/'control/checks.json').write_text(json.dumps(checks))
   (root/'control/protocol.json').write_text(json.dumps(protocol))
  error=None
  try:
   with redirect_stdout(io.StringIO()):runpy.run_path(str(root/'verify.py'),run_name='__main__')
  except Exception as exc:error=type(exc).__name__+': '+str(exc)
  rejected=error is not None
  results.append(dict(case=name,expected_reject=reject,rejected=rejected,passed=rejected==reject,error=error))
print(json.dumps(dict(verifier_sha256=hashlib.sha256(verify).hexdigest(),results=results),indent=2))
assert all(r['passed'] for r in results),'Mutation audit exposed a verifier gap'
