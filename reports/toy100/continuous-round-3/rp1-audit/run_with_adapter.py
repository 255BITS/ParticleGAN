"""Observation-only wrapper. Usage: python run_with_adapter.py DRIVER.py DRIVER_ARGS..."""
from pathlib import Path
import sys,hashlib,json,atexit
wrapper=Path(__file__).resolve();driver=Path(sys.argv.pop(1)).resolve();sys.argv[0]=str(driver)
sys.path.insert(0,str(driver.parent));sys.path.insert(1,str(wrapper.parent))
source=driver.read_text();lines=source.splitlines(True)
locations=[i for i,line in enumerate(lines) if line.startswith('import mechanism')]
if len(locations)!=1:raise RuntimeError('Expected one explicit mechanism import; refusing unknown driver')
lines.insert(locations[0]+1,'import observation_adapter\nobservation_adapter.install()\n')
executed=''.join(lines)
def provenance():
 if '--output' in sys.argv:
  out=Path(sys.argv[sys.argv.index('--output')+1])
  if out.is_dir():
   info={'driver':str(driver),'driver_sha256':hashlib.sha256(source.encode()).hexdigest(),
         'executed_wrapper_source_sha256':hashlib.sha256(executed.encode()).hexdigest(),
         'wrapper_sha256':hashlib.sha256(wrapper.read_bytes()).hexdigest(),
         'adapter_sha256':hashlib.sha256(wrapper.with_name('observation_adapter.py').read_bytes()).hexdigest(),
         'change':'One import/install after unchanged learner imports; no training/quality/budget code changed'}
   (out/'observation-wrapper-provenance.json').write_text(json.dumps(info,indent=2)+'\n')
atexit.register(provenance)
exec(compile(executed,str(driver),'exec'),{'__name__':'__main__','__file__':str(driver)})
