from pathlib import Path
import json,os,subprocess,time,sys,xml.etree.ElementTree as ET
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent
name=sys.argv[1]
cmd=[sys.executable,'-m','pytest','-q','tests/test_training.py','tests/test_recipe_defaults.py','tests/test_ka2.py','tests/test_serial_backward.py','tests/test_precision_game.py','--junitxml='+str(O/(name+'.xml'))]
t=time.monotonic()
with (O/(name+'.log')).open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,cwd=R)
x=ET.parse(O/(name+'.xml')).getroot();m={k:sum(int(s.get(k,0)) for s in x.iter('testsuite')) for k in ['tests','failures','errors','skipped']}
row=dict(candidate='regression',gate=name,status='PASS' if p.returncode==0 else 'FAIL',seconds=time.monotonic()-t,metrics=m,artifact=str(O/(name+'.log')))
with (R.parent/'tests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row),flush=True)
