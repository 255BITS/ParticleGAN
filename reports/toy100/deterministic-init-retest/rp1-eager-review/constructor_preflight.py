"""Run the reviewed diagnostic constructor check, retaining both setup receipts."""
from pathlib import Path
import json,sys
E=Path(__file__).resolve().parent.parent;B=E/'rp1-eager-screen';O=E/'rp1-eager-review'
sys.dont_write_bytecode=True;sys.path.insert(0,str(B))
import preflight
sys.argv=['preflight.py','--package-root',str(E/'port-source/api-rp1-cuda-eager/package'),'--declaration',str(B/'candidate-declaration.json'),'--cpu-init','--output',str(O/'cpu-preflight.json')]
preflight.main()
import reviewed_setup
assert len(reviewed_setup.RECEIPTS)==2
assert all(r['counts']=={'G':9,'D':8} and r['completed_steps']==0 and r['non_optimizer_and_rng_before']==r['non_optimizer_and_rng_after'] and r['only_declared_zero_optimizer_state_created'] for r in reviewed_setup.RECEIPTS)
(O/'setup-receipts.json').write_text(json.dumps(reviewed_setup.RECEIPTS,indent=2)+'\n')
