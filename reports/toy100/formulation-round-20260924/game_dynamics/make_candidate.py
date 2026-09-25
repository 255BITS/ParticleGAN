import argparse, datetime, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('name'); p.add_argument('--description', required=True); a=p.parse_args()
root=Path(__file__).resolve().parent; repo=root.parents[2]; c=root/'candidates'/a.name
probe=(repo/'reports/toy100/cpu-recipe-gpu-port/probe.py').read_text()
probe=probe.replace('proof = {', 'from mechanism import Correction\ncorrection = Correction()\n\nproof = {',1)
probe=probe.replace('result = original_step(opt, *args, **kwargs)', 'result = correction.step(opt, original_step, *args, **kwargs)\n    if item["calls"] % 250 == 0:\n        print(json.dumps(dict(event="progress", optimizer=list(proof["optimizers"]).index(str(id(opt))), updates=item["calls"])), flush=True)',1)
probe=probe.replace('record.update(seconds=', 'proof["game_correction"] = correction.receipt()\nrecord.update(seconds=',1)
if 'def context(' in (c/'mechanism.py').read_text():
    probe=probe.replace('with audit, patch.object', 'with correction.context(), audit, patch.object', 1)
(c/'probe.py').write_text(probe)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
config=repo/'configs/toy100/constraints_simple_regularization.json'
declaration=dict(candidate=a.name, declared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), description=a.description,
 source='prepared/repos/cuda', prepared_manifest_sha256=sha(root/'prepared/prepared-sources.json'),
 config=str(config), config_sha256=sha(config), file_hashes={str(path.relative_to(c)):sha(path) for path in sorted(c.rglob('*')) if path.is_file() and (path.suffix == '.py' or path.name == 'transforms.json')},
 initialization='retained audited CPU fixtures; native CUDA training and random draws',
 invariants=['architectures','data','fixed seeds','loss and critic regularizer','all auxiliary AE/token losses','noise schedules','learning-rate schedules','evaluation','thresholds','outer step budgets'],
 compute=dict(extra_forward_evaluations_per_outer_step=0,extra_backward_evaluations_per_outer_step=0,extra_optimizer_updates=0,base_updates='one D and one G per host outer step; no added critic steps',comparison='equal outer steps and equal model/gradient evaluations; added tensor arithmetic/memory, wall time measured'),
 checkpoint='memory stored in optimizer state; qualification continuation required after all 22')
(c/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
print(json.dumps(dict(candidate=a.name,hashes=declaration['file_hashes'])))
