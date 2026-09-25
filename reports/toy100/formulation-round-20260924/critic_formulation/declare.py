"""Snapshot candidate code/config/declaration before any candidate execution."""
import argparse, datetime, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('--overrides',default='{}');p.add_argument('--mechanism',required=True);p.add_argument('--install',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parent;checkout=root.parents[2]
candidate=root/'candidates'/a.candidate;candidate.mkdir(parents=True,exist_ok=False)
config=json.loads((checkout/'configs/toy100/constraints_simple_regularization.json').read_text());overrides=json.loads(a.overrides);config.update(overrides)
(candidate/'config.json').write_text(json.dumps(config,indent=2)+'\n')
probe=(checkout/'reports/toy100/cpu-recipe-gpu-port/probe.py').read_text()
if a.install:
 install=a.install.read_text();assert probe.count('config = json.loads(a.config.read_text())')==1
 probe=probe.replace('config = json.loads(a.config.read_text())',install+'\n\nconfig = json.loads(a.config.read_text())')
(candidate/'probe.py').write_text(probe)
h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
d=dict(candidate=a.candidate,declared_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),mechanism=a.mechanism,changed_fields=overrides,preserved='Adam, architecture, frozen data/seeds/evaluation/budgets, original decay/noise schedules, auxiliary AE/token losses, native CUDA RNG, retained CPU initialization',compute='One D and one G/prior Adam update per frozen step; same two sample-point critic input-gradient forwards/backwards and two optimization backwards as b_cap. No added model forwards, backwards, updates, or random draws. Equal-step and equal nominal network compute; scalar loss arithmetic may differ.',probe_sha256=h(candidate/'probe.py'),config_sha256=h(candidate/'config.json'),prepared_sources_sha256=h(root/'prepared/prepared-sources.json'))
(candidate/'declaration.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'candidate':a.candidate,'probe_sha256':d['probe_sha256'],'config_sha256':d['config_sha256']}))
