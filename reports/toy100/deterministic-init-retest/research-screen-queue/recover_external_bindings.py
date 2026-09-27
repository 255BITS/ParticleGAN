#!/usr/bin/env python3
"""Recover exact old external config/runtime bindings from retained launch lines."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
OUTPUT=HERE/'external-binding-recovery'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def pin(path):return dict(path=str(path),sha256=sha(path))
def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    OUTPUT.mkdir(exist_ok=True);(OUTPUT/'configs').mkdir(exist_ok=True)
    template=read(HERE.parent/'research-mode-hold-preparation/source-plan.json')
    rows=[]
    for definition in read(HERE/'queue.json')['rows']:
        if definition['status']!='PENDING_ADAPTER_OR_BINDING_REVIEW' or 'config.json' not in definition.get('missing_required_files',[]):continue
        directory=Path(definition['source_directory']);declaration=directory/'declaration.json'
        old=read(declaration) if declaration.exists() else {}
        receipts=[]
        for parent in list(directory.parents)[:3]:
            for path in parent.glob('*command*.jsonl'):
                for number,line in enumerate(path.read_text().splitlines(),1):
                    if not line.strip():continue
                    record=json.loads(line)
                    if record.get('candidate')!=definition['candidate']:continue
                    command=record.get('command',[])
                    if not all(k in command for k in ('--config','--repo')):continue
                    def arg(key):return Path(command[command.index(key)+1])
                    config=arg('--config');runtime=arg('--repo')
                    if not config.exists() or not runtime.exists():continue
                    probes=[Path(x) for x in command if x.endswith('/probe.py')]
                    if len(probes)!=1 or not probes[0].exists() or sha(probes[0])!=definition['probe_sha256']:continue
                    if old.get('config_sha256') and sha(config)!=old['config_sha256']:raise ValueError('launch/config declaration mismatch')
                    if record.get('declaration_sha256') and sha(declaration)!=record['declaration_sha256']:raise ValueError('launch/declaration mismatch')
                    differences=[rel for rel,want in template['historical_runtime_files'].items() if not (runtime/rel).exists() or sha(runtime/rel)!=want]
                    config_dest=OUTPUT/'configs'/(sha(config)+'.json')
                    if config_dest.exists():assert config_dest.read_bytes()==config.read_bytes()
                    else:config_dest.write_bytes(config.read_bytes())
                    receipts.append(dict(receipt=pin(path),line=number,raw_line_sha256=hashlib.sha256(line.encode()).hexdigest(),command=command,config=pin(config),retained_config=str(config_dest.relative_to(OUTPUT)),declared_config_sha256=old.get('config_sha256'),declaration=pin(declaration) if declaration.exists() else None,probe=pin(probes[0]),runtime=str(runtime),canonical_runtime_files_verified=len(template['historical_runtime_files'])-len(differences),runtime_differences=differences,old_tensor_fixture_used='--initial-state' in command,new_tensor_fixture_policy='Never reuse old weights; new initializer and original constructor cursor required'))
        configs={x['config']['sha256'] for x in receipts}
        rows.append(dict(queue_row=definition['id'],candidate=definition['candidate'],status='EXACT_CONFIG_AND_RUNTIME_RECOVERED_REQUIRES_PROBE_ADAPTER' if len(configs)==1 and all(not x['runtime_differences'] for x in receipts) else 'PENDING_EXACT_BINDING_REVIEW',receipts=receipts,active_local_sources=definition.get('active_local_sources'),configuration=read(OUTPUT/receipts[0]['retained_config']) if len(configs)==1 else None))
    dump(OUTPUT/'index.json',dict(schema=1,source_queue=pin(HERE/'queue.json'),canonical_source_plan=pin(HERE.parent/'research-mode-hold-preparation/source-plan.json'),rows=rows,quality_inherited=False,training=False))
    report=['# External configuration recovery','','Original launch records identify these configurations; no nearest-name substitution or old tensor loading is permitted. All source files remain unchanged. A recovered config does not clear its different probe interface for execution.','','| Candidate | Binding status | Matching launch receipts |','|---|---|---|']
    for row in rows:report.append('| '+row['candidate']+' | '+row['status']+' | '+str(len(row['receipts']))+' |')
    report+=['','The JSON preserves each original launch line hash, config/declaration/probe hash, and exact comparison against all200 canonical runtime files. Historical launch arguments using old CPU initialization fixtures are recorded for provenance only. The new retest must initialize every tensor afresh.']
    (OUTPUT/'README.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(rows=len(rows),recovered=sum(x['status'].startswith('EXACT_') for x in rows),configs=len(list((OUTPUT/'configs').glob('*.json'))))))

if __name__=='__main__':main()
