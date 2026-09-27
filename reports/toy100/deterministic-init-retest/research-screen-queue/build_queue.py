#!/usr/bin/env python3
"""Inventory exact research host groups. Standard library only; never trains."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import hashlib
import json
import re
import zipfile

HERE=Path(__file__).resolve().parent
HARNESS=HERE.parent/'research-mode-hold-preparation'
INVENTORY=HERE.parent.parent/'continuous-api-search/fixed-init-retest-inventory'
READ_ONLY_RUNTIME_KEYS=('repo','source','runtime','cuda','cuda_repo','cudarepo')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path):return json.loads(path.read_text())
def pin(path):return dict(path=str(path),sha256=sha(path))
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')

def source_closure(directory,probe_fallback=None):
    """Retain full local Python/config files, including imports absent from old pins."""
    files={p.name:p for p in directory.glob('*') if p.is_file() and p.suffix in ('.py','.json')}
    if 'probe.py' not in files and probe_fallback:files['probe.py']=probe_fallback
    required=('config.json','probe.py','mechanism.py','latent.py','response.py','checkpoint.py')
    missing=[name for name in required if name not in files]
    return files,missing

def active_local_sources(files):
    """Hash the actual local import closure, excluding result/docs/helper copies."""
    active={'config.json'} if 'config.json' in files else set()
    pending=['probe.py']
    while pending:
        name=pending.pop()
        if name in active or name not in files:continue
        active.add(name)
        for node in ast.walk(ast.parse(files[name].read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):modules=[node.module or '']
            for module in modules:
                local=module.split('.')[0]+'.py'
                if local in files and local not in active:pending.append(local)
    return {name:sha(files[name]) for name in sorted(active)}

def init_binding_concerns(files):
    """Conservative routing only. No claim of whole-mechanism correctness."""
    concerns=[];imports=[]
    receipt=HERE/'binding-exceptions.json'
    exceptions=read(receipt) if receipt.exists() else {}
    for name,path in files.items():
        if name not in ('mechanism.py','latent.py','response.py') and name not in ('stationary_policy.py','control.py'):
            continue
        source=path.read_text();tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                imports.append(dict(file=name,line=node.lineno,statement=ast.get_source_segment(source,node)))
            if isinstance(node,ast.Name) and node.id in ('exec','eval','setattr','__import__'):
                enclosing=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.lineno<=node.lineno<=n.end_lineno]
                function=min(enclosing,key=lambda x:x.end_lineno-x.lineno) if enclosing else None
                body=ast.get_source_segment(source,function) if function else source
                identity=hashlib.sha256(body.encode()).hexdigest()
                reviewed=identity in exceptions.get('function_sha256' if function else 'module_sha256',[])
                if not reviewed:concerns.append(dict(file=name,line=node.lineno,reason='Dynamic binding needs manual source review'))
            if isinstance(node,ast.Attribute) and node.attr in ('train_mode_hold','SimpleMLPGenerator','SimpleMLPDiscriminator','make_prior','_initialize','_initialize_prior','reset_parameters'):
                concerns.append(dict(file=name,line=node.lineno,reason='Constructor/host binding access needs review'))
        identifiers={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        identifiers.update(a.name for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names)
        if name!='response.py' and identifiers.intersection({'ParticlePrior','init_registry','initialize_'}):
            concerns.append(dict(file=name,reason='Prior/initializer class access needs manual source review'))
    return concerns,imports

def find_runtimes(directory,entry):
    """Read nearby declared metadata/launch receipts, never infer from scores."""
    receipts=set()
    for folder in [directory,*list(directory.parents)[:3]]:
        if folder.name in ('toy100','reports','gan-attempts','hypergan'):break
        receipts.update(folder.glob('*runtime*.json'))
        receipts.update(folder.glob('*manifest*.json'))
        receipts.update(folder.glob('*sources*.json'))
        receipts.update(folder.glob('*command*.json'))
    if (directory/'declaration.json').exists():receipts.add(directory/'declaration.json')
    for artifact in entry.get('historical_artifacts',[]):
        if not artifact.get('artifact'):continue
        path=Path(artifact['artifact'])
        receipts.update(path.parent.parent.glob('*command*.json'))
    found=[]
    def visit(value,pointer,receipt):
        if isinstance(value,dict):
            command=value.get('command')
            if isinstance(command,list) and '--repo' in command:
                i=command.index('--repo')
                if i+1<len(command):add(command[i+1],pointer+'/command/--repo',receipt)
            for k,v in value.items():
                if k.lower() in READ_ONLY_RUNTIME_KEYS and isinstance(v,str):add(v,pointer+'/'+k,receipt)
                if isinstance(v,(dict,list)):visit(v,pointer+'/'+k,receipt)
        elif isinstance(value,list):
            for i,v in enumerate(value):
                if isinstance(v,(dict,list)):visit(v,pointer+'/'+str(i),receipt)
    def add(value,pointer,receipt):
        path=Path(value)
        if path.is_absolute() and (path/'benchmarks/locked_shared/mode_hold.py').is_file():
            found.append(dict(runtime=str(path),receipt=pin(receipt),json_pointer=pointer))
    for receipt in sorted(receipts):
        try:visit(read(receipt),'',receipt)
        except (json.JSONDecodeError,UnicodeDecodeError):pass
    unique={digest(v):v for v in found}
    return list(unique.values())

def main():
    inventory=read(INVENTORY/'inventory.json');template=read(HARNESS/'source-plan.json')
    records=[];aliases=[]
    for entry in inventory['research_candidate_entries']:
        records.append(dict(id='research:'+entry['id'],candidate=entry['candidate'],entry=entry,sources=entry['source_bindings'],priority=1 if entry['cohort']=='overnight-20260925' else 3))
    for entry in inventory['research_priority_entries']:
        name=entry['candidate'];links=entry.get('research_entries',[])
        late=next((x for x in inventory['late_research_recovery'] if x['candidate']==name),None)
        if late and not links:
            aliases.append(dict(id='priority:'+entry.get('id',name),candidate=name,linked_rows=['late-research:'+name],meaning='Exact retained late-result/source identity; no duplicated launch'))
            continue
        if links:
            aliases.append(dict(id='priority:'+entry.get('id',name),candidate=name,linked_rows=['research:'+x for x in links],meaning='Coverage mapping, not another run'))
            continue
        sources=[]
        if entry.get('source_binding'):sources=[entry['source_binding']]
        elif entry.get('definition_source'):sources=[entry['definition_source']]
        for source in entry.get('sources',[]):
            if source.get('directory'):sources.append(dict(directory=source['directory'],files=source.get('recorded_sha256',{}),recorded_receipt=source.get('receipt')))
        records.append(dict(id='priority:'+entry.get('id',name),candidate=name,entry=entry,sources=sources,priority=0 if name in ('KA2','R2','B3-belief','SG3','B2','G1','B3 guarded reseed') else 2))
    # Legacy PR labels remain visible; never substitute a nearby current mechanism.
    for entry in inventory['legacy_pr_summary_entries']:
        records.append(dict(id=entry['id'],candidate=entry['candidate'],entry=entry,sources=[],priority=4))
    for entry in inventory['late_research_recovery']:
        config=next((x for x in entry['source_and_result_receipts'] if Path(x['source']).name=='config.json'),None)
        sources=[]
        if config:
            parent=Path(config['source']).parent
            files={Path(x['source']).name:x['sha256'] for x in entry['source_and_result_receipts'] if Path(x['source']).parent==parent}
            sources=[dict(directory=str(parent),files=files,recorded_receipt=entry['source_and_result_receipts'])]
        records.append(dict(id='late-research:'+entry['candidate'],candidate=entry['candidate'],entry=entry,sources=sources,priority=4))

    runtime_cache={};runtime_groups={};probe_groups=defaultdict(list);rows=[]
    for record in records:
        row={k:record[k] for k in ('id','candidate','priority')};entry=record['entry']
        row.update(status='PENDING_EXACT_SOURCE_BINDING',new_quality='NOT_RUN',prior_scores_inherited=False,historical_eligibility=entry.get('historical_continuous_eligibility',entry.get('eligibility','UNVERIFIED')))
        sources=record['sources']
        if entry.get('definition') and entry.get('source_archive_receipt'):
            definition=entry['definition'];receipt=Path(entry['source_archive_receipt']['path'])
            assert sha(receipt)==entry['source_archive_receipt']['sha256']
            original=Path(entry['source_receipt']['path']);assert sha(original)==entry['source_receipt']['sha256']
            archive_record=read(receipt)[definition['name']]
            row.update(status='EXACT_PACKAGE_DEFINITION_REQUIRES_DIFFERENT_ADAPTER',definition=definition,source_receipt=entry['source_receipt'],source_archive_receipt=entry['source_archive_receipt'],archive_record=archive_record,reason='Original package plus explicit config/options is defined; it is not the shared six-hook probe interface. Preserve loss and optimizer options when preparing its adapter.')
            rows.append(row);continue
        if len(sources)>1:
            core=('config.json','mechanism.py','latent.py','response.py')
            shared={tuple(s.get('files',{}).get(name) for name in core) for s in sources}
            probe_sources=[s for s in sources if 'probe.py' in s.get('files',{})]
            if len(shared)==1 and None not in next(iter(shared)) and len(probe_sources)==1:
                row['equivalent_task_source_bindings']=sources
                row['source_selection']='Same exact config/mechanism/latent/response across task directories; select the unique retained probe interface for the tiny screen.'
                sources=probe_sources
        if len(sources)!=1:
            row['reason']='No unique exact source directory binding';rows.append(row);continue
        source=sources[0];directory=Path(source['directory']);row['source_directory']=str(directory)
        if not directory.exists():row['reason']='Original directory unavailable; archived candidate source requires restoration';rows.append(row);continue
        recorded=source.get('files',{})
        if any(not (directory/n).exists() or sha(directory/n)!=want for n,want in recorded.items()):
            row['status']='BLOCKED_SOURCE_HASH_MISMATCH';rows.append(row);continue
        files,missing=source_closure(directory,HARNESS/'ka2-source/probe.py' if record['candidate']=='KA2' else None)
        hashes={n:sha(p) for n,p in files.items()};row['candidate_files']=hashes;row['missing_required_files']=missing
        row['active_local_sources']=active_local_sources(files)
        row['active_learner_digest']=digest(row['active_local_sources'])
        row['source_binding']=dict(recorded_files_verified=recorded,availability_snapshot=source.get('immutable_availability_snapshot'),receipt=source.get('recorded_receipt'),scope='Exact retained candidate definition; measurement provenance remains in original inventory')
        if record['candidate']=='KA2':row['probe_binding']='Exact shared parent probe staged as in independently reviewed KA2 preparation; not a missing algorithm substitution'
        probe=hashes.get('probe.py','MISSING');probe_groups[probe].append(row['id']);row['probe_sha256']=probe
        concerns,imports=init_binding_concerns(files);row['constructor_binding_concerns']=concerns;row['hook_imports']=imports
        row['configuration']=read(files['config.json']) if 'config.json' in files else None
        runtime_refs=find_runtimes(directory,entry);row['historical_runtime_receipts']=runtime_refs
        roots=sorted({v['runtime'] for v in runtime_refs})
        # Current common benchmark is separately declared. It is not evidence that
        # an unbound historical result used this runtime.
        if not roots:
            roots=[template['historical_runtime']]
            row['runtime_binding']='DECLARED_COMMON_RETEST_HOST; HISTORICAL_RUNTIME_RECEIPT_UNRESOLVED'
        else:row['runtime_binding']='EXPLICIT_RETAINED_RUNTIME_RECEIPT'
        identities=[]
        for runtime in roots:
            if runtime not in runtime_cache:
                root=Path(runtime);available={rel:sha(root/rel) for rel in template['historical_runtime_files'] if (root/rel).exists()}
                missing_runtime=sorted(set(template['historical_runtime_files'])-set(available))
                runtime_cache[runtime]=dict(files=available,digest=digest(available),missing=missing_runtime,canonical_equal=available==template['historical_runtime_files'])
            info=runtime_cache[runtime];identities.append(info['digest'])
            runtime_groups.setdefault(info['digest'],dict(paths=[],files=info['files'],missing=info['missing'],canonical_equal=info['canonical_equal']))
            if runtime not in runtime_groups[info['digest']]['paths']:runtime_groups[info['digest']]['paths'].append(runtime)
        row['runtime_digests']=sorted(set(identities));row['full_frozen_host_sha256']=template['frozen_host_sha256'] if all(runtime_cache[x]['canonical_equal'] for x in roots) else None
        row['binding_group']=digest(dict(probe=probe,runtime=row['runtime_digests'],host=row['full_frozen_host_sha256'],prior_registration=hashes.get('response.py'),latent=hashes.get('latent.py')))
        simple=(not missing and probe==template['candidate_files']['probe.py'] and hashes.get('response.py')==template['candidate_files']['response.py'] and hashes.get('latent.py')==template['candidate_files']['latent.py'] and hashes.get('checkpoint.py')==template['candidate_files']['checkpoint.py'] and not concerns and all(runtime_cache[x]['canonical_equal'] for x in roots))
        if simple:
            row['status']='COMPATIBLE_SOURCE_GROUP_REQUIRES_CANDIDATE_CONSTRUCTOR_REVIEW'
            row['benchmark_runtime']=template['historical_runtime']
        else:row['status']='PENDING_ADAPTER_OR_BINDING_REVIEW'
        if entry.get('classification')=='UNTESTED_DRAFT':row['status']='OUT_OF_SCOPE_UNTESTED_DRAFT';row['reason']='Retained original draft is not an already-tested leaderboard candidate; no launch under this retest queue.'
        # Preserve definition snapshots; unused helper files remain identifiable.
        archive_dir=HERE/'candidate-sources';archive_dir.mkdir(exist_ok=True)
        identity=digest(hashes);archive=archive_dir/(identity+'.zip')
        if not archive.exists():
            with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
                for name,path in sorted(files.items()):
                    info=zipfile.ZipInfo(name,(2026,9,27,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,path.read_bytes())
        with zipfile.ZipFile(archive) as z:
            assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}==hashes
        row['candidate_archive']=pin(archive);row['source_definition_digest']=identity
        rows.append(row)
    seen={}
    for row in sorted(rows,key=lambda r:(r['priority'],r['id'])):
        if 'binding_group' not in row:continue
        # Only exact complete source/config identity is deduplicated. Same label
        # or same mechanism with a different schedule is never enough.
        key=(row['binding_group'],row['active_learner_digest'])
        if key in seen:row['exact_duplicate_of']=seen[key]
        else:seen[key]=row['id']
    groups=defaultdict(list)
    for row in rows:
        if 'binding_group' in row:groups[row['binding_group']].append(row['id'])
    result=dict(schema=1,status='SOURCE_QUEUE_ONLY_NOT_TRAINING_AUTHORIZATION',inventory=pin(INVENTORY/'inventory.json'),initializer_template_manifest=pin(HARNESS/'manifest.json'),rows=rows,aliases=aliases,probe_groups=dict(probe_groups),binding_groups=dict(groups),runtime_groups=runtime_groups,counts=dict(rows=len(rows),aliases=len(aliases),by_status=dict(Counter(x['status'] for x in rows)),probe_groups=len(probe_groups),binding_groups=len(groups)),policy=['Use one fresh process per candidate; original global research hooks cannot share a process.','Keep original configuration and all declared schedule/noise policies. Source eligibility remains separate from retest quality.','Only an exact full host/runtime/probe/prior binding may reuse the tiny initialization bridge.','Common retest runtime selection is explicit where historical runtime provenance is unresolved; do not claim a pure historical initialization ablation.','No old weights or inherited quality; all candidate constructors require their own source-bound proof before execution.'])
    dump(HERE/'queue.json',result)
    lines=['# Historical research retest source queue','','This source-only queue groups exact probe, runtime, frozen host and prior-registration contracts. Group compatibility is not a quality result or launch authorization. The first group can reuse the reviewed tiny initialization bridge after each candidate receives a matching constructor/source receipt.','',f"{len(rows)} definition/recovery rows and {len(aliases)} linked aliases; {len(probe_groups)} probe identities and {len(groups)} full binding groups. Exact source duplicates are linked, while different configurations remain separate.",'','| Candidate | Identity | Source binding status |','|---|---|---|']
    for row in sorted(rows,key=lambda r:(r['priority'],r['id'])):lines.append(f"| {row['candidate']} | {row['id']} | {row['status']} |")
    lines+=['','All source/config hashes, retained archives, runtime receipts, explicit unresolved bindings and group membership are in [queue.json](queue.json). A runtime found in a launch receipt is distinguished from a newly declared common retest host. No score selects the host. All historical source/eligibility and score records remain intact.']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['counts'],indent=2))

if __name__=='__main__':main()
