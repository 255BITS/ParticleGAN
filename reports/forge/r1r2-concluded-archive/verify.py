"""Verify archived Git identities and preserved reports without training or regrading."""
import hashlib,json,subprocess,re,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(root))
out=Path(__file__).resolve().parent
a=json.loads((out/'archive.json').read_text());s=a['source_commit']
def git(*args): return subprocess.check_output(['git',*args],cwd=root)
def read(path): return git('show',f'{s}:{path}')
assert git('rev-parse',s+'^{tree}').decode().strip()==a['source_root_tree']
assert git('rev-parse',a['archive_ref']).decode().strip()==s
pins=0; gifs=0; media_bindings=0; sidecars=0
for entry in a['reports']:
 assert git('rev-parse',s+':'+entry['path']).decode().strip()==entry['git_tree']
 paths=git('ls-tree','-r','--name-only',s,entry['path']).decode().splitlines()
 assert len(paths)==entry['tracked_files']
 for p in paths:
  if p.endswith('.gif'):
   assert read(p)[:6] in (b'GIF87a',b'GIF89a');gifs+=1
 for pin in entry['pins']:
  data=read(pin['path']); assert len(data)==pin['bytes'];assert hashlib.sha256(data).hexdigest()==pin['sha256']
  assert git('rev-parse',s+':'+pin['path']).decode().strip()==pin['git_blob'];pins+=1
  if pin['path'].endswith('/results.json'):
   pub=json.loads(data)
   for m in pub.get('media',[]):
    assert hashlib.sha256(read(m['path'])).hexdigest()==m['sha256'];media_bindings+=1
    if 'receipt' in m:
     sidecar=str(Path(m['path']).with_suffix('.json'));receipt=read(sidecar)
     assert hashlib.sha256(receipt).hexdigest()==m['receipt_sha256']
     assert json.loads(receipt)==m['receipt']; sidecars+=1
readme=(out/'README.md').read_text();links=0
for kind,commit,path in re.findall(r'https://github.com/255BITS/ParticleGAN/(blob|tree)/([0-9a-f]{40})/([^\s)]+)',readme):
 git('cat-file','-e',commit+':'+path);links+=1
record=json.loads((root/'reports/forge/records/r1r2-pr401-concluded-archive-v1.json').read_text())
assert record['qualification_input'] is False and record['qualification_reuse'] is False and record['task_results']==[]
for binding in [record['source'],record['provenance']['archive_index']]:assert hashlib.sha256((root/binding['path']).read_bytes()).hexdigest()==binding['sha256']
baseline=sys.argv[1] if len(sys.argv)>1 else json.loads((out/'validation.json').read_text())['develop_base']
base=git('rev-parse',baseline).decode().strip();saved={}
for p in git('ls-tree','-r','--name-only',base,'reports/forge').decode().splitlines():
 if ('/leaderboards/' in p or p in ['reports/forge/technique-inventory.md','reports/forge/technique-inventory.json','reports/forge/publication-records.json']):
  data=git('show',base+':'+p);assert (root/p).read_bytes()==data;saved[p]=hashlib.sha256(data).hexdigest()
from experiments.forge.knowledge import freshness
fresh=freshness(root);assert fresh['status']=='CURRENT'
r={'schema_version':1,'scope':'archive_identity_and_reporting_validation','source_commit':s,'develop_base':base,'qualification_input':False,'original_gates_recomputed':False,'archive_root_tree_verified':True,'report_trees_verified':len(a['reports']),'document_result_blob_pins_verified':pins,'gif_objects_readable':gifs,'publication_gif_hash_bindings_verified':media_bindings,'publication_media_sidecars_verified':sidecars,'pinned_markdown_links_verified':links,'local_record_bindings_verified':True,'unchanged_published_artifacts_sha256':saved,'compilation_status':'CURRENT','publication_record_count_preserved':fresh['publication_record_count'],'original_inventory_coverage':{'valid':fresh['inventory_coverage']['valid'],'missing_count':len(fresh['inventory_coverage']['missing']),'unclassified_count':len(fresh['inventory_coverage']['unclassified']),'note':'Existing develop catalog drift; no unrelated history rebuild performed.'},'cleanup_training_updates':0,'cleanup_sampling_draws':0,'cleanup_paid_seconds':0,'result':'PASS'}
(out/'validation.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='unchanged_published_artifacts_sha256'},indent=2))
