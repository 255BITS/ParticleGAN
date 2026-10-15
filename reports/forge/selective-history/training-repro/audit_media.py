"""Read-only provenance audit of derived media; does not render or evaluate models."""
from pathlib import Path
import json,hashlib,torch
b=Path('/ml2/hypergan/combination-training-20261015');idx=json.load(open(b/'media/index.json'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
 if torch.is_tensor(a) and torch.is_tensor(b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
 if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b
assert idx['script_sha256']==sha(b/'render_media.py');alignments=[];omissions=[]
for row in idx['attempts']:
 d=b/'queue'/row['campaign']/row['attempt_id'];result=json.load(open(d/'result.json'))
 if not row['media']:
  assert result['raw']['attempt_status'] in ['timeout','error','cancelled'];assert all(r['gate_status']=='INCOMPLETE' for r in result['task_results']);assert not list(Path(row['output_directory']).rglob('*.gif'))
  omissions.append({'attempt':row['attempt_id'],'reason':result['raw']['attempt_status'],'gate':'INCOMPLETE','gif_count':0});continue
 raw=json.load(open(d/'raw-result.json'));observed=raw['evidence'].get('observations',[])
 for m in row['media']:
  dest=Path(row['output_directory']);assert sha(dest/(m['task_id']+'.gif'))==m['gif_sha256'];assert m['postprocessor_sha256']==idx['script_sha256'];assert not m['qualification_input'] and m['optimizer_updates_added']==m['sampling_draws_added']==0
  assert json.load(open(dest/(m['task_id']+'.json')))==m
  assert m['recorded_grade']==next(r['gate_status'] for r in result['task_results'] if r['task_id']==m['task_id'])
  for src,h in m['source_inputs'].items():assert sha(src)==h
  for alignment in m['scored_frame_alignment']:
   values=torch.load(alignment['source_path'],map_location='cpu',weights_only=True);assert len(values)==alignment['source_frame_count'];selected=[]
   for ob in observed:
    matches=[i for i,v in enumerate(values) if v.get('step')==ob['step'] and all(v.get('metrics',{}).get(k)==z for k,z in ob.items() if k!='step')];assert matches and all(same(values[matches[0]],values[i]) for i in matches);selected.append(matches[0])
   assert selected==alignment['selected_source_indices'];alignments.append({'attempt':row['attempt_id'],'task':m['task_id'],'selected_indices':selected,'original_frames':len(values),'aligned_frames':len(selected)})
report={'status':'PASS_RENDERING_PROVENANCE_AUDIT','attempts':len(idx['attempts']),'gifs_verified':sum(len(r['media']) for r in idx['attempts']),'media_omissions':omissions,'postprocessor_sha256':idx['script_sha256'],'source_hashes_verified':True,'derived_sidecars_exact':True,'alignment_maps_independently_verified':alignments,'model_evaluations_or_updates_added':0,'visual_qa':'root owns visual inspection'}
(b/'independent-media-audit.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,indent=2))
