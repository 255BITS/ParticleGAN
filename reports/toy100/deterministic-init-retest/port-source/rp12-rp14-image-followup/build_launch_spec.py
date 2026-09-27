"""Bind the fixed RP12/RP14/RP15 image batch to completed independent reviews."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PORT=ROOT/'port-source'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
harness=PORT/'new-init-image-screen-logging-v2'
files=read(harness/'bundle-sha256.json')
assert all(sha(harness/n)==h for n,h in files.items())
files={**files,'bundle-sha256.json':sha(harness/'bundle-sha256.json')}
sources={'probe':dict(root=str(harness),files=files), 'binding':dict(root=str(HERE),files={n:sha(HERE/n) for n in ['source-binding.json','source-binding.md']})}
reviews=[];cases=[]
for i,name in enumerate(['api-rp12','api-rp14','api-rp15']):
    port=PORT/name
    declaration=read(port/'candidate-declaration.json')
    cpu_dir=ROOT/'image-screen-review'/(name+'-intensity2-cpu')
    cpu=read(cpu_dir/'cpu-preflight.json')
    review_path=ROOT/'image-screen-review'/(name+'-image-source-review.json')
    review=read(review_path)
    assert cpu['status']=='PASS_CPU_ZERO_STEP' and cpu['cuda_initialized'] is False
    assert cpu['candidate_declaration_sha256']==sha(port/'candidate-declaration.json')
    assert cpu['source_sha256']['bundle-sha256.json']==files['bundle-sha256.json']
    assert cpu['task']=='img_intensity2'
    assert review['status'].startswith('PASS'), review['status']
    sources[f'case{i}']=dict(root=str(port),files={**{'package/'+n:h for n,h in declaration['package_sha256'].items()},'candidate-declaration.json':sha(port/'candidate-declaration.json')})
    sources[f'cpu{i}']=dict(root=str(cpu_dir),files={'cpu-preflight.json':sha(cpu_dir/'cpu-preflight.json')})
    sources[f'review{i}']=dict(root=str(review_path.parent),files={review_path.name:sha(review_path),review_path.with_suffix('.md').name:sha(review_path.with_suffix('.md'))})
    reviews.append(dict(path=str(review_path),sha256=sha(review_path),required_status=review['status']))
    cases.append(dict(id=name+'-img-intensity2',argv=['/tmp/pr38-default-env/bin/python','{inputs}/probe/image_screen.py','--package-root',f'{{inputs}}/case{i}/package','--declaration',f'{{inputs}}/case{i}/candidate-declaration.json','--task','img_intensity2','--cpu-receipt',f'{{inputs}}/cpu{i}/cpu-preflight.json','--output',f'{{output}}/{name}-img-intensity2']))
output=dict(status='REVIEWED_READY_FOR_EXTERNAL_RUN',lane='precision-three-intensity2',sources=sources,independent_reviews=reviews,cases=cases,instructions='Run exactly these three unchanged candidates sequentially, one GPU worker. Each own img_intensity2 runs all600 public updates and all24 observations. Use exact package and new deterministic initialization; frozen image dimensions, scoring and sample streams; full-step serial context. Preserve source, own CPU construction binding, actual CUDA initial-state equality, every accepted sample cursor and rate, game correction/controller telemetry, initial/final checkpoints and eager parameter-device clocks. RP12/RP14 have exact old image failures; RP15 has no old image result. Do not inherit quality or alter original mode_hold scores. No retries, seed changes, learner fixes, replacement models, additional tasks, extra windows, coefficient search or early score-based stopping. Preserve ERROR as terminal evidence if a runtime guard fails.')
(HERE/'launch-spec.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(path=str(HERE/'launch-spec.json'),sha256=sha(HERE/'launch-spec.json'),cases=len(cases))))
